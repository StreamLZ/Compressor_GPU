// Host driver for the 2026-06-10 tANS upgrade variants (see
// tans_upgrade_kernels.cuh). Builds the permuted expected output for
// the 4-byte output interleave, transcodes the snapshot's sub-streams
// into the BIL word-interleaved layout, verifies both kernels
// byte-exact, and benches them. Returns nonzero on verify failure.

#define UCK(call) do { \
    cudaError_t ue_ = (call); \
    if (ue_ != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ue_)); \
        return 1; \
    } \
} while (0)

static int runUpgradeBenches(
    const uint8_t* comp, uint32_t comp_size,
    const TansDecChunkDesc* descs, uint32_t n_chunks,
    const uint8_t* ref, uint64_t total_dst,
    const uint8_t* d_comp, const TansDecChunkDesc* d_descs,
    uint32_t* d_status, TansLutEnt* d_lut, const TansTableMeta* d_meta)
{
    dim3 block(32, 2);
    dim3 grid((n_chunks + 1) / 2);
    const int N_RUNS = 50;

    // ── Output slots: one 128-aligned slot per chunk in a fresh buffer ──
    uint32_t max_dst = 0;
    for (uint32_t i = 0; i < n_chunks; i++)
        if (descs[i].dst_size > max_dst) max_dst = descs[i].dst_size;
    uint32_t slot = (max_dst + 127u) & ~127u;
    uint64_t out2_bytes = (uint64_t)n_chunks * slot;

    uint32_t* new_off = (uint32_t*)malloc(sizeof(uint32_t) * n_chunks);
    for (uint32_t i = 0; i < n_chunks; i++) new_off[i] = i * slot;

    // Expected output in the new layout: q = (i/4)*128 + lane*4 + (i%4)
    // where the old logical position is p = i*32 + lane. Holes (when
    // dst_size % 128 != 0) stay zero in both buffers.
    uint8_t* exp2 = (uint8_t*)calloc(out2_bytes, 1);
    {
        uint64_t old_off = 0;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint32_t dsz = descs[c].dst_size;
            uint8_t* dstc = exp2 + (uint64_t)c * slot;
            const uint8_t* srcc = ref + old_off;
            for (uint32_t p = 0; p < dsz; p++) {
                uint32_t i = p >> 5, lane = p & 31u;
                uint32_t q = (i >> 2) * 128u + lane * 4u + (i & 3u);
                dstc[q] = srcc[p];
            }
            old_off += dsz;
        }
    }

    uint8_t* d_out2 = NULL;
    uint32_t* d_newoff = NULL;
    UCK(cudaMalloc(&d_out2, out2_bytes));
    UCK(cudaMalloc(&d_newoff, sizeof(uint32_t) * n_chunks));
    UCK(cudaMemcpy(d_newoff, new_off, sizeof(uint32_t) * n_chunks, cudaMemcpyHostToDevice));
    uint8_t* gpu2 = (uint8_t*)malloc(out2_bytes);
    uint32_t* status2 = (uint32_t*)malloc(sizeof(uint32_t) * n_chunks);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // ── Variant (a): u32 stores, original comp layout ──
    UCK(cudaMemset(d_out2, 0, out2_bytes));
    UCK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    slzTans32DecodeU32StoreKernel<<<grid, block>>>(d_comp, d_out2, d_descs, d_status, n_chunks, d_lut, d_meta, d_newoff);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { printf("u32store launch error: %s\n", cudaGetErrorString(le)); return 1; }
    UCK(cudaDeviceSynchronize());
    UCK(cudaMemcpy(gpu2, d_out2, out2_bytes, cudaMemcpyDeviceToHost));
    UCK(cudaMemcpy(status2, d_status, sizeof(uint32_t) * n_chunks, cudaMemcpyDeviceToHost));
    {
        int sb = 0, mm = 0;
        for (uint32_t c = 0; c < n_chunks; c++) {
            if (status2[c] != TANS_OK) sb++;
            uint32_t cmp_len = (descs[c].dst_size + 127u) & ~127u;
            if (cmp_len > slot) cmp_len = slot;
            if (memcmp(gpu2 + (uint64_t)c * slot, exp2 + (uint64_t)c * slot, cmp_len) != 0) {
                if (mm < 3) {
                    uint32_t j = 0;
                    const uint8_t* g = gpu2 + (uint64_t)c * slot;
                    const uint8_t* x = exp2 + (uint64_t)c * slot;
                    while (j < cmp_len && g[j] == x[j]) j++;
                    printf("  u32store chunk %u mismatch at %u/%u: got %02x exp %02x\n",
                           c, j, descs[c].dst_size, g[j], x[j]);
                }
                mm++;
            }
        }
        printf("verify (u32store): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  -- BYTE-EXACT" : "");
        if (sb || mm) return 1;
    }
    {
        float best_ms = 1e9f;
        for (int r = 0; r < N_RUNS; r++) {
            cudaEventRecord(e0);
            slzTans32DecodeU32StoreKernel<<<grid, block>>>(d_comp, d_out2, d_descs, d_status, n_chunks, d_lut, d_meta, d_newoff);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best_ms) best_ms = ms;
        }
        printf("[tans32-u32store] %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
               n_chunks, total_dst / 1e6, best_ms, (double)total_dst / (best_ms * 1e6));
    }

    // ── Transcode comp into the BIL layout: [128B hdr][il K*128][tails] ──
    // Sub-streams live at meta.src_after_table_off (past the FSE prob
    // table), not at desc.src_offset — pull the device-built meta back.
    TansTableMeta* meta_h = (TansTableMeta*)malloc(sizeof(TansTableMeta) * n_chunks);
    UCK(cudaMemcpy(meta_h, d_meta, sizeof(TansTableMeta) * n_chunks, cudaMemcpyDeviceToHost));
    uint64_t src2_cap = (uint64_t)comp_size + (uint64_t)n_chunks * 384u;
    uint8_t* src2 = (uint8_t*)malloc(src2_cap);
    uint32_t* src2_off = (uint32_t*)malloc(sizeof(uint32_t) * n_chunks);
    uint32_t* Karr = (uint32_t*)malloc(sizeof(uint32_t) * n_chunks);
    uint64_t pos2 = 0;
    for (uint32_t c = 0; c < n_chunks; c++) {
        const uint8_t* hdr = comp + descs[c].src_offset - 128;
        const uint8_t* sub0 = comp + meta_h[c].src_after_table_off;
        uint16_t sizes[32];
        uint32_t offs[32];
        uint32_t run = 0;
        uint32_t K = 0xFFFFFFFFu;
        for (int l = 0; l < 32; l++) {
            sizes[l] = (uint16_t)hdr[l * 2] | ((uint16_t)hdr[l * 2 + 1] << 8);
            offs[l] = run;
            run += sizes[l];
            uint32_t wl = (uint32_t)sizes[l] >> 2;
            if (wl < K) K = wl;
        }
        if (K == 0xFFFFFFFFu) K = 0;
        pos2 = (pos2 + 127u) & ~127ull;          // 128-align the header
        src2_off[c] = (uint32_t)pos2;
        Karr[c] = K;
        memcpy(src2 + pos2, hdr, 128);
        pos2 += 128;
        uint8_t* il = src2 + pos2;
        for (uint32_t w = 0; w < K; w++)
            for (int l = 0; l < 32; l++)
                memcpy(il + (uint64_t)w * 128u + (uint32_t)l * 4u, sub0 + offs[l] + w * 4u, 4);
        pos2 += (uint64_t)K * 128u;
        for (int l = 0; l < 32; l++) {
            uint32_t tl = (uint32_t)sizes[l] - K * 4u;
            memcpy(src2 + pos2, sub0 + offs[l] + K * 4u, tl);
            pos2 += tl;
        }
    }
    printf("[bil-transcode] %u chunks, %.1f MB -> %.1f MB (il layout)\n",
           n_chunks, comp_size / 1e6, pos2 / 1e6);

    uint8_t* d_src2 = NULL;
    uint32_t* d_src2off = NULL;
    uint32_t* d_Karr = NULL;
    UCK(cudaMalloc(&d_src2, pos2));
    UCK(cudaMalloc(&d_src2off, sizeof(uint32_t) * n_chunks));
    UCK(cudaMalloc(&d_Karr, sizeof(uint32_t) * n_chunks));
    UCK(cudaMemcpy(d_src2, src2, pos2, cudaMemcpyHostToDevice));
    UCK(cudaMemcpy(d_src2off, src2_off, sizeof(uint32_t) * n_chunks, cudaMemcpyHostToDevice));
    UCK(cudaMemcpy(d_Karr, Karr, sizeof(uint32_t) * n_chunks, cudaMemcpyHostToDevice));

    // ── Variants (a)+(b): BIL refill + u32 stores ──
    UCK(cudaMemset(d_out2, 0, out2_bytes));
    UCK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    slzTans32DecodeBilKernel<<<grid, block>>>(d_src2, d_out2, d_descs, d_status, n_chunks, d_lut, d_meta, d_src2off, d_Karr, d_newoff);
    le = cudaGetLastError();
    if (le != cudaSuccess) { printf("bil launch error: %s\n", cudaGetErrorString(le)); return 1; }
    UCK(cudaDeviceSynchronize());
    UCK(cudaMemcpy(gpu2, d_out2, out2_bytes, cudaMemcpyDeviceToHost));
    UCK(cudaMemcpy(status2, d_status, sizeof(uint32_t) * n_chunks, cudaMemcpyDeviceToHost));
    {
        int sb = 0, mm = 0;
        for (uint32_t c = 0; c < n_chunks; c++) {
            if (status2[c] != TANS_OK) sb++;
            uint32_t cmp_len = (descs[c].dst_size + 127u) & ~127u;
            if (cmp_len > slot) cmp_len = slot;
            if (memcmp(gpu2 + (uint64_t)c * slot, exp2 + (uint64_t)c * slot, cmp_len) != 0) {
                if (mm < 3) {
                    uint32_t j = 0;
                    const uint8_t* g = gpu2 + (uint64_t)c * slot;
                    const uint8_t* x = exp2 + (uint64_t)c * slot;
                    while (j < cmp_len && g[j] == x[j]) j++;
                    printf("  bil chunk %u mismatch at %u/%u: got %02x exp %02x (K=%u)\n",
                           c, j, descs[c].dst_size, g[j], x[j], Karr[c]);
                }
                mm++;
            }
        }
        printf("verify (bil+u32): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  -- BYTE-EXACT" : "");
        if (sb || mm) return 1;
    }
    {
        float best_ms = 1e9f;
        for (int r = 0; r < N_RUNS; r++) {
            cudaEventRecord(e0);
            slzTans32DecodeBilKernel<<<grid, block>>>(d_src2, d_out2, d_descs, d_status, n_chunks, d_lut, d_meta, d_src2off, d_Karr, d_newoff);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best_ms) best_ms = ms;
        }
        printf("[tans32-bil+u32]  %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
               n_chunks, total_dst / 1e6, best_ms, (double)total_dst / (best_ms * 1e6));
    }

    // ── Parallel FSE table build (5-scan spread) vs production build ──
    // Verification is via the decode path: rebuild every LUT + meta with
    // the parallel kernel into fresh buffers, then run the (already
    // byte-exact) BIL decode against them — matching output proves the
    // tables identical in effect.
    {
        TansLutEnt* d_lut2 = NULL;
        TansTableMeta* d_meta2 = NULL;
        UCK(cudaMalloc(&d_lut2, (uint64_t)n_chunks * 2048 * sizeof(TansLutEnt)));
        UCK(cudaMalloc(&d_meta2, sizeof(TansTableMeta) * n_chunks));
        UCK(cudaMemset(d_lut2, 0, (uint64_t)n_chunks * 2048 * sizeof(TansLutEnt)));

        dim3 fse_grid((n_chunks + 1) / 2);
        slzTansFseBuildParKernel<<<fse_grid, block>>>((const uint8_t*)d_comp, d_descs, d_lut2, d_meta2, n_chunks, NULL);
        le = cudaGetLastError();
        if (le != cudaSuccess) { printf("fse-par launch error: %s\n", cudaGetErrorString(le)); return 1; }
        UCK(cudaDeviceSynchronize());

        UCK(cudaMemset(d_out2, 0, out2_bytes));
        UCK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
        slzTans32DecodeBilKernel<<<grid, block>>>(d_src2, d_out2, d_descs, d_status, n_chunks, d_lut2, d_meta2, d_src2off, d_Karr, d_newoff);
        UCK(cudaDeviceSynchronize());
        UCK(cudaMemcpy(gpu2, d_out2, out2_bytes, cudaMemcpyDeviceToHost));
        UCK(cudaMemcpy(status2, d_status, sizeof(uint32_t) * n_chunks, cudaMemcpyDeviceToHost));
        int sb = 0, mm = 0;
        for (uint32_t c = 0; c < n_chunks; c++) {
            if (status2[c] != TANS_OK) sb++;
            uint32_t cmp_len = (descs[c].dst_size + 127u) & ~127u;
            if (cmp_len > slot) cmp_len = slot;
            if (memcmp(gpu2 + (uint64_t)c * slot, exp2 + (uint64_t)c * slot, cmp_len) != 0) mm++;
        }
        printf("verify (fse-build-par via decode): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  -- BYTE-EXACT" : "");
        if (sb || mm) return 1;

        float best_ms = 1e9f;
        for (int r = 0; r < N_RUNS; r++) {
            cudaEventRecord(e0);
            slzTansFseBuildParKernel<<<fse_grid, block>>>((const uint8_t*)d_comp, d_descs, d_lut2, d_meta2, n_chunks, NULL);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best_ms) best_ms = ms;
        }
        printf("[fse-build-par] %u chunks -> best %.3f ms\n", n_chunks, best_ms);

        cudaFree(d_lut2); cudaFree(d_meta2);
    }

    cudaFree(d_out2); cudaFree(d_newoff); cudaFree(d_src2); cudaFree(d_src2off); cudaFree(d_Karr);
    free(new_off); free(exp2); free(gpu2); free(status2); free(src2); free(src2_off); free(Karr);
    return 0;
}
