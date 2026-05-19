// Measure full-file Huffman ratio in 64 KB chunks (the production sub-chunk
// size). Each chunk gets its own canonical Huffman table; sum the encoded
// sizes plus per-chunk weight overhead (256 bytes worst case, less with
// canonical-code compression).
//
// Usage: file_ratio <file>

#define HUFF_REF_NO_MAIN
#include "huff_ref.c"

#define CHUNK_SIZE (64 * 1024)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size_t total = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *chunk = (uint8_t*)malloc(CHUNK_SIZE);
    uint8_t *enc = (uint8_t*)malloc(CHUNK_SIZE * 2 + 64);
    uint8_t *dec = (uint8_t*)malloc(CHUNK_SIZE + 64);
    uint32_t *lut = (uint32_t*)malloc(LUT_SIZE * sizeof(uint32_t));

    size_t total_raw = 0;
    size_t total_huff = 0;
    size_t total_weights = 0;
    size_t num_chunks = 0;
    int max_clen = 0;
    int fails = 0;

    while (1) {
        size_t got = fread(chunk, 1, CHUNK_SIZE, f);
        if (got == 0) break;

        uint32_t hist[256] = {0};
        for (size_t i = 0; i < got; i++) hist[chunk[i]]++;

        uint8_t code_lengths[256] = {0};
        uint32_t codes[256] = {0};
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) {
            height_limit(hist, code_lengths, MAX_CODE_LEN);
            max_len = 0;
            for (int s = 0; s < 256; s++) if (code_lengths[s] > max_len) max_len = code_lengths[s];
        }
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut);

        int enc_bytes = huff_encode_4s(chunk, got, code_lengths, codes,
                                        enc, CHUNK_SIZE * 2 + 64);
        if (enc_bytes < 0) { printf("encode overflow chunk %zu\n", num_chunks); fails++; break; }

        int dec_bytes = huff_decode_4s(enc, (size_t)enc_bytes, lut, dec, got);
        if (dec_bytes != (int)got || memcmp(chunk, dec, got) != 0) {
            printf("roundtrip fail chunk %zu\n", num_chunks);
            fails++;
            break;
        }

        // Weights overhead: 256 code lengths, each ≤ 4 bits → 128 bytes
        // (nibble-packed). Real implementation would compress this further
        // (zstd uses ~64 B average). Conservative: 128 B per chunk.
        size_t weights_overhead = 128;

        total_raw += got;
        total_huff += enc_bytes;
        total_weights += weights_overhead;
        num_chunks++;
        if (max_len > max_clen) max_clen = max_len;
    }

    fclose(f);
    free(chunk); free(enc); free(dec); free(lut);

    if (fails) { printf("%d failures\n", fails); return 1; }

    size_t total_with_weights = total_huff + total_weights;
    double huff_pct = 100.0 * total_huff / total_raw;
    double total_pct = 100.0 * total_with_weights / total_raw;

    printf("\n=== %s ===\n", argv[1]);
    printf("  size:          %zu bytes (%.1f MB)\n", total_raw, total_raw / 1e6);
    printf("  chunks (64KB): %zu\n", num_chunks);
    printf("  max code len:  %d\n", max_clen);
    printf("  Huffman 4-stream payload:  %zu bytes (%.3f%%)\n", total_huff, huff_pct);
    printf("  + per-chunk weights (128B): %zu bytes\n", total_weights);
    printf("  TOTAL with weights:        %zu bytes (%.3f%%)\n",
           total_with_weights, total_pct);
    return 0;
}
