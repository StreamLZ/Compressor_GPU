// Bridge Zig's ___chkstk_ms to MSVC's __chkstk
// Both do the same thing but Zig emits the MinGW name.
extern void __chkstk(void);
void ___chkstk_ms(void) {
    __chkstk();
}
