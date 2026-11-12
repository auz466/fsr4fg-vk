/* AMD's shaders and weights, as tools/setup.py packed them (build/gen/assets.bin) */
    .section .rodata
    .global fsr4fg_assets
    .hidden fsr4fg_assets
    .type fsr4fg_assets, @object
    .balign 16
fsr4fg_assets:
    .incbin "assets.bin"
    .size fsr4fg_assets, . - fsr4fg_assets
    .section .note.GNU-stack,"",@progbits
