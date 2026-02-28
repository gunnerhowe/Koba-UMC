/*
 * UMC C Extension Kernels
 * High-performance compression primitives for the Universal Manifold Codec.
 *
 * These kernels replace the Python/NumPy hot loops with cache-friendly C
 * implementations. Compiled to a shared library and loaded via ctypes.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o _umc_kernels.so kernels.c
 *   cl /O2 /LD kernels.c /Fe_umc_kernels.dll          (MSVC)
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
  #define UMC_EXPORT __declspec(dllexport)
#else
  #define UMC_EXPORT __attribute__((visibility("default")))
#endif

/* ========================================================================
 * Byte Transpose / Untranspose
 *
 * Reorder bytes so that byte-lane 0 of every element is contiguous,
 * then byte-lane 1, etc.  This dramatically improves entropy coding
 * because sign+exponent bytes (lane 3 for float32) cluster together.
 *
 * Normal:     [B0 B1 B2 B3] [B0 B1 B2 B3] ...
 * Transposed: [B0 B0 B0 ...] [B1 B1 B1 ...] [B2 B2 ...] [B3 B3 ...]
 * ====================================================================== */

UMC_EXPORT void byte_transpose(
    const uint8_t *src, uint8_t *dst,
    int64_t n_elements, int element_size)
{
    int64_t i;
    int b;

    if (element_size == 4) {
        /* Specialized fast path for float32 (most common case). */
        const int64_t n = n_elements;
        for (i = 0; i < n; i++) {
            dst[          i] = src[i * 4    ];
            dst[    n + i] = src[i * 4 + 1];
            dst[2 * n + i] = src[i * 4 + 2];
            dst[3 * n + i] = src[i * 4 + 3];
        }
    } else if (element_size == 2) {
        /* Fast path for float16 / int16. */
        const int64_t n = n_elements;
        for (i = 0; i < n; i++) {
            dst[      i] = src[i * 2    ];
            dst[n + i] = src[i * 2 + 1];
        }
    } else {
        /* Generic path. */
        for (i = 0; i < n_elements; i++) {
            for (b = 0; b < element_size; b++) {
                dst[b * n_elements + i] = src[i * element_size + b];
            }
        }
    }
}

UMC_EXPORT void byte_untranspose(
    const uint8_t *src, uint8_t *dst,
    int64_t n_elements, int element_size)
{
    int64_t i;
    int b;

    if (element_size == 4) {
        const int64_t n = n_elements;
        for (i = 0; i < n; i++) {
            dst[i * 4    ] = src[          i];
            dst[i * 4 + 1] = src[    n + i];
            dst[i * 4 + 2] = src[2 * n + i];
            dst[i * 4 + 3] = src[3 * n + i];
        }
    } else if (element_size == 2) {
        const int64_t n = n_elements;
        for (i = 0; i < n; i++) {
            dst[i * 2    ] = src[      i];
            dst[i * 2 + 1] = src[n + i];
        }
    } else {
        for (i = 0; i < n_elements; i++) {
            for (b = 0; b < element_size; b++) {
                dst[i * element_size + b] = src[b * n_elements + i];
            }
        }
    }
}

/* ========================================================================
 * Delta Encoding / Decoding (Order 1-4)
 *
 * Operates on (N, W, F) arrays stored in row-major order as int32.
 * Delta-encodes along the W (time) axis for each (n, f) lane.
 *
 *   encode: r[t] = x[t] - x[t-1],  r[0] = x[0]
 *   decode: x[t] = x[t-1] + r[t]   (cumulative sum, sequential!)
 * ====================================================================== */

UMC_EXPORT void delta_encode_order1(
    const int32_t *src, int32_t *dst,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t stride_w = n_features;                   /* offset between t and t+1 */
    int64_t stride_n = window_size * n_features;     /* offset between windows */

    for (n = 0; n < n_windows; n++) {
        const int32_t *in  = src + n * stride_n;
        int32_t       *out = dst + n * stride_n;

        /* First timestep: copy as-is */
        memcpy(out, in, n_features * sizeof(int32_t));

        /* Remaining timesteps: delta */
        for (t = 1; t < window_size; t++) {
            const int32_t *cur  = in  + t * stride_w;
            const int32_t *prev = in  + (t - 1) * stride_w;
            int32_t       *o    = out + t * stride_w;
            for (f = 0; f < n_features; f++) {
                o[f] = cur[f] - prev[f];
            }
        }
    }
}

UMC_EXPORT void delta_decode_order1(
    int32_t *data,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    /* In-place cumulative sum along W axis. Sequential — cannot be parallelized. */
    int64_t n, t, f;
    int64_t stride_w = n_features;
    int64_t stride_n = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        int32_t *d = data + n * stride_n;
        for (t = 1; t < window_size; t++) {
            int32_t *cur  = d + t * stride_w;
            int32_t *prev = d + (t - 1) * stride_w;
            for (f = 0; f < n_features; f++) {
                cur[f] += prev[f];
            }
        }
    }
}

UMC_EXPORT void delta_encode_order2(
    const int32_t *src, int32_t *dst,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        const int32_t *in  = src + n * sn;
        int32_t       *out = dst + n * sn;

        memcpy(out, in, sw * sizeof(int32_t));   /* t=0 */
        if (window_size > 1) {
            for (f = 0; f < n_features; f++)
                out[sw + f] = in[sw + f] - in[f];   /* t=1: order-1 delta */
        }
        for (t = 2; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                int64_t predicted = 2 * (int64_t)in[(t-1)*sw + f]
                                  -     (int64_t)in[(t-2)*sw + f];
                out[t*sw + f] = (int32_t)((int64_t)in[t*sw + f] - predicted);
            }
        }
    }
}

UMC_EXPORT void delta_decode_order2(
    int32_t *data,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        int32_t *d = data + n * sn;
        if (window_size > 1) {
            for (f = 0; f < n_features; f++)
                d[sw + f] += d[f];
        }
        for (t = 2; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                int64_t predicted = 2 * (int64_t)d[(t-1)*sw + f]
                                  -     (int64_t)d[(t-2)*sw + f];
                d[t*sw + f] = (int32_t)((int64_t)d[t*sw + f] + predicted);
            }
        }
    }
}

UMC_EXPORT void delta_decode_order3(
    int32_t *data,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        int32_t *d = data + n * sn;
        if (window_size > 1) {
            for (f = 0; f < n_features; f++)
                d[sw + f] += d[f];
        }
        if (window_size > 2) {
            for (f = 0; f < n_features; f++) {
                int64_t predicted = 2 * (int64_t)d[sw + f] - (int64_t)d[f];
                d[2*sw + f] = (int32_t)((int64_t)d[2*sw + f] + predicted);
            }
        }
        for (t = 3; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                int64_t predicted = 3 * (int64_t)d[(t-1)*sw + f]
                                  - 3 * (int64_t)d[(t-2)*sw + f]
                                  +     (int64_t)d[(t-3)*sw + f];
                d[t*sw + f] = (int32_t)((int64_t)d[t*sw + f] + predicted);
            }
        }
    }
}

UMC_EXPORT void delta_decode_order4(
    int32_t *data,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        int32_t *d = data + n * sn;
        if (window_size > 1) {
            for (f = 0; f < n_features; f++)
                d[sw + f] += d[f];
        }
        if (window_size > 2) {
            for (f = 0; f < n_features; f++) {
                int64_t pred = 2*(int64_t)d[sw+f] - (int64_t)d[f];
                d[2*sw+f] = (int32_t)((int64_t)d[2*sw+f] + pred);
            }
        }
        if (window_size > 3) {
            for (f = 0; f < n_features; f++) {
                int64_t pred = 3*(int64_t)d[2*sw+f] - 3*(int64_t)d[sw+f] + (int64_t)d[f];
                d[3*sw+f] = (int32_t)((int64_t)d[3*sw+f] + pred);
            }
        }
        for (t = 4; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                int64_t pred = 4*(int64_t)d[(t-1)*sw+f]
                             - 6*(int64_t)d[(t-2)*sw+f]
                             + 4*(int64_t)d[(t-3)*sw+f]
                             -   (int64_t)d[(t-4)*sw+f];
                d[t*sw+f] = (int32_t)((int64_t)d[t*sw+f] + pred);
            }
        }
    }
}

/* ========================================================================
 * XOR Delta Encoding / Decoding
 * ====================================================================== */

UMC_EXPORT void xor_encode(
    const uint32_t *src, uint32_t *dst,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        const uint32_t *in  = src + n * sn;
        uint32_t       *out = dst + n * sn;
        memcpy(out, in, sw * sizeof(uint32_t));
        for (t = 1; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                out[t*sw + f] = in[t*sw + f] ^ in[(t-1)*sw + f];
            }
        }
    }
}

UMC_EXPORT void xor_decode(
    uint32_t *data,
    int64_t n_windows, int64_t window_size, int64_t n_features)
{
    int64_t n, t, f;
    int64_t sw = n_features;
    int64_t sn = window_size * n_features;

    for (n = 0; n < n_windows; n++) {
        uint32_t *d = data + n * sn;
        for (t = 1; t < window_size; t++) {
            for (f = 0; f < n_features; f++) {
                d[t*sw + f] ^= d[(t-1)*sw + f];
            }
        }
    }
}

/* ========================================================================
 * Zigzag Encoding / Decoding
 *
 * Maps signed integers to unsigned:  0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ...
 * This makes small residuals (positive or negative) into small unsigned
 * values, which compress better.
 * ====================================================================== */

UMC_EXPORT void zigzag_encode(
    const int32_t *src, uint32_t *dst, int64_t count)
{
    int64_t i;
    for (i = 0; i < count; i++) {
        int32_t v = src[i];
        dst[i] = (uint32_t)((v >> 31) ^ (v << 1));
    }
}

UMC_EXPORT void zigzag_decode(
    const uint32_t *src, int32_t *dst, int64_t count)
{
    int64_t i;
    for (i = 0; i < count; i++) {
        uint32_t v = src[i];
        dst[i] = (int32_t)((v >> 1) ^ -(int32_t)(v & 1));
    }
}

/* ========================================================================
 * Version / Capability Check
 * ====================================================================== */

UMC_EXPORT int umc_kernels_version(void)
{
    return 1;  /* Increment when ABI changes */
}
