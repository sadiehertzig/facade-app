// NativePixels.c
//
// Two 8-bit image operations used to imitate what a photo goes through after
// it is shared online, so the protection can be checked against those copies
// (see RGBImage.handled in App/NativeTensor.swift):
//
//  1. facade_jpeg_roundtrip: save as a JPEG in memory and decode it again,
//     using libjpeg-turbo.
//  2. facade_resize_rgb: resize with exactly the arithmetic Pillow (the Python
//     imaging library) uses, so the phone produces the same bytes Pillow
//     would. For 8-bit images Pillow computes each filter weight in double,
//     turns it into a fixed-point integer (the weight times 2^22), adds the
//     weighted pixels as integers, then rounds and clamps. Copying those steps
//     exactly is what makes the output match byte for byte.
//
// Pixels are packed RGB8 (R, G, B bytes, row by row) throughout.
//
// Resampling coefficient and rounding conventions follow Pillow 12.3.0
// src/libImaging/Resample.c. See THIRD_PARTY_NOTICES.md and licenses/ at the
// repository root for its HPND license.
// jpeglib.h comes from libjpeg-turbo, built by app/ios/scripts/build_native_deps.sh.
#include "NativePixels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <setjmp.h>
#include <jpeglib.h>

// Accepts 1 to 8192 pixels on each side.
static int dimensions(int w, int h) { return w > 0 && h > 0 && w <= 8192 && h <= 8192; }
// libjpeg reports a fatal error by calling err->error_exit, which by default
// ends the whole program. `failure` extends libjpeg's error manager with a
// jmp_buf to jump back to and the caller's message buffer.
struct failure { struct jpeg_error_mgr base; jmp_buf jump; char *message; size_t capacity; };
// Replacement error_exit: copy libjpeg's message for the caller, then longjmp
// back to the setjmp in facade_jpeg_roundtrip, which cleans up and returns -3.
static void failed(j_common_ptr p) {
    struct failure *f = (struct failure *)p->err;
    char text[JMSG_LENGTH_MAX];
    (*p->err->format_message)(p, text);
    if (f->message && f->capacity) snprintf(f->message, f->capacity, "%s", text);
    longjmp(f->jump, 1);
}

// Encodes rgb (w x h) as a JPEG at `quality`, decodes it into `decoded`.
// Returns 0 on success, -1 for bad arguments, -2 if memory runs out, -3 if
// libjpeg failed or the decoded image did not have the expected size.
int facade_jpeg_roundtrip(const uint8_t *rgb, int w, int h, int quality,
                          uint8_t *decoded, char *error, size_t error_size) {
    if (!rgb || !decoded || rgb == decoded || !dimensions(w,h) || quality < 1 || quality > 100) return -1;
    // Everything libjpeg touches is on the heap. After a longjmp, C only
    // guarantees the values of local variables that were not changed since
    // setjmp. These pointers are set before setjmp and never changed after, so
    // the cleanup code can trust them; the data they point to lives on the heap.
    struct jpeg_compress_struct *enc = calloc(1, sizeof(*enc));
    struct jpeg_decompress_struct *dec = calloc(1, sizeof(*dec));
    struct failure *f = calloc(1, sizeof(*f));
    // `*bytes` will point at the compressed JPEG that jpeg_mem_dest allocates;
    // `length` is its size in bytes.
    unsigned char **bytes = calloc(1, sizeof(*bytes));
    unsigned long length = 0;
    if (!enc || !dec || !f || !bytes) { free(enc); free(dec); free(f); free(bytes); return -2; }
    // Both structs share one error manager, with our error_exit installed.
    enc->err = dec->err = jpeg_std_error(&f->base);
    f->base.error_exit = failed; f->message = error; f->capacity = error_size;
    int result = -3;
    // setjmp returns 0 now; if libjpeg fails later, longjmp makes it return 1 and
    // the whole block is skipped.
    if (!setjmp(f->jump)) {
        // ---- Encode ----
        jpeg_create_compress(enc);
        jpeg_mem_dest(enc, bytes, &length);
        enc->image_width=w; enc->image_height=h; enc->input_components=3; enc->in_color_space=JCS_RGB;
        jpeg_set_defaults(enc); jpeg_set_quality(enc, quality, TRUE);
        // Every setting is fixed explicitly so the result never depends on library
        // defaults: 4:2:0 chroma subsampling (luma Y at full resolution, the two
        // colour channels Cb and Cr at half width and half height), the standard
        // Huffman tables (no optimize pass), and the accurate integer DCT.
        enc->comp_info[0].h_samp_factor=enc->comp_info[0].v_samp_factor=2;
        enc->comp_info[1].h_samp_factor=enc->comp_info[1].v_samp_factor=1;
        enc->comp_info[2].h_samp_factor=enc->comp_info[2].v_samp_factor=1;
        enc->optimize_coding=FALSE; enc->dct_method=JDCT_ISLOW;
        // Feed the image one row (scanline) at a time.
        jpeg_start_compress(enc, TRUE);
        while (enc->next_scanline < enc->image_height) {
            JSAMPROW row=(JSAMPROW)(rgb + (size_t)enc->next_scanline*w*3);
            jpeg_write_scanlines(enc, &row, 1);
        }
        jpeg_finish_compress(enc);
        // ---- Decode ---- Same accurate integer DCT; "fancy" upsampling smoothly
        // interpolates the half-resolution colour channels back to full size.
        jpeg_create_decompress(dec); jpeg_mem_src(dec, *bytes, length);
        jpeg_read_header(dec, TRUE); dec->out_color_space=JCS_RGB; dec->dct_method=JDCT_ISLOW;
        dec->do_fancy_upsampling=TRUE; jpeg_start_decompress(dec);
        // Only accept the decode if it has the original size and 3 channels.
        if (dec->output_width == (unsigned)w && dec->output_height == (unsigned)h && dec->output_components == 3) {
            while (dec->output_scanline < dec->output_height) {
                JSAMPROW row=decoded + (size_t)dec->output_scanline*w*3;
                jpeg_read_scanlines(dec, &row, 1);
            }
            jpeg_finish_decompress(dec); result=0;
        }
    }
    // Cleanup for every path. `mem` is only set once jpeg_create_* has run, so it
    // tells us whether there is anything to destroy. free(NULL) is harmless, so
    // *bytes is safe to free even if the encoder never allocated it.
    if (enc->mem) jpeg_destroy_compress(enc);
    if (dec->mem) jpeg_destroy_decompress(dec);
    free(*bytes); free(bytes); free(f); free(enc); free(dec);
    return result;
}

// Precomputed resize weights for one axis, the same layout Pillow uses. For
// output pixel i, the inputs start[i] .. start[i]+count[i]-1 are combined
// using weights[i*size + j]. `size` is the widest window any output can need.
// Weights are fixed-point integers: real weight * 2^22.
typedef struct { int size; int *start; int *count; int32_t *weights; } Coefficients;
static void dispose(Coefficients *p) { free(p->start); free(p->count); free(p->weights); }
// The filter shapes, as in Pillow:
//  filter 1, bilinear: a triangle, 1 - |x| for |x| < 1, else 0.
//  filter 3, Lanczos: sinc(x) * sinc(x/3) for -3 <= x < 3, else 0, where
//  sinc(x) = sin(pi*x)/(pi*x) and sinc(0) = 1.
// The filter number is also the filter's radius (support), which coefficients()
// relies on.
static double kernel(double x, int filter) {
    if (filter == 1) { x=fabs(x); return x < 1 ? 1-x : 0; }
    if (x < -3 || x >= 3) return 0;
    if (x == 0) return 1;
    double a=x*M_PI, b=a/3;
    return (sin(a)/a)*(sin(b)/b);
}
// Builds the weights for resizing `input` pixels to `output` pixels along one
// axis, following Pillow's precompute_coeffs. Returns 0, or -2 if out of
// memory (the caller frees with dispose()).
static int coefficients(int input, int output, int filter, Coefficients *p) {
    // scale = input pixels per output pixel. When shrinking (scale > 1) the filter
    // is stretched by fscale = scale so every input pixel is covered (this is the
    // antialiasing); when enlarging it keeps its normal width (fscale = 1).
    // support = how far, in input pixels, the stretched filter reaches.
    double scale=(double)input/output, fscale=fmax(scale,1), support=filter*fscale;
    // Widest possible window: support on both sides, rounded up, plus one.
    p->size=(int)ceil(support)*2+1;
    p->start=calloc(output,sizeof(int)); p->count=calloc(output,sizeof(int));
    p->weights=calloc((size_t)output*p->size,sizeof(int32_t));
    // Scratch space for one output pixel's unnormalized weights, in double.
    double *work=calloc(p->size,sizeof(double));
    if (!p->start || !p->count || !p->weights || !work) { free(work); return -2; }
    for (int i=0;i<output;i++) {
        // center = position of output pixel i's centre in input coordinates.
        double center=(i+.5)*scale, total=0;
        // The window of input pixels the filter reaches, clipped to the image. The
        // (int) casts truncate toward zero, exactly like Pillow's code.
        int first=(int)(center-support+.5), last=(int)(center+support+.5);
        if(first<0)first=0; if(last>input)last=input;
        p->start[i]=first; p->count[i]=last-first;
        // Weight of input pixel first+j: the filter at the distance between that
        // pixel's centre (first+j+0.5) and `center`, measured in stretched units.
        for(int j=0;j<last-first;j++) { work[j]=kernel((j+first-center+.5)/fscale,filter); total+=work[j]; }
        // Normalize so the weights add up to 1 (unless they are all zero), then
        // convert to fixed point: multiply by 4194304 = 2^22 and round half away from
        // zero. This is Pillow's normalize_coeffs_8bpc with PRECISION_BITS = 22.
        for(int j=0;j<last-first;j++) {
            double v=total!=0 ? work[j]/total : work[j];
            p->weights[(size_t)i*p->size+j]=(int32_t)(v*4194304.0+(v<0 ? -.5 : .5));
        }
    }
    free(work); return 0;
}
// Converts a fixed-point sum back to a byte: shift right by 22 bits (divide
// by 2^22, rounding down) and clamp to 0..255. The sums start at 2^21, half of
// 2^22, so the shift rounds to nearest instead of always down.
static uint8_t clipped(int64_t value) { value >>= 22; return value < 0 ? 0 : value > 255 ? 255 : (uint8_t)value; }
// Resizes a packed RGB8 image from w x h to ow x oh. Pillow's order: a
// horizontal pass (w -> ow) into a temporary image `middle` (ow x h), rounded to
// bytes, then a vertical pass (h -> oh) into `out`. A pass whose size does not
// change is skipped and just copies, as Pillow does. Returns 0, -1 for bad
// arguments, -2 if memory runs out.
int facade_resize_rgb(const uint8_t *rgb, int w, int h, uint8_t *out, int ow, int oh, int filter) {
    if (!rgb || !out || rgb == out || !dimensions(w,h) || !dimensions(ow,oh) || (filter!=1 && filter!=3)) return -1;
    // Same size: nothing to do but copy.
    if(w==ow && h==oh) { memcpy(out,rgb,(size_t)w*h*3); return 0; }
    // {0} so dispose() is safe even for a pass that never ran.
    Coefficients horizontal={0}, vertical={0};
    uint8_t *middle=malloc((size_t)ow*h*3);
    if (!middle) return -2;
    int status=0;
    // Horizontal pass, or a copy if the width does not change.
    if(w==ow) memcpy(middle,rgb,(size_t)w*h*3);
    else {
        status=coefficients(w,ow,filter,&horizontal); if(status) goto done;
        // For every row y, output column x and channel c: a weighted sum over the
        // input columns start[x] .. start[x]+count[x]-1 of the same row.
        for(int y=0;y<h;y++) for(int x=0;x<ow;x++) for(int c=0;c<3;c++) {
            // 2097152 = 2^21: the rounding offset (half of 2^22, see clipped()).
            int64_t sum=2097152;
            for(int j=0;j<horizontal.count[x];j++)
                sum+=(int64_t)rgb[((size_t)y*w+horizontal.start[x]+j)*3+c]*horizontal.weights[(size_t)x*horizontal.size+j];
            middle[((size_t)y*ow+x)*3+c]=clipped(sum);
        }
    }
    // Vertical pass, or a copy if the height does not change.
    if(h==oh) memcpy(out,middle,(size_t)ow*h*3);
    else {
        status=coefficients(h,oh,filter,&vertical); if(status) goto done;
        // Same as above, but down the columns of `middle`: a weighted sum over rows
        // start[y] .. start[y]+count[y]-1 at the same column x and channel c.
        for(int y=0;y<oh;y++) for(int x=0;x<ow;x++) for(int c=0;c<3;c++) {
            int64_t sum=2097152;
            for(int j=0;j<vertical.count[y];j++)
                sum+=(int64_t)middle[((size_t)(vertical.start[y]+j)*ow+x)*3+c]*vertical.weights[(size_t)y*vertical.size+j];
            out[((size_t)y*ow+x)*3+c]=clipped(sum);
        }
    }
// Shared cleanup for success and failure.
done:
    dispose(&horizontal); dispose(&vertical); free(middle); return status;
}
