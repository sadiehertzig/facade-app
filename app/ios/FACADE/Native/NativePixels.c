// Resampling coefficient and rounding conventions follow Pillow 12.3.0
// src/libImaging/Resample.c. See THIRD_PARTY_NOTICES.md and licenses/ at the
// repository root for its HPND license.
#include "NativePixels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <setjmp.h>
#include <jpeglib.h>

static int dimensions(int w, int h) { return w > 0 && h > 0 && w <= 8192 && h <= 8192; }
struct failure { struct jpeg_error_mgr base; jmp_buf jump; char *message; size_t capacity; };
static void failed(j_common_ptr p) {
    struct failure *f = (struct failure *)p->err;
    char text[JMSG_LENGTH_MAX];
    (*p->err->format_message)(p, text);
    if (f->message && f->capacity) snprintf(f->message, f->capacity, "%s", text);
    longjmp(f->jump, 1);
}

int facade_jpeg_roundtrip(const uint8_t *rgb, int w, int h, int quality,
                          uint8_t *decoded, char *error, size_t error_size) {
    if (!rgb || !decoded || rgb == decoded || !dimensions(w,h) || quality < 1 || quality > 100) return -1;
    // Heap state remains defined across libjpeg's longjmp error recovery.
    struct jpeg_compress_struct *enc = calloc(1, sizeof(*enc));
    struct jpeg_decompress_struct *dec = calloc(1, sizeof(*dec));
    struct failure *f = calloc(1, sizeof(*f));
    unsigned char **bytes = calloc(1, sizeof(*bytes));
    unsigned long length = 0;
    if (!enc || !dec || !f || !bytes) { free(enc); free(dec); free(f); free(bytes); return -2; }
    enc->err = dec->err = jpeg_std_error(&f->base);
    f->base.error_exit = failed; f->message = error; f->capacity = error_size;
    int result = -3;
    if (!setjmp(f->jump)) {
        jpeg_create_compress(enc);
        jpeg_mem_dest(enc, bytes, &length);
        enc->image_width=w; enc->image_height=h; enc->input_components=3; enc->in_color_space=JCS_RGB;
        jpeg_set_defaults(enc); jpeg_set_quality(enc, quality, TRUE);
        enc->comp_info[0].h_samp_factor=enc->comp_info[0].v_samp_factor=2;
        enc->comp_info[1].h_samp_factor=enc->comp_info[1].v_samp_factor=1;
        enc->comp_info[2].h_samp_factor=enc->comp_info[2].v_samp_factor=1;
        enc->optimize_coding=FALSE; enc->dct_method=JDCT_ISLOW;
        jpeg_start_compress(enc, TRUE);
        while (enc->next_scanline < enc->image_height) {
            JSAMPROW row=(JSAMPROW)(rgb + (size_t)enc->next_scanline*w*3);
            jpeg_write_scanlines(enc, &row, 1);
        }
        jpeg_finish_compress(enc);
        jpeg_create_decompress(dec); jpeg_mem_src(dec, *bytes, length);
        jpeg_read_header(dec, TRUE); dec->out_color_space=JCS_RGB; dec->dct_method=JDCT_ISLOW;
        dec->do_fancy_upsampling=TRUE; jpeg_start_decompress(dec);
        if (dec->output_width == (unsigned)w && dec->output_height == (unsigned)h && dec->output_components == 3) {
            while (dec->output_scanline < dec->output_height) {
                JSAMPROW row=decoded + (size_t)dec->output_scanline*w*3;
                jpeg_read_scanlines(dec, &row, 1);
            }
            jpeg_finish_decompress(dec); result=0;
        }
    }
    if (enc->mem) jpeg_destroy_compress(enc);
    if (dec->mem) jpeg_destroy_decompress(dec);
    free(*bytes); free(bytes); free(f); free(enc); free(dec);
    return result;
}

typedef struct { int size; int *start; int *count; int32_t *weights; } Coefficients;
static void dispose(Coefficients *p) { free(p->start); free(p->count); free(p->weights); }
static double kernel(double x, int filter) {
    if (filter == 1) { x=fabs(x); return x < 1 ? 1-x : 0; }
    if (x < -3 || x >= 3) return 0;
    if (x == 0) return 1;
    double a=x*M_PI, b=a/3;
    return (sin(a)/a)*(sin(b)/b);
}
static int coefficients(int input, int output, int filter, Coefficients *p) {
    double scale=(double)input/output, fscale=fmax(scale,1), support=filter*fscale;
    p->size=(int)ceil(support)*2+1;
    p->start=calloc(output,sizeof(int)); p->count=calloc(output,sizeof(int));
    p->weights=calloc((size_t)output*p->size,sizeof(int32_t));
    double *work=calloc(p->size,sizeof(double));
    if (!p->start || !p->count || !p->weights || !work) { free(work); return -2; }
    for (int i=0;i<output;i++) {
        double center=(i+.5)*scale, total=0;
        int first=(int)(center-support+.5), last=(int)(center+support+.5);
        if(first<0)first=0; if(last>input)last=input;
        p->start[i]=first; p->count[i]=last-first;
        for(int j=0;j<last-first;j++) { work[j]=kernel((j+first-center+.5)/fscale,filter); total+=work[j]; }
        for(int j=0;j<last-first;j++) {
            double v=total!=0 ? work[j]/total : work[j];
            p->weights[(size_t)i*p->size+j]=(int32_t)(v*4194304.0+(v<0 ? -.5 : .5));
        }
    }
    free(work); return 0;
}
static uint8_t clipped(int64_t value) { value >>= 22; return value < 0 ? 0 : value > 255 ? 255 : (uint8_t)value; }
int facade_resize_rgb(const uint8_t *rgb, int w, int h, uint8_t *out, int ow, int oh, int filter) {
    if (!rgb || !out || rgb == out || !dimensions(w,h) || !dimensions(ow,oh) || (filter!=1 && filter!=3)) return -1;
    if(w==ow && h==oh) { memcpy(out,rgb,(size_t)w*h*3); return 0; }
    Coefficients horizontal={0}, vertical={0};
    uint8_t *middle=malloc((size_t)ow*h*3);
    if (!middle) return -2;
    int status=0;
    if(w==ow) memcpy(middle,rgb,(size_t)w*h*3);
    else {
        status=coefficients(w,ow,filter,&horizontal); if(status) goto done;
        for(int y=0;y<h;y++) for(int x=0;x<ow;x++) for(int c=0;c<3;c++) {
            int64_t sum=2097152;
            for(int j=0;j<horizontal.count[x];j++)
                sum+=(int64_t)rgb[((size_t)y*w+horizontal.start[x]+j)*3+c]*horizontal.weights[(size_t)x*horizontal.size+j];
            middle[((size_t)y*ow+x)*3+c]=clipped(sum);
        }
    }
    if(h==oh) memcpy(out,middle,(size_t)ow*h*3);
    else {
        status=coefficients(h,oh,filter,&vertical); if(status) goto done;
        for(int y=0;y<oh;y++) for(int x=0;x<ow;x++) for(int c=0;c<3;c++) {
            int64_t sum=2097152;
            for(int j=0;j<vertical.count[y];j++)
                sum+=(int64_t)middle[((size_t)(vertical.start[y]+j)*ow+x)*3+c]*vertical.weights[(size_t)y*vertical.size+j];
            out[((size_t)y*ow+x)*3+c]=clipped(sum);
        }
    }
done:
    dispose(&horizontal); dispose(&vertical); free(middle); return status;
}
