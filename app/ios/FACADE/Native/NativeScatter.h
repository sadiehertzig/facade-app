// NativeScatter.h
//
// The backward half of a convolution's im2col step ("col2im"). It is the
// mirror image of NativeColumns.h.
//
// Where it sits. In the backward pass, NativeNetwork.cpp first does one matrix
// multiply that gives the gradient of every entry of the column matrix (one
// row per input channel and kernel position, one column per output pixel; see
// NativeColumns.h for the layout). Each column-matrix entry was a copy of one
// input pixel, or a padding zero. So to get the gradient of the input pixels,
// scatterColumns adds every entry back onto the pixel it was copied from, and
// drops the entries that were padding. A pixel that sat in up to nine patches
// collects up to nine contributions.
//
// Order matters. Those contributions are added with +=, and float addition is
// not associative, so the order of the additions decides the exact result.
// Both the general loop and the 3x3 fast loop walk channel z, kernel row ky,
// kernel column kx, output row y, output column x, in that nesting, so each
// pixel receives its contributions in the same sequence and the results are
// bit-identical.
//
// Accumulate, never overwrite. The destination gradient may already hold
// contributions from another layer that also reads this tensor (for example
// the shortcut branch of a residual block). scatterColumns only ever adds.
#pragma once
#include <vector>
#include <limits>

namespace facade {
// Checks one axis (rows or columns) of the convolution: is there any output
// position o and kernel tap k that reads a real input index
// i = o*stride - padding + k, rather than padding? It tries every pair, so it
// is correct even for strides that skip pixels or for very large padding.
// scatterColumns uses this to decide whether to create an empty gradient
// buffer. Network::add only creates the buffer when a contribution actually
// arrives, so if every column entry is padding the buffer must stay empty.
inline bool scatterAxisTouches(int input, int kernel, int output, int stride, int padding) {
    for (int k = 0; k < kernel; ++k)
        for (int o = 0; o < output; ++o) {
            int i = o * stride - padding + k;
            if (i >= 0 && i < input) return true;
        }
    return false;
}

// True for the 3x3 / stride 1 / padding 1 / same-size case that
// scatterRows3x3 handles. Like columns3x3Eligible in NativeColumns.h, plus a
// check that the destination tensor really has c*h*w elements. The size_t
// divisions check that c*9*h*w fits in an int without overflowing.
inline bool scatter3x3Eligible(int tensorCount,int c,int h,int w,int kh,int kw,
                              int oh,int ow,int sh,int sw,int ph,int pw) {
    if (!(c>0&&h>0&&w>0&&kh==3&&kw==3&&sh==1&&sw==1&&ph==1&&pw==1&&oh==h&&ow==w)) return false;
    const size_t plane=size_t(h)*size_t(w),limit=size_t(std::numeric_limits<int>::max());
    return plane<=limit/9 && size_t(c)<=limit/(9*plane) && size_t(tensorCount)==size_t(c)*plane;
}
// The fast 3x3 scatter. dst = the input gradient (c*h*w floats, CHW);
// col = the column-matrix gradient (c*9 rows of h*w floats). For matrix row
// (z, ky, kx) and output pixel (y, x), the source pixel is
// (z, y-1+ky, x-1+kx); entries whose source is in the padding border are
// skipped instead of being tested one by one.
inline void scatterRows3x3(float* dst,const float* col,int c,int h,int w) {
    const size_t plane=size_t(h)*size_t(w);
    for (int z=0;z<c;++z)
        for (int ky=0;ky<3;++ky)
            for (int kx=0;kx<3;++kx)
                for (int y=0;y<h;++y) {
                    // sy = input row this kernel row read for output row y; skip padding rows.
                    const int sy=y-1+ky;
                    if (sy<0||sy>=h) continue;
                    // kx = 0 means output column 0 read the left padding, so start at x = 1;
                    // kx = 2 means the last output column read the right padding, so stop one
                    // early. If nothing is left (w = 1), skip.
                    const int begin=kx==0?1:0,end=w-(kx==2?1:0);
                    if (begin>=end) continue;
                    // source = this output row's slice of matrix row (z, ky, kx);
                    // row = input row sy of channel z in the destination.
                    const float* source=col+((size_t(z)*3+ky)*3+kx)*plane+size_t(y)*w;
                    float* row=dst+(size_t(z)*h+sy)*w;
                    // Output column x read input column x-1+kx.
                    for (int x=begin;x<end;++x) row[x-1+kx]+=source[x];
                }
}

// Adds the column-matrix gradient `col` back onto `gradient`, the gradient of
// the convolution's input (tensorCount = c*h*w elements). `col` is a fresh
// buffer made by the caller, so it can never be the same memory as `gradient`.
// The loops keep the z / ky / kx / y / x nesting and the += order of the
// original code, so the result is bit-identical in either path.
// An existing gradient is never cleared: it may already hold another layer's
// contribution. optimized = false forces the general loop (the self-checks
// use this to compare the two paths).
inline void scatterColumns(std::vector<float>& gradient, bool constant, int tensorCount,
                           const float* col, int c, int h, int w, int kh, int kw,
                           int oh, int ow, int sh, int sw, int ph, int pw,bool optimized=true) {
    // Constant nodes never get a gradient.
    if (constant) return;
    // No buffer yet: create one filled with +0.0, but only if at least one column
    // entry lands inside the image on both axes. Otherwise leave it empty, as
    // Network::add would.
    if (gradient.empty()) {
        if (!scatterAxisTouches(h, kh, oh, sh, ph) ||
            !scatterAxisTouches(w, kw, ow, sw, pw)) return;
        gradient.assign(tensorCount, 0);
    }
    // Fast path for the common 3x3 case.
    if(optimized&&scatter3x3Eligible(tensorCount,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw)){scatterRows3x3(gradient.data(),col,c,h,w);return;}
    // General case, any kernel, stride and padding. l = number of output pixels =
    // the length of one matrix row.
    float* dst = gradient.data();
    int l = oh * ow;
    for (int z = 0; z < c; ++z)
        for (int ky = 0; ky < kh; ++ky)
            for (int kx = 0; kx < kw; ++kx)
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x) {
                        // (sy, sx) = the input pixel that matrix entry (z, ky, kx, y, x) was copied
                        // from. Only real pixels get a contribution; padding entries are dropped.
                        int sy = y * sh - ph + ky, sx = x * sw - pw + kx;
                        if (sy >= 0 && sy < h && sx >= 0 && sx < w)
                            dst[(z * h + sy) * w + sx] +=
                                col[((z * kh + ky) * kw + kx) * l + y * ow + x];
                    }
}
// End of namespace facade.
}
