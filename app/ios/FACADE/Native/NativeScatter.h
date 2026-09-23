#pragma once
#include <vector>
#include <limits>

namespace facade {
// Test for any valid contribution, including stride gaps and excessive padding.
// This preserves Network.add's lazy initialization when every column is padding.
inline bool scatterAxisTouches(int input, int kernel, int output, int stride, int padding) {
    for (int k = 0; k < kernel; ++k)
        for (int o = 0; o < output; ++o) {
            int i = o * stride - padding + k;
            if (i >= 0 && i < input) return true;
        }
    return false;
}

inline bool scatter3x3Eligible(int tensorCount,int c,int h,int w,int kh,int kw,
                              int oh,int ow,int sh,int sw,int ph,int pw) {
    if (!(c>0&&h>0&&w>0&&kh==3&&kw==3&&sh==1&&sw==1&&ph==1&&pw==1&&oh==h&&ow==w)) return false;
    const size_t plane=size_t(h)*size_t(w),limit=size_t(std::numeric_limits<int>::max());
    return plane<=limit/9 && size_t(c)<=limit/(9*plane) && size_t(tensorCount)==size_t(c)*plane;
}
inline void scatterRows3x3(float* dst,const float* col,int c,int h,int w) {
    const size_t plane=size_t(h)*size_t(w);
    for (int z=0;z<c;++z)
        for (int ky=0;ky<3;++ky)
            for (int kx=0;kx<3;++kx)
                for (int y=0;y<h;++y) {
                    const int sy=y-1+ky;
                    if (sy<0||sy>=h) continue;
                    const int begin=kx==0?1:0,end=w-(kx==2?1:0);
                    if (begin>=end) continue;
                    const float* source=col+((size_t(z)*3+ky)*3+kx)*plane+size_t(y)*w;
                    float* row=dst+(size_t(z)*h+sy)*w;
                    for (int x=begin;x<end;++x) row[x-1+kx]+=source[x];
                }
}

// The source column allocation and destination tensor are distinct in the graph.
// Keep z/ky/kx/y/x and each += in the original order. Never clear a gradient
// already contributed by another consumer of this tensor.
inline void scatterColumns(std::vector<float>& gradient, bool constant, int tensorCount,
                           const float* col, int c, int h, int w, int kh, int kw,
                           int oh, int ow, int sh, int sw, int ph, int pw,bool optimized=true) {
    if (constant) return;
    if (gradient.empty()) {
        if (!scatterAxisTouches(h, kh, oh, sh, ph) ||
            !scatterAxisTouches(w, kw, ow, sw, pw)) return;
        gradient.assign(tensorCount, 0);
    }
    if(optimized&&scatter3x3Eligible(tensorCount,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw)){scatterRows3x3(gradient.data(),col,c,h,w);return;}
    float* dst = gradient.data();
    int l = oh * ow;
    for (int z = 0; z < c; ++z)
        for (int ky = 0; ky < kh; ++ky)
            for (int kx = 0; kx < kw; ++kx)
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x) {
                        int sy = y * sh - ph + ky, sx = x * sw - pw + kx;
                        if (sy >= 0 && sy < h && sx >= 0 && sx < w)
                            dst[(z * h + sy) * w + sx] +=
                                col[((z * kh + ky) * kw + kx) * l + y * ow + x];
                    }
}
}
