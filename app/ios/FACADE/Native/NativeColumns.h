// NativeColumns.h
//
// Fast "im2col" for the most common convolution in both recognizers: a 3x3
// kernel, stride 1, padding 1, so the output is the same size as the input.
// (InSwapper's recognizer has 45 of these, SimSwap's 63.)
//
// What im2col does. A convolution output pixel is a dot product between the
// kernel weights and the 3x3 patch of input pixels around it, across all input
// channels. im2col copies every patch into one big matrix so that a single
// matrix multiply (cblas_sgemm in NativeNetwork.cpp) computes all of the dot
// products at once.
//
// Matrix layout. The matrix has c*3*3 rows and h*w columns. Row
// (z*3 + ky)*3 + kx holds, for every output pixel (y, x), the input value at
// channel z, row y-1+ky, column x-1+kx, or 0 when that position falls in the
// one-pixel padding border. So each row is a copy of the input channel shifted
// by (ky-1, kx-1).
//
// Why a special version. The general loop in Network::columns tests every
// single element for "inside the image or padding". Here each w-long piece of
// a matrix row is either all padding (the shifted row is above or below the
// image) or one memcpy of w-1 or w input values plus at most one zero at an
// edge. The copied values are identical, so the result is bit-identical to
// the general loop; it only does less work per element.
#ifndef FACADE_NATIVE_COLUMNS_H
#define FACADE_NATIVE_COLUMNS_H
#include <algorithm>
#include <cstring>
#include <limits>
// Everything in these kernel headers lives in namespace facade.
namespace facade {
// True when the convolution is the 3x3 / stride 1 / padding 1 / same-size case
// (c, h, w = input channels and size; kh, kw = kernel; sh, sw = stride;
// ph, pw = padding; oh, ow = output size).
inline bool columns3x3Eligible(int c,int h,int w,int kh,int kw,int sh,int sw,int ph,int pw,int oh,int ow) {
 if(!(c>0&&h>0&&w>0&&kh==3&&kw==3&&sh==1&&sw==1&&ph==1&&pw==1&&oh==h&&ow==w))return false;
 // The column matrix has c*9*plane entries. The two divisions check, without
 // overflowing, that this count fits in an int, because the rest of the network
 // code indexes buffers with int.
 const size_t plane=size_t(h)*size_t(w),limit=size_t(std::numeric_limits<int>::max());
 return plane<=limit/9&&size_t(c)<=limit/(9*plane);
}
// Fills `col` (c*9 rows of h*w floats each) from `input` (c channels of h*w
// floats, CHW order). The caller has already checked columns3x3Eligible.
inline void fillColumns3x3(float* col,const float* input,int c,int h,int w) {
 const size_t plane=size_t(h)*size_t(w);
 // One pass per (channel z, kernel row ky, kernel column kx, output row y):
 // fill one h*w-long matrix row, w floats at a time.
 for(int z=0;z<c;z++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int y=0;y<h;y++) {
  // row = start of the w floats for output row y inside matrix row (z, ky, kx).
  float* row=col+((size_t(z)*3+ky)*3+kx)*plane+size_t(y)*w;
  // sy = the input row this kernel row reads for output row y.
  const int sy=y-1+ky;
  // Above or below the image: the whole row is padding.
  if(sy<0||sy>=h) {std::fill(row,row+w,0.0f);continue;}
  // Horizontal shift. kx = 0 reads column x-1, so output column 0 is padding
  // (left = 1). kx = 2 reads column x+1, so the last output column is padding
  // (right = 1). kx = 1 reads column x, no padding. count = values to copy.
  const int left=kx==0 ? 1:0,right=kx==2 ? 1:0,count=w-left-right;
  if(left)row[0]=0.0f;
  if(right)row[w-1]=0.0f;
  // Copy input[z][sy][left-1+kx ...] into row[left ...]. The source offset
  // left-1+kx is 0 for kx = 0 and kx = 1, and 1 for kx = 2.
  if(count>0)std::memcpy(row+left,input+(size_t(z)*h+sy)*w+size_t(left-1+kx),size_t(count)*sizeof(float));
 }
}
// End of namespace facade.
}
#endif
