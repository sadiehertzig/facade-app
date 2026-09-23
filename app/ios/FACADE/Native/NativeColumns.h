#ifndef FACADE_NATIVE_COLUMNS_H
#define FACADE_NATIVE_COLUMNS_H
#include <algorithm>
#include <cstring>
#include <limits>
namespace facade {
inline bool columns3x3Eligible(int c,int h,int w,int kh,int kw,int sh,int sw,int ph,int pw,int oh,int ow) {
 if(!(c>0&&h>0&&w>0&&kh==3&&kw==3&&sh==1&&sw==1&&ph==1&&pw==1&&oh==h&&ow==w))return false;
 const size_t plane=size_t(h)*size_t(w),limit=size_t(std::numeric_limits<int>::max());
 return plane<=limit/9&&size_t(c)<=limit/(9*plane);
}
inline void fillColumns3x3(float* col,const float* input,int c,int h,int w) {
 const size_t plane=size_t(h)*size_t(w);
 for(int z=0;z<c;z++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int y=0;y<h;y++) {
  float* row=col+((size_t(z)*3+ky)*3+kx)*plane+size_t(y)*w;
  const int sy=y-1+ky;
  if(sy<0||sy>=h) {std::fill(row,row+w,0.0f);continue;}
  const int left=kx==0 ? 1:0,right=kx==2 ? 1:0,count=w-left-right;
  if(left)row[0]=0.0f;
  if(right)row[w-1]=0.0f;
  if(count>0)std::memcpy(row+left,input+(size_t(z)*h+sy)*w+size_t(left-1+kx),size_t(count)*sizeof(float));
 }
}
}
#endif
