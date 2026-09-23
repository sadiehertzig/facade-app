#include "NativeGeometry.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <Accelerate/Accelerate.h>

static bool valid(int w,int h){return w>0&&h>0&&w<=8192&&h<=8192;}
int facade_cv_resize_rgb(const uint8_t* in,int w,int h,uint8_t* out,int ow,int oh){
    if(!in||!out||!valid(w,h)||!valid(ow,oh))return -1;
    try{cv::Mat a(h,w,CV_8UC3,const_cast<uint8_t*>(in)),b(oh,ow,CV_8UC3,out);cv::resize(a,b,b.size(),0,0,cv::INTER_LINEAR);return 0;}catch(...){return -2;}
}
int facade_cv_warp_rgb(const uint8_t* in,int w,int h,const float* matrix,uint8_t* out,int size){
    if(!in||!out||!matrix||!valid(w,h)||!valid(size,size))return -1;
    try{cv::Mat a(h,w,CV_8UC3,const_cast<uint8_t*>(in)),b(size,size,CV_8UC3,out),m(2,3,CV_32F,const_cast<float*>(matrix));cv::warpAffine(a,b,m,b.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);return 0;}catch(...){return -2;}
}
int facade_cv_lift(const float* in,int w,int h,float* out,int ow,int oh){
    if(!in||!out||!valid(w,h)||!valid(ow,oh))return -1;
    try{cv::Mat a(h,w,CV_32FC3,const_cast<float*>(in)),b(oh,ow,CV_32FC3,out);cv::resize(a,b,b.size(),0,0,cv::INTER_LANCZOS4);return 0;}catch(...){return -2;}
}
int facade_similarity(const float* s,const float* d,float* m){
    double sx=0,sy=0,dx=0,dy=0;
    for(int i=0;i<5;i++){sx+=s[2*i];sy+=s[2*i+1];dx+=d[2*i];dy+=d[2*i+1];}
    sx/=5;sy/=5;dx/=5;dy/=5;double a=0,b=0,den=0;
    for(int i=0;i<5;i++){double x=s[2*i]-sx,y=s[2*i+1]-sy,u=d[2*i]-dx,v=d[2*i+1]-dy;a+=x*u+y*v;b+=x*v-y*u;den+=x*x+y*y;}
    if(!std::isfinite(den)||den<1e-12)return -1;a/=den;b/=den;
    m[0]=a;m[1]=-b;m[2]=dx-a*sx+b*sy;m[3]=b;m[4]=a;m[5]=dy-b*sx-a*sy;return 0;
}
int facade_inverse(const float* m,float* n){
    double det=double(m[0])*m[4]-double(m[1])*m[3];if(!std::isfinite(det)||std::abs(det)<1e-12)return -1;
    double a=m[4]/det,b=-m[1]/det,c=-m[3]/det,d=m[0]/det;
    n[0]=a;n[1]=b;n[2]=-a*m[2]-b*m[5];n[3]=c;n[4]=d;n[5]=-c*m[2]-d*m[5];return 0;
}
static std::vector<float> crop_coordinates(const float* m,int size){
    int n=size*size;std::vector<float> pix(n*3),out(n*2);
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){int i=(y*size+x)*3;pix[i]=x;pix[i+1]=y;pix[i+2]=1;}
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,2,n,3,1,m,3,pix.data(),3,0,out.data(),n);
    return out;
}
void facade_float_crop(const float* in,int w,int h,const float* m,float* out,int size){
    auto coordinates=crop_coordinates(m,size);
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){
        float xx=coordinates[y*size+x],yy=coordinates[size*size+y*size+x];int x0=floorf(xx),y0=floorf(yy);float fx=xx-x0,fy=yy-y0;
        for(int c=0;c<3;c++){
            auto get=[&](int ix,int iy){return ix>=0&&iy>=0&&ix<w&&iy<h?in[(c*h+iy)*w+ix]:0.f;};
            out[(c*size+y)*size+x]=get(x0,y0)*((1-fx)*(1-fy))+get(x0,y0+1)*((1-fx)*fy)+get(x0+1,y0)*(fx*(1-fy))+get(x0+1,y0+1)*(fx*fy);
        }
    }
}
void facade_crop_backward(const float* gradient,int size,const float* m,float* out,int w,int h){
    std::fill(out,out+3*w*h,0.f);auto coordinates=crop_coordinates(m,size);
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){
        float xx=coordinates[y*size+x],yy=coordinates[size*size+y*size+x];int x0=floorf(xx),y0=floorf(yy);float fx=xx-x0,fy=yy-y0;
        int xs[4]={x0,x0,x0+1,x0+1},ys[4]={y0,y0+1,y0,y0+1};float weights[4]={(1-fx)*(1-fy),(1-fx)*fy,fx*(1-fy),fx*fy};
        for(int c=0;c<3;c++)for(int j=0;j<4;j++)if(xs[j]>=0&&ys[j]>=0&&xs[j]<w&&ys[j]<h)out[(c*h+ys[j])*w+xs[j]]+=gradient[(c*size+y)*size+x]*weights[j];
    }
}
struct ResizeWeights {std::vector<int> first,count;std::vector<std::vector<float>> weights;};
static ResizeWeights resize_weights(int n,int out,bool antialias){
    ResizeWeights r;r.first.resize(out);r.count.resize(out);r.weights.resize(out);double scale=double(n)/out;
    for(int i=0;i<out;i++){
        if(!antialias){double z=std::max(0.,(i+.5)*scale-.5);int a=std::min(int(z),n-1),b=std::min(a+1,n-1);float t=z-a;r.first[i]=a;r.count[i]=(b==a?1:2);r.weights[i]=b==a?std::vector<float>{1.f}:std::vector<float>{1-t,t};}
        else{double support=std::max(1.,scale),center=(i+.5)*scale;int a=std::max(0,int(center-support+.5)),b=std::min(n,int(center+support+.5));r.first[i]=a;r.count[i]=b-a;float sum=0;for(int j=a;j<b;j++){float v=std::max(0.,1-std::abs((j-center+.5)/support));r.weights[i].push_back(v);sum+=v;}for(auto& v:r.weights[i])v/=sum;}
    }return r;
}
void facade_resize_float(const float* in,int w,int h,float* out,int ow,int oh,int aa){
    auto xx=resize_weights(w,ow,aa),yy=resize_weights(h,oh,aa);std::vector<float> middle(3*h*ow);
    for(int c=0;c<3;c++)for(int y=0;y<h;y++)for(int x=0;x<ow;x++){float s=0;for(int j=0;j<xx.count[x];j++)s+=in[(c*h+y)*w+xx.first[x]+j]*xx.weights[x][j];middle[(c*h+y)*ow+x]=s;}
    for(int c=0;c<3;c++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++){float s=0;for(int j=0;j<yy.count[y];j++)s+=middle[(c*h+yy.first[y]+j)*ow+x]*yy.weights[y][j];out[(c*oh+y)*ow+x]=s;}
}
void facade_resize_backward(const float* gradient,int ow,int oh,float* out,int w,int h,int aa){
    auto xx=resize_weights(w,ow,aa),yy=resize_weights(h,oh,aa);std::vector<float> middle(3*h*ow,0.f);std::fill(out,out+3*w*h,0.f);
    for(int c=0;c<3;c++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++)for(int j=0;j<yy.count[y];j++)middle[(c*h+yy.first[y]+j)*ow+x]+=gradient[(c*oh+y)*ow+x]*yy.weights[y][j];
    for(int c=0;c<3;c++)for(int y=0;y<h;y++)for(int x=0;x<ow;x++)for(int j=0;j<xx.count[x];j++)out[(c*h+y)*w+xx.first[x]+j]+=middle[(c*h+y)*ow+x]*xx.weights[x][j];
}
void facade_dct_low(const float* in,float* out,int channels,int w,int h,int rescale){
    float m[8][8];for(int u=0;u<8;u++)for(int x=0;x<8;x++)m[u][x]=(u?0.5:1/std::sqrt(8.))*std::cos(M_PI/8*(x+.5)*u);
    for(int c=0;c<channels;c++)for(int by=0;by<h;by+=8)for(int bx=0;bx<w;bx+=8){
        float coeff[4][4]={},temp[4][8]={};
        for(int u=0;u<4;u++)for(int y=0;y<8;y++){float s=0;for(int x=0;x<8;x++)s+=m[u][x]*in[(c*h+by+x)*w+bx+y];temp[u][y]=s;}
        for(int u=0;u<4;u++)for(int v=0;v<4-u;v++){float s=0;for(int y=0;y<8;y++)s+=temp[u][y]*m[v][y];coeff[u][v]=s;}
        for(int x=0;x<8;x++)for(int y=0;y<8;y++){float s=0;for(int u=0;u<4;u++)for(int v=0;v<4-u;v++)s+=m[u][x]*coeff[u][v]*m[v][y];out[(c*h+by+x)*w+bx+y]=s;}
    }
    if(rescale){double a=0,b=0;int n=channels*w*h;for(int i=0;i<n;i++){a+=std::abs(in[i]);b+=std::abs(out[i]);}float scale=float(a/n)/std::max(float(b/n),1e-12f);for(int i=0;i<n;i++)out[i]*=scale;}
}
