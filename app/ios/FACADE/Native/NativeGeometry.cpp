// NativeGeometry.cpp
//
// Image-geometry helpers for the FACADE engine. Three groups:
//
//  1. OpenCV wrappers on 8-bit images (resize, face-crop warp) and on one float
//     image (the Lanczos "lift"). These wrap the caller's buffers in cv::Mat
//     headers without copying, so OpenCV reads and writes Swift's memory
//     directly. Any OpenCV exception becomes the return code -2.
//
//  2. Face alignment math: fitting a similarity transform to five face
//     landmarks (eyes, nose tip, mouth corners), inverting it, and a float
//     bilinear crop with its exact gradient. The optimizer needs the gradient
//     of the recognizer's score with respect to the WHOLE photo, so every step
//     between the photo and the 112x112 crop must be differentiable: the
//     gradient of the crop is pushed back through *_backward functions here.
//
//  3. A separable float resize (and its gradient) and an 8x8 block DCT filter
//     that keeps only smooth, low-frequency changes.
//
// Float images in groups 2 and 3 are CHW: element (c, y, x) of a w x h image
// is at (c*h + y)*w + x. Pixel centres sit on whole-number coordinates
// (pixel (x, y) covers x-0.5..x+0.5), so coordinate (3, 4) is exactly the
// centre of pixel (3, 4).
//
// The *_backward functions are the transposes ("adjoints") of their forward
// functions: if forward computes out = W * in for some fixed weights W,
// backward computes grad_in = W^T * grad_out, using the same weights.
#include "NativeGeometry.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <Accelerate/Accelerate.h>

// Same size limit as NativePixels.c: 1 to 8192 pixels on each side.
static bool valid(int w,int h){return w>0&&h>0&&w<=8192&&h<=8192;}
// Bilinear resize of a packed RGB8 image with OpenCV (used to make the face
// detector's 640-pixel input). Because b already has the target size and
// type, cv::resize writes straight into `out` instead of allocating.
// const_cast is only there because cv::Mat has no read-only constructor;
// the input is not modified.
int facade_cv_resize_rgb(const uint8_t* in,int w,int h,uint8_t* out,int ow,int oh){
    if(!in||!out||!valid(w,h)||!valid(ow,oh))return -1;
    try{cv::Mat a(h,w,CV_8UC3,const_cast<uint8_t*>(in)),b(oh,ow,CV_8UC3,out);cv::resize(a,b,b.size(),0,0,cv::INTER_LINEAR);return 0;}catch(...){return -2;}
}
// Cuts a size x size face crop out of a packed RGB8 photo. `matrix` maps photo
// coordinates to crop coordinates; warpAffine inverts it internally and, for
// each crop pixel, samples the photo bilinearly. Pixels that fall outside the
// photo become black (BORDER_CONSTANT, 0).
int facade_cv_warp_rgb(const uint8_t* in,int w,int h,const float* matrix,uint8_t* out,int size){
    if(!in||!out||!matrix||!valid(w,h)||!valid(size,size))return -1;
    try{cv::Mat a(h,w,CV_8UC3,const_cast<uint8_t*>(in)),b(size,size,CV_8UC3,out),m(2,3,CV_32F,const_cast<float*>(matrix));cv::warpAffine(a,b,m,b.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);return 0;}catch(...){return -2;}
}
// "Lift": resizes an interleaved 3-float-per-pixel image (CV_32FC3) with
// Lanczos interpolation over an 8x8 neighbourhood. The pipeline optimizes a
// 256x256 copy of the photo; this scales that copy's pixel change back up to
// the photo's full size (see App/NativePipeline.swift).
int facade_cv_lift(const float* in,int w,int h,float* out,int ow,int oh){
    if(!in||!out||!valid(w,h)||!valid(ow,oh))return -1;
    try{cv::Mat a(h,w,CV_32FC3,const_cast<float*>(in)),b(oh,ow,CV_32FC3,out);cv::resize(a,b,b.size(),0,0,cv::INTER_LANCZOS4);return 0;}catch(...){return -2;}
}
// Least-squares similarity transform (rotation + uniform scale + translation,
// no shear or mirroring) taking the five source points s onto the five
// destination points d. s and d are x0,y0,x1,y1,...,x4,y4. Output m is the 2x3
// matrix [a -b tx; b a ty]. Here a = scale*cos(angle), b = scale*sin(angle).
// In App/NativeVision.swift, s holds the detected landmarks in the photo and d
// a fixed template of where those landmarks belong in the crop. Everything is
// computed in double.
int facade_similarity(const float* s,const float* d,float* m){
    // (sx, sy) and (dx, dy) are the centroids (average points) of the two sets.
    double sx=0,sy=0,dx=0,dy=0;
    for(int i=0;i<5;i++){sx+=s[2*i];sy+=s[2*i+1];dx+=d[2*i];dy+=d[2*i+1];}
    // With both sets centred on their centroids, minimizing
    // sum |[a -b; b a](x, y) - (u, v)|^2 has the closed-form answer
    //     a = sum(x*u + y*v) / sum(x^2 + y^2)
    //     b = sum(x*v - y*u) / sum(x^2 + y^2)
    sx/=5;sy/=5;dx/=5;dy/=5;double a=0,b=0,den=0;
    for(int i=0;i<5;i++){double x=s[2*i]-sx,y=s[2*i+1]-sy,u=d[2*i]-dx,v=d[2*i+1]-dy;a+=x*u+y*v;b+=x*v-y*u;den+=x*x+y*y;}
    // den is about 0 when all source points coincide; no unique fit exists then.
    if(!std::isfinite(den)||den<1e-12)return -1;a/=den;b/=den;
    // Translation: chosen so that the source centroid lands on the destination
    // centroid.
    m[0]=a;m[1]=-b;m[2]=dx-a*sx+b*sy;m[3]=b;m[4]=a;m[5]=dy-b*sx-a*sy;return 0;
}
// Inverse of the 2x3 affine matrix m. Invert the 2x2 part [m0 m1; m3 m4]
// (divide the swapped/negated entries by the determinant), then the new
// translation is -(inverse 2x2) * (m2, m5). Returns -1 if the matrix is
// (nearly) singular.
int facade_inverse(const float* m,float* n){
    double det=double(m[0])*m[4]-double(m[1])*m[3];if(!std::isfinite(det)||std::abs(det)<1e-12)return -1;
    double a=m[4]/det,b=-m[1]/det,c=-m[3]/det,d=m[0]/det;
    n[0]=a;n[1]=b;n[2]=-a*m[2]-b*m[5];n[3]=c;n[4]=d;n[5]=-c*m[2]-d*m[5];return 0;
}
// For every pixel (x, y) of a size x size crop, where it lands in the photo.
// m maps crop coordinates to photo coordinates. One sgemm computes all of them:
// the (2 x 3) matrix m times the (3 x n) matrix of [x, y, 1] columns. Result:
// the first n floats are photo x-coordinates, the next n are y-coordinates,
// both in crop row-major order (index y*size + x).
static std::vector<float> crop_coordinates(const float* m,int size){
    int n=size*size;std::vector<float> pix(n*3),out(n*2);
    // pix holds one row [x, y, 1] per crop pixel (n x 3); CblasTrans reads it as
    // 3 x n.
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){int i=(y*size+x)*3;pix[i]=x;pix[i+1]=y;pix[i+2]=1;}
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,2,n,3,1,m,3,pix.data(),3,0,out.data(),n);
    return out;
}
// Float face crop, the differentiable twin of facade_cv_warp_rgb. `m` is the
// INVERSE alignment (crop -> photo). For each crop pixel it finds the photo
// position (xx, yy) and blends the four surrounding photo pixels:
//     x0 = floor(xx), fx = xx - x0 (and the same for y)
//     out = p(x0,y0)(1-fx)(1-fy) + p(x0,y0+1)(1-fx)fy
//         + p(x0+1,y0)fx(1-fy)  + p(x0+1,y0+1)fx*fy
// A neighbour outside the photo counts as 0 (black), like the OpenCV warp.
// Input is w x h CHW, output is size x size CHW.
void facade_float_crop(const float* in,int w,int h,const float* m,float* out,int size){
    auto coordinates=crop_coordinates(m,size);
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){
        // coordinates[y*size+x] is the photo x, coordinates[size*size + y*size+x] the
        // photo y, for crop pixel (x, y).
        float xx=coordinates[y*size+x],yy=coordinates[size*size+y*size+x];int x0=floorf(xx),y0=floorf(yy);float fx=xx-x0,fy=yy-y0;
        for(int c=0;c<3;c++){
            // get(ix, iy) = photo pixel (ix, iy) of channel c, or 0 outside the photo.
            auto get=[&](int ix,int iy){return ix>=0&&iy>=0&&ix<w&&iy<h?in[(c*h+iy)*w+ix]:0.f;};
            out[(c*size+y)*size+x]=get(x0,y0)*((1-fx)*(1-fy))+get(x0,y0+1)*((1-fx)*fy)+get(x0+1,y0)*(fx*(1-fy))+get(x0+1,y0+1)*(fx*fy);
        }
    }
}
// Gradient of facade_float_crop. Each crop pixel took fixed fractions of four
// photo pixels, so each photo pixel's gradient gets the same fractions of that
// crop pixel's gradient. Neighbours outside the photo were never read and get
// nothing. `out` (the photo gradient, 3 x w x h) is zeroed first and then only
// added to, in a fixed order (crop pixels in row order, then channel, then the
// four neighbours), so the result is the same on every run.
void facade_crop_backward(const float* gradient,int size,const float* m,float* out,int w,int h){
    std::fill(out,out+3*w*h,0.f);auto coordinates=crop_coordinates(m,size);
    for(int y=0;y<size;y++)for(int x=0;x<size;x++){
        float xx=coordinates[y*size+x],yy=coordinates[size*size+y*size+x];int x0=floorf(xx),y0=floorf(yy);float fx=xx-x0,fy=yy-y0;
        // The same four neighbours and weights as the forward crop, in the same order.
        int xs[4]={x0,x0,x0+1,x0+1},ys[4]={y0,y0+1,y0,y0+1};float weights[4]={(1-fx)*(1-fy),(1-fx)*fy,fx*(1-fy),fx*fy};
        for(int c=0;c<3;c++)for(int j=0;j<4;j++)if(xs[j]>=0&&ys[j]>=0&&xs[j]<w&&ys[j]<h)out[(c*h+ys[j])*w+xs[j]]+=gradient[(c*size+y)*size+x]*weights[j];
    }
}
// Resize weights for one axis: for each output index i, `first` is the first
// input index used, `count` how many consecutive inputs are used, and
// `weights` the weight of each of them.
struct ResizeWeights {std::vector<int> first,count;std::vector<std::vector<float>> weights;};
// Builds the weights for resizing n input samples to `out` samples.
// scale = input samples per output sample (> 1 means shrinking).
static ResizeWeights resize_weights(int n,int out,bool antialias){
    ResizeWeights r;r.first.resize(out);r.count.resize(out);r.weights.resize(out);double scale=double(n)/out;
    for(int i=0;i<out;i++){
        // Plain bilinear (antialias off). Output sample i sits at input coordinate
        // z = (i + 0.5)*scale - 0.5, clamped at 0 (the "half-pixel centres" rule,
        // align_corners = False in PyTorch terms). It blends input a = floor(z) and
        // b = a + 1 with weights 1-t and t; at the right edge, where b would run off
        // the end, only one input with weight 1 is used.
        if(!antialias){double z=std::max(0.,(i+.5)*scale-.5);int a=std::min(int(z),n-1),b=std::min(a+1,n-1);float t=z-a;r.first[i]=a;r.count[i]=(b==a?1:2);r.weights[i]=b==a?std::vector<float>{1.f}:std::vector<float>{1-t,t};}
        // Antialiased (used when shrinking). A triangle filter centred on the output
        // sample, widened to `support` = max(1, scale) input samples on each side, so
        // every input sample in that range contributes instead of just the nearest
        // two. Weights are then divided by their sum so they add up to 1. This follows
        // the weight formula of PyTorch's antialiased bilinear interpolation.
        else{double support=std::max(1.,scale),center=(i+.5)*scale;int a=std::max(0,int(center-support+.5)),b=std::min(n,int(center+support+.5));r.first[i]=a;r.count[i]=b-a;float sum=0;for(int j=a;j<b;j++){float v=std::max(0.,1-std::abs((j-center+.5)/support));r.weights[i].push_back(v);sum+=v;}for(auto& v:r.weights[i])v/=sum;}
    }return r;
}
// Resizes a 3-channel CHW float image from w x h to ow x oh in two passes:
// first along each row (w -> ow) into `middle` (3 x h x ow), then along each
// column (h -> oh) into `out`. aa != 0 turns on antialiasing. Each output value
// is a weighted sum computed in a fixed order.
void facade_resize_float(const float* in,int w,int h,float* out,int ow,int oh,int aa){
    auto xx=resize_weights(w,ow,aa),yy=resize_weights(h,oh,aa);std::vector<float> middle(3*h*ow);
    // Pass 1, horizontal: middle[c][y][x] = sum over j of
    // in[c][y][first[x]+j] * weight[x][j].
    for(int c=0;c<3;c++)for(int y=0;y<h;y++)for(int x=0;x<ow;x++){float s=0;for(int j=0;j<xx.count[x];j++)s+=in[(c*h+y)*w+xx.first[x]+j]*xx.weights[x][j];middle[(c*h+y)*ow+x]=s;}
    // Pass 2, vertical: out[c][y][x] = sum over j of
    // middle[c][first[y]+j][x] * weight[y][j].
    for(int c=0;c<3;c++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++){float s=0;for(int j=0;j<yy.count[y];j++)s+=middle[(c*h+yy.first[y]+j)*ow+x]*yy.weights[y][j];out[(c*oh+y)*ow+x]=s;}
}
// Gradient of facade_resize_float. `gradient` is ow x oh (the resized size),
// `out` receives the w x h gradient of the original. The passes run in reverse
// order (vertical first, then horizontal) and each one spreads every value
// back to the inputs it came from, with the same weights.
void facade_resize_backward(const float* gradient,int ow,int oh,float* out,int w,int h,int aa){
    auto xx=resize_weights(w,ow,aa),yy=resize_weights(h,oh,aa);std::vector<float> middle(3*h*ow,0.f);std::fill(out,out+3*w*h,0.f);
    // Undo pass 2: spread each output row's gradient onto the `middle` rows.
    for(int c=0;c<3;c++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++)for(int j=0;j<yy.count[y];j++)middle[(c*h+yy.first[y]+j)*ow+x]+=gradient[(c*oh+y)*ow+x]*yy.weights[y][j];
    // Undo pass 1: spread each `middle` value onto the input columns.
    for(int c=0;c<3;c++)for(int y=0;y<h;y++)for(int x=0;x<ow;x++)for(int j=0;j<xx.count[x];j++)out[(c*h+y)*w+xx.first[x]+j]+=middle[(c*h+y)*ow+x]*xx.weights[x][j];
}
// Low-frequency filter on 8x8 blocks. For every channel and every 8x8 block
// it takes the 2D DCT (the frequency content of the block), keeps only the 10
// lowest-frequency coefficients (those with u + v <= 3, out of 64), and
// transforms back. Sharp, pixel-to-pixel changes are removed; smooth changes
// survive. The optimizer runs its pixel changes and gradients through this
// filter so that the protection pattern stays smooth (TensorMath.project in
// App/NativeTensor.swift, used by App/NativeOptimizer.swift and
// App/NativePipeline.swift).
// w and h must be multiples of 8 (the Swift caller checks this).
// rescale != 0 then scales the result so its average absolute value equals
// the input's, because removing frequencies also shrinks the overall size of
// the change.
void facade_dct_low(const float* in,float* out,int channels,int w,int h,int rescale){
    // m is the 8x8 orthonormal DCT-II matrix: m[u][x] = c(u) * cos(pi/8 * (x + 0.5) * u),
    // with c(0) = 1/sqrt(8) and c(u > 0) = 1/2 (= sqrt(2/8)). Orthonormal means its
    // inverse is its transpose, which is what the reconstruction step uses.
    float m[8][8];for(int u=0;u<8;u++)for(int x=0;x<8;x++)m[u][x]=(u?0.5:1/std::sqrt(8.))*std::cos(M_PI/8*(x+.5)*u);
    for(int c=0;c<channels;c++)for(int by=0;by<h;by+=8)for(int bx=0;bx<w;bx+=8){
        // In this function x is the row inside the block and y the column (the
        // reverse of the usual naming). (by, bx) = the block's top-left corner.
        float coeff[4][4]={},temp[4][8]={};
        // Step 1: temp[u][y] = sum over rows x of m[u][x] * block[x][y]
        // (DCT down each column, only for the 4 lowest vertical frequencies).
        for(int u=0;u<4;u++)for(int y=0;y<8;y++){float s=0;for(int x=0;x<8;x++)s+=m[u][x]*in[(c*h+by+x)*w+bx+y];temp[u][y]=s;}
        // Step 2: coeff[u][v] = sum over columns y of temp[u][y] * m[v][y]
        // (DCT along the rows). v < 4 - u keeps only u + v <= 3.
        for(int u=0;u<4;u++)for(int v=0;v<4-u;v++){float s=0;for(int y=0;y<8;y++)s+=temp[u][y]*m[v][y];coeff[u][v]=s;}
        // Step 3: rebuild every pixel from the kept coefficients:
        // block[x][y] = sum over kept (u, v) of m[u][x] * coeff[u][v] * m[v][y].
        for(int x=0;x<8;x++)for(int y=0;y<8;y++){float s=0;for(int u=0;u<4;u++)for(int v=0;v<4-u;v++)s+=m[u][x]*coeff[u][v]*m[v][y];out[(c*h+by+x)*w+bx+y]=s;}
    }
    // Optional rescale: mean |in| / mean |out|, in double. The 1e-12 floor avoids
    // dividing by zero when the filtered result is all zeros.
    if(rescale){double a=0,b=0;int n=channels*w*h;for(int i=0;i<n;i++){a+=std::abs(in[i]);b+=std::abs(out[i]);}float scale=float(a/n)/std::max(float(b/n),1e-12f);for(int i=0;i<n;i++)out[i]*=scale;}
}
