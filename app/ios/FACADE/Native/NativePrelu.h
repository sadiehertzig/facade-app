// NativePrelu.h
//
// Fast kernel for the network's PReLU activation layers, forward and backward.
//
// What PReLU does: y = x when x > 0, otherwise y = x * slope. The slope is a
// learned number. Backward, the gradient passes through unchanged where x > 0
// and is multiplied by the slope elsewhere. (x = +0.0 and x = -0.0 are not
// greater than 0, so both take the slope branch, in both paths.)
//
// The two recognizers store the slope in two different ways, and the kernel
// handles both:
//  * "owned": the slope lives in the PReLU node itself (Node::weight). In the
//    SimSwap recognizer this is a single slope for the whole tensor. The
//    kernel treats that as one "plane" covering every element.
//  * a separate constant input node holding one slope per channel, shape
//    [C, 1, 1] (the InSwapper recognizer). Then channel c's plane of H*W
//    values uses slope[c].
// An owned slope with more than one value, or a slope node that is not a
// constant, is left to the plain loop in NativeNetwork.cpp.
//
// Tensor layout: flat CHW with a batch of 1, so channel c is the contiguous
// range [c*H*W, (c+1)*H*W). The kernel visits elements in the same order as
// the plain loop and does the same single multiply, so the results are
// bit-identical.
#pragma once
#include <cstddef>
#include <vector>
// Everything in these kernel headers lives in namespace facade.
namespace facade {
// Element count of `shape`, or 0 if the rank is not 1 to 4, a dimension is not
// positive, or the total would exceed 10,000,000 (checked before each
// multiply so it cannot overflow). Shapes outside this range use the plain loop.
inline int preluBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>10000000/n)return 0;n*=size_t(d);}
 return int(n);
}
// Decides whether the fast kernel applies, and if so how the slope is laid out.
// as/ss/os = shapes of the input, the slope node and the output.
// aid/sid = graph indices of the input and of the slope node (sid = -1 when
// there is no slope node). owned = the slope is stored in the PReLU node.
// constantSlope = the slope node is a constant.
// Outputs: count = elements, planes = how many slope values,
// plane = how many consecutive elements share one slope value.
inline bool preluPlaneEligible(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,
 int& count,int& planes,int& plane) {
 count=preluBoundedCount(os);planes=0;plane=0;
 // The input must be a different node from the slope, have the output's shape,
 // and be [1, C] or [1, C, H, W].
 if(aid<0||sid < -1||aid==sid||!count||as!=os||
 (os.size()!=2&&os.size()!=4)||os[0]!=1)return false;
 // Owned slope: one plane covering everything. (The forward and backward
 // functions then also require exactly one slope value.)
 if(owned){planes=1;plane=count;return true;}
 // Separate slope node: it must be a constant, the output must be rank 4, and
 // there must be exactly one slope per channel (C > 1).
 if(sid<0||!constantSlope||os.size()!=4)return false;
 planes=os[1];if(planes<=1||preluBoundedCount(ss)!=planes)return false;
 plane=os[2]*os[3];return plane>0&&size_t(planes)*size_t(plane)==size_t(count);
}
// Forward: y[i] = a[i] > 0 ? a[i] : a[i] * slope[c]. Returns false without
// writing if not eligible, if a buffer has the wrong length, or if y is the
// same vector as an input.
inline bool preluPlaneForward(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,
 const std::vector<float>& a,const std::vector<float>& slope,std::vector<float>& y) {
 int count=0,planes=0,plane=0;
 if(!preluPlaneEligible(as,ss,os,aid,sid,owned,constantSlope,count,planes,plane)||
 a.size()!=size_t(count)||slope.size()!=size_t(planes)||y.size()!=size_t(count)||
 &a==&y||&slope==&y)return false;
 const float* ap=a.data();const float* sp=slope.data();float* yp=y.data();
 // One plane at a time; `coefficient` is that plane's slope.
 for(int c=0;c<planes;c++){const float coefficient=sp[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);yp[i]=ap[i]>0?ap[i]:ap[i]*coefficient;}
 }return true;
}
// Backward: ag[i] += (a[i] > 0 ? g[i] : g[i] * slope[c]). g is the gradient at
// the output, ag the gradient of the input. The slope gets no gradient (the
// weights are fixed; only the pixels are being optimized). constantA = the
// input is a constant node and gets nothing.
inline bool preluPlaneBackward(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,bool constantA,
 const std::vector<float>& a,const std::vector<float>& slope,const std::vector<float>& g,
 std::vector<float>& ag) {
 int count=0,planes=0,plane=0;
 if(!preluPlaneEligible(as,ss,os,aid,sid,owned,constantSlope,count,planes,plane)||
 a.size()!=size_t(count)||slope.size()!=size_t(planes)||g.size()!=size_t(count)||
 (!ag.empty()&&ag.size()!=size_t(count))||&ag==&a||&ag==&slope||&ag==&g)return false;
 // Create a missing gradient buffer (filled with +0.0, like Network::add) before
 // taking pointers, then add every value with +=. Adding onto +0.0 is what makes
 // the first contribution match the plain loop, including the sign of zero.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 const float* ap=a.data();const float* sp=slope.data();const float* gp=g.data();
 // Null destination = constant input: values are computed but not stored.
 float* destination=constantA?nullptr:ag.data();
 for(int c=0;c<planes;c++){const float coefficient=sp[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float value=ap[i]>0?gp[i]:gp[i]*coefficient;
   if(destination)destination[i]+=value;
  }
 }return true;
}
// End of namespace facade.
}
