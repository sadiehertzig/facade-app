// NativeBatchNorm.h
//
// Fast kernel for the network's batch-norm layers ("bn" nodes in
// NativeNetwork.cpp), forward and backward.
//
// What batch norm does here. When the network was exported, each batch-norm
// layer's learned statistics were folded into two numbers per channel: a scale
// alpha[c] and an offset beta[c]. So at run time the layer is just
//     forward:   y = x * alpha[c] + beta[c]
//     backward:  dx += dy * alpha[c]
// where c is the channel the element belongs to.
//
// Tensor layout. Tensors are flat float arrays in CHW order with a batch of 1,
// so for shape [1, C, H, W] channel c owns one contiguous "plane" of H*W
// values starting at index c*H*W. A rank-2 shape [1, C] is the same idea with
// a plane of one value. That lets the kernel load alpha[c] and beta[c] once
// and then run a tight inner loop over the plane, instead of working out
// i / plane for every element like the plain loop in NativeNetwork.cpp does.
//
// Fast path and fallback. Each function first checks that the shapes and
// buffers are exactly the simple case it handles. If anything is different it
// returns false without touching any output, and the caller runs the plain
// loop instead. The two paths give bit-identical results:
//  * the elements are visited in the same order,
//  * the multiply and the add are written as two statements, so the product
//    is rounded to float before beta is added. A fused multiply-add would
//    round only once and give different bits. (The build also passes
//    -ffp-contract=off, which forbids fusing anyway.) NativeBatchNormFixtures.h
//    tests this with inputs where fusing would change the answer.
#pragma once
#include <cstddef>
#include <vector>
// Everything in these kernel headers lives in namespace facade.
namespace facade {
// Returns the number of elements in `shape`, or 0 if the shape is not
// something the fast kernel accepts: rank 1 to 4, every dimension positive,
// and the total at most 9,999,999. The d > 9999999/n test checks the bound
// before multiplying, so the product can never overflow.
inline int bnBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>9999999/n)return 0;n*=size_t(d);}
 return int(n);
}
// Decides whether the fast kernel can handle this layer. `as` is the input
// shape, `os` the output shape. Accepted: input and output shapes identical,
// rank 2 ([1, C]) or rank 4 ([1, C, H, W]), batch of 1, and exactly one alpha
// and one beta per channel. On success it also hands back the element count,
// the number of channels, and the plane size (count / channels).
inline bool bnPlaneEligible(const std::vector<int>& as,const std::vector<int>& os,
 size_t alphaCount,size_t betaCount,int& count,int& channels,int& plane) {
 count=bnBoundedCount(os);channels=0;plane=0;
 if(!count||as!=os||(os.size()!=2&&os.size()!=4)||os[0]!=1)return false;
 // Channels are always dimension 1 (after the batch dimension).
 channels=os[1];
 if(alphaCount!=size_t(channels)||betaCount!=size_t(channels))return false;
 // plane = H*W (or 1 for rank 2). The last test is a sanity check that
 // channels * plane really covers every element.
 plane=count/channels;return plane>0&&size_t(channels)*size_t(plane)==size_t(count);
}
// Forward: y[i] = a[i] * alpha[c] + beta[c]. Returns false (and writes
// nothing) if the layer is not eligible, if a buffer has the wrong length, or
// if any two of the vectors are the same object. That last check guarantees
// that writing y can never change a value the loop still has to read.
inline bool bnPlaneForward(const std::vector<int>& as,const std::vector<int>& os,
 const std::vector<float>& a,const std::vector<float>& alpha,const std::vector<float>& beta,
 std::vector<float>& y) {
 int count=0,channels=0,plane=0;
 if(!bnPlaneEligible(as,os,alpha.size(),beta.size(),count,channels,plane)||
 a.size()!=size_t(count)||y.size()!=size_t(count)||
 &a==&alpha||&a==&beta||&alpha==&beta||&y==&a||&y==&alpha||&y==&beta)return false;
 // Raw pointers so the inner loop is plain array indexing.
 const float* ap=a.data();const float* al=alpha.data();const float* be=beta.data();float* yp=y.data();
 // Outer loop: one channel at a time. start = index of this channel's first
 // element.
 for(int c=0;c<channels;c++){const float scale=al[size_t(c)],offset=be[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  // v is the rounded product; adding beta is a separate statement on purpose
  // (see the top of this file).
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);float v=ap[i]*scale;yp[i]=v+offset;}
 }return true;
}
// Backward: ag[i] += g[i] * alpha[c], where g is the gradient arriving at the
// layer's output and ag is the gradient of its input. y and beta are not
// needed for the math (the derivative of x*alpha + beta with respect to x is
// just alpha); they are passed in only so the same size and no-overlap checks
// can be made on them. constantA = the input is a constant node, which never
// receives a gradient.
inline bool bnPlaneBackward(const std::vector<int>& as,const std::vector<int>& os,
 const std::vector<float>& a,const std::vector<float>& alpha,const std::vector<float>& beta,
 const std::vector<float>& y,const std::vector<float>& g,bool constantA,std::vector<float>& ag) {
 int count=0,channels=0,plane=0;
 if(!bnPlaneEligible(as,os,alpha.size(),beta.size(),count,channels,plane)||
 a.size()!=size_t(count)||y.size()!=size_t(count)||g.size()!=size_t(count)||
 (!ag.empty()&&ag.size()!=size_t(count))||
 &a==&alpha||&a==&beta||&alpha==&beta||&y==&a||&y==&alpha||&y==&beta||
 &g==&a||&g==&alpha||&g==&beta||&g==&y||
 &ag==&a||&ag==&alpha||&ag==&beta||&ag==&y||&ag==&g)return false;
 // If the input has no gradient buffer yet, create one filled with +0.0, the
 // same way Network::add does on its first contribution. Every value is then
 // added with +=, including the first one. Writing the first value with = would
 // not be the same: +0.0 + (-0.0) is +0.0, but storing -0.0 directly keeps the
 // minus sign, and the plain loop never does that.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 // destination is null for a constant input, so the loop below computes the
 // values but stores nothing, exactly like add() ignoring a constant node.
 const float* gp=g.data();const float* al=alpha.data();float* destination=constantA?nullptr:ag.data();
 for(int c=0;c<channels;c++){const float scale=al[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  // Same visiting order as the plain loop: channel by channel, element by
  // element, so every += happens in the same sequence.
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float value=gp[i]*scale;
   if(destination)destination[i]+=value;
  }
 }return true;
}
// End of namespace facade.
}
