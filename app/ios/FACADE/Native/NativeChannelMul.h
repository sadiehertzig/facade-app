// NativeChannelMul.h
//
// Fast kernel for one kind of "mul" node in the network: a full tensor times
// one value per channel,
//     y[c][h][w] = a[c][h][w] * b[c].
// In the SimSwap recognizer this is the last step of every
// squeeze-and-excitation block: b holds one sigmoid "gate" value per channel,
// computed from the image, and each channel is rescaled by its gate. (The
// InSwapper recognizer's only mul is by a single scalar, which this kernel
// does not handle; it uses the general loop.) NativeNetwork.cpp's
// binaryForward / binaryReverse try this kernel first and fall back to their
// general loop when it returns false.
//
// Tensor layout. a and y are flat CHW arrays with a batch of 1, so channel c
// owns the contiguous plane a[c*H*W ... (c+1)*H*W - 1]. b has exactly C values.
// The general loop finds b's index with Network::at(), which divides every
// index by H*W; this kernel instead loops channel by channel and reads b[c]
// once per plane. Both visit the elements in the same order, so the results
// are bit-identical.
//
// The gradient of b is a sum over a whole plane, b_grad[c] += sum of
// g[i] * a[i]. Float addition depends on order, so the kernel adds the terms
// one at a time in increasing i, exactly as the general loop does.
#pragma once
#include <cstddef>
#include <vector>
// Everything in these kernel headers lives in namespace facade.
namespace facade {
// Returns the number of elements in `shape`, or 0 if the shape is outside
// what the kernel accepts: rank 1 to 4, positive dimensions, total at most
// 10,000,000 (checked before each multiply so it cannot overflow). This kernel
// only handles a narrower set of shapes than Network::at() allows; anything it
// refuses still works through the general loop.
inline int mulBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>10000000/n)return 0;n*=size_t(d);}
 return int(n);
}
// Decides whether this mul node is the "tensor times per-channel vector" case.
// as/bs/os = shapes of a, b and the output; aid/bid = graph indices of the two
// input nodes. Requirements:
//  * aid != bid: the two inputs are different nodes (squaring a tensor is not
//    this case, and separate gradient buffers are needed below);
//  * a has as many elements as the output;
//  * b is neither full size nor a single scalar: it must have exactly one value
//    per channel;
//  * the output is [1, C, H, W] with C > 1 and H*W > 1.
// On success it reports the element count, channel count and plane size H*W.
inline bool channelMulEligible(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,int& count,int& channels,int& plane) {
 count=mulBoundedCount(os);const int ac=mulBoundedCount(as),bc=mulBoundedCount(bs);
 if(aid==bid||!count||ac!=count||bc==count||bc==1||!bc||os.size()!=4||os[0]!=1)return false;
 channels=os[1];if(bc!=channels)return false;
 plane=os[2]*os[3];return plane>1&&channels>1&&size_t(channels)*size_t(plane)==size_t(count);
}
// Forward: y[i] = a[i] * b[c] for every element i in channel c. Returns false
// without writing anything if the case or any buffer size does not match.
inline bool channelMulForward(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,const std::vector<float>& a,
 const std::vector<float>& b,std::vector<float>& y) {
 int count=0,channels=0,plane=0;
 if(!channelMulEligible(as,bs,os,aid,bid,count,channels,plane)||
 a.size()!=size_t(count)||b.size()!=size_t(channels)||y.size()!=size_t(count))return false;
 // start = index of channel c's first element.
 for(int c=0;c<channels;c++){const float bb=b[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);y[i]=a[i]*bb;}
 }return true;
}
// Backward. g is the gradient arriving at the output; ag and bg are the
// gradients of a and b. For y = a*b the derivatives are dy/da = b and
// dy/db = a, so
//     ag[i] += g[i] * b[c]          (one term per element)
//     bg[c] += g[i] * a[i]          (summed over the whole plane of channel c)
// constantA / constantB = that input is a constant node and gets no gradient.
inline bool channelMulBackward(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,bool constantA,bool constantB,
 const std::vector<float>& a,const std::vector<float>& b,const std::vector<float>& g,
 std::vector<float>& ag,std::vector<float>& bg) {
 int count=0,channels=0,plane=0;
 if(!channelMulEligible(as,bs,os,aid,bid,count,channels,plane)||
 a.size()!=size_t(count)||b.size()!=size_t(channels)||g.size()!=size_t(count)||
 (!ag.empty()&&ag.size()!=size_t(count))||(!bg.empty()&&bg.size()!=size_t(channels)))return false;
 // Create missing gradient buffers (filled with +0.0, like Network::add) before
 // taking any data() pointer: assign() can move a vector's storage, which would
 // leave an earlier pointer dangling. Because aid != bid, ag and bg are two
 // different vectors and cannot overwrite each other.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 if(!constantB&&bg.empty())bg.assign(size_t(channels),0);
 // Null pointer = constant input: the value is computed but not stored.
 float* ap=constantA?nullptr:ag.data();float* bp=constantB?nullptr:bg.data();
 for(int c=0;c<channels;c++){const float bb=b[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  // For each element: a's contribution first, then b's, the same order the
  // general loop uses.
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float aa=a[i];
   const float av=g[i]*bb;if(ap)ap[i]+=av;
   const float bv=g[i]*aa;if(bp)bp[size_t(c)]+=bv;
  }
 }return true;
}
}
