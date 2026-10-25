// NativeEqualAdd.h
//
// Fast kernel for "add" nodes whose two inputs have the same number of
// elements: y[i] = a[i] + b[i]. These are the residual (shortcut) additions of
// both recognizers, e.g. [1, 256, 14, 14] + [1, 256, 14, 14]. Every add node in
// the two recognizer graphs the app loads is this case.
//
// Why it exists. The general add in NativeNetwork.cpp fetches each operand
// through Network::at() (which handles broadcasting) and adds each gradient
// through Network::add() (which checks for constants and creates buffers),
// for every single element. When both inputs have the output's element count,
// at() always returns i unchanged, so this kernel can skip that work and run a
// straight loop. Because at() would have returned i anyway, the results are
// bit-identical.
//
// The functions are templates over the tensor type so this header does not
// need to see the Tensor struct, which is defined inside NativeNetwork.cpp. T
// must have `shape` (vector<int>), `x` and `g` (vector<float>).
#pragma once
#include <cstddef>
#include <limits>
// The kernel only runs when the element counts are bounded and equal and the
// two inputs are different nodes.
// Anything else (the same node added to itself, unequal counts, a shape it
// cannot check) returns false and the general loop runs instead.
namespace facade {
// Element count of tensor.shape, written to `count`. False if the rank is not
// 1 to 4, a dimension is not positive, or the total would exceed 9,999,999
// (each multiply is checked before it happens, so it cannot overflow).
template<class T> inline bool equalAddCount(const T& tensor,size_t& count) {
 if(tensor.shape.empty()||tensor.shape.size()>4)return false;
 count=1;
 for(int d:tensor.shape){if(d<=0||count>9999999u/size_t(d))return false;count*=size_t(d);}
 return count>0&&count<10000000u;
}
// True when the kernel applies: `distinct` (the two inputs are different
// nodes), and a, b and the output all have the same element count with value
// buffers of that length. Only the counts must match, not the shapes, because
// at() only compares counts too.
template<class T> inline bool equalAddEligible(const T& a,const T& b,const T& t,bool distinct,size_t& count) {
 size_t ac=0,bc=0;
 return distinct&&equalAddCount(t,count)&&equalAddCount(a,ac)&&equalAddCount(b,bc)&&
 ac==count&&bc==count&&a.x.size()==count&&b.x.size()==count&&t.x.size()==count;
}
// Forward: y[i] = a[i] + b[i]. Returns false without writing if not eligible.
template<class T> inline bool equalAddForward(const T& a,const T& b,T& t,bool distinct) {
 size_t count=0;if(!equalAddEligible(a,b,t,distinct,count))return false;
 const float* ax=a.x.data();const float* bx=b.x.data();float* y=t.x.data();
 for(size_t i=0;i<count;i++)y[i]=ax[i]+bx[i];
 return true;
}
// Backward: the derivative of a + b is 1 for both inputs, so the output
// gradient is added unchanged to each: ag[i] += g[i] and bg[i] += g[i].
// constantA / constantB = that input is a constant node and gets no gradient.
// Also returns false if an existing gradient buffer has the wrong length.
template<class T> inline bool equalAddBackward(T& a,T& b,T& t,bool constantA,bool constantB,bool distinct) {
 size_t count=0;if(!equalAddEligible(a,b,t,distinct,count)||t.g.size()!=count||
 (!constantA&&!a.g.empty()&&a.g.size()!=count)||(!constantB&&!b.g.empty()&&b.g.size()!=count))return false;
 // Create any missing gradient buffer (filled with +0.0, like Network::add)
 // before taking data() pointers, because assign() may move the storage.
 // Constant inputs are skipped completely.
 if(!constantA&&a.g.empty())a.g.assign(count,0.0f);
 if(!constantB&&b.g.empty())b.g.assign(count,0.0f);
 float* ag=constantA?nullptr:a.g.data();float* bg=constantB?nullptr:b.g.data();const float* g=t.g.data();
 // Per element, a's contribution first and then b's, as in the general loop.
 for(size_t i=0;i<count;i++){if(!constantA)ag[i]+=g[i];if(!constantB)bg[i]+=g[i];}
 return true;
}
// End of namespace facade.
}
