#pragma once
#include <cstddef>
#include <limits>
// Only bounded equal-count, distinct-input tensors enter these loops.
// Aliases, unequal counts and unproved geometry retain the original scalar path.
namespace facade {
template<class T> inline bool equalAddCount(const T& tensor,size_t& count) {
 if(tensor.shape.empty()||tensor.shape.size()>4)return false;
 count=1;
 for(int d:tensor.shape){if(d<=0||count>9999999u/size_t(d))return false;count*=size_t(d);}
 return count>0&&count<10000000u;
}
template<class T> inline bool equalAddEligible(const T& a,const T& b,const T& t,bool distinct,size_t& count) {
 size_t ac=0,bc=0;
 return distinct&&equalAddCount(t,count)&&equalAddCount(a,ac)&&equalAddCount(b,bc)&&
 ac==count&&bc==count&&a.x.size()==count&&b.x.size()==count&&t.x.size()==count;
}
template<class T> inline bool equalAddForward(const T& a,const T& b,T& t,bool distinct) {
 size_t count=0;if(!equalAddEligible(a,b,t,distinct,count))return false;
 const float* ax=a.x.data();const float* bx=b.x.data();float* y=t.x.data();
 for(size_t i=0;i<count;i++)y[i]=ax[i]+bx[i];
 return true;
}
template<class T> inline bool equalAddBackward(T& a,T& b,T& t,bool constantA,bool constantB,bool distinct) {
 size_t count=0;if(!equalAddEligible(a,b,t,distinct,count)||t.g.size()!=count||
 (!constantA&&!a.g.empty()&&a.g.size()!=count)||(!constantB&&!b.g.empty()&&b.g.size()!=count))return false;
 // Finish any allocation before taking pointers. Never touch constant gradients.
 if(!constantA&&a.g.empty())a.g.assign(count,0.0f);
 if(!constantB&&b.g.empty())b.g.assign(count,0.0f);
 float* ag=constantA?nullptr:a.g.data();float* bg=constantB?nullptr:b.g.data();const float* g=t.g.data();
 for(size_t i=0;i<count;i++){if(!constantA)ag[i]+=g[i];if(!constantB)bg[i]+=g[i];}
 return true;
}
}
