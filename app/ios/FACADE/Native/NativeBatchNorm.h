#pragma once
#include <cstddef>
#include <vector>
namespace facade {
// Narrow finite-graph BN family. Unsupported valid cases use the original router.
inline int bnBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>9999999/n)return 0;n*=size_t(d);}
 return int(n);
}
inline bool bnPlaneEligible(const std::vector<int>& as,const std::vector<int>& os,
 size_t alphaCount,size_t betaCount,int& count,int& channels,int& plane) {
 count=bnBoundedCount(os);channels=0;plane=0;
 if(!count||as!=os||(os.size()!=2&&os.size()!=4)||os[0]!=1)return false;
 channels=os[1];
 if(alphaCount!=size_t(channels)||betaCount!=size_t(channels))return false;
 plane=count/channels;return plane>0&&size_t(channels)*size_t(plane)==size_t(count);
}
inline bool bnPlaneForward(const std::vector<int>& as,const std::vector<int>& os,
 const std::vector<float>& a,const std::vector<float>& alpha,const std::vector<float>& beta,
 std::vector<float>& y) {
 int count=0,channels=0,plane=0;
 if(!bnPlaneEligible(as,os,alpha.size(),beta.size(),count,channels,plane)||
 a.size()!=size_t(count)||y.size()!=size_t(count)||
 &a==&alpha||&a==&beta||&alpha==&beta||&y==&a||&y==&alpha||&y==&beta)return false;
 const float* ap=a.data();const float* al=alpha.data();const float* be=beta.data();float* yp=y.data();
 for(int c=0;c<channels;c++){const float scale=al[size_t(c)],offset=be[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);float v=ap[i]*scale;yp[i]=v+offset;}
 }return true;
}
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
 // Initialize +0 only when empty. Every contribution, including the first, uses +=.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 const float* gp=g.data();const float* al=alpha.data();float* destination=constantA?nullptr:ag.data();
 for(int c=0;c<channels;c++){const float scale=al[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float value=gp[i]*scale;
   if(destination)destination[i]+=value;
  }
 }return true;
}
}
