#pragma once
#include <cstddef>
#include <vector>
namespace facade {
// Eligibility is deliberately narrower than at(); failures retain its original path.
inline int mulBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>10000000/n)return 0;n*=size_t(d);}
 return int(n);
}
inline bool channelMulEligible(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,int& count,int& channels,int& plane) {
 count=mulBoundedCount(os);const int ac=mulBoundedCount(as),bc=mulBoundedCount(bs);
 if(aid==bid||!count||ac!=count||bc==count||bc==1||!bc||os.size()!=4||os[0]!=1)return false;
 channels=os[1];if(bc!=channels)return false;
 plane=os[2]*os[3];return plane>1&&channels>1&&size_t(channels)*size_t(plane)==size_t(count);
}
inline bool channelMulForward(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,const std::vector<float>& a,
 const std::vector<float>& b,std::vector<float>& y) {
 int count=0,channels=0,plane=0;
 if(!channelMulEligible(as,bs,os,aid,bid,count,channels,plane)||
 a.size()!=size_t(count)||b.size()!=size_t(channels)||y.size()!=size_t(count))return false;
 for(int c=0;c<channels;c++){const float bb=b[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);y[i]=a[i]*bb;}
 }return true;
}
inline bool channelMulBackward(const std::vector<int>& as,const std::vector<int>& bs,
 const std::vector<int>& os,int aid,int bid,bool constantA,bool constantB,
 const std::vector<float>& a,const std::vector<float>& b,const std::vector<float>& g,
 std::vector<float>& ag,std::vector<float>& bg) {
 int count=0,channels=0,plane=0;
 if(!channelMulEligible(as,bs,os,aid,bid,count,channels,plane)||
 a.size()!=size_t(count)||b.size()!=size_t(channels)||g.size()!=size_t(count)||
 (!ag.empty()&&ag.size()!=size_t(count))||(!bg.empty()&&bg.size()!=size_t(channels)))return false;
 // Distinct IDs are required, and all allocation completes before pointer capture.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 if(!constantB&&bg.empty())bg.assign(size_t(channels),0);
 float* ap=constantA?nullptr:ag.data();float* bp=constantB?nullptr:bg.data();
 for(int c=0;c<channels;c++){const float bb=b[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float aa=a[i];
   const float av=g[i]*bb;if(ap)ap[i]+=av;
   const float bv=g[i]*aa;if(bp)bp[size_t(c)]+=bv;
  }
 }return true;
}
}
