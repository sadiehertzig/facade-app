#pragma once
#include <cstddef>
#include <vector>
namespace facade {
// Narrow finite-graph PReLU families; unsupported valid cases keep the scalar router.
inline int preluBoundedCount(const std::vector<int>& shape) {
 if(shape.empty()||shape.size()>4)return 0;size_t n=1;
 for(int d:shape){if(d<=0||size_t(d)>10000000/n)return 0;n*=size_t(d);}
 return int(n);
}
inline bool preluPlaneEligible(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,
 int& count,int& planes,int& plane) {
 count=preluBoundedCount(os);planes=0;plane=0;
 if(aid<0||sid < -1||aid==sid||!count||as!=os||
 (os.size()!=2&&os.size()!=4)||os[0]!=1)return false;
 if(owned){planes=1;plane=count;return true;}
 if(sid<0||!constantSlope||os.size()!=4)return false;
 planes=os[1];if(planes<=1||preluBoundedCount(ss)!=planes)return false;
 plane=os[2]*os[3];return plane>0&&size_t(planes)*size_t(plane)==size_t(count);
}
inline bool preluPlaneForward(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,
 const std::vector<float>& a,const std::vector<float>& slope,std::vector<float>& y) {
 int count=0,planes=0,plane=0;
 if(!preluPlaneEligible(as,ss,os,aid,sid,owned,constantSlope,count,planes,plane)||
 a.size()!=size_t(count)||slope.size()!=size_t(planes)||y.size()!=size_t(count)||
 &a==&y||&slope==&y)return false;
 const float* ap=a.data();const float* sp=slope.data();float* yp=y.data();
 for(int c=0;c<planes;c++){const float coefficient=sp[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);yp[i]=ap[i]>0?ap[i]:ap[i]*coefficient;}
 }return true;
}
inline bool preluPlaneBackward(const std::vector<int>& as,const std::vector<int>& ss,
 const std::vector<int>& os,int aid,int sid,bool owned,bool constantSlope,bool constantA,
 const std::vector<float>& a,const std::vector<float>& slope,const std::vector<float>& g,
 std::vector<float>& ag) {
 int count=0,planes=0,plane=0;
 if(!preluPlaneEligible(as,ss,os,aid,sid,owned,constantSlope,count,planes,plane)||
 a.size()!=size_t(count)||slope.size()!=size_t(planes)||g.size()!=size_t(count)||
 (!ag.empty()&&ag.size()!=size_t(count))||&ag==&a||&ag==&slope||&ag==&g)return false;
 // Finish allocation before pointers; +0 followed by += preserves the first update.
 if(!constantA&&ag.empty())ag.assign(size_t(count),0);
 const float* ap=a.data();const float* sp=slope.data();const float* gp=g.data();
 float* destination=constantA?nullptr:ag.data();
 for(int c=0;c<planes;c++){const float coefficient=sp[size_t(c)];const size_t start=size_t(c)*size_t(plane);
  for(int x=0;x<plane;x++){const size_t i=start+size_t(x);const float value=ap[i]>0?gp[i]:gp[i]*coefficient;
   if(destination)destination[i]+=value;
  }
 }return true;
}
}
