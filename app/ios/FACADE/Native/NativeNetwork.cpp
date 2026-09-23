#include "NativeProfile.h"
#include "NativeChannelMul.h"
#include "NativeScatter.h"
#include "NativeColumns.h"
#include "NativeEqualAdd.h"
#include "NativePrelu.h"
#include "NativeBatchNorm.h"
// Fixed-weight CPU forward and reverse operations. No training or external runtime.
#include "NativeNetwork.h"
#include <opencv2/core.hpp>
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <arm_neon.h>
#include <sleef.h>
#include <stdexcept>
#include <string>
#include <vector>
using Vec=std::vector<float>;
struct Tensor {std::vector<int> shape;Vec x,g;int count()const{return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>());}};
struct Node {std::string op,name;std::vector<int> inputs,kernel,stride,padding;Tensor t;Vec weight,bias,alpha,beta;std::vector<int> selected;};
static void require(bool b,const char *message){if(!b)throw std::runtime_error(message);}
static std::vector<int> integers(const cv::FileNode& x){std::vector<int> v;for(auto i:x)v.push_back((int)i);return v;}
static Vec values(const cv::FileNode& spec,const Vec& weights){if(spec.empty())return {};int off=spec["offset"],n=spec["count"];require(off>=0&&n>=0&&(size_t)off+n<=weights.size(),"Weight bounds");return Vec(weights.begin()+off,weights.begin()+off+n);}
// PyTorch 2.8 CPU contiguous float sum order (four-lane NEON, four ILP rows,
// four cascade levels). Derived from its BSD-licensed SumKernel.cpp (see
// THIRD_PARTY_NOTICES.md at the repository root).
static float torch_sum(const float* x,int n){
 int vectors=n/4,rows=vectors/4,ceilLog=0;while((1<<ceilLog)<rows)ceilLog++;
 int power=std::max(4,ceilLog/4),step=1<<power,mask=step-1;float acc[4][4][4]={};int i=0;
 for(;i+step<=rows;){
  for(int j=0;j<step;j++,i++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=x[(i*4+k)*4+lane];
  for(int level=1;level<4;level++){for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++){acc[level][k][lane]+=acc[level-1][k][lane];acc[level-1][k][lane]=0;}if(i&(mask<<(level*power)))break;}
 }
 for(;i<rows;i++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=x[(i*4+k)*4+lane];
 for(int level=1;level<4;level++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=acc[level][k][lane];
 for(int v=rows*4;v<vectors;v++)for(int lane=0;lane<4;lane++)acc[0][0][lane]+=x[v*4+lane];
 for(int k=1;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][0][lane]+=acc[0][k][lane];
 float total=0;for(int j=vectors*4;j<n;j++)total+=x[j];for(int lane=0;lane<4;lane++)total+=acc[0][0][lane];return total;
}
struct Network {
 std::vector<Node> nodes;int output;
 Network(const char* graph,const char* path){ProfileScope profile("native.load");
  std::ifstream file(path,std::ios::binary|std::ios::ate);require(bool(file),"Missing weights");auto size=file.tellg();require(size>0&&size%4==0&&size<1000000000,"Invalid weights size");file.seekg(0);Vec weights(size/4);file.read((char*)weights.data(),size);require(bool(file),"Truncated weights");
  cv::FileStorage s(graph,cv::FileStorage::READ);require(s.isOpened(),"Invalid native graph");require((int)s["version"]==1,"Graph version");output=s["output"];
  for(auto v:s["nodes"]){Node n;n.op=(std::string)v["op"];n.name=(std::string)v["name"];n.t.shape=integers(v["shape"]);n.inputs=integers(v["inputs"]);n.kernel=integers(v["kernel"]);n.stride=integers(v["stride"]);n.padding=integers(v["padding"]);n.weight=values(v["weights"],weights);n.bias=values(v["bias"],weights);n.alpha=values(v["alpha"],weights);n.beta=values(v["beta"],weights);
   require(n.t.shape.size()<=4&&!n.t.shape.empty()&&n.t.count()>0&&n.t.count()<10000000,"Tensor bounds");for(int a:n.inputs)require(a>=0&&a<(int)nodes.size(),"Graph order");nodes.push_back(std::move(n));}
  require(output>=0&&output<(int)nodes.size()&&nodes[output].t.count()==512,"Embedding shape");
 }
 // Shapes are frozen; broadcasting is limited to scalars, equal tensors, or channels.
 static int at(const Tensor& a,const Tensor& out,int i){
  if(a.count()==out.count())return i;if(a.count()==1)return 0;
  require(out.shape.size()==4&&a.count()==out.shape[1],"Unsupported broadcast");return i/(out.shape[2]*out.shape[3]);
 }
 void add(int node,int i,float v){auto& t=nodes[node].t;if(nodes[node].op=="constant")return;if(t.g.empty())t.g.assign(t.count(),0);t.g[i]+=v;}
 static Vec columns(const Tensor& a,const Node& n,bool optimized=true){ProfileScope profile("native.forward.columns");
  int c=a.shape[1],h=a.shape[2],w=a.shape[3],kh=n.kernel[0],kw=n.kernel[1],oh=n.t.shape[2],ow=n.t.shape[3],l=oh*ow;facade_profile_allocation("allocation.native.forward.columns",uint64_t(c)*kh*kw*l*sizeof(float));Vec col(c*kh*kw*l);
  if(optimized&&a.shape.size()==4&&a.shape[0]==1&&n.t.shape.size()==4&&n.t.shape[0]==1&&n.kernel.size()==2&&n.stride.size()==2&&n.padding.size()==2&&facade::columns3x3Eligible(c,h,w,kh,kw,n.stride[0],n.stride[1],n.padding[0],n.padding[1],oh,ow)){facade::fillColumns3x3(col.data(),a.x.data(),c,h,w);return col;}
  for(int z=0;z<c;z++)for(int ky=0;ky<kh;ky++)for(int kx=0;kx<kw;kx++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++){
   int sy=y*n.stride[0]-n.padding[0]+ky,sx=x*n.stride[1]-n.padding[1]+kx;col[((z*kh+ky)*kw+kx)*l+y*ow+x]=(sy>=0&&sy<h&&sx>=0&&sx<w)?a.x[(z*h+sy)*w+sx]:0;
  }return col;
 }
 // BN_ROUTER_BEGIN: original scalar loops remain the false/fallback arms.
 bool bnForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;auto& a=nodes[n.inputs[0]].t;
  if(optimized&&n.inputs.size()==1&& &a!=&t&&
   facade::bnPlaneForward(a.shape,t.shape,a.x,n.alpha,n.beta,y))return true;
    int plane=t.count()/n.alpha.size();for(int i=0;i<t.count();i++){float v=a.x[i]*n.alpha[i/plane];y[i]=v+n.beta[i/plane];}
  return false;
 }
 bool bnReverse(Node& n,bool optimized=true){
  auto& t=n.t;auto& g=t.g;int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.inputs.size()==1&& &a!=&t&&
   facade::bnPlaneBackward(a.shape,t.shape,a.x,n.alpha,n.beta,t.x,g,nodes[id].op=="constant",a.g))return true;
    int plane=t.count()/n.alpha.size();for(int i=0;i<t.count();i++)add(id,i,g[i]*n.alpha[i/plane]);
  return false;
 }
 // BN_ROUTER_END
 // PRELU_ROUTER_BEGIN: original scalar loops remain the false/fallback arms.
 bool preluForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;const int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.inputs.size()<=2){
   const bool owned=!n.weight.empty();const int sid=n.inputs.size()==2?n.inputs[1]:-1;
   if((owned||sid>=0)&&(n.inputs.size()==1||(sid>=0&&size_t(sid)<nodes.size()))){
    const auto& slope=owned?n.weight:nodes[sid].t.x;const auto& shape=owned?t.shape:nodes[sid].t.shape;
    if(facade::preluPlaneForward(a.shape,shape,t.shape,id,sid,owned,sid>=0&&nodes[sid].op=="constant",a.x,slope,y))return true;
   }
  }
    const auto& slope=n.weight.empty()?nodes[n.inputs[1]].t.x:n.weight;int plane=t.count()/slope.size();for(int i=0;i<t.count();i++)y[i]=a.x[i]>0?a.x[i]:a.x[i]*slope[i/plane];
  return false;
 }
 bool preluReverse(Node& n,bool optimized=true){
  auto& t=n.t;auto& g=t.g;int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.inputs.size()<=2){
   const bool owned=!n.weight.empty();const int sid=n.inputs.size()==2?n.inputs[1]:-1;
   if((owned||sid>=0)&&(n.inputs.size()==1||(sid>=0&&size_t(sid)<nodes.size()))){
    const auto& slope=owned?n.weight:nodes[sid].t.x;const auto& shape=owned?t.shape:nodes[sid].t.shape;
    if(facade::preluPlaneBackward(a.shape,shape,t.shape,id,sid,owned,sid>=0&&nodes[sid].op=="constant",nodes[id].op=="constant",a.x,slope,g,a.g))return true;
   }
  }
    const auto& slope=n.weight.empty()?nodes[n.inputs[1]].t.x:n.weight;int plane=t.count()/slope.size();for(int i=0;i<t.count();i++)add(id,i,a.x[i]>0?g[i]:g[i]*slope[i/plane]);
  return false;
 }
 // PRELU_ROUTER_END
 // MUL_ROUTER_BEGIN: original scalar loops remain the false/fallback arms.
 Network():output(2){};
 bool binaryForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;auto& a=nodes[n.inputs[0]].t;
  if(optimized&&n.op=="mul"&&facade::channelMulForward(a.shape,nodes[n.inputs[1]].t.shape,t.shape,n.inputs[0],n.inputs[1],a.x,nodes[n.inputs[1]].t.x,y))return true;
  auto& b=nodes[n.inputs[1]].t;for(int i=0;i<t.count();i++){float aa=a.x[at(a,t,i)],bb=b.x[at(b,t,i)];y[i]=n.op=="add"?aa+bb:n.op=="sub"?aa-bb:n.op=="mul"?aa*bb:aa/bb;}
  return false;
 }
 bool binaryReverse(Node& n,bool optimized=true){
  auto& t=n.t;auto& g=t.g;int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.op=="mul"&&facade::channelMulBackward(a.shape,nodes[n.inputs[1]].t.shape,t.shape,id,n.inputs[1],nodes[id].op=="constant",nodes[n.inputs[1]].op=="constant",a.x,nodes[n.inputs[1]].t.x,g,a.g,nodes[n.inputs[1]].t.g))return true;
  int bid=n.inputs[1];auto& b=nodes[bid].t;for(int i=0;i<t.count();i++){int ai=at(a,t,i),bi=at(b,t,i);float aa=a.x[ai],bb=b.x[bi];add(id,ai,n.op=="mul"?g[i]*bb:n.op=="div"?g[i]/bb:g[i]);add(bid,bi,n.op=="add"?g[i]:n.op=="sub"?-g[i]:n.op=="mul"?g[i]*aa:-g[i]*aa/(bb*bb));}
  return false;
 }
 // MUL_ROUTER_END
 void forward(const float* rgb){ProfileScope profile("native.forward");
  for(auto& n:nodes){auto& t=n.t;t.g.clear();{ProfileScope p("native.activation.resize");t.x.resize(t.count());}auto& y=t.x;
   if(n.op=="input"){require(t.count()==3*112*112,"Input shape");std::copy(rgb,rgb+t.count(),y.begin());continue;}
   if(n.op=="constant"){y=n.weight;continue;}
   auto& a=nodes[n.inputs[0]].t;
   if(n.op=="identity")y=a.x;
   else if(n.op=="conv"){
    auto col=columns(a,n);int oc=t.shape[1],k=a.shape[1]*n.kernel[0]*n.kernel[1],l=t.shape[2]*t.shape[3];
    for(int c=0;c<oc;c++)std::fill(y.begin()+c*l,y.begin()+(c+1)*l,n.bias.empty()?0:n.bias[c]);
    {ProfileScope p("native.forward.conv.gemm");cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,oc,l,k,1,n.weight.data(),k,col.data(),l,n.bias.empty()?0:1,y.data(),l);}
   }else if(n.op=="linear"){
    int out=t.count(),in=a.count();y=n.bias.empty()?Vec(out,0):n.bias;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,out,in,1,a.x.data(),in,n.weight.data(),in,n.bias.empty()?0:1,y.data(),out);
   }else if(n.op=="matmul"){
    auto& b=nodes[n.inputs[1]].t;int out=t.count(),in=a.count();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,out,in,1,a.x.data(),in,b.x.data(),out,0,y.data(),out);
   }else if(n.op=="bn"){
    bnForward(n);
   }else if(n.op=="prelu"){
    preluForward(n);
   }else if(n.op=="pool2"){
    int h=a.shape[2],w=a.shape[3],oh=t.shape[2],ow=t.shape[3];n.selected.resize(t.count());
    for(int c=0;c<t.shape[1];c++)for(int yy=0;yy<oh;yy++)for(int xx=0;xx<ow;xx++){
     int j=(c*oh+yy)*ow+xx,b=(c*h+yy*2)*w+xx*2,selected=b;float best=a.x[b];for(int p:{b+1,b+w,b+w+1})if(a.x[p]>best){best=a.x[p];selected=p;}y[j]=best;n.selected[j]=selected;
    }
   }else if(n.op=="average"){
    int plane=a.count()/t.count();for(int c=0;c<t.count();c++){float sum=torch_sum(a.x.data()+c*plane,plane);y[c]=sum/plane;}
   }else if(n.op=="sigmoid"){
    int i=0;const auto one=vdupq_n_f32(1);for(;i+4<=t.count();i+=4){auto e=Sleef_expf4_u10(vnegq_f32(vld1q_f32(a.x.data()+i)));vst1q_f32(y.data()+i,vdivq_f32(one,vaddq_f32(one,e)));}
    for(;i<t.count();i++)y[i]=1.f/(1.f+std::exp(-a.x[i]));
   }
   else if(n.op=="normalize"){
    double sum=0;for(float v:a.x)sum+=double(v)*v;float norm=std::max(float(std::sqrt(sum)),1e-12f);for(int i=0;i<t.count();i++)y[i]=a.x[i]/norm;
   }else if(n.op=="add"&&facade::equalAddForward(a,nodes[n.inputs[1]].t,t,n.inputs[0]!=n.inputs[1])){
   }else if(n.op=="add"||n.op=="sub"||n.op=="mul"||n.op=="div"){
    binaryForward(n);
   }else throw std::runtime_error("Unknown native operation "+n.op);
  }
 }
 void reverse(const float* ref){ProfileScope profile("native.backward");
  nodes[output].t.g.assign(ref,ref+512);
  for(int ni=(int)nodes.size()-1;ni>=0;ni--){auto& n=nodes[ni];auto& t=n.t;auto& g=t.g;if(g.empty()||n.op=="input"||n.op=="constant")continue;int id=n.inputs[0];auto& a=nodes[id].t;
   if(n.op=="identity"){for(int i=0;i<t.count();i++)add(id,i,g[i]);}
   else if(n.op=="conv"){
    int oc=t.shape[1],c=a.shape[1],h=a.shape[2],w=a.shape[3],kh=n.kernel[0],kw=n.kernel[1],oh=t.shape[2],ow=t.shape[3],l=oh*ow,k=c*kh*kw;facade_profile_allocation("allocation.native.backward.columns",uint64_t(k)*l*sizeof(float));Vec col;{ProfileScope p("native.backward.buffer");col=Vec(k*l);}
    {ProfileScope p("native.backward.conv.gemm");cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,k,l,oc,1,n.weight.data(),k,g.data(),l,0,col.data(),l);}
    ProfileScope scatter("native.backward.conv.scatter");
    facade::scatterColumns(a.g,nodes[id].op=="constant",a.count(),col.data(),c,h,w,kh,kw,oh,ow,n.stride[0],n.stride[1],n.padding[0],n.padding[1]);
   }else if(n.op=="linear"||n.op=="matmul"){
    Vec ga(a.count());const float* w=n.op=="linear"?n.weight.data():nodes[n.inputs[1]].t.x.data();bool trans=n.op=="matmul";
    cblas_sgemm(CblasRowMajor,CblasNoTrans,trans?CblasTrans:CblasNoTrans,1,a.count(),t.count(),1,g.data(),t.count(),w,trans?t.count():a.count(),0,ga.data(),a.count());for(int i=0;i<a.count();i++)add(id,i,ga[i]);
   }else if(n.op=="bn"){
    bnReverse(n);
   }else if(n.op=="prelu"){
    preluReverse(n);
   }else if(n.op=="pool2"){for(int i=0;i<t.count();i++)add(id,n.selected[i],g[i]);}
   else if(n.op=="average"){int plane=a.count()/t.count();for(int i=0;i<a.count();i++)add(id,i,g[i/plane]/plane);}
   else if(n.op=="sigmoid"){for(int i=0;i<t.count();i++)add(id,i,g[i]*(1-t.x[i])*t.x[i]);}
   else if(n.op=="normalize"){
    double ss=0,dot=0;for(int i=0;i<a.count();i++){ss+=double(a.x[i])*a.x[i];dot+=double(g[i])*a.x[i];}float norm=std::max(float(std::sqrt(ss)),1e-12f),factor=float(dot)/(norm*norm*norm);for(int i=0;i<a.count();i++)add(id,i,g[i]/norm-a.x[i]*factor);
   }else if(n.op=="add"&&facade::equalAddBackward(a,nodes[n.inputs[1]].t,t,nodes[id].op=="constant",nodes[n.inputs[1]].op=="constant",id!=n.inputs[1])){
   }else if(n.op=="add"||n.op=="sub"||n.op=="mul"||n.op=="div"){
    binaryReverse(n);
   }else throw std::runtime_error("Unknown reverse operation "+n.op);
   Vec().swap(g);
  }
 }
};
static void error_text(char* out,size_t n,const char* text){if(out&&n)snprintf(out,n,"%s",text);}
void* facade_network_load(const char* graph,const char* weights,char* error,size_t n){try{return new Network(graph,weights);}catch(const std::exception& e){error_text(error,n,e.what());return nullptr;}}
void facade_network_free(void* p){delete static_cast<Network*>(p);}
int facade_network_predict(void* p,const float* rgb,const float* ref,float* e,float* score,float* g,char* error,size_t n){
 try{require(p&&rgb&&ref&&e&&score&&g,"Null native input");auto& model=*static_cast<Network*>(p);model.forward(rgb);auto& emb=model.nodes[model.output].t.x;std::copy(emb.begin(),emb.end(),e);double s=0;for(int i=0;i<512;i++)s+=double(e[i])*ref[i];*score=float(s);model.reverse(ref);auto& gradient=model.nodes[0].t.g;require(gradient.size()==3*112*112,"Native gradient shape");std::copy(gradient.begin(),gradient.end(),g);return 0;}catch(const std::exception& ex){error_text(error,n,ex.what());return -1;}
}
int facade_network_forward(void* p,const float* rgb,const float* ref,float* e,float* score,char* error,size_t n){
 try{require(p&&rgb&&ref&&e&&score,"Null native input");auto& model=*static_cast<Network*>(p);model.forward(rgb);auto& emb=model.nodes[model.output].t.x;std::copy(emb.begin(),emb.end(),e);double s=0;for(int i=0;i<512;i++)s+=double(e[i])*ref[i];*score=float(s);return 0;}catch(const std::exception& ex){error_text(error,n,ex.what());return -1;}
}
// Read-only layer diagnostics used by the port verifier; never changes inference.
extern "C" const float* facade_network_layer(void* p,int i,int* count){auto& ns=static_cast<Network*>(p)->nodes;if(i<0||i>=(int)ns.size()){*count=0;return nullptr;}*count=ns[i].t.x.size();return ns[i].t.x.data();}

// Fixture-only geometry adapter. Production graph validation and inference APIs are unchanged.
static void columns_fixture_input(const float* input,size_t inputCount,const int* g,size_t geometryCount,
 float* out,size_t outputCount,Tensor& a,Node& node) {
 require(input&&g&&out,"Null column fixture input");
 require(geometryCount==11,"Column fixture geometry count");
 for(int i=0;i<11;i++)require(g[i]>=0&&g[i]<=4096,"Column fixture geometry range");
 int c=g[0],h=g[1],w=g[2],kh=g[3],kw=g[4],sh=g[5],sw=g[6],ph=g[7],pw=g[8],oh=g[9],ow=g[10];
 require(c>0&&h>0&&w>0&&kh>0&&kw>0&&sh>0&&sw>0&&oh>0&&ow>0,"Column fixture positive dimensions");
 require(h+2*ph>=kh&&w+2*pw>=kw&&oh==(h+2*ph-kh)/sh+1&&ow==(w+2*pw-kw)/sw+1,"Column fixture output geometry");
 size_t in=size_t(c)*h*w,n=size_t(c)*kh*kw*oh*ow;
 require(in<=10000000&&n<=50000000&&inputCount==in&&outputCount==n,"Column fixture buffer size");
 a.shape={1,c,h,w};a.x.assign(input,input+inputCount);node.kernel={kh,kw};node.stride={sh,sw};node.padding={ph,pw};node.t.shape={1,1,oh,ow};
}
extern "C" int facade_columns_fixture(const float* input,size_t inputCount,const int* geometry,size_t geometryCount,
 int optimized,float* out,size_t outputCount,int* specialized,char* error,size_t capacity) {
 try {
  require((optimized==0||optimized==1)&&specialized,"Column fixture mode");
  Tensor a;Node node;columns_fixture_input(input,inputCount,geometry,geometryCount,out,outputCount,a,node);
  auto values=Network::columns(a,node,optimized!=0);
  std::memcpy(out,values.data(),values.size()*sizeof(float));
  *specialized=optimized&&facade::columns3x3Eligible(geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[5],geometry[6],geometry[7],geometry[8],geometry[9],geometry[10]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}
extern "C" int facade_columns_benchmark(const float* input,size_t inputCount,const int* geometry,size_t geometryCount,
 int optimized,int repeats,float* out,size_t outputCount,int* specialized,double* seconds,char* error,size_t capacity) {
 try {
  require((optimized==0||optimized==1)&&specialized&&seconds,"Column fixture mode");
  require(repeats>0&&repeats<=10000,"Column benchmark repeat count");
  Tensor a;Node node;columns_fixture_input(input,inputCount,geometry,geometryCount,out,outputCount,a,node);
  std::vector<Tensor> variants(4,a);
  for(size_t v=1;v<variants.size();v++)for(float& x:variants[v].x){uint32_t bits;std::memcpy(&bits,&x,4);bits^=uint32_t(v)*0x1011u;std::memcpy(&x,&bits,4);}
  Vec retained;
  const uint64_t start=facade_profile_now();
  for(int i=0;i<repeats;i++){auto current=Network::columns(variants[size_t(i)%4],node,optimized!=0);retained.swap(current);asm volatile("" : : "r"(retained.data()) : "memory");}
  *seconds=double(facade_profile_now()-start)/1e9;
  // Copy/hash only after the timed repeat loop. Profile counters observe every iteration.
  std::memcpy(out,retained.data(),retained.size()*sizeof(float));
  *specialized=optimized&&facade::columns3x3Eligible(geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[5],geometry[6],geometry[7],geometry[8],geometry[9],geometry[10]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}

// Fixture-only scatter adapter; no production graph validation or API changes.
static int scatter_fixture_geometry(const float* col,size_t columnCount,const float* initial,size_t initialCount,
 const int* g,size_t geometryCount,float* out,size_t outputCapacity,bool benchmark) {
 require(col&&g&&out,"Null scatter fixture input");
 require(geometryCount==11,"Scatter fixture geometry count");
 for(int i=0;i<11;i++)require(g[i]>=0&&g[i]<=4096,"Scatter fixture geometry range");
 int c=g[0],h=g[1],w=g[2],kh=g[3],kw=g[4],sh=g[5],sw=g[6],ph=g[7],pw=g[8],oh=g[9],ow=g[10];
 require(c>0&&h>0&&w>0&&kh>0&&kw>0&&sh>0&&sw>0&&oh>0&&ow>0,"Scatter fixture positive dimensions");
 require(h+2*ph>=kh&&w+2*pw>=kw&&oh==(h+2*ph-kh)/sh+1&&ow==(w+2*pw-kw)/sw+1,"Scatter fixture output geometry");
 size_t n=size_t(c)*h*w,m=size_t(c)*kh*kw*oh*ow;
 require(n<=10000000&&m<=50000000&&columnCount==m&&outputCapacity==n*(benchmark?4:1),"Scatter fixture buffer size");
 require(initialCount==0||(initial&&initialCount==n),"Scatter initial gradient size");
 return int(n);
}
static void scatter_fixture_call(Vec& gradient,bool constant,int n,const float* col,const int* g,bool optimized) {
 ProfileScope scope("native.backward.conv.scatter");
 facade::scatterColumns(gradient,constant,n,col,g[0],g[1],g[2],g[3],g[4],g[9],g[10],g[5],g[6],g[7],g[8],optimized);
}
extern "C" int facade_scatter_fixture(const float* col,size_t columnCount,const float* initial,size_t initialCount,
 const int* geometry,size_t geometryCount,int constant,int optimized,float* out,size_t outputCapacity,
 size_t* actualCount,int* specialized,char* error,size_t capacity) {
 try {
  require((constant==0||constant==1)&&(optimized==0||optimized==1)&&actualCount&&specialized,"Scatter fixture mode");
  int n=scatter_fixture_geometry(col,columnCount,initial,initialCount,geometry,geometryCount,out,outputCapacity,false);
  Vec gradient;if(initialCount)gradient.assign(initial,initial+initialCount);
  scatter_fixture_call(gradient,constant!=0,n,col,geometry,optimized!=0);
  if(!gradient.empty())std::memcpy(out,gradient.data(),gradient.size()*sizeof(float));
  *actualCount=gradient.size();
  *specialized=optimized&&!constant&&facade::scatter3x3Eligible(n,geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[9],geometry[10],geometry[5],geometry[6],geometry[7],geometry[8]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}
extern "C" int facade_scatter_benchmark(const float* col,size_t columnCount,const float* initial,size_t initialCount,
 const int* geometry,size_t geometryCount,int optimized,int repeats,float* out,size_t outputCapacity,
 int* specialized,double* seconds,char* error,size_t capacity) {
 try {
  require((optimized==0||optimized==1)&&specialized&&seconds,"Scatter fixture mode");
  require(repeats>0&&repeats<=10000&&repeats%4==0,"Scatter benchmark repeat count");
  int n=scatter_fixture_geometry(col,columnCount,initial,initialCount,geometry,geometryCount,out,outputCapacity,true);
  require(initialCount==size_t(n),"Scatter benchmark nonzero initial gradient");
  std::vector<Vec> columns(4,Vec(col,col+columnCount)),gradients(4,Vec(initial,initial+initialCount));
  for(size_t v=1;v<columns.size();v++)for(float& x:columns[v]){uint32_t bits;std::memcpy(&bits,&x,4);bits^=uint32_t(v)*0x1011u;std::memcpy(&x,&bits,4);}
  const uint64_t start=facade_profile_now();
  for(int i=0;i<repeats;i++){
   const size_t slot=size_t(i)%4;
   if(slot%2==0)Vec().swap(gradients[slot]);
   scatter_fixture_call(gradients[slot],false,n,columns[slot].data(),geometry,optimized!=0);
   asm volatile("" : : "r"(gradients[slot].data()) : "memory");
  }
  *seconds=double(facade_profile_now()-start)/1e9;
  // All four destinations, setup excluded and final copies/digests after timing.
  for(size_t slot=0;slot<4;slot++){
   require(gradients[slot].size()==size_t(n),"Scatter benchmark destination size");
   std::memcpy(out+slot*size_t(n),gradients[slot].data(),size_t(n)*sizeof(float));
  }
  *specialized=optimized&&facade::scatter3x3Eligible(n,geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[9],geometry[10],geometry[5],geometry[6],geometry[7],geometry[8]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}

// MUL_FIXTURE_ADAPTER_BEGIN: bounded synthetic fixtures; not production graph validation.
static std::vector<int> mul_fixture_shape(const int* geometry,int rankIndex,int start) {
 require(geometry[rankIndex]>=1&&geometry[rankIndex]<=4,"Mul fixture rank");
 std::vector<int> shape;for(int j=0;j<4;j++){
  if(j<geometry[rankIndex])shape.push_back(geometry[start+j]);
  else require(geometry[start+j]==0,"Mul fixture shape padding");
 }require(facade::mulBoundedCount(shape)>0,"Mul fixture shape bounds");return shape;
}
static Network mul_fixture_network(const float* a,size_t ac,const float* b,size_t bc,const float* g,size_t n,
 const float* ag,size_t agc,const float* bg,size_t bgc,const int* geometry,size_t geometryCount,
 int constantA,int constantB,float* out,size_t capacity,bool benchmark) {
 require(a&&b&&g&&geometry&&out,"Null mul fixture input");
 require(geometryCount==16,"Mul fixture geometry count");
 require((constantA==0||constantA==1)&&(constantB==0||constantB==1)&&(geometry[3]==0||geometry[3]==1),"Mul fixture flags");
 auto as=mul_fixture_shape(geometry,0,4),bs=mul_fixture_shape(geometry,1,8),os=mul_fixture_shape(geometry,2,12);
 require(ac==size_t(facade::mulBoundedCount(as))&&bc==size_t(facade::mulBoundedCount(bs))&&n==size_t(facade::mulBoundedCount(os)),"Mul fixture buffer size");
 require(capacity==(benchmark?4:1)*(n+ac+bc),"Mul fixture output capacity");
 require((agc==0||(ag&&agc==ac))&&(bgc==0||(bg&&bgc==bc)),"Mul fixture initial gradient");
 if(geometry[3])require(as==bs&&ac==bc&&constantA==constantB&&agc==bgc&&
 std::memcmp(a,b,ac*sizeof(float))==0&&(agc==0||std::memcmp(ag,bg,agc*sizeof(float))==0),"Mul fixture alias inputs");
 Network model;model.nodes.resize(3);
 for(int j=0;j<2;j++){auto& nd=model.nodes[j];nd.op=(j==0?constantA:constantB)?"constant":"input";nd.t.shape=j==0?as:bs;}
 model.nodes[0].t.x.assign(a,a+ac);model.nodes[1].t.x.assign(b,b+bc);
 if(agc)model.nodes[0].t.g.assign(ag,ag+agc);if(bgc)model.nodes[1].t.g.assign(bg,bg+bgc);
 auto& nd=model.nodes[2];nd.op="mul";nd.inputs={0,geometry[3]?0:1};nd.t.shape=os;nd.t.x.resize(n);nd.t.g.assign(g,g+n);
 return model;
}
static void mul_fixture_run(Network& model,bool optimized,int* flags) {
 auto& node=model.nodes[2];
 {ProfileScope p("fixture.mul.forward");flags[0]=model.binaryForward(node,optimized);}
 {ProfileScope p("fixture.mul.backward");flags[1]=model.binaryReverse(node,optimized);Vec().swap(node.t.g);}
 flags[2]=node.t.g.empty();
}
static void mul_fixture_export(Network& model,float* out,size_t* counts) {
 auto& node=model.nodes[2];const auto& a=model.nodes[node.inputs[0]].t;const auto& b=model.nodes[node.inputs[1]].t;
 counts[0]=node.t.x.size();counts[1]=a.g.size();counts[2]=b.g.size();
 std::copy(node.t.x.begin(),node.t.x.end(),out);out+=node.t.x.size();
 if(!a.g.empty())std::copy(a.g.begin(),a.g.end(),out);out+=a.x.size();
 if(!b.g.empty())std::copy(b.g.begin(),b.g.end(),out);
}
extern "C" int facade_mul_fixture(const float* a,size_t ac,const float* b,size_t bc,const float* g,size_t n,
 const float* ag,size_t agc,const float* bg,size_t bgc,const int* geometry,size_t geometryCount,
 int constantA,int constantB,int optimized,float* out,size_t capacity,size_t* counts,int* flags,char* error,size_t errorCapacity) {
 try {
  require((optimized==0||optimized==1)&&counts&&flags,"Mul fixture mode");
  auto model=mul_fixture_network(a,ac,b,bc,g,n,ag,agc,bg,bgc,geometry,geometryCount,constantA,constantB,out,capacity,false);
  int values[3];mul_fixture_run(model,optimized!=0,values);size_t sizes[3];
  mul_fixture_export(model,out,sizes);std::copy(sizes,sizes+3,counts);std::copy(values,values+3,flags);return 0;
 }catch(const std::exception& ex){error_text(error,errorCapacity,ex.what());return -1;}
}
extern "C" int facade_mul_benchmark(const float* a,size_t ac,const float* b,size_t bc,const float* g,size_t n,
 const float* ag,size_t agc,const float* bg,size_t bgc,const int* geometry,size_t geometryCount,
 int optimized,int direction,int repeats,float* out,size_t capacity,size_t* counts,int* dispatched,int* gradientCleared,double* seconds,char* error,size_t errorCapacity) {
 try {
  require((optimized==0||optimized==1)&&(direction==0||direction==1)&&counts&&dispatched&&gradientCleared&&seconds,"Mul benchmark mode");
  require(repeats==128,"Mul benchmark repeat count");
  require(agc==ac&&bgc==bc,"Mul benchmark initial gradients");
  auto base=mul_fixture_network(a,ac,b,bc,g,n,ag,agc,bg,bgc,geometry,geometryCount,0,0,out,capacity,true);
  int nn=0,cc=0,pp=0;require(facade::channelMulEligible(base.nodes[0].t.shape,base.nodes[1].t.shape,base.nodes[2].t.shape,0,base.nodes[2].inputs[1],nn,cc,pp),"Mul benchmark eligible geometry");
  std::vector<Network> models(4,base);std::vector<Vec> gradients(4,base.nodes[2].t.g);
  for(size_t v=1;v<4;v++){
   for(auto* values:{&models[v].nodes[0].t.x,&models[v].nodes[1].t.x,&gradients[v]})
    for(float& value:*values){uint32_t bits;std::memcpy(&bits,&value,4);bits^=uint32_t(v)*0x1011u;std::memcpy(&value,&bits,4);}
  }
  // Populate y for the backward-only case outside all measured intervals.
  if(direction==1)for(auto& model:models)model.binaryForward(model.nodes[2],false);
  uint64_t total=0;int seen=-1;
  for(int iteration=0;iteration<repeats;iteration++){
   const size_t slot=size_t(iteration)%4;auto& model=models[slot];auto& node=model.nodes[2];
   if(slot%2==0){Vec().swap(model.nodes[0].t.g);Vec().swap(model.nodes[1].t.g);}
   else{model.nodes[0].t.g.assign(ag,ag+agc);model.nodes[1].t.g.assign(bg,bg+bgc);}
   if(direction==1)node.t.g=gradients[slot];else node.t.g.clear(); // All resets and source-gradient setup are outside timing.
   const uint64_t start=facade_profile_now();
   if(direction==0){ProfileScope p("fixture.mul.forward");seen=model.binaryForward(node,optimized!=0);}
   else{ProfileScope p("fixture.mul.backward");seen=model.binaryReverse(node,optimized!=0);Vec().swap(node.t.g);}
   asm volatile("" : : "r"(node.t.x.data()),"r"(model.nodes[0].t.g.data()),"r"(model.nodes[1].t.g.data()),"r"(node.t.g.data()) : "memory");
   total+=facade_profile_now()-start;
  }
  *seconds=double(total)/1e9;*dispatched=seen;
  *gradientCleared=std::all_of(models.begin(),models.end(),[](const Network& model){return model.nodes[2].t.g.empty();});
  for(size_t v=0;v<4;v++)mul_fixture_export(models[v],out+v*(n+ac+bc),counts+3*v);
  return 0;
 }catch(const std::exception& ex){error_text(error,errorCapacity,ex.what());return -1;}
}
// MUL_FIXTURE_ADAPTER_END

// Fixture-only equal-add driver; production graph API unchanged.
#include "NativeEqualAddFixtures.h"

// Fixture-only ADD/MUL seam; production graph API unchanged.
#include "NativeAddMulSeam.h"

// Fixture-only PReLU driver; production graph API unchanged.
#include "NativePreluFixtures.h"

// Fixture-only BatchNorm driver; production graph API unchanged.
#include "NativeBatchNormFixtures.h"
