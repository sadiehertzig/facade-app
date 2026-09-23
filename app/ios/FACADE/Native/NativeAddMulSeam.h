#pragma once
// Fixture-only mixed DAG executor. Include after Network/error_text; no production flags.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <iomanip>
#include <sstream>
#include <utility>
namespace add_mul_seam {
struct Graph {std::string id;std::vector<Node> nodes;int output;std::vector<int> forward,backward;};
struct Snapshot {std::vector<Vec> x,g,incoming;std::vector<int> forward,backward;};
static size_t count(const std::vector<int>& shape){
 require(!shape.empty()&&shape.size()<=4,"Seam rank");size_t n=1;
 for(int d:shape){require(d>0&&n<=256u/size_t(d),"Seam bounded shape");n*=size_t(d);}return n;
}
static bool bits(const Vec& a,const Vec& b){
 return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
static std::string hash(const Vec& v){
 unsigned char h[CC_SHA256_DIGEST_LENGTH];CC_SHA256(v.data(),CC_LONG(v.size()*sizeof(float)),h);
 const char* hex="0123456789abcdef";std::string out;for(unsigned char x:h){out+=hex[x>>4];out+=hex[x&15];}return out;
}
static Graph graph(const cv::FileNode& row){
 Graph out;out.id=(std::string)row["id"];out.output=int(row["output"]);
 out.forward=integers(row["expectedForwardDispatch"]);out.backward=integers(row["expectedBackwardDispatch"]);
 for(auto entry:row["nodes"]){
  Node n;n.name=(std::string)entry["name"];n.op=(std::string)entry["op"];
  n.t.shape=integers(entry["shape"]);count(n.t.shape);n.inputs=integers(entry["inputs"]);
  const int id=int(out.nodes.size());
  if(id<3)require(n.op=="input"&&n.inputs.empty(),"Seam leaf");
  else {
   require((n.op=="add"||n.op=="mul")&&n.inputs.size()==2,"Seam operation");
   for(int input:n.inputs)require(input>=0&&input<id,"Seam graph order");
   for(int input:n.inputs){const auto& a=out.nodes[size_t(input)].t;
    const size_t ac=count(a.shape),oc=count(n.t.shape);
    require(ac==oc||ac==1||(n.t.shape.size()==4&&ac==size_t(n.t.shape[1])),"Seam broadcast");
   }
  }
  out.nodes.push_back(std::move(n));
 }
 require(out.nodes.size()>=5&&out.nodes.size()<=12&&out.output==int(out.nodes.size())-1,"Seam graph bounds");
 require(out.forward.size()==out.nodes.size()&&out.backward.size()==out.nodes.size(),"Seam dispatch count");
 return out;
}
static Vec values(size_t n,uint32_t variant){
 Vec v(n);for(size_t i=0;i<n;i++){
  uint32_t u=0x3e800000u|((uint32_t(i)*104729u+variant*8191u)&0x003fffffu)|((i+variant)%2?0x80000000u:0u);
  std::memcpy(&v[i],&u,4);
 }return v;
}
static bool nonzero(const std::string& policy){
 return policy=="nonzero-finite"||policy=="signed-zero-dest"||policy=="constant-all-nonzero";
}
static bool constant(const std::string& policy,int leaf){
 return policy=="constant-all-empty"||policy=="constant-all-nonzero"||
  (policy=="constant-x-empty"&&leaf==0)||(policy=="constant-channel-empty"&&leaf==2);
}
static void leaves(Network& net,const Graph& geom,const std::string& policy,uint32_t variant){
 for(size_t j=0;j<3;j++){
  auto& n=net.nodes[j];n.op=constant(policy,int(j))?"constant":"input";n.t.x=values(count(n.t.shape),variant+uint32_t(j)*7u);
 }
 if(policy=="fresh-negative-zero"||policy=="signed-zero-dest"){
  for(size_t j=0;j<2;j++)for(size_t i=0;i<net.nodes[j].t.x.size();i++){
   uint32_t u=(i+j+variant)%2?0x80000000u:0u;std::memcpy(&net.nodes[j].t.x[i],&u,4);
  }
  std::fill(net.nodes[2].t.x.begin(),net.nodes[2].t.x.end(),1.0f);
 }
 if(geom.id=="channel-cancellation"&&variant==0&&(policy=="fresh-finite"||policy=="nonzero-finite")){
  const float fresh[]={16777216.0f,1.0f,-16777216.0f,1.0f};
  const float old[]={1.0f,-16777216.0f,1.0f,0.0f};
  auto& x=net.nodes[0].t.x;for(size_t i=0;i<x.size();i++)x[i]=(policy=="fresh-finite"?fresh:old)[i%4];
  std::fill(net.nodes[1].t.x.begin(),net.nodes[1].t.x.end(),0.0f);
  std::fill(net.nodes[2].t.x.begin(),net.nodes[2].t.x.end(),1.0f);
 }
}
static Vec upstream(const Graph& geom,const std::string& policy,uint32_t variant){
 Vec g=values(count(geom.nodes[size_t(geom.output)].t.shape),variant+97u);
 if(policy=="fresh-negative-zero"||policy=="signed-zero-dest")for(float& x:g){uint32_t u=0x80000000u;std::memcpy(&x,&u,4);}
 if(geom.id=="channel-cancellation"&&variant==0&&(policy=="fresh-finite"||policy=="nonzero-finite"))std::fill(g.begin(),g.end(),1.0f);
 return g;
}
static void gradients(Network& net,const Graph& geom,const std::string& policy,uint32_t variant){
 // Seed only leaves, after the forward pass clears all g. Internal nonzero states
 // arise from actual repeated/branched consumers in the reverse walk.
 for(size_t j=0;j<3;j++){
  auto& g=net.nodes[j].t.g;
  if(nonzero(policy))g=values(count(net.nodes[j].t.shape),variant+uint32_t(j)*13u+41u);
  if(policy=="signed-zero-dest")for(size_t i=0;i<g.size();i++){uint32_t u=(i+j)%2?0x80000000u:0u;std::memcpy(&g[i],&u,4);}
 }
 if(geom.id=="channel-cancellation"&&variant==0&&policy=="nonzero-finite")
  std::fill(net.nodes[2].t.g.begin(),net.nodes[2].t.g.end(),16777216.0f);
}
static bool forwardNode(Network& net,Node& n,bool optimized){
 auto& t=n.t;auto& a=net.nodes[size_t(n.inputs[0])].t;auto& b=net.nodes[size_t(n.inputs[1])].t;
 if(optimized&&n.op=="add"&&facade::equalAddForward(a,b,t,n.inputs[0]!=n.inputs[1]))return true;
 // false suppresses the MUL path too; scalar mode never enters the ADD guard.
 return net.binaryForward(n,optimized);
}
static bool backwardNode(Network& net,Node& n,bool optimized){
 const int ai=n.inputs[0],bi=n.inputs[1];
 if(optimized&&n.op=="add"&&facade::equalAddBackward(net.nodes[size_t(ai)].t,net.nodes[size_t(bi)].t,n.t,
  net.nodes[size_t(ai)].op=="constant",net.nodes[size_t(bi)].op=="constant",ai!=bi))return true;
 return net.binaryReverse(n,optimized);
}
static std::vector<int> forward(Network& net,bool optimized,size_t& calls){
 std::vector<int> dispatch(net.nodes.size(),-1);
 for(size_t i=0;i<net.nodes.size();i++){
  auto& n=net.nodes[i];n.t.g.clear();n.t.x.resize(count(n.t.shape));
  if(n.op=="input"||n.op=="constant")continue;
  ProfileScope scope("fixture.add_mul_seam.forward");calls++;
  dispatch[i]=forwardNode(net,n,optimized)?1:0;
 }return dispatch;
}
static Snapshot reverse(Network& net,bool optimized,const Vec& rootGradient,const std::vector<int>& f,size_t& calls){
 net.nodes[size_t(net.output)].t.g=rootGradient;
 Snapshot out;out.forward=f;out.backward.assign(net.nodes.size(),-1);out.incoming.resize(net.nodes.size());
 for(int i=int(net.nodes.size())-1;i>=0;i--){
  auto& n=net.nodes[size_t(i)];auto& g=n.t.g;
  if(g.empty()||n.op=="input"||n.op=="constant")continue;
  out.incoming[size_t(i)]=g; // Owned cotangent before this operator or its cleanup.
  {ProfileScope scope("fixture.add_mul_seam.backward");calls++;
   out.backward[size_t(i)]=backwardNode(net,n,optimized)?1:0;
   Vec().swap(g); // Same placement after the numerical branch as production reverse.
  }
 }
 for(const auto& n:net.nodes){out.x.push_back(n.t.x);out.g.push_back(n.t.g);}
 return out;
}
static bool same(const Snapshot& a,const Snapshot& b){
 if(a.x.size()!=b.x.size()||a.g.size()!=b.g.size()||a.incoming.size()!=b.incoming.size())return false;
 for(size_t i=0;i<a.x.size();i++)if(!bits(a.x[i],b.x[i])||!bits(a.g[i],b.g[i])||!bits(a.incoming[i],b.incoming[i]))return false;
 return true;
}
static std::string fingerprint(const Snapshot& s){
 std::string v;for(size_t i=0;i<s.x.size();i++)v+=hash(s.x[i])+hash(s.g[i])+hash(s.incoming[i]);return v;
}
static void snapshotJSON(std::ostream& out,const Snapshot& s){
 out<<"[";
 for(size_t i=0;i<s.x.size();i++){
  if(i)out<<",";
  out<<"{\"node\":"<<i<<",\"activationCount\":"<<s.x[i].size()<<",\"activationSHA256\":"<<std::quoted(hash(s.x[i]))
   <<",\"gradientCount\":"<<s.g[i].size()<<",\"gradientSHA256\":"<<std::quoted(hash(s.g[i]))
   <<",\"incomingCount\":"<<s.incoming[i].size()<<",\"incomingSHA256\":"<<std::quoted(hash(s.incoming[i]))
   <<",\"forwardDispatch\":"<<s.forward[i]<<",\"backwardDispatch\":"<<s.backward[i]
   <<",\"reverseVisited\":"<<(s.backward[i]>=0)<<",\"gradientEmpty\":"<<s.g[i].empty()<<"}";
 }out<<"]";
}
static bool dispatch(const Snapshot& s,const Graph& g,bool optimized){
 for(size_t i=0;i<g.nodes.size();i++){
  const int ef=g.forward[i]<0?-1:(optimized?g.forward[i]:0);
  const int eb=g.backward[i]<0?-1:(optimized?g.backward[i]:0);
  if(s.forward[i]!=ef||s.backward[i]!=eb)return false;
 }return true;
}
static bool released(const Snapshot& s){for(size_t i=3;i<s.g.size();i++)if(!s.g[i].empty())return false;return true;}
static bool unchangedLeaves(const Snapshot& s,const std::array<Vec,3>& input){
 for(size_t i=0;i<3;i++)if(!bits(s.x[i],input[i]))return false;return true;
}
static bool constantsPreserved(const Snapshot& s,const std::array<Vec,3>& initial,const std::string& policy){
 for(int i=0;i<3;i++)if(constant(policy,i)&&!bits(s.g[size_t(i)],initial[size_t(i)]))return false;return true;
}
static bool positiveZeros(const Snapshot& s){
 for(size_t j=0;j<3;j++)for(float x:s.g[j]){uint32_t u;std::memcpy(&u,&x,4);if(u!=0)return false;}return true;
}
static std::string check(const cv::FileStorage& manifest){
 require(int(manifest["formatVersion"])==1,"Seam manifest version");
 std::ostringstream out;out<<std::boolalpha<<"{\"rows\":[";size_t rows=0,fc=0,bc=0;bool passed=true;
 for(auto entry:manifest["graphs"]){
  const Graph geom=graph(entry);
  for(auto pol:manifest["policies"]){
   const std::string policy=(std::string)pol;
   Network a,b;a.nodes=geom.nodes;b.nodes=geom.nodes;a.output=b.output=geom.output;
   leaves(a,geom,policy,0);leaves(b,geom,policy,0);
   const std::array<Vec,3> inputs={a.nodes[0].t.x,a.nodes[1].t.x,a.nodes[2].t.x};
   auto af=forward(a,false,fc),bf=forward(b,true,fc);
   gradients(a,geom,policy,0);gradients(b,geom,policy,0);
   const std::array<Vec,3> initial={a.nodes[0].t.g,a.nodes[1].t.g,a.nodes[2].t.g};
   const Vec g=upstream(geom,policy,0);
   Snapshot firstA=reverse(a,false,g,af,bc),firstB=reverse(b,true,g,bf,bc);
   const std::string retainedA=fingerprint(firstA),retainedB=fingerprint(firstB);
   Snapshot secondA=reverse(a,false,g,af,bc),secondB=reverse(b,true,g,bf,bc);
   leaves(b,geom,policy,31);auto otherF=forward(b,true,fc);gradients(b,geom,policy,31);
   Snapshot other=reverse(b,true,upstream(geom,policy,31),otherF,bc);
   leaves(b,geom,policy,0);auto againF=forward(b,true,fc);gradients(b,geom,policy,0);
   Snapshot again=reverse(b,true,g,againF,bc);
   bool first=same(firstA,firstB),second=same(secondA,secondB),repeat=same(firstB,again);
   bool retained=fingerprint(firstA)==retainedA&&fingerprint(firstB)==retainedB;
   bool inputsOK=unchangedLeaves(firstA,inputs)&&unchangedLeaves(firstB,inputs)&&unchangedLeaves(secondA,inputs)&&unchangedLeaves(secondB,inputs)&&unchangedLeaves(again,inputs);
   bool constantsOK=constantsPreserved(firstA,initial,policy)&&constantsPreserved(firstB,initial,policy)&&constantsPreserved(secondA,initial,policy)&&constantsPreserved(secondB,initial,policy)&&constantsPreserved(again,initial,policy);
   bool release=released(firstA)&&released(firstB)&&released(secondA)&&released(secondB)&&released(other)&&released(again);
   bool routes=dispatch(firstA,geom,false)&&dispatch(firstB,geom,true)&&dispatch(secondA,geom,false)&&dispatch(secondB,geom,true)&&dispatch(other,geom,true)&&dispatch(again,geom,true);
   bool changed=false;for(size_t i=0;i<3;i++)changed=changed||!bits(firstB.x[i],other.x[i]);
   bool zeros=policy!="fresh-negative-zero"||(positiveZeros(firstA)&&positiveZeros(firstB)&&positiveZeros(secondA)&&positiveZeros(secondB)&&positiveZeros(again));
   bool witness=true;
   if(geom.id=="channel-cancellation"&&(policy=="fresh-finite"||policy=="nonzero-finite")){
    witness=firstA.g[2].size()==2&&firstB.g[2].size()==2;
    for(float x:firstA.g[2])witness=witness&&x==1.0f;for(float x:firstB.g[2])witness=witness&&x==1.0f;
   }
   bool ok=first&&second&&repeat&&retained&&inputsOK&&constantsOK&&release&&routes&&changed&&zeros&&witness;passed=passed&&ok;
   if(rows++)out<<",";
   out<<"{\"id\":"<<std::quoted(geom.id+"/"+policy)<<",\"checks\":{\"firstBitsExact\":"<<first<<",\"secondAccumulationBitsExact\":"<<second
    <<",\"repeatAfterOtherInputExact\":"<<repeat<<",\"retainedOwnedCopiesUnchanged\":"<<retained<<",\"inputsUnchanged\":"<<inputsOK
    <<",\"constantsPreserved\":"<<constantsOK<<",\"outputGradientReleaseExact\":"<<release<<",\"dispatchAndVisitExact\":"<<routes
    <<",\"changedInputObserved\":"<<changed<<",\"freshNegativeZeroPositive\":"<<zeros<<",\"channelCancellationWitness\":"<<witness<<"}";
   for(auto named:{std::make_pair("firstLegacy",&firstA),std::make_pair("firstCandidate",&firstB),std::make_pair("secondLegacy",&secondA),std::make_pair("secondCandidate",&secondB),std::make_pair("changedCandidate",&other),std::make_pair("replayCandidate",&again)}){
    out<<","<<std::quoted(named.first)<<":";snapshotJSON(out,*named.second);
   }
   out<<"}";
  }
 }
 out<<"],\"passed\":"<<passed<<",\"fixtureRows\":"<<rows<<",\"forwardCalls\":"<<fc<<",\"backwardCalls\":"<<bc
  <<",\"domain\":\"Small finite DAG fixtures using actual ADD helpers and MUL Network routers. Scalar mode bypasses both fast paths. No model execution, scoped timing claim, malformed-graph/FENV or allocation-failure equivalence.\"}";
 return out.str();
}
}
extern "C" const char* facade_add_mul_seam_report(const char* path,char* error,size_t capacity){
 try{
  require(path,"Seam manifest path");cv::FileStorage manifest(path,cv::FileStorage::READ);require(manifest.isOpened(),"Seam manifest");
  static thread_local std::string report;report=add_mul_seam::check(manifest);return report.c_str();
 }catch(const std::exception& e){error_text(error,capacity,e.what());return nullptr;}
}
