#pragma once
// Fixture-only executor; include after Network/error_text. No startup work.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <iomanip>
#include <map>
#include <sstream>
namespace prelu_fixture {
struct Geometry {
 std::string id;std::vector<int> as,os,ss;int owned=0;bool constant=false,alias=false,extra=false,specialized=false;
};
struct Snapshot {std::array<Vec,8> buffers;int forward=-1,backward=-1;};
static size_t count(const std::vector<int>& shape){
 require(!shape.empty()&&shape.size()<=4,"Prelu fixture rank");size_t n=1;
 for(int d:shape){require(d>0&&size_t(d)<=10000000u/n,"Prelu fixture shape bounds");n*=size_t(d);}return n;
}
static bool bits(const Vec& a,const Vec& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),4*a.size())==0);}
static std::string hash(const Vec& v){
 unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(v.data(),CC_LONG(v.size()*sizeof(float)),digest);
 const char* hex="0123456789abcdef";std::string out;for(unsigned char c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
}
static float fromBits(uint32_t bits){float v;std::memcpy(&v,&bits,4);return v;}
static Vec values(size_t n,uint32_t variant){
 Vec v(n);for(size_t i=0;i<n;i++)v[i]=fromBits(0x3e800000u|((uint32_t(i)*104729u+variant*8191u)&0x003fffffu)|((i+variant)%2?0x80000000u:0));return v;
}
static Geometry geometry(const cv::FileNode& r){
 Geometry q;q.id=(std::string)r["id"];q.as=integers(r["aShape"]);q.os=integers(r["outShape"]);q.ss=integers(r["slopeShape"]);
 q.owned=int(r["ownedCount"]);q.constant=int(r["coefficientConstant"]);q.alias=int(r["alias"]);q.extra=int(r["hasExtraInput"]);q.specialized=int(r["specialized"]);
 require(count(q.as)==count(q.os),"Prelu fixture element count");require(q.owned>=0,"Prelu fixture owned count");
 if(q.extra)count(q.ss);else require(q.ss.empty()&&q.owned>0,"Prelu fixture coefficient input");
 if(q.alias)require(q.extra&&q.as==q.ss,"Prelu fixture alias");
 size_t m=q.owned?size_t(q.owned):count(q.ss);require(m>0&&m<=count(q.os)&&count(q.os)%m==0,"Prelu fixture slope");return q;
}
static std::map<std::string,Geometry> geometries(const cv::FileStorage& manifest){
 require(int(manifest["formatVersion"])==1,"Prelu manifest version");std::map<std::string,Geometry> out;
 for(auto row:manifest["geometries"]){auto q=geometry(row);require(out.emplace(q.id,q).second,"Prelu duplicate geometry");}return out;
}
static bool initial(const std::string& p){return p=="nonzero-finite"||p=="constant-nonzero"||p=="nonzero-cancellation";}
static bool constantA(const std::string& p){return p=="constant-empty"||p=="constant-nonzero";}
static const Vec& coefficient(const Network& n){const auto& node=n.nodes[2];return node.weight.empty()?n.nodes[size_t(node.inputs[1])].t.x:node.weight;}
static Vec& coefficient(Network& n){auto& node=n.nodes[2];return node.weight.empty()?n.nodes[size_t(node.inputs[1])].t.x:node.weight;}
static Network network(const Geometry& q,const std::string& policy,uint32_t variant=0){
 Network n;n.nodes.resize(3);n.output=2;
 n.nodes[0].op=constantA(policy)?"constant":"input";n.nodes[0].t.shape=q.as;n.nodes[0].t.x=values(count(q.as),variant);
 n.nodes[1].op=q.constant?"constant":"input";n.nodes[1].t.shape=q.extra?q.ss:std::vector<int>{1};
 if(q.extra&&!q.alias)n.nodes[1].t.x=values(count(q.ss),variant+7);
 auto& out=n.nodes[2];out.op="prelu";out.t.shape=q.os;out.t.x.resize(count(q.os));out.inputs={0};if(q.extra)out.inputs.push_back(q.alias?0:1);
 if(q.owned)out.weight=values(size_t(q.owned),variant+13);
 if(policy=="fresh-negative-zero"||policy=="nonzero-cancellation")std::fill(n.nodes[0].t.x.begin(),n.nodes[0].t.x.end(),1.0f);
 if(policy=="signed-zero-input"){
  for(size_t i=0;i<n.nodes[0].t.x.size();i++)n.nodes[0].t.x[i]=fromBits(i%2?0x80000000u:0u);
  if(!q.alias){auto& s=coefficient(n);const uint32_t pattern[]={0xbf800000u,0u,0x80000000u,0x3f800000u};for(size_t i=0;i<s.size();i++)s[i]=fromBits(pattern[i%4]);}
 }
 return n;
}
static void validate(const Network& n,bool reverse){
 const auto& node=n.nodes[2];size_t ac=count(n.nodes[0].t.shape),oc=count(node.t.shape);
 require(ac==oc&&n.nodes[0].t.x.size()==ac&&node.t.x.size()==oc,"Prelu fixture activation buffer");
 require(!node.inputs.empty()&&node.inputs[0]==0&&node.inputs.size()<=2,"Prelu fixture input IDs");
 require(node.inputs.size()!=2||(node.inputs[1]>=0&&node.inputs[1]<2),"Prelu fixture input IDs");
 require(!node.weight.empty()||node.inputs.size()==2,"Prelu fixture slope");
 const auto& slope=coefficient(n);require(!slope.empty()&&slope.size()<=oc&&oc%slope.size()==0,"Prelu fixture slope");
 if(node.inputs.size()==2)require(n.nodes[size_t(node.inputs[1])].t.x.size()==count(n.nodes[size_t(node.inputs[1])].t.shape),"Prelu fixture coefficient buffer");
 require(n.nodes[0].t.g.empty()||n.nodes[0].t.g.size()==ac,"Prelu fixture destination buffer");
 if(reverse)require(node.t.g.empty()||node.t.g.size()==oc,"Prelu fixture upstream buffer");
}
static void seed(Network& n,const Geometry& q,const std::string& p){
 for(auto& node:n.nodes)node.t.g.clear();
 if(initial(p)){
  n.nodes[0].t.g=values(count(q.as),41);
  if(q.extra&&!q.alias)n.nodes[1].t.g=values(count(q.ss),59);
 }
 if(p=="nonzero-cancellation")std::fill(n.nodes[0].t.g.begin(),n.nodes[0].t.g.end(),16777216.0f);
}
static Vec upstream(const Geometry& q,const std::string& p,int accumulation=0,uint32_t variant=0){
 if(p=="empty-upstream")return {};
 Vec g=values(count(q.os),97+variant);
 if(p=="fresh-negative-zero")std::fill(g.begin(),g.end(),fromBits(0x80000000u));
 if(p=="signed-zero-input")for(size_t i=0;i<g.size();i++)g[i]=i%2?-1.0f:1.0f;
 if(p=="nonzero-cancellation")std::fill(g.begin(),g.end(),accumulation==1?-16777216.0f:1.0f);
 return g;
}
static int forward(Network& n,bool optimized,size_t& calls){
 validate(n,false);for(auto& node:n.nodes)node.t.g.clear();n.nodes[2].t.x.resize(count(n.nodes[2].t.shape));
 ProfileScope p("fixture.prelu.forward");calls++;return n.preluForward(n.nodes[2],optimized)?1:0;
}
static Snapshot snapshot(const Network& n,const Vec& incoming,int f,int b){
 const auto& node=n.nodes[2];Snapshot s;s.forward=f;s.backward=b;
 s.buffers[0]=node.t.x;s.buffers[1]=n.nodes[0].t.g;
 if(node.inputs.size()==2){s.buffers[2]=n.nodes[size_t(node.inputs[1])].t.g;s.buffers[4]=n.nodes[size_t(node.inputs[1])].t.x;}
 s.buffers[3]=n.nodes[0].t.x;s.buffers[5]=node.weight;s.buffers[6]=incoming;s.buffers[7]=node.t.g;return s;
}
static Snapshot reverse(Network& n,const Vec& g,bool optimized,int f,size_t& calls){
 n.nodes[2].t.g=g;validate(n,true);int b=-1;
 if(!g.empty()){ProfileScope p("fixture.prelu.backward");calls++;b=n.preluReverse(n.nodes[2],optimized)?1:0;Vec().swap(n.nodes[2].t.g);}
 return snapshot(n,g,f,b);
}
static bool same(const Snapshot& a,const Snapshot& b){for(size_t i=0;i<8;i++)if(!bits(a.buffers[i],b.buffers[i]))return false;return true;}
static std::string fingerprint(const Snapshot& s){std::string out;for(const auto& b:s.buffers)out+=hash(b);return out;}
static void snapshotJSON(std::ostream& out,const Snapshot& s){
 out<<"{\"hashes\":[";for(size_t i=0;i<8;i++){if(i)out<<",";out<<std::quoted(hash(s.buffers[i]));}
 out<<"],\"counts\":[";for(size_t i=0;i<8;i++){if(i)out<<",";out<<s.buffers[i].size();}
 out<<"],\"forwardDispatch\":"<<s.forward<<",\"backwardDispatch\":"<<s.backward<<",\"reverseVisited\":"<<(s.backward>=0)<<",\"outputGradientCleared\":"<<s.buffers[7].empty()<<"}";
}
static void changedCoefficient(Network& n,const Vec& original,bool changed){
 coefficient(n)=original;if(changed)for(float& v:coefficient(n)){uint32_t bits;std::memcpy(&bits,&v,4);bits^=31u*0x1011u;v=fromBits(bits);}
}
static bool positiveZero(const Vec& v){for(float f:v){uint32_t bits;std::memcpy(&bits,&f,4);if(bits!=0)return false;}return true;}
static std::string check(const cv::FileStorage& manifest){
 auto shapes=geometries(manifest);std::ostringstream out;out<<std::boolalpha<<std::setprecision(17)<<"{\"rows\":[";
 size_t rows=0,fc=0,bc=0;bool passed=true;
 for(auto entry:manifest["cases"]){
  const std::string id=(std::string)entry["id"],key=(std::string)entry["geometry"],policy=(std::string)entry["policy"];
  const auto& q=shapes.at(key);Network a=network(q,policy),b=network(q,policy);
  const Vec originalCoefficient=coefficient(a);std::array<Snapshot,10> s;
  int af=forward(a,false,fc),bf=forward(b,true,fc);seed(a,q,policy);seed(b,q,policy);
  const Snapshot beforeA=snapshot(a,{},af,-1),beforeB=snapshot(b,{},bf,-1);
  for(int step=0;step<3;step++){auto g=upstream(q,policy,step);s[size_t(step)*2]=reverse(a,g,false,af,bc);s[size_t(step)*2+1]=reverse(b,g,true,bf,bc);}
  const std::string retained=fingerprint(s[0])+fingerprint(s[1]);
  changedCoefficient(a,originalCoefficient,true);changedCoefficient(b,originalCoefficient,true);
  af=forward(a,false,fc);bf=forward(b,true,fc);seed(a,q,policy);seed(b,q,policy);
  s[6]=reverse(a,upstream(q,policy,0,31),false,af,bc);s[7]=reverse(b,upstream(q,policy,0,31),true,bf,bc);
  changedCoefficient(a,originalCoefficient,false);changedCoefficient(b,originalCoefficient,false);
  af=forward(a,false,fc);bf=forward(b,true,fc);seed(a,q,policy);seed(b,q,policy);
  s[8]=reverse(a,upstream(q,policy),false,af,bc);s[9]=reverse(b,upstream(q,policy),true,bf,bc);
  bool exact=true;for(size_t i=0;i<10;i+=2)exact=exact&&same(s[i],s[i+1]);
  const bool replay=same(s[0],s[8])&&same(s[1],s[9]);
  bool routes=true,cleanup=true,constantOK=true,coefficientOK=true,inputOK=true;
  for(size_t i=0;i<10;i++){
   int expected=i%2&&q.specialized?1:0;routes=routes&&s[i].forward==expected&&s[i].backward==(policy=="empty-upstream"?-1:expected);
   cleanup=cleanup&&s[i].buffers[7].empty();
   if(constantA(policy))constantOK=constantOK&&bits(s[i].buffers[1],beforeA.buffers[1]);
   if(!q.alias)coefficientOK=coefficientOK&&bits(s[i].buffers[2],beforeA.buffers[2]);
   if(i<6||i>=8)inputOK=inputOK&&bits(s[i].buffers[3],beforeA.buffers[3])&&bits(s[i].buffers[4],beforeA.buffers[4])&&bits(s[i].buffers[5],beforeA.buffers[5]);
  }
  const bool retainedOK=retained==fingerprint(s[0])+fingerprint(s[1]);
  bool zero=policy!="fresh-negative-zero"||(positiveZero(s[0].buffers[1])&&positiveZero(s[1].buffers[1]));
  bool cancellation=true;if(policy=="nonzero-cancellation")for(float x:s[5].buffers[1])cancellation=cancellation&&x==1.0f;
  const bool changed=!bits(q.owned?s[0].buffers[5]:s[0].buffers[4],q.owned?s[6].buffers[5]:s[6].buffers[4]);
  bool ok=exact&&replay&&routes&&cleanup&&constantOK&&coefficientOK&&inputOK&&retainedOK&&zero&&cancellation&&changed;passed=passed&&ok;
  if(rows++)out<<",";out<<"{\"id\":"<<std::quoted(id)<<",\"checks\":{\"allPairsBitsExact\":"<<exact<<",\"replayExact\":"<<replay<<",\"dispatchExact\":"<<routes
   <<",\"outputGradientCleared\":"<<cleanup<<",\"constantActivationPreserved\":"<<constantOK<<",\"separateCoefficientGradientPreserved\":"<<coefficientOK
   <<",\"originalInputsPreserved\":"<<inputOK<<",\"retainedOwnedCopiesPreserved\":"<<retainedOK<<",\"freshNegativeZeroPositive\":"<<zero
   <<",\"cancellationOrderObserved\":"<<cancellation<<",\"changedCoefficientObserved\":"<<changed<<"}";
  out<<",\"initial\":";snapshotJSON(out,beforeA);
  const char* names[]={"firstLegacy","firstCandidate","secondLegacy","secondCandidate","thirdLegacy","thirdCandidate","changedLegacy","changedCandidate","replayLegacy","replayCandidate"};
  for(size_t i=0;i<10;i++){out<<","<<std::quoted(names[i])<<":";snapshotJSON(out,s[i]);}out<<"}";
 }
 out<<"],\"errors\":[";size_t errorRows=0;
 for(const char* errorID:{"rank","shape-bounds","empty-slope","nondivisible-slope","short-activation","short-gradient"}){
  std::array<std::string,2> messages;bool unchanged=true;
  for(int arm=0;arm<2;arm++){
   Network n=network(shapes.at("channel-row3"),"fresh-finite");n.nodes[2].t.g=values(6,1);
   std::string id=errorID;if(id=="rank")n.nodes[0].t.shape.clear();if(id=="shape-bounds")n.nodes[0].t.shape[0]=0;
   if(id=="empty-slope")n.nodes[1].t.x.clear();if(id=="nondivisible-slope"){n.nodes[1].t.shape={4};n.nodes[1].t.x.resize(4);}
   if(id=="short-activation")n.nodes[0].t.x.pop_back();if(id=="short-gradient")n.nodes[2].t.g.pop_back();
   const std::string before=fingerprint(snapshot(n,{},-1,-1));
   try{validate(n,true);messages[size_t(arm)]="NO_ERROR";}catch(const std::exception& e){messages[size_t(arm)]=e.what();}
   unchanged=unchanged&&before==fingerprint(snapshot(n,{},-1,-1));
  }
  bool ok=messages[0]!="NO_ERROR"&&messages[0]==messages[1]&&unchanged;passed=passed&&ok;
  if(errorRows++)out<<",";out<<"{\"id\":"<<std::quoted(errorID)<<",\"legacyCode\":-1,\"candidateCode\":-1,\"legacyError\":"<<std::quoted(messages[0])
   <<",\"candidateError\":"<<std::quoted(messages[1])<<",\"outputUnchanged\":"<<unchanged<<"}";
 }
 out<<"],\"passed\":"<<passed<<",\"fixtureRows\":"<<rows<<",\"forwardCalls\":"<<fc<<",\"backwardCalls\":"<<bc
  <<",\"domain\":\"Finite bounded PReLU router fixtures. Actual scalar/candidate routers; no duplicate optimized math. Defined adapter errors precede routers. No malformed legacy graph, nonfinite/FENV or allocation-failure equivalence.\"}";return out.str();
}
static std::string benchmark(const cv::FileStorage& manifest,bool optimized){
 require(int(manifest["repeats"])==128&&int(manifest["rotatingPayloads"])==4,"Prelu benchmark schedule");
 auto shapes=geometries(manifest);std::ostringstream out;out<<std::boolalpha<<std::setprecision(17)<<"{\"mode\":"<<std::quoted(optimized?"prelu-bench-b":"prelu-bench-a")<<",\"rows\":[";
 size_t rows=0,fc=0,bc=0,setup=0;double scoped=0;
 for(auto entry:manifest["benchmark"]){const auto& q=shapes.at((std::string)entry);require(q.specialized,"Prelu benchmark geometry");
  for(int direction=0;direction<2;direction++){
   std::vector<Network> models;std::array<Vec,4> gs,ags,cgs;
   for(uint32_t slot=0;slot<4;slot++){
    models.push_back(network(q,"fresh-finite",slot));auto& n=models.back();gs[slot]=upstream(q,"fresh-finite",0,slot);
   }
   for(size_t slot=0;slot<4;slot++){
    auto& n=models[slot];ags[slot]=values(count(q.as),41+uint32_t(slot));
    if(q.extra)cgs[slot]=values(count(q.ss),59+uint32_t(slot));
    if(direction==1){n.preluForward(n.nodes[2],false);setup++;}
   }
   uint64_t total=0;bool consistent=true;int dispatched=-1;
   for(int i=0;i<128;i++){
    size_t slot=size_t(i)%4;auto& n=models[slot];auto& node=n.nodes[2];
    if(slot%2==0){Vec().swap(n.nodes[0].t.g);Vec().swap(n.nodes[1].t.g);}
    else{n.nodes[0].t.g=ags[slot];n.nodes[1].t.g=cgs[slot];}
    if(direction==1)node.t.g=gs[slot];else node.t.g.clear();
    const uint64_t start=facade_profile_now();int used;
    if(direction==0){ProfileScope scope("fixture.prelu.forward");used=n.preluForward(node,optimized)?1:0;fc++;}
    else{ProfileScope scope("fixture.prelu.backward");used=n.preluReverse(node,optimized)?1:0;Vec().swap(node.t.g);bc++;}
    asm volatile("" : : "r"(node.t.x.data()),"r"(n.nodes[0].t.g.data()),"r"(n.nodes[1].t.g.data()),"r"(node.t.g.data()) : "memory");
    total+=facade_profile_now()-start;
    consistent=consistent&&(used==(optimized?1:0));dispatched=used;
   }
   const double seconds=double(total)/1e9;scoped+=seconds;
   bool cleared=true;for(const auto& n:models)cleared=cleared&&n.nodes[2].t.g.empty();
   if(rows++)out<<",";out<<"{\"id\":"<<std::quoted(q.id)<<",\"direction\":"<<std::quoted(direction==0?"forward":"backward")<<",\"seconds\":"<<seconds
    <<",\"dispatch\":"<<(dispatched==1)<<",\"dispatchConsistent\":"<<consistent<<",\"outputGradientCleared\":"<<cleared<<",\"outputSHA256\":[";
   for(size_t slot=0;slot<4;slot++){if(slot)out<<",";auto s=snapshot(models[slot],{},-1,-1);out<<"[";for(size_t j=0;j<3;j++){if(j)out<<",";out<<std::quoted(hash(s.buffers[j]));}out<<"]";}
   out<<"],\"outputCounts\":[";for(size_t slot=0;slot<4;slot++){if(slot)out<<",";auto s=snapshot(models[slot],{},-1,-1);out<<"[";for(size_t j=0;j<3;j++){if(j)out<<",";out<<s.buffers[j].size();}out<<"]";}
   out<<"],\"slotObservations\":[";for(size_t slot=0;slot<4;slot++){if(slot)out<<",";snapshotJSON(out,snapshot(models[slot],direction==1?gs[slot]:Vec{},direction==0?dispatched:0,direction==1?dispatched:-1));}out<<"]}";
  }
 }
 out<<"],\"repeats\":128,\"rotatingPayloads\":4,\"scopedSeconds\":"<<scoped<<",\"forwardCalls\":"<<fc<<",\"backwardCalls\":"<<bc<<",\"untimedScalarSetupForwardCalls\":"<<setup
  <<",\"barrier\":\"Four finite inputs/coefficient/upstream payloads rotate. Symmetric fresh/nonzero resets and500 backward-output setup forwards outside timing. Actual router, lazy activation-gradient preparation and single reverse cleanup inside; opaque complete-output barrier before stopclock. All4 slot buffers observed after each timed row. Sequential binary64 sum of row seconds.\"}";return out.str();
}
}
extern "C" const char* facade_prelu_report(const char* path,int mode,char* error,size_t capacity){
 try{
  require(path,"Prelu fixture manifest path");require(mode>=0&&mode<=2,"Prelu fixture report mode");
  cv::FileStorage manifest(path,cv::FileStorage::READ);require(manifest.isOpened(),"Prelu fixture manifest");
  static thread_local std::string report;report=mode==0?prelu_fixture::check(manifest):prelu_fixture::benchmark(manifest,mode==2);return report.c_str();
 }catch(const std::exception& e){error_text(error,capacity,e.what());return nullptr;}
}
