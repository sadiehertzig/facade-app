#pragma once
// Fixture-only adapters. Tensor/Network are already declared by NativeNetwork.cpp.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <iomanip>
#include <sstream>
namespace equal_add_fixture {
struct Geometry {std::string id;std::vector<int> a,b,out;bool alias=false,eligible=false;};
struct State {Tensor a,b,t;bool alias=false,constantA=false,constantB=false;Tensor& rhs(){return alias?a:b;}const Tensor& rhs()const{return alias?a:b;}};
static size_t countShape(const std::vector<int>& s){
 require(!s.empty()&&s.size()<=4,"Add fixture rank");
 size_t n=1;for(int d:s){require(d>0,"Add fixture positive dimensions");require(n<=9999999u/size_t(d),"Add fixture count bounds");n*=size_t(d);}return n;
}
static std::string hash(const Vec& v){
 unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(v.data(),CC_LONG(v.size()*sizeof(float)),bytes);
 static const char* hex="0123456789abcdef";std::string s;for(unsigned char c:bytes){s+=hex[c>>4];s+=hex[c&15];}return s;
}
static Vec finiteValues(size_t n,uint32_t variant=0){
 Vec out(n);for(size_t i=0;i<n;i++){uint32_t bits=0x3f000000u|((uint32_t(i)*1664525u+1013904223u+variant*104729u)&0x007fffffu)|(i%2?0x80000000u:0);std::memcpy(&out[i],&bits,4);}return out;
}
static State make(const Geometry& geom,const std::string& policy,uint32_t variant=0){
 size_t na=countShape(geom.a),nb=countShape(geom.b),no=countShape(geom.out);
 require(!geom.alias||geom.a==geom.b,"Add fixture alias geometry");
 State s;s.a.shape=geom.a;s.b.shape=geom.b;s.t.shape=geom.out;s.alias=geom.alias;
 s.a.x=finiteValues(na,variant);s.b.x=finiteValues(nb,variant+7);s.t.x=Vec(no,0.0f);s.t.g=finiteValues(no,variant+13);
 s.constantA=policy.find("constant-a")!=std::string::npos||policy.find("constant-both")!=std::string::npos;
 s.constantB=policy.find("constant-b")!=std::string::npos;
 if(s.alias){s.constantA=s.constantA||s.constantB;s.constantB=s.constantA;s.b.x=s.a.x;}
 if(policy.find("nonzero")!=std::string::npos||policy=="cancellation"){s.a.g=finiteValues(na,variant+19);s.b.g=finiteValues(nb,variant+23);}
 if(policy=="mixed-a-empty")s.b.g=finiteValues(nb,variant+23);
 if(policy=="mixed-b-empty")s.a.g=finiteValues(na,variant+19);
 if(policy=="fresh-negative-zero"||policy=="signed-zero-dest"){
  for(float& x:s.t.g){uint32_t bits=0x80000000u;std::memcpy(&x,&bits,4);}
  for(size_t i=0;i<na;i++){uint32_t bits=i%2?0u:0x80000000u;std::memcpy(&s.a.x[i],&bits,4);}
  for(size_t i=0;i<nb;i++){uint32_t bits=i%2?0x80000000u:0u;std::memcpy(&s.b.x[i],&bits,4);}
  if(policy=="signed-zero-dest"){s.a.g=s.a.x;s.b.g=s.b.x;}
 }
 if(policy=="cancellation"){const float values[]={16777216.0f,1.0f,-16777216.0f,1.0f};for(size_t i=0;i<no;i++)s.t.g[i]=values[i%4];std::fill(s.a.g.begin(),s.a.g.end(),16777216.0f);std::fill(s.b.g.begin(),s.b.g.end(),16777216.0f);}
 if(s.alias){s.b.x=s.a.x;s.b.g=s.a.g;}
 return s;
}
static void validate(const State& s){
 const size_t na=countShape(s.a.shape),nb=countShape(s.rhs().shape),no=countShape(s.t.shape);
 require(s.a.x.size()==na&&s.rhs().x.size()==nb&&s.t.x.size()==no&&s.t.g.size()==no,"Add fixture buffer size");
 require((s.a.g.empty()||s.a.g.size()==na)&&(s.rhs().g.empty()||s.rhs().g.size()==nb),"Add fixture gradient size");
}
static void scalarForward(State& s){
 auto& a=s.a;auto& b=s.rhs();auto& t=s.t;auto& y=t.x;
 for(int i=0;i<t.count();i++){float aa=a.x[Network::at(a,t,i)],bb=b.x[Network::at(b,t,i)];y[i]=aa+bb;}
}
static void scalarBackward(State& s){
 auto add=[](Tensor& t,bool constant,int i,float v){if(constant)return;if(t.g.empty())t.g.assign(t.count(),0);t.g[i]+=v;};
 auto& a=s.a;auto& b=s.rhs();auto& t=s.t;auto& g=t.g;
 for(int i=0;i<t.count();i++){int ai=Network::at(a,t,i),bi=Network::at(b,t,i);float aa=a.x[ai],bb=b.x[bi];(void)aa;(void)bb;add(a,s.constantA,ai,g[i]);add(b,s.constantB,bi,g[i]);}
}
static bool invoke(State& s,bool optimized,bool reverse){
 ProfileScope scope(reverse?"fixture.equal_add.backward":"fixture.equal_add.forward");
 bool specialized=false;
 if(reverse){
  specialized=optimized&&facade::equalAddBackward(s.a,s.rhs(),s.t,s.constantA,s.constantB,!s.alias);
  if(!specialized)scalarBackward(s);
  Vec().swap(s.t.g); // Same final output-gradient release as production reverse.
 }else{
  specialized=optimized&&facade::equalAddForward(s.a,s.rhs(),s.t,!s.alias);
  if(!specialized)scalarForward(s);
 }
 return specialized;
}
static std::array<const Vec*,3> outputs(const State& s){return {&s.t.x,&s.a.g,&s.rhs().g};}
static bool same(const State& a,const State& b){
 auto aa=outputs(a),bb=outputs(b);
 for(size_t i=0;i<3;i++)if(aa[i]->size()!=bb[i]->size()||(!aa[i]->empty()&&std::memcmp(aa[i]->data(),bb[i]->data(),aa[i]->size()*4)))return false;
 return a.t.g.size()==b.t.g.size()&&(a.t.g.empty()||std::memcmp(a.t.g.data(),b.t.g.data(),a.t.g.size()*4)==0);
}
static void hashes(std::ostream& out,const State& s){
 out<<"[";auto v=outputs(s);for(size_t i=0;i<3;i++){if(i)out<<",";out<<std::quoted(hash(*v[i]));}out<<"]";
}
static void counts(std::ostream& out,const State& s){
 out<<"[";auto v=outputs(s);for(size_t i=0;i<3;i++){if(i)out<<",";out<<v[i]->size();}out<<"]";
}
static Geometry geometry(const cv::FileNode& node){
 Geometry g;g.id=(std::string)node["id"];g.a=integers(node["aShape"]);g.b=integers(node["bShape"]);g.out=integers(node["outShape"]);g.alias=int(node["alias"])!=0;g.eligible=int(node["eligible"])!=0;return g;
}
static std::string check(const cv::FileStorage& manifest){
 std::ostringstream out;out<<std::boolalpha<<"{\"rows\":[";size_t rowCount=0,calls[2]={0,0};bool allPassed=true;
 for(auto node:manifest["fixtures"]){const Geometry geom=geometry(node);for(auto pol:manifest["policies"]){const std::string policy=(std::string)pol;for(bool reverse:{false,true}){
  State seed=make(geom,policy);validate(seed);State a=seed,b=seed;
  bool aspec=invoke(a,false,reverse),bspec=invoke(b,true,reverse);calls[reverse]+=2;
  State savedA=a,savedB=b;
  a.t.g=seed.t.g;b.t.g=seed.t.g;invoke(a,false,reverse);invoke(b,true,reverse);calls[reverse]+=2;
  State other=make(geom,policy,31);validate(other);invoke(other,true,reverse);calls[reverse]++;
  State again=seed;bool againSpec=invoke(again,true,reverse);calls[reverse]++;
  bool first=same(savedA,savedB),second=same(a,b),retained=same(savedA,savedB)&&same(savedA,again);
  bool inputs=seed.a.x==savedA.a.x&&seed.a.x==savedB.a.x&&hash(seed.a.x)==hash(savedB.a.x)&&hash(seed.rhs().x)==hash(savedB.rhs().x);
  bool constants=(!seed.constantA||(hash(seed.a.g)==hash(savedB.a.g)))&&(!seed.constantB||(hash(seed.rhs().g)==hash(savedB.rhs().g)));
  bool cleanup=!reverse||(savedA.t.g.empty()&&savedB.t.g.empty()&&a.t.g.empty()&&b.t.g.empty());
  bool positiveZero=true;
  if(reverse&&policy=="fresh-negative-zero"){for(auto ptr:{&savedB.a.g,&savedB.rhs().g})for(float x:*ptr){uint32_t bits;std::memcpy(&bits,&x,4);positiveZero=positiveZero&&bits==0;}}
  bool dispatch=!aspec&&bspec==geom.eligible&&againSpec==geom.eligible;
  bool pass=first&&second&&retained&&inputs&&constants&&cleanup&&positiveZero&&dispatch;
  allPassed=allPassed&&pass;
  if(rowCount++)out<<",";
  out<<"{\"id\":"<<std::quoted(geom.id+"/"+policy+(reverse?"/backward":"/forward"))<<",\"checks\":{"
     <<"\"firstBitsExact\":"<<first<<",\"secondAccumulationBitsExact\":"<<second<<",\"retainedOutputUnchanged\":"<<retained
     <<",\"repeatAfterOtherInputExact\":"<<same(savedA,again)<<",\"inputsUnchanged\":"<<inputs<<",\"constantsPreserved\":"<<constants
     <<",\"outputGradientReleaseExact\":"<<cleanup<<",\"freshNegativeZeroPositive\":"<<positiveZero<<",\"dispatchExact\":"<<dispatch
     <<"},\"firstSHA256\":";hashes(out,savedA);out<<",\"secondSHA256\":";hashes(out,a);out<<",\"firstCounts\":";counts(out,savedA);out<<",\"secondCounts\":";counts(out,a);out<<"}";
 }}}
 out<<"],\"errors\":[";
 const std::vector<std::string> errorNames={"zero-dimension","rank-five","oversized-count","short-input","short-gradient","alias-geometry","unsupported-broadcast"};
 size_t errorIndex=0;
 for(const auto& name:errorNames){
  Geometry g{"error",{3},{3},{3},false,true};
  if(name=="zero-dimension")g.a={0};
  if(name=="rank-five")g.a={1,1,1,1,3};
  if(name=="oversized-count")g.a={4096,4096,4096};
  if(name=="alias-geometry"){g.alias=true;g.a={1,3};}
  if(name=="unsupported-broadcast")g.a={2};
  std::array<std::string,2> messages;bool untouched=true;
  for(int optimized=0;optimized<2;optimized++){
   try{
    State s=make(g,"nonzero-finite");if(name=="short-input")s.a.x.pop_back();if(name=="short-gradient")s.a.g.pop_back();State before=s;
    try{validate(s);calls[0]++;invoke(s,optimized!=0,false);}catch(...){untouched=untouched&&same(s,before);throw;}
    throw std::runtime_error("Unexpected error fixture success");
   }catch(const std::exception& e){messages[optimized]=e.what();}
  }
  allPassed=allPassed&&messages[0]==messages[1]&&messages[0]!="Unexpected error fixture success"&&untouched;
  if(errorIndex++)out<<",";
  out<<"{\"id\":"<<std::quoted(name)<<",\"legacyError\":"<<std::quoted(messages[0])<<",\"candidateError\":"<<std::quoted(messages[1])<<",\"outputsUnchanged\":"<<untouched<<"}";
 }
 out<<"],\"passed\":"<<allPassed<<",\"fixtureRows\":"<<rowCount<<",\"forwardCalls\":"<<calls[0]<<",\"backwardCalls\":"<<calls[1]
    <<",\"domain\":\"Finite bounded tensors; equal total counts precede shape rank. Aliases and unsupported eligible geometry use the unchanged scalar path. Constant gradients untouched; fresh gradients add from positive zero. Nonfinite arithmetic, invalid original graphs and allocation failure behavior are outside the exact claim.\"}";
 return out.str();
}
static std::string bench(const cv::FileStorage& manifest,bool optimized){
 const int repeats=manifest["repeats"];require(repeats==128,"Frozen add benchmark repetitions");
 std::ostringstream out;out<<std::boolalpha<<std::setprecision(17)<<"{\"mode\":"<<std::quoted(optimized?"add-bench-b":"add-bench-a")<<",\"repeats\":"<<repeats<<",\"rotatingPayloads\":4,\"rows\":[";
 size_t rowCount=0,calls[2]={0,0};double total=0;
 for(auto node:manifest["benchmark"]){const Geometry geom=geometry(node);require(geom.eligible&&!geom.alias,"Add benchmark eligibility");for(bool reverse:{false,true}){
  std::vector<State> states,seeds;for(uint32_t slot=0;slot<4;slot++)seeds.push_back(make(geom,slot%2?"nonzero-finite":"fresh-finite",slot+1));states=seeds;
  for(auto& s:states)validate(s);
  double seconds=0;bool dispatch=true,cleanup=true;
  for(int i=0;i<repeats;i++){
   const size_t slot=size_t(i)%4;auto& s=states[slot];const auto& seed=seeds[slot];
   // Equal per-arm reset/copy setup is outside measured operation boundaries.
   if(seed.a.g.empty())Vec().swap(s.a.g);else s.a.g=seed.a.g;
   if(seed.b.g.empty())Vec().swap(s.b.g);else s.b.g=seed.b.g;
   s.t.g=seed.t.g;std::fill(s.t.x.begin(),s.t.x.end(),0.0f);
   const uint64_t start=facade_profile_now();
   const bool specialized=invoke(s,optimized,reverse);
   asm volatile("" : : "r"(s.t.x.data()),"r"(s.a.g.data()),"r"(s.rhs().g.data()) : "memory");
   seconds+=double(facade_profile_now()-start)/1e9;
   dispatch=dispatch&&(specialized==optimized);cleanup=cleanup&&(!reverse||s.t.g.empty());
  }
  calls[reverse]+=size_t(repeats);require(seconds>0&&dispatch&&cleanup,"Add benchmark state");total+=seconds;
  if(rowCount++)out<<",";
  out<<"{\"id\":"<<std::quoted(geom.id)<<",\"direction\":"<<std::quoted(reverse?"backward":"forward")<<",\"seconds\":"<<seconds<<",\"dispatch\":"<<dispatch
     <<",\"outputGradientCleared\":"<<cleanup<<",\"outputSHA256\":[";
  for(size_t i=0;i<4;i++){if(i)out<<",";hashes(out,states[i]);}out<<"],\"outputCounts\":[";
  for(size_t i=0;i<4;i++){if(i)out<<",";counts(out,states[i]);}out<<"]}";
 }}
 out<<"],\"forwardCalls\":"<<calls[0]<<",\"backwardCalls\":"<<calls[1]<<",\"scopedSeconds\":"<<total
    <<",\"timingBoundary\":\"Reset/payload construction outside each call. Frozen scalar fixture oracle or production helper dispatch, lazy gradient preparation, numeric loop and reverse output-gradient release are timed, with identical profile and opaque memory barrier. All four complete output states copied/hashed after timing; no model execution.\"}";
 return out.str();
}
}
extern "C" const char* facade_equal_add_report(const char* path,int mode,char* error,size_t capacity){
 try{
  require(path&&(mode==0||mode==1||mode==2),"Add fixture mode");
  cv::FileStorage manifest(path,cv::FileStorage::READ);require(manifest.isOpened(),"Add fixture manifest");
  static thread_local std::string report;
  report=mode==0?equal_add_fixture::check(manifest):equal_add_fixture::bench(manifest,mode==2);
  return report.c_str();
 }catch(const std::exception& e){error_text(error,capacity,e.what());return nullptr;}
}
