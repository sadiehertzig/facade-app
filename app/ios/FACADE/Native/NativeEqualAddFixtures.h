// NativeEqualAddFixtures.h
//
// Self-check for the equal-shape add kernel (NativeEqualAdd.h). It is not part
// of photo protection: nothing in the app calls it. An outside test calls
// facade_equal_add_report with the path of a JSON manifest and reads back a
// JSON report.
//
// What it proves. For every test case it runs the same add node twice, once
// through the plain per-element loop (called "legacy" in the report) and once
// through the fast kernel ("candidate"), and checks that every output and
// gradient buffer is identical bit for bit. It also checks that repeated runs
// accumulate gradients identically, that a run with different inputs leaves
// nothing behind, that inputs and constant gradients are never modified, that
// the output gradient is released afterwards, that the expected path (fast or
// plain) was taken, and that bad inputs give the same error message on both
// paths.
//
// Where the numbers come from. The manifest only lists shapes ("fixtures":
// id, aShape, bShape, outShape, alias, eligible) and the names of the gradient
// situations to try ("policies"). All tensor values are generated here by
// finiteValues(): exact floats built from bit patterns with a simple integer
// formula, so every machine produces the same inputs and no random numbers are
// involved. Some policies use special values on purpose:
//  * "fresh-negative-zero" / "signed-zero-dest": -0.0 and +0.0, to check that a
//    newly created gradient starts at +0.0 (because +0.0 + -0.0 = +0.0);
//  * "cancellation": 16777216 = 2^24, the point where adding 1.0 to a float is
//    lost to rounding, so a sum done in a different order would change;
//  * "constant-a/-b/-both": inputs marked constant, which must get no gradient.
//
// Mode 0 runs the checks. Modes 1 and 2 time the plain loop and the fast
// kernel over 128 runs that rotate between 4 different data sets.
//
// This header is #included at the end of NativeNetwork.cpp, which provides
// Tensor, Network, Vec, require(), integers() and error_text().
#pragma once
// The includes below are only needed by this self-check (CommonCrypto gives
// SHA-256 for the report's buffer fingerprints).
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <iomanip>
#include <sstream>
namespace equal_add_fixture {
// One test shape from the manifest. alias = both inputs are the same node;
// eligible = the fast kernel is expected to run.
struct Geometry {std::string id;std::vector<int> a,b,out;bool alias=false,eligible=false;};
// The tensors of one add node: inputs a and b, output t, plus which inputs are
// constants. rhs() is the second input, which is `a` itself when aliased.
struct State {Tensor a,b,t;bool alias=false,constantA=false,constantB=false;Tensor& rhs(){return alias?a:b;}const Tensor& rhs()const{return alias?a:b;}};
// Element count of a shape, with the same limits as the kernel; throws on a
// bad shape (this is how the error cases fail).
static size_t countShape(const std::vector<int>& s){
 require(!s.empty()&&s.size()<=4,"Add fixture rank");
 size_t n=1;for(int d:s){require(d>0,"Add fixture positive dimensions");require(n<=9999999u/size_t(d),"Add fixture count bounds");n*=size_t(d);}return n;
}
// SHA-256 of a float buffer's raw bytes, as hex. Used in the report so two
// runs can be compared by fingerprint.
static std::string hash(const Vec& v){
 unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(v.data(),CC_LONG(v.size()*sizeof(float)),bytes);
 static const char* hex="0123456789abcdef";std::string s;for(unsigned char c:bytes){s+=hex[c>>4];s+=hex[c&15];}return s;
}
// Deterministic test values: sign bit alternates, exponent fixed so the
// magnitude is between 0.5 and 1, and the 23 mantissa bits come from an
// integer formula (1664525 and 1013904223 are classic random-number-generator
// constants). `variant` gives a different but equally reproducible set.
static Vec finiteValues(size_t n,uint32_t variant=0){
 Vec out(n);for(size_t i=0;i<n;i++){uint32_t bits=0x3f000000u|((uint32_t(i)*1664525u+1013904223u+variant*104729u)&0x007fffffu)|(i%2?0x80000000u:0);std::memcpy(&out[i],&bits,4);}return out;
}
// Builds the tensors for one shape and policy (see the list at the top).
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
// Checks every buffer has the length its shape says.
static void validate(const State& s){
 const size_t na=countShape(s.a.shape),nb=countShape(s.rhs().shape),no=countShape(s.t.shape);
 require(s.a.x.size()==na&&s.rhs().x.size()==nb&&s.t.x.size()==no&&s.t.g.size()==no,"Add fixture buffer size");
 require((s.a.g.empty()||s.a.g.size()==na)&&(s.rhs().g.empty()||s.rhs().g.size()==nb),"Add fixture gradient size");
}
// Reference forward: the plain loop from NativeNetwork.cpp, using Network::at.
static void scalarForward(State& s){
 auto& a=s.a;auto& b=s.rhs();auto& t=s.t;auto& y=t.x;
 for(int i=0;i<t.count();i++){float aa=a.x[Network::at(a,t,i)],bb=b.x[Network::at(b,t,i)];y[i]=aa+bb;}
}
// Reference backward: the plain loop, with its own copy of Network::add's
// rules (skip constants, create a +0.0 buffer on first use, then +=).
static void scalarBackward(State& s){
 auto add=[](Tensor& t,bool constant,int i,float v){if(constant)return;if(t.g.empty())t.g.assign(t.count(),0);t.g[i]+=v;};
 auto& a=s.a;auto& b=s.rhs();auto& t=s.t;auto& g=t.g;
 for(int i=0;i<t.count();i++){int ai=Network::at(a,t,i),bi=Network::at(b,t,i);float aa=a.x[ai],bb=b.x[bi];(void)aa;(void)bb;add(a,s.constantA,ai,g[i]);add(b,s.constantB,bi,g[i]);}
}
// Runs one direction through the fast kernel (if `optimized` and it accepts
// the case) or the reference loop. Returns true if the fast kernel ran.
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
// The three buffers compared between runs: output, gradient of a, gradient of b.
static std::array<const Vec*,3> outputs(const State& s){return {&s.t.x,&s.a.g,&s.rhs().g};}
// Bitwise equality of those buffers (memcmp, so -0.0 and +0.0 count as
// different).
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
// Reads one fixture row from the manifest.
static Geometry geometry(const cv::FileNode& node){
 Geometry g;g.id=(std::string)node["id"];g.a=integers(node["aShape"]);g.b=integers(node["bShape"]);g.out=integers(node["outShape"]);g.alias=int(node["alias"])!=0;g.eligible=int(node["eligible"])!=0;return g;
}
// The comparison run (mode 0). Builds one JSON row per fixture, policy and
// direction, then one row per deliberate error case.
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
// The timing run (modes 1 and 2). Only invoke() is timed (the add itself and,
// backward, the release of the output gradient); resetting the buffers
// happens outside the timed interval.
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
   // Reset this slot's buffers the same way for both modes, before the clock
   // starts, so the reset is not part of the measured time.
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
// C entry point. mode 0 = checks, 1 = time the plain loop, 2 = time the fast
// kernel. The returned string stays valid until the next call on this thread.
extern "C" const char* facade_equal_add_report(const char* path,int mode,char* error,size_t capacity){
 try{
  require(path&&(mode==0||mode==1||mode==2),"Add fixture mode");
  cv::FileStorage manifest(path,cv::FileStorage::READ);require(manifest.isOpened(),"Add fixture manifest");
  static thread_local std::string report;
  report=mode==0?equal_add_fixture::check(manifest):equal_add_fixture::bench(manifest,mode==2);
  return report.c_str();
 }catch(const std::exception& e){error_text(error,capacity,e.what());return nullptr;}
}
