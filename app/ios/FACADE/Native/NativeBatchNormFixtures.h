// NativeBatchNormFixtures.h
//
// Self-check for the batch-norm router (Network::bnForward / bnReverse in
// NativeNetwork.cpp and the fast kernel in NativeBatchNorm.h). It is not part
// of photo protection; nothing in the app calls it. An outside test calls
// facade_bn_report with the path of a JSON manifest.
//
// What it proves. Each case builds a four-node network twice: node 0 is the
// activation input, node 1 a one-value extra input, and nodes 2 and 3 are two
// batch-norm layers that both read node 0. Two consumers means node 0's
// gradient receives two contributions, so the check covers accumulation in
// the order the real backward pass uses. One copy runs through the router with
// the fast kernel off ("Legacy" in the report), the other with it on
// ("Candidate"); eight buffers must match bit for bit (output, the two input
// gradients, input values, alpha, beta, incoming gradient, released output
// gradient). Further runs change alpha, beta or both, and a replay must equal
// the first run. It also checks that changing beta alone leaves the gradient
// unchanged (the gradient depends only on alpha).
//
// Where the numbers come from. The manifest lists shapes ("geometries":
// input/output shapes, input ids, alpha/beta counts, whether the fast kernel is
// expected) and "cases" (geometry + policy). All values come from values()
// (exact bit patterns, magnitude 0.25 to 0.375, alternating sign), so they are
// the same on every machine. Some policies pick values that expose rounding:
//  * "separate-rounding": x = 1 + 2^-23, alpha = 1 - 2^-23, beta = -1. The
//    product rounds to exactly 1.0, so y = 1.0 - 1 = +0.0. A fused
//    multiply-add would skip that rounding and give -2^-46 instead. The
//    backward pass is set up the same way. This is why NativeBatchNorm.h
//    writes the multiply and the add as two statements.
//  * "identity-coefficients": x = -0.0, alpha = 1, beta = 0, so y must be +0.0.
//  * "signed-zero-coefficients": mixes +0.0, -0.0, +/-0.5 and +/-1 in x,
//    alpha and beta.
//  * "nonzero-cancellation": gradients around 2^24, where adding 1.0 is lost
//    to rounding, so a different addition order would give a different answer.
//
// The "errors" rows build deliberately broken networks and check that the
// input validation rejects them with the same message both times.
//
// Mode 0 runs the checks; modes 1 and 2 time the plain loop and the fast kernel
// over 128 calls rotating between 4 data sets.
//
// #included at the end of NativeNetwork.cpp, which provides Network, Vec,
// ProfileScope, require(), integers() and error_text().
#pragma once
// Used only by this self-check; nothing here runs when the app starts.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <iomanip>
#include <map>
#include <sstream>
namespace bn_fixture {
// One shape from the manifest.
struct Geometry {std::string id;std::vector<int> as,os,inputs;size_t alpha=0,beta=0;bool specialized=false;};
// The eight compared buffers plus the node's input ids, which batch-norm node
// ran (2 or 3), which path each step took, and whether both batch-norm nodes'
// output gradients were released.
struct Snapshot {std::array<Vec,8> buffers;std::vector<int> inputs;int consumer=2,forward=-1,backward=-1;bool allCleared=false;};
static size_t count(const std::vector<int>& shape){
 require(!shape.empty()&&shape.size()<=4,"BN fixture rank");size_t n=1;
 for(int d:shape){require(d>0&&size_t(d)<=9999999u/n,"BN fixture shape bounds");n*=size_t(d);}return n;
}
static bool bits(const Vec& a,const Vec& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),4*a.size())==0);}
static std::string hash(const Vec& v){
 unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(v.data(),CC_LONG(v.size()*sizeof(float)),digest);
 const char* hex="0123456789abcdef";std::string out;for(unsigned char c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
}
// Reinterprets 32 bits as a float, to build exact test values.
static float fromBits(uint32_t bits){float v;std::memcpy(&v,&bits,4);return v;}
// Deterministic test values (same formula as in NativeAddMulSeam.h).
static Vec values(size_t n,uint32_t variant){
 Vec v(n);for(size_t i=0;i<n;i++)v[i]=fromBits(0x3e800000u|((uint32_t(i)*104729u+variant*8191u)&0x003fffffu)|((i+variant)%2?0x80000000u:0));return v;
}
// Flips the bits in `mask` of every value (0x80000000 = the sign bit).
static void xorValues(Vec& v,uint32_t mask){for(float& x:v){uint32_t u;std::memcpy(&u,&x,4);x=fromBits(u^mask);}}
static Geometry geometry(const cv::FileNode& r){
 Geometry q;q.id=(std::string)r["id"];q.as=integers(r["aShape"]);q.os=integers(r["outShape"]);q.inputs=integers(r["inputs"]);
 int alpha=int(r["alphaCount"]),beta=int(r["betaCount"]);require(alpha>0&&beta>=alpha,"BN fixture coefficients");q.alpha=size_t(alpha);q.beta=size_t(beta);q.specialized=int(r["specialized"]);
 require(!q.inputs.empty()&&q.inputs[0]==0&&q.inputs.size()<=2,"BN fixture input IDs");for(int id:q.inputs)require(id>=0&&id<2,"BN fixture input IDs");
 require(count(q.as)>=count(q.os),"BN fixture activation buffer");require(q.alpha<=count(q.os)&&count(q.os)%q.alpha==0,"BN fixture coefficients");return q;
}
static std::map<std::string,Geometry> geometries(const cv::FileStorage& m){
 require(int(m["formatVersion"])==1,"BN manifest version");std::map<std::string,Geometry> out;
 for(auto r:m["geometries"]){auto q=geometry(r);require(out.emplace(q.id,q).second,"BN duplicate geometry");}return out;
}
static bool initial(const std::string& p){return p=="nonzero-finite"||p=="constant-nonzero"||p=="nonzero-cancellation"||p=="separate-rounding"||p=="signed-zero-coefficients";}
static bool constantA(const std::string& p){return p=="constant-empty"||p=="constant-nonzero";}
// Builds the four-node network. Node 3's alpha has its sign flipped (except
// for "nonzero-cancellation") so the two consumers contribute different
// gradients. dual = false leaves node 3 without
// coefficients (the timing run uses only node 2).
static Network network(const Geometry& q,const std::string& policy,uint32_t variant=0,bool dual=true){
 Network n;n.nodes.resize(4);n.output=2;n.nodes[0].op=constantA(policy)?"constant":"input";n.nodes[0].t.shape=q.as;n.nodes[0].t.x=values(count(q.as),variant);
 n.nodes[1].op="input";n.nodes[1].t.shape={1};n.nodes[1].t.x=values(1,7+variant);
 if(policy=="signed-zero-coefficients"){const uint32_t v[]={0u,0x80000000u,0x3f000000u,0xbf000000u};for(size_t i=0;i<n.nodes[0].t.x.size();i++)n.nodes[0].t.x[i]=fromBits(v[i%4]);}
 if(policy=="identity-coefficients")std::fill(n.nodes[0].t.x.begin(),n.nodes[0].t.x.end(),fromBits(0x80000000u));
 if(policy=="separate-rounding")std::fill(n.nodes[0].t.x.begin(),n.nodes[0].t.x.end(),fromBits(0x3f800001u));
 for(int id=2;id<4;id++){
  auto& node=n.nodes[size_t(id)];node.op="bn";node.inputs=q.inputs;node.t.shape=q.os;
  if(id==3&&!dual)continue;
  node.t.x.resize(count(q.os));node.alpha=values(q.alpha,13+variant);node.beta=values(q.beta,19+variant);
  if(policy=="signed-zero-coefficients"){
   const uint32_t a[]={0xbf800000u,0u,0x80000000u,0x3f800000u};for(size_t i=0;i<q.alpha;i++)node.alpha[i]=fromBits(a[i%4]);
   for(size_t i=0;i<q.beta;i++)node.beta[i]=fromBits(i%2?0x80000000u:0);
  }
  if(policy=="identity-coefficients"||policy=="nonzero-cancellation"){std::fill(node.alpha.begin(),node.alpha.end(),1.0f);std::fill(node.beta.begin(),node.beta.end(),0.0f);}
  if(policy=="separate-rounding"){std::fill(node.alpha.begin(),node.alpha.end(),fromBits(0x3f7ffffeu));std::fill(node.beta.begin(),node.beta.end(),-1.0f);}
  if(id==3&&policy!="nonzero-cancellation")xorValues(node.alpha,0x80000000u);
 }
 return n;
}
// Rejects inputs the plain loop could not handle safely, before either router
// runs, so both paths fail with exactly the same message.
static void validate(const Network& n,int which,bool reverse){
 require(which==2||which==3,"BN fixture consumer");const auto& node=n.nodes[size_t(which)];
 require(!node.inputs.empty()&&node.inputs[0]==0&&node.inputs.size()<=2,"BN fixture input IDs");for(int id:node.inputs)require(id>=0&&id<2,"BN fixture input IDs");
 size_t ac=count(n.nodes[0].t.shape),oc=count(node.t.shape);
 require(ac>=oc&&n.nodes[0].t.x.size()==ac&&node.t.x.size()==oc,"BN fixture activation buffer");
 require(!node.alpha.empty()&&node.alpha.size()<=oc&&oc%node.alpha.size()==0&&node.beta.size()>=node.alpha.size(),"BN fixture coefficients");
 require(n.nodes[0].t.g.empty()||n.nodes[0].t.g.size()==ac,"BN fixture destination buffer");
 if(reverse)require(node.t.g.empty()||node.t.g.size()==oc,"BN fixture upstream buffer");
}
// Sets each policy's starting gradients (after clearing all of them).
static void seed(Network& n,const Geometry& q,const std::string& p){
 for(auto& node:n.nodes)node.t.g.clear();
 if(initial(p)){n.nodes[0].t.g=values(count(q.as),41);if(q.inputs.size()==2&&q.inputs[1]==1)n.nodes[1].t.g=values(1,59);}
 if(p=="signed-zero-coefficients"){
  for(size_t i=0;i<n.nodes[0].t.g.size();i++)n.nodes[0].t.g[i]=fromBits(i%2?0x80000000u:0);
  if(!n.nodes[1].t.g.empty())n.nodes[1].t.g[0]=fromBits(0x80000000u);
 }
 if(p=="nonzero-cancellation")std::fill(n.nodes[0].t.g.begin(),n.nodes[0].t.g.end(),16777216.0f);
 if(p=="separate-rounding")std::fill(n.nodes[0].t.g.begin(),n.nodes[0].t.g.end(),-1.0f);
}
// The incoming gradient for each backward step of a policy.
static Vec upstream(const Geometry& q,const std::string& p,int step=0,uint32_t variant=0){
 if(p=="empty-upstream")return {};Vec g=values(count(q.os),97+variant);
 if(p=="fresh-negative-zero"||p=="identity-coefficients")std::fill(g.begin(),g.end(),fromBits(0x80000000u));
 if(p=="signed-zero-coefficients")for(size_t i=0;i<g.size();i++)g[i]=i%2?-1.0f:1.0f;
 if(p=="nonzero-cancellation")std::fill(g.begin(),g.end(),step==1?-16777216.0f:1.0f);
 if(p=="separate-rounding")std::fill(g.begin(),g.end(),fromBits(0x3f800001u));return g;
}
// One forward pass of node `which` through the router; returns 1 if fast.
static int forward(Network& n,int which,bool optimized,size_t& calls){
 validate(n,which,false);for(auto& node:n.nodes)node.t.g.clear();
 ProfileScope p("fixture.bn.forward");calls++;return n.bnForward(n.nodes[size_t(which)],optimized)?1:0;
}
static Snapshot snapshot(const Network& n,int which,const Vec& incoming,int f,int b){
 const auto& node=n.nodes[size_t(which)];Snapshot s;s.consumer=which;s.inputs=node.inputs;s.forward=f;s.backward=b;
 s.buffers[0]=node.t.x;s.buffers[1]=n.nodes[0].t.g;if(node.inputs.size()==2)s.buffers[2]=n.nodes[size_t(node.inputs[1])].t.g;
 s.buffers[3]=n.nodes[0].t.x;s.buffers[4]=node.alpha;s.buffers[5]=node.beta;s.buffers[6]=incoming;s.buffers[7]=node.t.g;
 s.allCleared=n.nodes[2].t.g.empty()&&n.nodes[3].t.g.empty();return s;
}
// One backward pass of node `which` (skipped when the incoming gradient is
// empty), then its output gradient is released as in the real backward pass.
static Snapshot reverse(Network& n,int which,const Vec& incoming,bool optimized,int f,size_t& calls){
 auto& node=n.nodes[size_t(which)];node.t.g=incoming;validate(n,which,true);int b=-1;
 if(!incoming.empty()){ProfileScope p("fixture.bn.backward");calls++;b=n.bnReverse(node,optimized)?1:0;Vec().swap(node.t.g);}
 return snapshot(n,which,incoming,f,b);
}
static bool same(const Snapshot& a,const Snapshot& b){for(size_t i=0;i<8;i++)if(!bits(a.buffers[i],b.buffers[i]))return false;return a.consumer==b.consumer&&a.inputs==b.inputs;}
static std::string fingerprint(const Snapshot& s){std::string out;for(const auto& b:s.buffers)out+=hash(b);return out;}
static void snapshotJSON(std::ostream& out,const Snapshot& s){
 out<<"{\"hashes\":[";for(size_t i=0;i<8;i++){if(i)out<<",";out<<std::quoted(hash(s.buffers[i]));}out<<"],\"counts\":[";
 for(size_t i=0;i<8;i++){if(i)out<<",";out<<s.buffers[i].size();}out<<"],\"inputIDs\":[";
 for(size_t i=0;i<s.inputs.size();i++){if(i)out<<",";out<<s.inputs[i];}
 out<<"],\"consumerNode\":"<<s.consumer<<",\"forwardDispatch\":"<<s.forward<<",\"backwardDispatch\":"<<s.backward
  <<",\"reverseVisited\":"<<(s.backward>=0)<<",\"outputGradientCleared\":"<<s.buffers[7].empty()<<",\"allConsumerGradientsCleared\":"<<s.allCleared<<"}";
}
static void coefficients(Network& n,const Vec& alpha,const Vec& beta,int phase){
 n.nodes[2].alpha=alpha;n.nodes[2].beta=beta;
 if(phase==1||phase==3)xorValues(n.nodes[2].alpha,31u*0x1011u);
 if(phase==2||phase==3)xorValues(n.nodes[2].beta,23u*0x1011u);
}
static bool prefixValue(const Vec& v,size_t end,uint32_t bits){
 if(v.size()<end)return false;for(size_t i=0;i<end;i++){uint32_t u;std::memcpy(&u,&v[i],4);if(u!=bits)return false;}return true;
}
static bool tailPreserved(const Snapshot& s,const Snapshot& initial,size_t start){
 if(s.buffers[1].empty())return initial.buffers[1].empty();
 for(size_t i=start;i<s.buffers[1].size();i++){uint32_t got,want=0;std::memcpy(&got,&s.buffers[1][i],4);if(!initial.buffers[1].empty())std::memcpy(&want,&initial.buffers[1][i],4);if(got!=want)return false;}return true;
}
static std::array<Vec,3> secondaryValues(const Network& n){return {n.nodes[3].t.x,n.nodes[3].alpha,n.nodes[3].beta};}
// Runs every case and builds the JSON report.
static std::string check(const cv::FileStorage& manifest){
 auto shapes=geometries(manifest);std::ostringstream out;out<<std::boolalpha<<std::setprecision(17)<<"{\"rows\":[";
 size_t rows=0,fc=0,bc=0;bool passed=true;
 for(auto entry:manifest["cases"]){
  const std::string id=(std::string)entry["id"],key=(std::string)entry["geometry"],policy=(std::string)entry["policy"];const auto& q=shapes.at(key);
  Network a=network(q,policy),b=network(q,policy);const Vec alpha=a.nodes[2].alpha,beta=a.nodes[2].beta;std::array<Snapshot,14> s;
  int af=forward(a,2,false,fc),bf=forward(b,2,true,fc),as=forward(a,3,false,fc),bs=forward(b,3,true,fc);
  seed(a,q,policy);seed(b,q,policy);const Snapshot initialA=snapshot(a,2,{},af,-1),initialSecondary=snapshot(a,3,{},as,-1);
  const auto secondary=secondaryValues(a);
  for(int step=0;step<3;step++){int which=step==1?3:2;const auto g=upstream(q,policy,step);
   s[size_t(step)*2]=reverse(a,which,g,false,which==2?af:as,bc);s[size_t(step)*2+1]=reverse(b,which,g,true,which==2?bf:bs,bc);}
  std::string retained;for(size_t i=0;i<4;i++)retained+=fingerprint(s[i]);
  for(int phase=1;phase<=4;phase++){
   coefficients(a,alpha,beta,phase==4?0:phase);coefficients(b,alpha,beta,phase==4?0:phase);
   af=forward(a,2,false,fc);bf=forward(b,2,true,fc);seed(a,q,policy);seed(b,q,policy);
   s[4+size_t(phase)*2]=reverse(a,2,upstream(q,policy),false,af,bc);s[5+size_t(phase)*2]=reverse(b,2,upstream(q,policy),true,bf,bc);
  }
  bool exact=true;for(size_t i=0;i<14;i+=2)exact=exact&&same(s[i],s[i+1]);
  bool replay=same(s[0],s[12])&&same(s[1],s[13]),routes=true,cleanup=true,constantsOK=true,unusedOK=true,inputsOK=true,tails=true;
  for(size_t i=0;i<14;i++){
   int expected=i%2&&q.specialized?1:0;
   routes=routes&&s[i].forward==expected&&s[i].backward==(policy=="empty-upstream"?-1:expected)&&s[i].consumer==((i==2||i==3)?3:2)&&s[i].inputs==q.inputs;
   cleanup=cleanup&&s[i].buffers[7].empty()&&s[i].allCleared;
   if(constantA(policy))constantsOK=constantsOK&&bits(s[i].buffers[1],initialA.buffers[1]);
   if(q.inputs.size()<2||q.inputs[1]!=0)unusedOK=unusedOK&&bits(s[i].buffers[2],initialA.buffers[2]);
   else unusedOK=unusedOK&&bits(s[i].buffers[2],s[i].buffers[1]);
   inputsOK=inputsOK&&bits(s[i].buffers[3],initialA.buffers[3]);
   tails=tails&&tailPreserved(s[i],initialA,count(q.os));
  }
  std::string after;for(size_t i=0;i<4;i++)after+=fingerprint(s[i]);bool retainedOK=retained==after;
  bool secondaryOK=true;auto finalSecondary=secondaryValues(b);for(size_t i=0;i<3;i++)secondaryOK=secondaryOK&&bits(secondary[i],finalSecondary[i]);
  bool betaInvariant=bits(s[0].buffers[1],s[8].buffers[1])&&bits(s[1].buffers[1],s[9].buffers[1])&&bits(s[0].buffers[6],s[8].buffers[6]);
  bool refresh=!bits(s[0].buffers[4],s[6].buffers[4])&&bits(s[0].buffers[5],s[6].buffers[5])&&bits(s[0].buffers[4],s[8].buffers[4])&&!bits(s[0].buffers[5],s[8].buffers[5])&&!bits(s[0].buffers[4],s[10].buffers[4])&&!bits(s[0].buffers[5],s[10].buffers[5]);
  bool zero=policy!="fresh-negative-zero"||prefixValue(s[1].buffers[1],count(q.as),0);
  bool identity=policy!="identity-coefficients"||prefixValue(s[1].buffers[0],count(q.os),0);
  bool rounding=policy!="separate-rounding"||(prefixValue(s[1].buffers[0],count(q.os),0)&&prefixValue(s[1].buffers[1],count(q.os),0));
  bool cancellation=policy!="nonzero-cancellation"||prefixValue(s[5].buffers[1],count(q.os),0x3f800000u);
  bool ok=exact&&replay&&routes&&cleanup&&constantsOK&&unusedOK&&inputsOK&&tails&&retainedOK&&secondaryOK&&betaInvariant&&refresh&&zero&&identity&&rounding&&cancellation;passed=passed&&ok;
  if(rows++)out<<",";out<<"{\"id\":"<<std::quoted(id)<<",\"checks\":{\"allPairsBitsExact\":"<<exact<<",\"replayExact\":"<<replay<<",\"dispatchAndConsumerOrderExact\":"<<routes
   <<",\"outputGradientsCleared\":"<<cleanup<<",\"constantActivationPreserved\":"<<constantsOK<<",\"unusedInputGradientCorrect\":"<<unusedOK<<",\"activationUnchanged\":"<<inputsOK
   <<",\"completeGradientTailPreserved\":"<<tails<<",\"retainedOwnedCopiesPreserved\":"<<retainedOK<<",\"secondaryConsumerPreserved\":"<<secondaryOK<<",\"betaOnlyLocalDerivativeInvariant\":"<<betaInvariant
   <<",\"coefficientRefreshObserved\":"<<refresh<<",\"freshNegativeZeroPositive\":"<<zero<<",\"identityZeroWitness\":"<<identity<<",\"separateRoundingWitness\":"<<rounding<<",\"cancellationOrderObserved\":"<<cancellation<<"}";
  out<<",\"initial\":";snapshotJSON(out,initialA);out<<",\"initialSecondary\":";snapshotJSON(out,initialSecondary);
  const char* names[]={"firstLegacy","firstCandidate","secondConsumerLegacy","secondConsumerCandidate","thirdLegacy","thirdCandidate","alphaChangedLegacy","alphaChangedCandidate","betaChangedLegacy","betaChangedCandidate","bothChangedLegacy","bothChangedCandidate","replayLegacy","replayCandidate"};
  for(size_t i=0;i<14;i++){out<<","<<std::quoted(names[i])<<":";snapshotJSON(out,s[i]);}
  out<<",\"secondaryRetained\":{\"initialHashes\":[";for(size_t i=0;i<3;i++){if(i)out<<",";out<<std::quoted(hash(secondary[i]));}
  out<<"],\"finalHashes\":[";for(size_t i=0;i<3;i++){if(i)out<<",";out<<std::quoted(hash(finalSecondary[i]));}
  out<<"],\"counts\":[";for(size_t i=0;i<3;i++){if(i)out<<",";out<<secondary[i].size();}out<<"]}}";
 }
 // Error rows: break one thing in a small network and check that validate()
 // rejects it, without changing any buffer.
 out<<"],\"errors\":[";size_t errorRows=0;
 for(const char* name:{"rank","shape-bounds","empty-alpha","short-beta","nondivisible-alpha","short-activation","short-gradient","input-id","short-destination"}){
  std::array<std::string,2> messages;std::array<int,2> codes={0,0};bool unchanged=true;
  for(size_t arm=0;arm<2;arm++){
   Network n=network(shapes.at("channel-row3"),"fresh-finite");n.nodes[2].t.g=values(6,1);std::string id=name;
   if(id=="rank")n.nodes[0].t.shape.clear();if(id=="shape-bounds")n.nodes[0].t.shape[0]=0;
   if(id=="empty-alpha")n.nodes[2].alpha.clear();if(id=="short-beta")n.nodes[2].beta.resize(1);
   if(id=="nondivisible-alpha"){n.nodes[2].alpha.resize(4);n.nodes[2].beta.resize(4);}
   if(id=="short-activation")n.nodes[0].t.x.pop_back();if(id=="short-gradient")n.nodes[2].t.g.pop_back();
   if(id=="input-id")n.nodes[2].inputs[0]=5;if(id=="short-destination")n.nodes[0].t.g=values(5,41);
   const std::string before=fingerprint(snapshot(n,2,{},-1,-1));
   try{validate(n,2,true);messages[arm]="NO_ERROR";}catch(const std::exception& e){codes[arm]=-1;messages[arm]=e.what();}
   unchanged=unchanged&&before==fingerprint(snapshot(n,2,{},-1,-1));
  }
  bool ok=codes[0]==-1&&codes[1]==-1&&messages[0]==messages[1]&&unchanged;passed=passed&&ok;
  if(errorRows++)out<<",";out<<"{\"id\":"<<std::quoted(name)<<",\"legacyCode\":"<<codes[0]<<",\"candidateCode\":"<<codes[1]<<",\"legacyError\":"<<std::quoted(messages[0])<<",\"candidateError\":"<<std::quoted(messages[1])<<",\"outputUnchanged\":"<<unchanged<<"}";
 }
 out<<"],\"passed\":"<<passed<<",\"fixtureRows\":"<<rows<<",\"forwardCalls\":"<<fc<<",\"backwardCalls\":"<<bc
  <<",\"domain\":\"Finite bounded BN routers with two actual ordered consumers. Full coefficient/activation/gradient/tail state; fixed local incoming gradient for beta-only derivative comparison. Adapter errors precede routers. No undefined legacy indexing, FENV/nonfinite or allocation-failure claim.\"}";return out.str();
}
// Timing run: only the router call (and, backward, the release of the output
// gradient) is inside the timed interval; resets happen outside it.
static std::string benchmark(const cv::FileStorage& manifest,bool optimized){
 require(int(manifest["repeats"])==128&&int(manifest["rotatingPayloads"])==4,"BN benchmark schedule");auto shapes=geometries(manifest);
 std::ostringstream out;out<<std::boolalpha<<std::setprecision(17)<<"{\"mode\":"<<std::quoted(optimized?"bn-bench-b":"bn-bench-a")<<",\"rows\":[";
 size_t rows=0,fc=0,bc=0,setup=0;double scoped=0;
 for(auto entry:manifest["benchmark"]){const auto& q=shapes.at((std::string)entry);require(q.specialized&&q.inputs.size()==1,"BN benchmark geometry");
  for(int direction=0;direction<2;direction++){
   std::vector<Network> models;std::array<Vec,4> gs,ags;
   for(uint32_t slot=0;slot<4;slot++){models.push_back(network(q,"fresh-finite",slot,false));gs[slot]=upstream(q,"fresh-finite",0,slot);ags[slot]=values(count(q.as),41+slot);}
   if(direction==1)for(auto& n:models){n.bnForward(n.nodes[2],false);setup++;}
   uint64_t total=0;bool consistent=true;int dispatched=-1;
   for(int iteration=0;iteration<128;iteration++){
    size_t slot=size_t(iteration)%4;auto& n=models[slot];auto& node=n.nodes[2];
    if(slot%2==0)Vec().swap(n.nodes[0].t.g);else n.nodes[0].t.g=ags[slot];
    if(direction==1)node.t.g=gs[slot];else node.t.g.clear();
    const uint64_t start=facade_profile_now();int used;
    if(direction==0){ProfileScope p("fixture.bn.forward");used=n.bnForward(node,optimized)?1:0;fc++;}
    else{ProfileScope p("fixture.bn.backward");used=n.bnReverse(node,optimized)?1:0;Vec().swap(node.t.g);bc++;}
    asm volatile("" : : "r"(node.t.x.data()),"r"(n.nodes[0].t.g.data()),"r"(node.alpha.data()),"r"(node.beta.data()),"r"(node.t.g.data()) : "memory");
    total+=facade_profile_now()-start;consistent=consistent&&(used==(optimized?1:0));dispatched=used;
   }
   double seconds=double(total)/1e9;scoped+=seconds;bool cleared=true;for(const auto& n:models)cleared=cleared&&n.nodes[2].t.g.empty()&&n.nodes[3].t.g.empty();
   if(rows++)out<<",";out<<"{\"id\":"<<std::quoted(q.id)<<",\"direction\":"<<std::quoted(direction==0?"forward":"backward")<<",\"seconds\":"<<seconds<<",\"dispatch\":"<<(dispatched==1)
    <<",\"dispatchConsistent\":"<<consistent<<",\"outputGradientCleared\":"<<cleared<<",\"outputSHA256\":[";
   for(size_t slot=0;slot<4;slot++){if(slot)out<<",";auto s=snapshot(models[slot],2,{},-1,-1);out<<"[";for(size_t j=0;j<3;j++){if(j)out<<",";out<<std::quoted(hash(s.buffers[j]));}out<<"]";}
   out<<"],\"outputCounts\":[";for(size_t slot=0;slot<4;slot++){if(slot)out<<",";auto s=snapshot(models[slot],2,{},-1,-1);out<<"[";for(size_t j=0;j<3;j++){if(j)out<<",";out<<s.buffers[j].size();}out<<"]";}
   out<<"],\"slotObservations\":[";for(size_t slot=0;slot<4;slot++){if(slot)out<<",";snapshotJSON(out,snapshot(models[slot],2,direction==1?gs[slot]:Vec{},direction==0?dispatched:0,direction==1?dispatched:-1));}out<<"]}";
  }
 }
 out<<"],\"repeats\":128,\"rotatingPayloads\":4,\"scopedSeconds\":"<<scoped<<",\"forwardCalls\":"<<fc<<",\"backwardCalls\":"<<bc<<",\"untimedScalarSetupForwardCalls\":"<<setup
  <<",\"barrier\":\"4 distinct finite activation/alpha/beta/upstream payloads. Fresh/nonzero resets and524 scalar y-setup forwards outside timing; router/lazy allocation/single cleanup inside; complete-output barrier before stopclock; all4 full8-buffer observations after row. Strict sequential binary64 sum of row seconds.\"}";return out.str();
}
}
// C entry point. mode 0 = checks, 1 = time the plain loop, 2 = time the fast
// kernel. The returned string stays valid until the next call on this thread.
extern "C" const char* facade_bn_report(const char* path,int mode,char* error,size_t capacity){
 try{require(path,"BN fixture manifest path");require(mode>=0&&mode<=2,"BN fixture report mode");cv::FileStorage manifest(path,cv::FileStorage::READ);require(manifest.isOpened(),"BN fixture manifest");
  static thread_local std::string report;report=mode==0?bn_fixture::check(manifest):bn_fixture::benchmark(manifest,mode==2);return report.c_str();
 }catch(const std::exception& e){error_text(error,capacity,e.what());return nullptr;}
}
