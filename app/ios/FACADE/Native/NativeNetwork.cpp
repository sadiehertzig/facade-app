// NativeNetwork.cpp
//
// The face-recognition network that FACADE runs on the phone, written as plain
// C++ with no machine-learning framework behind it. This one file does three
// jobs:
//
//   1. Load a network description (graph.json, read with OpenCV's FileStorage)
//      and its weights (weights.f32, one flat array of 32-bit floats).
//   2. Run the network forward on a 112x112 face crop to get a 512-number
//      "embedding" (the network's description of whose face this is) and a
//      score: the dot product of that embedding with a reference embedding of
//      the clean, unprotected face.
//   3. Run the network backward to get the gradient of that score with respect
//      to every input pixel. The Swift optimizer (NativeOptimizer.swift and
//      NativePipeline.swift) nudges the pixels against that gradient, so the
//      protected photo still looks like the same person to a human but less
//      like the reference to the kind of network a face-swap tool relies on.
//
// Key ideas needed to read this file:
//
// * Tensor layout. Every tensor is a flat std::vector<float> in "CHW" order:
//   all of channel 0's rows, then channel 1's rows, and so on. For a shape
//   [1, C, H, W], element (c, y, x) sits at index (c*H + y)*W + x. The input is
//   [1, 3, 112, 112] with values in 0..1; the output is 512 numbers.
//
// * Each Node owns its output tensor `t`. `t.x` is the forward value and `t.g`
//   is the gradient flowing back into it. Nodes are stored in evaluation order
//   (every node's inputs come earlier in the list), so forward() walks them
//   first-to-last and reverse() walks them last-to-first. Gradients are
//   accumulated with += so a tensor read by two later nodes receives both
//   contributions. `t.g` starts empty and is only allocated when the first
//   contribution arrives (see add()).
//
// * Determinism. The app promises that the same photo produces the same
//   protected pixels on every build and every phone. Floating-point addition is
//   not associative, so the ORDER of every sum matters. The project is compiled
//   with -ffp-contract=off -fno-fast-math (Config/Base.xcconfig) so the
//   compiler cannot reorder or fuse operations, and torch_sum() below
//   reproduces PyTorch's exact summation order so the phone matches the desktop
//   model the protection was tuned against.
//
// * Fast paths and fallbacks. Several layer types have a hand-tuned kernel in a
//   Native*.h header (batch norm, PReLU, channel multiply, equal-shape add, 3x3
//   convolution columns and their reverse scatter). Each kernel first checks
//   that the tensor shapes are exactly the case it handles; if not, it returns
//   false and the original plain loop in this file runs instead. Both paths
//   produce bit-identical results, which the *Fixtures.h drivers verify.
//
// * Convolution is done as "im2col + GEMM": the input is unrolled into a matrix
//   of patches (columns()), then one matrix multiply (cblas_sgemm from Apple's
//   Accelerate framework) does all the dot products at once. The backward pass
//   is the transpose: a GEMM into a column matrix, then scatterColumns() adds
//   the pieces back onto the input pixels.
//
// Everything after facade_network_layer() is self-check code: C entry points
// that run one kernel in isolation, with the fast path on or off, so an outside
// test can confirm the two agree. The app itself never calls them (see the
// Swift callers in App/NativeTensor.swift, which use only load, free, forward
// and predict).
#include "NativeProfile.h"
#include "NativeChannelMul.h"
#include "NativeScatter.h"
#include "NativeColumns.h"
#include "NativeEqualAdd.h"
#include "NativePrelu.h"
#include "NativeBatchNorm.h"
// Everything here runs on the CPU with fixed weights: there is no training and
// no machine-learning framework behind it.
#include "NativeNetwork.h"
// OpenCV is used only to read the JSON graph; Accelerate supplies cblas_sgemm;
// arm_neon and SLEEF are used by the vectorized sigmoid.
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
// Shorthand: every tensor buffer is a flat vector of floats.
using Vec=std::vector<float>;
// A tensor is a shape (up to four dimensions, e.g. [1, C, H, W]) plus two flat
// buffers: x holds the forward value, g the gradient flowing back into it.
// g stays empty until some node contributes to it. count() is the number of
// elements, the product of the shape.
struct Tensor {std::vector<int> shape;Vec x,g;int count()const{return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>());}};
// One layer of the network. `op` names the operation ("conv", "bn", "prelu",
// "add", ...). `inputs` are indices of earlier nodes whose outputs feed this
// one. kernel/stride/padding are only used by "conv". weight/bias/alpha/beta
// are this node's parameters, copied out of weights.f32 at load time (alpha and
// beta are the per-channel scale and offset of a folded batch norm).
// `selected` remembers which input element won each 2x2 max-pool window so the
// backward pass can route the gradient to it. `t` is this node's output.
struct Node {std::string op,name;std::vector<int> inputs,kernel,stride,padding;Tensor t;Vec weight,bias,alpha,beta;std::vector<int> selected;};
// Throws a std::runtime_error with a short message. Every C entry point catches
// it and hands the message back to Swift instead of crashing.
static void require(bool b,const char *message){if(!b)throw std::runtime_error(message);}
// Reads an integer list out of the graph file, e.g. "shape": [1, 64, 56, 56].
static std::vector<int> integers(const cv::FileNode& x){std::vector<int> v;for(auto i:x)v.push_back((int)i);return v;}
// Reads one parameter slice out of the flat weights file. The graph names each
// slice as {"offset": start, "count": n} into weights.f32; the bounds check
// stops a bad graph from reading past the end of the array.
static Vec values(const cv::FileNode& spec,const Vec& weights){if(spec.empty())return {};int off=spec["offset"],n=spec["count"];require(off>=0&&n>=0&&(size_t)off+n<=weights.size(),"Weight bounds");return Vec(weights.begin()+off,weights.begin()+off+n);}
// torch_sum: adds n floats in exactly the order PyTorch 2.8's CPU kernel does.
// A plain left-to-right loop would give a slightly different float result, and
// the phone must match the desktop model bit for bit. PyTorch's kernel works in
// chunks of 16 (four NEON lanes x four independent rows) and pours partial sums
// into a four-level cascade so no single accumulator grows too long; the code
// below mirrors that structure in scalar form. Derived from its BSD-licensed
// SumKernel.cpp (see THIRD_PARTY_NOTICES.md at the repository root). Used by
// the "average" pooling layer.
static float torch_sum(const float* x,int n){
 // vectors = how many groups of 4 lanes; rows = how many groups of 16 values.
 // ceilLog = ceil(log2(rows)); PyTorch derives the cascade step from it.
 int vectors=n/4,rows=vectors/4,ceilLog=0;while((1<<ceilLog)<rows)ceilLog++;
 // step = how many 16-value rows go into level 0 before it is poured into
 // level 1. acc[level][row][lane] is the cascade: 4 levels x 4 rows x 4 lanes.
 int power=std::max(4,ceilLog/4),step=1<<power,mask=step-1;float acc[4][4][4]={};int i=0;
 // Main loop: whole chunks of `step` rows each.
 for(;i+step<=rows;){
  // Level 0: add `step` rows straight from the input, lane by lane.
  for(int j=0;j<step;j++,i++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=x[(i*4+k)*4+lane];
  // Pour level 0 into level 1, level 1 into level 2, and so on, but stop at the
  // first level whose counter bit is not set yet (the i&(mask<<...) test). This
  // bounds how many additions any single accumulator ever sees.
  for(int level=1;level<4;level++){for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++){acc[level][k][lane]+=acc[level-1][k][lane];acc[level-1][k][lane]=0;}if(i&(mask<<(level*power)))break;}
 }
 // Leftover rows that did not fill a whole chunk.
 for(;i<rows;i++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=x[(i*4+k)*4+lane];
 // Collapse the cascade back down into level 0.
 for(int level=1;level<4;level++)for(int k=0;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][k][lane]+=acc[level][k][lane];
 // Leftover groups of 4 that did not fill a whole row of 16.
 for(int v=rows*4;v<vectors;v++)for(int lane=0;lane<4;lane++)acc[0][0][lane]+=x[v*4+lane];
 // Fold the four rows into row 0.
 for(int k=1;k<4;k++)for(int lane=0;lane<4;lane++)acc[0][0][lane]+=acc[0][k][lane];
 // Finally the scalar tail (n not divisible by 4), then the four lanes, in this order.
 float total=0;for(int j=vectors*4;j<n;j++)total+=x[j];for(int lane=0;lane<4;lane++)total+=acc[0][0][lane];return total;
}
// The loaded network: a list of nodes in evaluation order plus the index of the
// node whose output is the 512-number embedding.
struct Network {
 std::vector<Node> nodes;int output;
 // Loads graph.json (through OpenCV's FileStorage, which reads JSON) and
 // weights.f32. Every bound is checked up front so a corrupt file fails here
 // with a message instead of crashing later.
 Network(const char* graph,const char* path){ProfileScope profile("native.load");
  // The weights file is a raw array of 32-bit floats; it must be a multiple of
  // 4 bytes and under 1 GB.
  std::ifstream file(path,std::ios::binary|std::ios::ate);require(bool(file),"Missing weights");auto size=file.tellg();require(size>0&&size%4==0&&size<1000000000,"Invalid weights size");file.seekg(0);Vec weights(size/4);file.read((char*)weights.data(),size);require(bool(file),"Truncated weights");
  // The graph must be version 1 and name its output node.
  cv::FileStorage s(graph,cv::FileStorage::READ);require(s.isOpened(),"Invalid native graph");require((int)s["version"]==1,"Graph version");output=s["output"];
  // Each node's operation, shape, inputs, hyper-parameters and parameter slices.
  for(auto v:s["nodes"]){Node n;n.op=(std::string)v["op"];n.name=(std::string)v["name"];n.t.shape=integers(v["shape"]);n.inputs=integers(v["inputs"]);n.kernel=integers(v["kernel"]);n.stride=integers(v["stride"]);n.padding=integers(v["padding"]);n.weight=values(v["weights"],weights);n.bias=values(v["bias"],weights);n.alpha=values(v["alpha"],weights);n.beta=values(v["beta"],weights);
   // Inputs may only refer to nodes earlier in the list, which is what makes a
   // simple first-to-last walk a valid evaluation order.
   require(n.t.shape.size()<=4&&!n.t.shape.empty()&&n.t.count()>0&&n.t.count()<10000000,"Tensor bounds");for(int a:n.inputs)require(a>=0&&a<(int)nodes.size(),"Graph order");nodes.push_back(std::move(n));}
  // The output must be a 512-element embedding.
  require(output>=0&&output<(int)nodes.size()&&nodes[output].t.count()==512,"Embedding shape");
 }
 // at(): maps an output index i to the matching index inside input tensor a
 // when a and the output have different sizes. The graph's shapes never change
 // at run time, so only three cases are allowed: same element count (index
 // unchanged), a single scalar (always index 0), or one value per channel
 // (index = channel = i / (H*W)).
 static int at(const Tensor& a,const Tensor& out,int i){
  if(a.count()==out.count())return i;if(a.count()==1)return 0;
  require(out.shape.size()==4&&a.count()==out.shape[1],"Unsupported broadcast");return i/(out.shape[2]*out.shape[3]);
 }
 // Adds v to element i of node `node`'s gradient. Constants have no gradient.
 // The gradient buffer is allocated (as zeros) on first use, so a tensor nobody
 // contributes to keeps an empty g and the reverse pass skips it.
 void add(int node,int i,float v){auto& t=nodes[node].t;if(nodes[node].op=="constant")return;if(t.g.empty())t.g.assign(t.count(),0);t.g[i]+=v;}
 // im2col: unrolls the input of a convolution into a matrix with one row per
 // (input channel, kernel row, kernel column) and one column per output pixel.
 // Padding positions become zeros. After this, the convolution is a single
 // matrix multiply. For the common 3x3, stride 1, pad 1 case a memcpy-based
 // kernel (NativeColumns.h) fills the matrix; otherwise the general loop does.
 static Vec columns(const Tensor& a,const Node& n,bool optimized=true){ProfileScope profile("native.forward.columns");
  // l = number of output pixels; the matrix is (c*kh*kw) rows by l columns.
  int c=a.shape[1],h=a.shape[2],w=a.shape[3],kh=n.kernel[0],kw=n.kernel[1],oh=n.t.shape[2],ow=n.t.shape[3],l=oh*ow;facade_profile_allocation("allocation.native.forward.columns",uint64_t(c)*kh*kw*l*sizeof(float));Vec col(c*kh*kw*l);
  if(optimized&&a.shape.size()==4&&a.shape[0]==1&&n.t.shape.size()==4&&n.t.shape[0]==1&&n.kernel.size()==2&&n.stride.size()==2&&n.padding.size()==2&&facade::columns3x3Eligible(c,h,w,kh,kw,n.stride[0],n.stride[1],n.padding[0],n.padding[1],oh,ow)){facade::fillColumns3x3(col.data(),a.x.data(),c,h,w);return col;}
  // General case: for every patch position, copy the input pixel it reads, or 0
  // when that pixel lies outside the image (sy/sx out of range means padding).
  for(int z=0;z<c;z++)for(int ky=0;ky<kh;ky++)for(int kx=0;kx<kw;kx++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++){
   int sy=y*n.stride[0]-n.padding[0]+ky,sx=x*n.stride[1]-n.padding[1]+kx;col[((z*kh+ky)*kw+kx)*l+y*ow+x]=(sy>=0&&sy<h&&sx>=0&&sx<w)?a.x[(z*h+sy)*w+sx]:0;
  }return col;
 }
 // Batch norm. At export time the batch-norm statistics were folded into one
 // per-channel scale (alpha) and offset (beta), so forward is y = x*alpha + beta
 // and backward is dx = dy*alpha. The fast kernel in NativeBatchNorm.h handles
 // the plane layout; the plain loop below is the fallback and the reference.
 // Both routers return true when the fast kernel ran and false otherwise (the
 // self-check drivers read that flag).
 bool bnForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;auto& a=nodes[n.inputs[0]].t;
  if(optimized&&n.inputs.size()==1&& &a!=&t&&
   facade::bnPlaneForward(a.shape,t.shape,a.x,n.alpha,n.beta,y))return true;
    // Fallback: i/plane is the channel of element i. The multiply and the add are
    // two statements on purpose; see NativeBatchNorm.h.
    int plane=t.count()/n.alpha.size();for(int i=0;i<t.count();i++){float v=a.x[i]*n.alpha[i/plane];y[i]=v+n.beta[i/plane];}
  return false;
 }
 bool bnReverse(Node& n,bool optimized=true){
  auto& t=n.t;auto& g=t.g;int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.inputs.size()==1&& &a!=&t&&
   facade::bnPlaneBackward(a.shape,t.shape,a.x,n.alpha,n.beta,t.x,g,nodes[id].op=="constant",a.g))return true;
    // Fallback backward: dx += dy * alpha[channel].
    int plane=t.count()/n.alpha.size();for(int i=0;i<t.count();i++)add(id,i,g[i]*n.alpha[i/plane]);
  return false;
 }
 // End of batch norm.
 // PReLU: y = x when x > 0, otherwise y = x * slope. The slope is either owned
 // by this node (n.weight: one value per channel, or one for the whole tensor)
 // or comes from a second input node (a constant node in the graph). The fast
 // kernel in NativePrelu.h handles the per-channel plane layout.
 bool preluForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;const int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.inputs.size()<=2){
   // `owned`: the slope lives in this node. `sid`: index of a separate slope
   // node, or -1 when there is none.
   const bool owned=!n.weight.empty();const int sid=n.inputs.size()==2?n.inputs[1]:-1;
   if((owned||sid>=0)&&(n.inputs.size()==1||(sid>=0&&size_t(sid)<nodes.size()))){
    const auto& slope=owned?n.weight:nodes[sid].t.x;const auto& shape=owned?t.shape:nodes[sid].t.shape;
    if(facade::preluPlaneForward(a.shape,shape,t.shape,id,sid,owned,sid>=0&&nodes[sid].op=="constant",a.x,slope,y))return true;
   }
  }
    // Fallback: each of slope.size() values covers `plane` consecutive elements.
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
    // Fallback backward: the gradient passes through unchanged where x > 0 and
    // is scaled by the slope elsewhere. The slope gets no gradient.
    const auto& slope=n.weight.empty()?nodes[n.inputs[1]].t.x:n.weight;int plane=t.count()/slope.size();for(int i=0;i<t.count();i++)add(id,i,a.x[i]>0?g[i]:g[i]*slope[i/plane]);
  return false;
 }
 // End of PReLU.
 // Element-wise binary operations follow (after the empty constructor).
 // Empty network, used only by the self-check drivers that build tiny graphs
 // by hand.
 Network():output(2){};
 // Element-wise binary operations (add, sub, mul, div) with the limited
 // broadcasting that at() allows. The fast kernel in NativeChannelMul.h covers
 // "tensor times one value per channel", which is how the SimSwap recognizer's
 // squeeze-and-excitation blocks rescale each channel by a computed gate value.
 bool binaryForward(Node& n,bool optimized=true){
  auto& t=n.t;auto& y=t.x;auto& a=nodes[n.inputs[0]].t;
  if(optimized&&n.op=="mul"&&facade::channelMulForward(a.shape,nodes[n.inputs[1]].t.shape,t.shape,n.inputs[0],n.inputs[1],a.x,nodes[n.inputs[1]].t.x,y))return true;
  // Fallback: fetch both operands through at() so scalars and per-channel
  // values broadcast against the full tensor.
  auto& b=nodes[n.inputs[1]].t;for(int i=0;i<t.count();i++){float aa=a.x[at(a,t,i)],bb=b.x[at(b,t,i)];y[i]=n.op=="add"?aa+bb:n.op=="sub"?aa-bb:n.op=="mul"?aa*bb:aa/bb;}
  return false;
 }
 bool binaryReverse(Node& n,bool optimized=true){
  auto& t=n.t;auto& g=t.g;int id=n.inputs[0];auto& a=nodes[id].t;
  if(optimized&&n.op=="mul"&&facade::channelMulBackward(a.shape,nodes[n.inputs[1]].t.shape,t.shape,id,n.inputs[1],nodes[id].op=="constant",nodes[n.inputs[1]].op=="constant",a.x,nodes[n.inputs[1]].t.x,g,a.g,nodes[n.inputs[1]].t.g))return true;
  // Fallback backward: the derivative of each operation with respect to each
  // operand. For mul: da += g*b and db += g*a. For div: da += g/b and
  // db += -g*a/b^2. For add/sub the gradient passes straight through.
  int bid=n.inputs[1];auto& b=nodes[bid].t;for(int i=0;i<t.count();i++){int ai=at(a,t,i),bi=at(b,t,i);float aa=a.x[ai],bb=b.x[bi];add(id,ai,n.op=="mul"?g[i]*bb:n.op=="div"?g[i]/bb:g[i]);add(bid,bi,n.op=="add"?g[i]:n.op=="sub"?-g[i]:n.op=="mul"?g[i]*aa:-g[i]*aa/(bb*bb));}
  return false;
 }
 // End of binary operations.
 // Forward pass. `rgb` is the 112x112 face crop as 3*112*112 floats in CHW
 // order, values 0..1. Walks every node in order and fills its t.x.
 void forward(const float* rgb){ProfileScope profile("native.forward");
  // Drop last run's gradient and make sure the output buffer has the right size.
  for(auto& n:nodes){auto& t=n.t;t.g.clear();{ProfileScope p("native.activation.resize");t.x.resize(t.count());}auto& y=t.x;
   // "input": copy the pixels in. "constant": the value was loaded into n.weight.
   if(n.op=="input"){require(t.count()==3*112*112,"Input shape");std::copy(rgb,rgb+t.count(),y.begin());continue;}
   if(n.op=="constant"){y=n.weight;continue;}
   auto& a=nodes[n.inputs[0]].t;
   // Every remaining operation reads at least node inputs[0], called `a` here.
   if(n.op=="identity")y=a.x;
   // conv: y[oc][pixel] = bias[oc] + sum over k of weight[oc][k] * col[k][pixel],
   // with k running over (input channel, kernel row, kernel column).
   else if(n.op=="conv"){
    auto col=columns(a,n);int oc=t.shape[1],k=a.shape[1]*n.kernel[0]*n.kernel[1],l=t.shape[2]*t.shape[3];
    // Pre-fill each output channel with its bias; beta=1 in the GEMM adds onto it.
    for(int c=0;c<oc;c++)std::fill(y.begin()+c*l,y.begin()+(c+1)*l,n.bias.empty()?0:n.bias[c]);
    // GEMM: (oc x k) times (k x l) -> (oc x l). Most of the forward time is here.
    {ProfileScope p("native.forward.conv.gemm");cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,oc,l,k,1,n.weight.data(),k,col.data(),l,n.bias.empty()?0:1,y.data(),l);}
   // linear: y = bias + x * W^T, with W stored as (out x in).
   }else if(n.op=="linear"){
    int out=t.count(),in=a.count();y=n.bias.empty()?Vec(out,0):n.bias;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,out,in,1,a.x.data(),in,n.weight.data(),in,n.bias.empty()?0:1,y.data(),out);
   // matmul: y = x * B where B is another node's output (no bias).
   }else if(n.op=="matmul"){
    auto& b=nodes[n.inputs[1]].t;int out=t.count(),in=a.count();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,out,in,1,a.x.data(),in,b.x.data(),out,0,y.data(),out);
   }else if(n.op=="bn"){
    bnForward(n);
   }else if(n.op=="prelu"){
    preluForward(n);
   // pool2: 2x2 max pooling with stride 2. b is the index of the window's
   // top-left element; the other three are b+1, b+w and b+w+1. The winner's
   // index is kept in n.selected so the backward pass can send the gradient to
   // it. On ties the first element wins (the strict > keeps the earliest).
   }else if(n.op=="pool2"){
    int h=a.shape[2],w=a.shape[3],oh=t.shape[2],ow=t.shape[3];n.selected.resize(t.count());
    for(int c=0;c<t.shape[1];c++)for(int yy=0;yy<oh;yy++)for(int xx=0;xx<ow;xx++){
     int j=(c*oh+yy)*ow+xx,b=(c*h+yy*2)*w+xx*2,selected=b;float best=a.x[b];for(int p:{b+1,b+w,b+w+1})if(a.x[p]>best){best=a.x[p];selected=p;}y[j]=best;n.selected[j]=selected;
    }
   // average: global average pooling, one mean per channel. The sum uses
   // torch_sum so it rounds exactly like PyTorch.
   }else if(n.op=="average"){
    int plane=a.count()/t.count();for(int c=0;c<t.count();c++){float sum=torch_sum(a.x.data()+c*plane,plane);y[c]=sum/plane;}
   // sigmoid: 1/(1+exp(-x)), four values at a time with SLEEF's vectorized exp
   // (the _u10 variant is accurate to 1 ulp; SLEEF is also what PyTorch's
   // vectorized CPU kernels use), then a scalar std::exp tail for the last
   // zero to three values.
   }else if(n.op=="sigmoid"){
    int i=0;const auto one=vdupq_n_f32(1);for(;i+4<=t.count();i+=4){auto e=Sleef_expf4_u10(vnegq_f32(vld1q_f32(a.x.data()+i)));vst1q_f32(y.data()+i,vdivq_f32(one,vaddq_f32(one,e)));}
    for(;i<t.count();i++)y[i]=1.f/(1.f+std::exp(-a.x[i]));
   }
   // normalize: divide by the L2 norm. The sum of squares is accumulated in
   // double to limit rounding; the 1e-12 floor guards against dividing by zero.
   else if(n.op=="normalize"){
    double sum=0;for(float v:a.x)sum+=double(v)*v;float norm=std::max(float(std::sqrt(sum)),1e-12f);for(int i=0;i<t.count();i++)y[i]=a.x[i]/norm;
   // add with two same-size, distinct inputs: the equal-add kernel. If it
   // declines (returns false) the general binary path below runs.
   }else if(n.op=="add"&&facade::equalAddForward(a,nodes[n.inputs[1]].t,t,n.inputs[0]!=n.inputs[1])){
   // Any other add, or sub/mul/div.
   }else if(n.op=="add"||n.op=="sub"||n.op=="mul"||n.op=="div"){
    binaryForward(n);
   }else throw std::runtime_error("Unknown native operation "+n.op);
  }
 }
 // Backward pass. `ref` is the 512-number reference embedding. The score is
 // embedding . ref, so d(score)/d(embedding) = ref: that is the gradient seeded
 // into the output node. Then every node is visited in reverse order and passes
 // its gradient back to its inputs with add(). When the walk finishes,
 // nodes[0].t.g (the input node) holds d(score)/d(pixel) for all 3*112*112
 // pixels.
 void reverse(const float* ref){ProfileScope profile("native.backward");
  nodes[output].t.g.assign(ref,ref+512);
  // Skip nodes nothing flowed into (g empty) and the leaves.
  for(int ni=(int)nodes.size()-1;ni>=0;ni--){auto& n=nodes[ni];auto& t=n.t;auto& g=t.g;if(g.empty()||n.op=="input"||n.op=="constant")continue;int id=n.inputs[0];auto& a=nodes[id].t;
   if(n.op=="identity"){for(int i=0;i<t.count();i++)add(id,i,g[i]);}
   // conv backward: col = W^T * g gives the gradient of every patch element
   // ((k x oc) times (oc x l) -> (k x l)); scatterColumns then adds each patch
   // element back onto the input pixel it came from (the transpose of columns()).
   else if(n.op=="conv"){
    int oc=t.shape[1],c=a.shape[1],h=a.shape[2],w=a.shape[3],kh=n.kernel[0],kw=n.kernel[1],oh=t.shape[2],ow=t.shape[3],l=oh*ow,k=c*kh*kw;facade_profile_allocation("allocation.native.backward.columns",uint64_t(k)*l*sizeof(float));Vec col;{ProfileScope p("native.backward.buffer");col=Vec(k*l);}
    {ProfileScope p("native.backward.conv.gemm");cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,k,l,oc,1,n.weight.data(),k,g.data(),l,0,col.data(),l);}
    ProfileScope scatter("native.backward.conv.scatter");
    facade::scatterColumns(a.g,nodes[id].op=="constant",a.count(),col.data(),c,h,w,kh,kw,oh,ow,n.stride[0],n.stride[1],n.padding[0],n.padding[1]);
   // linear/matmul backward: ga = g * W (or g * B^T for matmul), then accumulate.
   }else if(n.op=="linear"||n.op=="matmul"){
    Vec ga(a.count());const float* w=n.op=="linear"?n.weight.data():nodes[n.inputs[1]].t.x.data();bool trans=n.op=="matmul";
    cblas_sgemm(CblasRowMajor,CblasNoTrans,trans?CblasTrans:CblasNoTrans,1,a.count(),t.count(),1,g.data(),t.count(),w,trans?t.count():a.count(),0,ga.data(),a.count());for(int i=0;i<a.count();i++)add(id,i,ga[i]);
   }else if(n.op=="bn"){
    bnReverse(n);
   }else if(n.op=="prelu"){
    preluReverse(n);
   // pool2 backward: the whole gradient goes to the element that won the max.
   }else if(n.op=="pool2"){for(int i=0;i<t.count();i++)add(id,n.selected[i],g[i]);}
   // average backward: each input element gets 1/plane of its channel's gradient.
   else if(n.op=="average"){int plane=a.count()/t.count();for(int i=0;i<a.count();i++)add(id,i,g[i/plane]/plane);}
   // sigmoid backward: dy/dx = y*(1-y), using the saved output t.x.
   else if(n.op=="sigmoid"){for(int i=0;i<t.count();i++)add(id,i,g[i]*(1-t.x[i])*t.x[i]);}
   // normalize backward: for y = x/|x|, dx = g/|x| - x * (g.x)/|x|^3.
   else if(n.op=="normalize"){
    double ss=0,dot=0;for(int i=0;i<a.count();i++){ss+=double(a.x[i])*a.x[i];dot+=double(g[i])*a.x[i];}float norm=std::max(float(std::sqrt(ss)),1e-12f),factor=float(dot)/(norm*norm*norm);for(int i=0;i<a.count();i++)add(id,i,g[i]/norm-a.x[i]*factor);
   // Equal-add backward kernel, then the general binary fallback, as in forward.
   }else if(n.op=="add"&&facade::equalAddBackward(a,nodes[n.inputs[1]].t,t,nodes[id].op=="constant",nodes[n.inputs[1]].op=="constant",id!=n.inputs[1])){
   }else if(n.op=="add"||n.op=="sub"||n.op=="mul"||n.op=="div"){
    binaryReverse(n);
   }else throw std::runtime_error("Unknown reverse operation "+n.op);
   // This node's gradient is no longer needed. Swapping with an empty vector
   // releases the memory now rather than at the next forward pass.
   Vec().swap(g);
  }
 }
};
// Copies an error message into the caller's buffer, if one was given.
static void error_text(char* out,size_t n,const char* text){if(out&&n)snprintf(out,n,"%s",text);}
// ---- C entry points called from Swift (see App/NativeTensor.swift) ----
// facade_network_load returns an opaque pointer to a Network, or null with an
// error message. facade_network_free releases it.
void* facade_network_load(const char* graph,const char* weights,char* error,size_t n){try{return new Network(graph,weights);}catch(const std::exception& e){error_text(error,n,e.what());return nullptr;}}
void facade_network_free(void* p){delete static_cast<Network*>(p);}
// Forward + backward in one call: writes the 512-float embedding, the score
// (embedding . ref, accumulated in double) and the 3*112*112 pixel gradient.
// Returns 0 on success, -1 with a message on any failure.
int facade_network_predict(void* p,const float* rgb,const float* ref,float* e,float* score,float* g,char* error,size_t n){
 try{require(p&&rgb&&ref&&e&&score&&g,"Null native input");auto& model=*static_cast<Network*>(p);model.forward(rgb);auto& emb=model.nodes[model.output].t.x;std::copy(emb.begin(),emb.end(),e);double s=0;for(int i=0;i<512;i++)s+=double(e[i])*ref[i];*score=float(s);model.reverse(ref);auto& gradient=model.nodes[0].t.g;require(gradient.size()==3*112*112,"Native gradient shape");std::copy(gradient.begin(),gradient.end(),g);return 0;}catch(const std::exception& ex){error_text(error,n,ex.what());return -1;}
}
// Forward only: embedding and score, no gradient. Used when the optimizer
// only needs to measure, not to update.
int facade_network_forward(void* p,const float* rgb,const float* ref,float* e,float* score,char* error,size_t n){
 try{require(p&&rgb&&ref&&e&&score,"Null native input");auto& model=*static_cast<Network*>(p);model.forward(rgb);auto& emb=model.nodes[model.output].t.x;std::copy(emb.begin(),emb.end(),e);double s=0;for(int i=0;i<512;i++)s+=double(e[i])*ref[i];*score=float(s);return 0;}catch(const std::exception& ex){error_text(error,n,ex.what());return -1;}
}
// Read-only access to any node's forward output, for comparing the phone's
// layers one by one against the desktop model. Never changes inference.
extern "C" const float* facade_network_layer(void* p,int i,int* count){auto& ns=static_cast<Network*>(p)->nodes;if(i<0||i>=(int)ns.size()){*count=0;return nullptr;}*count=ns[i].t.x.size();return ns[i].t.x.data();}

// =====================================================================
// Self-check entry points. Nothing below runs while protecting a photo.
// Each one builds a tiny tensor or network from caller-supplied numbers, runs
// one kernel with `optimized` on or off, and hands the result back so an
// outside test can confirm the fast kernel and the plain loop agree bit for
// bit. The *_benchmark variants time repeated runs; they rotate between four
// copies of the data with their bits XOR-scrambled (bits ^= v*0x1011) so the
// CPU cache and branch predictor do not see the same values every time, and
// the `asm volatile` lines stop the compiler from skipping work whose result
// it thinks is unused.
// =====================================================================
// Column (im2col) self-check. The 11 geometry integers are: c, h, w, kh, kw,
// stride h, stride w, padding h, padding w, output h, output w.
// Validates the 11 geometry integers and builds a Tensor + Node that
// Network::columns can consume.
static void columns_fixture_input(const float* input,size_t inputCount,const int* g,size_t geometryCount,
 float* out,size_t outputCount,Tensor& a,Node& node) {
 require(input&&g&&out,"Null column fixture input");
 require(geometryCount==11,"Column fixture geometry count");
 for(int i=0;i<11;i++)require(g[i]>=0&&g[i]<=4096,"Column fixture geometry range");
 int c=g[0],h=g[1],w=g[2],kh=g[3],kw=g[4],sh=g[5],sw=g[6],ph=g[7],pw=g[8],oh=g[9],ow=g[10];
 require(c>0&&h>0&&w>0&&kh>0&&kw>0&&sh>0&&sw>0&&oh>0&&ow>0,"Column fixture positive dimensions");
 // The output size must be what the convolution formula gives:
 // (in + 2*pad - kernel)/stride + 1.
 require(h+2*ph>=kh&&w+2*pw>=kw&&oh==(h+2*ph-kh)/sh+1&&ow==(w+2*pw-kw)/sw+1,"Column fixture output geometry");
 size_t in=size_t(c)*h*w,n=size_t(c)*kh*kw*oh*ow;
 require(in<=10000000&&n<=50000000&&inputCount==in&&outputCount==n,"Column fixture buffer size");
 a.shape={1,c,h,w};a.x.assign(input,input+inputCount);node.kernel={kh,kw};node.stride={sh,sw};node.padding={ph,pw};node.t.shape={1,1,oh,ow};
}
// Runs columns() once. *specialized reports whether the 3x3 fast kernel was
// eligible for this geometry.
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
// Times `repeats` runs of columns(). Each run uses one of four bit-scrambled
// copies of the input so the data is not identical every time.
extern "C" int facade_columns_benchmark(const float* input,size_t inputCount,const int* geometry,size_t geometryCount,
 int optimized,int repeats,float* out,size_t outputCount,int* specialized,double* seconds,char* error,size_t capacity) {
 try {
  require((optimized==0||optimized==1)&&specialized&&seconds,"Column fixture mode");
  require(repeats>0&&repeats<=10000,"Column benchmark repeat count");
  Tensor a;Node node;columns_fixture_input(input,inputCount,geometry,geometryCount,out,outputCount,a,node);
  std::vector<Tensor> variants(4,a);
  // Copies 1..3 differ from copy 0 by flipped mantissa bits; the geometry is the same.
  for(size_t v=1;v<variants.size();v++)for(float& x:variants[v].x){uint32_t bits;std::memcpy(&bits,&x,4);bits^=uint32_t(v)*0x1011u;std::memcpy(&x,&bits,4);}
  Vec retained;
  const uint64_t start=facade_profile_now();
  for(int i=0;i<repeats;i++){auto current=Network::columns(variants[size_t(i)%4],node,optimized!=0);retained.swap(current);asm volatile("" : : "r"(retained.data()) : "memory");}
  *seconds=double(facade_profile_now()-start)/1e9;
  // Copy and hash only after the timed loop; the profile counters see every iteration.
  std::memcpy(out,retained.data(),retained.size()*sizeof(float));
  *specialized=optimized&&facade::columns3x3Eligible(geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[5],geometry[6],geometry[7],geometry[8],geometry[9],geometry[10]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}

// Scatter (the reverse of im2col) self-check. Same 11-integer geometry as the
// column fixture.
static int scatter_fixture_geometry(const float* col,size_t columnCount,const float* initial,size_t initialCount,
 const int* g,size_t geometryCount,float* out,size_t outputCapacity,bool benchmark) {
 require(col&&g&&out,"Null scatter fixture input");
 require(geometryCount==11,"Scatter fixture geometry count");
 for(int i=0;i<11;i++)require(g[i]>=0&&g[i]<=4096,"Scatter fixture geometry range");
 int c=g[0],h=g[1],w=g[2],kh=g[3],kw=g[4],sh=g[5],sw=g[6],ph=g[7],pw=g[8],oh=g[9],ow=g[10];
 require(c>0&&h>0&&w>0&&kh>0&&kw>0&&sh>0&&sw>0&&oh>0&&ow>0,"Scatter fixture positive dimensions");
 require(h+2*ph>=kh&&w+2*pw>=kw&&oh==(h+2*ph-kh)/sh+1&&ow==(w+2*pw-kw)/sw+1,"Scatter fixture output geometry");
 size_t n=size_t(c)*h*w,m=size_t(c)*kh*kw*oh*ow;
 // The benchmark variant keeps four destination gradients, so it needs 4x the space.
 require(n<=10000000&&m<=50000000&&columnCount==m&&outputCapacity==n*(benchmark?4:1),"Scatter fixture buffer size");
 require(initialCount==0||(initial&&initialCount==n),"Scatter initial gradient size");
 return int(n);
}
static void scatter_fixture_call(Vec& gradient,bool constant,int n,const float* col,const int* g,bool optimized) {
 ProfileScope scope("native.backward.conv.scatter");
 facade::scatterColumns(gradient,constant,n,col,g[0],g[1],g[2],g[3],g[4],g[9],g[10],g[5],g[6],g[7],g[8],optimized);
}
// Runs scatterColumns once. `initial` optionally pre-fills the destination
// gradient so accumulation (+=) onto existing values is tested too.
// `constant` = 1 means the destination is a constant node, which must stay
// untouched.
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
// Times scatterColumns. Even slots start from an empty gradient (so the lazy
// allocation is timed too), odd slots from the supplied initial one.
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
  // All four destinations are copied out after timing; setup is excluded.
  for(size_t slot=0;slot<4;slot++){
   require(gradients[slot].size()==size_t(n),"Scatter benchmark destination size");
   std::memcpy(out+slot*size_t(n),gradients[slot].data(),size_t(n)*sizeof(float));
  }
  *specialized=optimized&&facade::scatter3x3Eligible(n,geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[9],geometry[10],geometry[5],geometry[6],geometry[7],geometry[8]);
  return 0;
 }catch(const std::exception& e){error_text(error,capacity,e.what());return -1;}
}

// Multiply self-check. The 16 geometry integers are: rank of a, rank of b,
// rank of out, alias flag (1 = both inputs are the same node), then three
// zero-padded 4-integer shapes for a, b and out.
// Unpacks one zero-padded 4-integer shape and checks it is in bounds.
static std::vector<int> mul_fixture_shape(const int* geometry,int rankIndex,int start) {
 require(geometry[rankIndex]>=1&&geometry[rankIndex]<=4,"Mul fixture rank");
 std::vector<int> shape;for(int j=0;j<4;j++){
  if(j<geometry[rankIndex])shape.push_back(geometry[start+j]);
  else require(geometry[start+j]==0,"Mul fixture shape padding");
 }require(facade::mulBoundedCount(shape)>0,"Mul fixture shape bounds");return shape;
}
// Builds a three-node network (a, b, a*b) from the caller's buffers, including
// optional pre-existing gradients on a and b and the upstream gradient g.
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
// Runs forward and backward once. flags = {forward used the fast path,
// backward used the fast path, the output gradient was released afterwards}.
static void mul_fixture_run(Network& model,bool optimized,int* flags) {
 auto& node=model.nodes[2];
 {ProfileScope p("fixture.mul.forward");flags[0]=model.binaryForward(node,optimized);}
 {ProfileScope p("fixture.mul.backward");flags[1]=model.binaryReverse(node,optimized);Vec().swap(node.t.g);}
 flags[2]=node.t.g.empty();
}
// Copies out [y, grad a, grad b] back to back; counts says how many of each
// were present (a constant input has none).
static void mul_fixture_export(Network& model,float* out,size_t* counts) {
 auto& node=model.nodes[2];const auto& a=model.nodes[node.inputs[0]].t;const auto& b=model.nodes[node.inputs[1]].t;
 counts[0]=node.t.x.size();counts[1]=a.g.size();counts[2]=b.g.size();
 std::copy(node.t.x.begin(),node.t.x.end(),out);out+=node.t.x.size();
 if(!a.g.empty())std::copy(a.g.begin(),a.g.end(),out);out+=a.x.size();
 if(!b.g.empty())std::copy(b.g.begin(),b.g.end(),out);
}
// One forward + backward run of the mul node.
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
// Times 128 forward (direction 0) or backward (direction 1) runs of the mul
// node, rotating over four scrambled copies of the network. Only the kernel
// call itself is inside the timed interval.
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
  // The backward pass reads the forward output, so compute it once beforehand,
  // outside the timing.
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
// End of the multiply self-check.

// The remaining self-check drivers are large enough to live in their own
// headers. They are #included here rather than compiled separately because
// they use Tensor, Node, Network and error_text from this file.
#include "NativeEqualAddFixtures.h"

// Small graphs that mix add and mul nodes, with repeated consumers.
#include "NativeAddMulSeam.h"

// PReLU router: fast kernel against the plain loop.
#include "NativePreluFixtures.h"

// Batch-norm router: fast kernel against the plain loop.
#include "NativeBatchNormFixtures.h"
// No Swift code in the app calls any of the fixture entry points above.
