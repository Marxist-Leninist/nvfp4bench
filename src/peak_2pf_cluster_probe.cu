// Matched plain-vs-2CTA-cluster SM121 OMMA benchmark. Do not run while production owns the GPU.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
namespace cg=cooperative_groups;
static constexpr double FPO=2.0*16.0*8.0*128.0;
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))

template<bool CLUSTER> __device__ __forceinline__ void body(int iters,float* sink){
 uint32_t a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0,m=0xEEEEEEEEu,sf=0x38383838u; const uint16_t z=0;
 float d[16][4];
 #pragma unroll
 for(int j=0;j<16;++j) for(int q=0;q<4;++q)d[j][q]=0.f;
 for(int i=0;i<iters;++i){
  #pragma unroll
  for(int j=0;j<16;++j) O4(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,b2,b3,m,sf,z);
 }
 float s=0;
 #pragma unroll
 for(int j=0;j<16;++j) for(int q=0;q<4;++q)s+=d[j][q];
 if(threadIdx.x==0)sink[blockIdx.x]=s;
}
extern "C" __global__ __launch_bounds__(128,1) void plain4x(int iters,float* sink){body<false>(iters,sink);}
extern "C" __global__ __cluster_dims__(2,1,1) __launch_bounds__(128,1) void cluster2_4x(int iters,float* sink){auto c=cg::this_cluster();c.sync();body<true>(iters,sink);c.sync();}

template<class Launch> static double timed(const char* n,Launch launch,int blocks,int iters,float* sink,cudaEvent_t a,cudaEvent_t b){
 launch();cudaDeviceSynchronize();std::vector<float>v;for(int r=0;r<12;++r){cudaEventRecord(a);launch();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());double ms=.5*(v[5]+v[6]);
 double omma=(double)blocks*4.0*iters*16.0;double t=omma/(ms*1e-3)*FPO/1e12;std::printf("%-12s blocks=%d ms=%.4f TFLOPS=%.2f\n",n,blocks,ms,t);return t;
}
int main(){cudaDeviceProp p{};cudaGetDeviceProperties(&p,0);int iters=700;float*s;int maxb=p.multiProcessorCount*16;cudaMalloc(&s,sizeof(float)*maxb);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int bpsm:{2,4,8,16}){int blocks=p.multiProcessorCount*bpsm;blocks-=blocks%2;double x=timed("plain4x",[&]{plain4x<<<blocks,128>>>(iters,s);},blocks,iters,s,a,b);double y=timed("cluster2_4x",[&]{cluster2_4x<<<blocks,128>>>(iters,s);},blocks,iters,s,a,b);std::printf("b/SM=%d cluster/plain=%.4fx\n",bpsm,y/x);}cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(s);}
