// peak_2pf_probe.cu
// Honest SM121 2-PF research probe. Build/disassemble is safe while production owns GPU;
// executable must only be run in an explicit benchmark window.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

static constexpr double FLOP_PER_SPARSE_OMMA = 2.0 * 16.0 * 8.0 * 128.0; // 32768 dense-equivalent logical FLOPs

#define OMMA_2X(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,META,SF,Z) \
 asm volatile("mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X." \
 "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0 " \
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
 : "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3) \
 : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(META),"r"(SF),"h"(Z),"h"(Z),"r"(SF),"h"(Z),"h"(Z))
#define OMMA_4X(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,META,SF,Z) \
 asm volatile("mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
 "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
 : "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3) \
 : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(META),"r"(SF),"h"(Z),"h"(Z),"r"(SF),"h"(Z),"h"(Z))

template<int FORM, int BANKS, int ACC_PER_BANK>
__global__ __launch_bounds__(128,1) void peak_issue(int iters, float* sink) {
  uint32_t a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
  uint32_t c0=0x33333333,c1=c0,c2=c0,c3=c0,e0=c0,e1=e0,e2=e0,e3=e0;
  uint32_t meta0=0xEEEEEEEEu, meta1=0xDDDDDDDDu;
  uint32_t sf0 = FORM==2 ? 0x00008181u : 0x38383838u;
  uint32_t sf1 = FORM==2 ? 0x00008282u : 0x39393939u;
  const uint16_t z=0;
  float x[ACC_PER_BANK][4];
  float y[ACC_PER_BANK][4];
  #pragma unroll
  for(int j=0;j<ACC_PER_BANK;++j) for(int q=0;q<4;++q){x[j][q]=0.f; y[j][q]=0.f;}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for(int i=0;i<iters;++i) {
    #pragma unroll
    for(int j=0;j<ACC_PER_BANK;++j) {
      if constexpr (FORM==2) OMMA_2X(x[j][0],x[j][1],x[j][2],x[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta0,sf0,z);
      else                   OMMA_4X(x[j][0],x[j][1],x[j][2],x[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta0,sf0,z);
      if constexpr (BANKS==2) {
        if constexpr (FORM==2) OMMA_2X(y[j][0],y[j][1],y[j][2],y[j][3],c0,c1,c2,c3,e0,e1,e2,e3,meta1,sf1,z);
        else                   OMMA_4X(y[j][0],y[j][1],y[j][2],y[j][3],c0,c1,c2,c3,e0,e1,e2,e3,meta1,sf1,z);
      }
    }
  }
#endif
  float s=0.f;
  #pragma unroll
  for(int j=0;j<ACC_PER_BANK;++j) for(int q=0;q<4;++q) s += x[j][q] + (BANKS==2 ? y[j][q] : 0.f);
  if(threadIdx.x==0) sink[blockIdx.x]=s;
}

template<int FORM,int BANKS,int ACC>
static double bench(const char* name,int blocks,int block,int iters,float* sink,cudaEvent_t a,cudaEvent_t b) {
  peak_issue<FORM,BANKS,ACC><<<blocks,block>>>(iters,sink); cudaDeviceSynchronize();
  std::vector<float> ms;
  for(int r=0;r<12;++r){ cudaEventRecord(a); peak_issue<FORM,BANKS,ACC><<<blocks,block>>>(iters,sink); cudaEventRecord(b); cudaEventSynchronize(b); float x=0; cudaEventElapsedTime(&x,a,b); ms.push_back(x); }
  std::sort(ms.begin(),ms.end()); double med=0.5*(ms[5]+ms[6]);
  long long warps=(long long)blocks*(block/32); long long omma_per_warp=(long long)iters*BANKS*ACC;
  double total_omma=(double)warps*omma_per_warp;
  double omma_s=total_omma/(med*1e-3); double tf=omma_s*FLOP_PER_SPARSE_OMMA/1e12;
  std::printf("%-18s median_ms=%9.4f TFLOPS=%9.2f OMMA/s=%12.4e inst/warp=%lld\n",name,med,tf,omma_s,omma_per_warp);
  return tf;
}

int main(){
  int dev=0; cudaSetDevice(dev); cudaDeviceProp p{}; cudaGetDeviceProperties(&p,dev);
  int blocks=p.multiProcessorCount*8, block=128, iters=700; float* sink=nullptr; cudaMalloc(&sink,sizeof(float)*blocks);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::printf("device=%s SMs=%d fixed_flop_per_omma=%.0f\n",p.name,p.multiProcessorCount,FLOP_PER_SPARSE_OMMA);
  double a2=bench<2,1,16>("2X one-bank16",blocks,block,iters,sink,a,b);
  double b2=bench<2,2,8>("2X two-bank8",blocks,block,iters,sink,a,b);
  double c2=bench<2,2,16>("2X two-bank16",blocks,block,iters,sink,a,b);
  double a4=bench<4,1,16>("4X one-bank16",blocks,block,iters,sink,a,b);
  double b4=bench<4,2,8>("4X two-bank8",blocks,block,iters,sink,a,b);
  double c4=bench<4,2,16>("4X two-bank16",blocks,block,iters,sink,a,b);
  std::printf("fair same-instruction-count ratios: 2X two/one=%.4fx  4X two/one=%.4fx\n",b2/a2,b4/a4);
  std::printf("target ratios vs 2PF: 2Xbest=%.3f%% 4Xbest=%.3f%%\n",100.0*std::max({a2,b2,c2})/2000.0,100.0*std::max({a4,b4,c4})/2000.0);
  cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(sink); return 0;
}
