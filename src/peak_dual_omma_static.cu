#include <cstdint>
#define MMA_SPARSE(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,META,SF,Z) \
  asm volatile( \
    "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X." \
    "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15}," \
    "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
    : "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3) \
    : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3), \
      "f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(META),"r"(SF),"h"(Z),"h"(Z),"r"(SF),"h"(Z),"h"(Z))

extern "C" __global__ __launch_bounds__(128,1)
void peak_one_stream_static(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=0x02020202,b1=b0,b2=b0,b3=b0;
  uint32_t meta=0xEEEEEEEEu, sf=0x00008181u; const uint16_t z=0;
  float d[16][4];
  #pragma unroll
  for(int j=0;j<16;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for(int i=0;i<iters;++i) {
    #pragma unroll
    for(int j=0;j<16;++j) MMA_SPARSE(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta,sf,z);
  }
#endif
  float s=0.f;
  #pragma unroll
  for(int j=0;j<16;++j) s += d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if(threadIdx.x==0) sink[0]=s;
}

extern "C" __global__ __launch_bounds__(128,1)
void peak_two_stream_8x8_static(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=0x02020202,b1=b0,b2=b0,b3=b0;
  uint32_t c0=0x03030303,c1=c0,c2=c0,c3=c0,e0=0x03030303,e1=e0,e2=e0,e3=e0;
  uint32_t meta0=0xEEEEEEEEu,meta1=0xDDDDDDDDu,sf0=0x00008181u,sf1=0x00008282u; const uint16_t z=0;
  float x[8][4], y[8][4];
  #pragma unroll
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) x[j][q]=y[j][q]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for(int i=0;i<iters;++i) {
    #pragma unroll
    for(int j=0;j<8;++j) {
      MMA_SPARSE(x[j][0],x[j][1],x[j][2],x[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta0,sf0,z);
      MMA_SPARSE(y[j][0],y[j][1],y[j][2],y[j][3],c0,c1,c2,c3,e0,e1,e2,e3,meta1,sf1,z);
    }
  }
#endif
  float s=0.f;
  #pragma unroll
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) s += x[j][q]+y[j][q];
  if(threadIdx.x==0) sink[0]=s;
}

extern "C" __global__ __launch_bounds__(128,1)
void peak_two_stream_16x16_static(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=0x02020202,b1=b0,b2=b0,b3=b0;
  uint32_t c0=0x03030303,c1=c0,c2=c0,c3=c0,e0=0x03030303,e1=e0,e2=e0,e3=e0;
  uint32_t meta0=0xEEEEEEEEu,meta1=0xDDDDDDDDu,sf0=0x00008181u,sf1=0x00008282u; const uint16_t z=0;
  float x[16][4], y[16][4];
  #pragma unroll
  for(int j=0;j<16;++j) for(int q=0;q<4;++q) x[j][q]=y[j][q]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for(int i=0;i<iters;++i) {
    #pragma unroll
    for(int j=0;j<16;++j) {
      MMA_SPARSE(x[j][0],x[j][1],x[j][2],x[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta0,sf0,z);
      MMA_SPARSE(y[j][0],y[j][1],y[j][2],y[j][3],c0,c1,c2,c3,e0,e1,e2,e3,meta1,sf1,z);
    }
  }
#endif
  float s=0.f;
  #pragma unroll
  for(int j=0;j<16;++j) for(int q=0;q<4;++q) s += x[j][q]+y[j][q];
  if(threadIdx.x==0) sink[0]=s;
}
