#include <cstdint>
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))
#define IS8(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,C0,C1,C2,C3,M) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0;\n" \
: "=r"(D0),"=r"(D1),"=r"(D2),"=r"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"r"(C0),"r"(C1),"r"(C2),"r"(C3),"r"(M))

__device__ __forceinline__ float do_omma(int iters){
 uint32_t a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=0x33333333,b1=b0,b2=b0,b3=b0,m=0xeeeeeeee,s=0x38383838; const uint16_t z=0; float d[16][4]={{0}};
 #pragma unroll 1
 for(int i=0;i<iters;++i){
  #pragma unroll
  for(int j=0;j<16;++j) O4(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,b2,b3,m,s,z);
 }
 float x=0;
 #pragma unroll
 for(int j=0;j<16;++j)for(int q=0;q<4;++q)x+=d[j][q];
 return x;
}
__device__ __forceinline__ int do_imma(int iters){
 uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=0x03030303,b1=b0,b2=b0,b3=b0,m=0xeeeeeeee; int d[16][4]={{0}};
 #pragma unroll 1
 for(int i=0;i<iters;++i){
  #pragma unroll
  for(int j=0;j<16;++j) IS8(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,b2,b3,d[j][0],d[j][1],d[j][2],d[j][3],m);
 }
 int x=0;
 #pragma unroll
 for(int j=0;j<16;++j)for(int q=0;q<4;++q)x+=d[j][q];
 return x;
}
extern "C" __global__ __launch_bounds__(128,1) void dual_omma_even(int iters,float* fsink,int* isink){
 int warp=threadIdx.x>>5; if((warp&1)==0){ float x=do_omma(iters); if((threadIdx.x&31)==0) fsink[blockIdx.x*2+(warp>>1)]=x; }
}
extern "C" __global__ __launch_bounds__(128,1) void dual_imma_odd(int iters,float* fsink,int* isink){
 int warp=threadIdx.x>>5; if((warp&1)==1){ int x=do_imma(iters); if((threadIdx.x&31)==0) isink[blockIdx.x*2+(warp>>1)]=x; }
}
extern "C" __global__ __launch_bounds__(128,1) void dual_both(int iters,float* fsink,int* isink){
 int warp=threadIdx.x>>5;
 if((warp&1)==0){ float x=do_omma(iters); if((threadIdx.x&31)==0) fsink[blockIdx.x*2+(warp>>1)]=x; }
 else { int x=do_imma(iters); if((threadIdx.x&31)==0) isink[blockIdx.x*2+(warp>>1)]=x; }
}
