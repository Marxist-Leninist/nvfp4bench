#include <cstdint>
#define BX(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,C0,C1,C2,C3) asm volatile( \
"mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
: "=r"(D0),"=r"(D1),"=r"(D2),"=r"(D3) \
: "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(C0),"r"(C1),"r"(C2),"r"(C3))
#define BA(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,C0,C1,C2,C3) asm volatile( \
"mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
: "=r"(D0),"=r"(D1),"=r"(D2),"=r"(D3) \
: "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(C0),"r"(C1),"r"(C2),"r"(C3))
extern "C" __global__ __launch_bounds__(128,1) void bmma_xor(int iters,int* sink){
 uint32_t a0=0xaaaaaaaa,a1=0x55555555,a2=0xcccccccc,a3=0x33333333,b0=0xf0f0f0f0,b1=0x0f0f0f0f; int d[16][4]={{0}};
 #pragma unroll 1
 for(int i=0;i<iters;++i){
  #pragma unroll
  for(int j=0;j<16;++j) BX(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,d[j][0],d[j][1],d[j][2],d[j][3]);
 }
 int s=0;
 #pragma unroll
 for(int j=0;j<16;++j)for(int q=0;q<4;++q)s+=d[j][q]; if(threadIdx.x==0)sink[blockIdx.x]=s;
}
extern "C" __global__ __launch_bounds__(128,1) void bmma_and(int iters,int* sink){
 uint32_t a0=0xaaaaaaaa,a1=0x55555555,a2=0xcccccccc,a3=0x33333333,b0=0xf0f0f0f0,b1=0x0f0f0f0f; int d[16][4]={{0}};
 #pragma unroll 1
 for(int i=0;i<iters;++i){
  #pragma unroll
  for(int j=0;j<16;++j) BA(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,d[j][0],d[j][1],d[j][2],d[j][3]);
 }
 int s=0;
 #pragma unroll
 for(int j=0;j<16;++j)for(int q=0;q<4;++q)s+=d[j][q]; if(threadIdx.x==0)sink[blockIdx.x]=s;
}
