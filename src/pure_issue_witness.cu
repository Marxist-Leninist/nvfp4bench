#include <cstdint>
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))
#define DO(J) O4(d[J][0],d[J][1],d[J][2],d[J][3],a0,a1,a2,a3,b0,b1,b2,b3,m,s,z)
extern "C" __global__ __launch_bounds__(128,1) void pure_issue_witness(int iters,float* sink){
 uint32_t a0=0x22222222u,a1=a0,a2=a0,a3=a0,b0=0x33333333u,b1=b0,b2=b0,b3=b0,m=0xeeeeeeeeu,s=0x38383838u; const uint16_t z=0;
 float d[16][4];
 #pragma unroll
 for(int j=0;j<16;++j)for(int q=0;q<4;++q)d[j][q]=0.f;
 #pragma unroll 1
 for(int i=0;i<iters;++i){DO(0);DO(1);DO(2);DO(3);DO(4);DO(5);DO(6);DO(7);DO(8);DO(9);DO(10);DO(11);DO(12);DO(13);DO(14);DO(15);}
 // Keep every lane-0 accumulator component separately. This makes each donor's
 // arithmetic contribution independently checkable instead of hiding it in one sum.
 if(threadIdx.x==0){
  #pragma unroll
  for(int j=0;j<16;++j)for(int q=0;q<4;++q)sink[64*blockIdx.x + 4*j + q]=d[j][q];
 }
}
