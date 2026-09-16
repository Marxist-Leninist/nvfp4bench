#include <cstdint>
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))
extern "C" __global__ __launch_bounds__(128,1) void sass_patch_target(int iters,float* sink){
 uint32_t a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=0x33333333,b1=b0,b2=b0,b3=b0,m0=0xeeeeeeee,s0=0x38383838;
 uint32_t c0=0x44444444,c1=c0,c2=c0,c3=c0,e0=0x55555555,e1=e0,e2=e0,e3=e0,m1=0xdddddddd,s1=0x39393939; const uint16_t z=0;
 float x[8][4], y[8][4];
 #pragma unroll
 for(int j=0;j<8;++j)for(int q=0;q<4;++q){x[j][q]=0.f;y[j][q]=0.f;}
 #pragma unroll 1
 for(int i=0;i<iters;++i){
  O4(x[0][0],x[0][1],x[0][2],x[0][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[1][0],x[1][1],x[1][2],x[1][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[2][0],x[2][1],x[2][2],x[2][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[3][0],x[3][1],x[3][2],x[3][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[4][0],x[4][1],x[4][2],x[4][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[5][0],x[5][1],x[5][2],x[5][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[6][0],x[6][1],x[6][2],x[6][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(x[7][0],x[7][1],x[7][2],x[7][3],a0,a1,a2,a3,b0,b1,b2,b3,m0,s0,z);
  O4(y[0][0],y[0][1],y[0][2],y[0][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[1][0],y[1][1],y[1][2],y[1][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[2][0],y[2][1],y[2][2],y[2][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[3][0],y[3][1],y[3][2],y[3][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[4][0],y[4][1],y[4][2],y[4][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[5][0],y[5][1],y[5][2],y[5][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[6][0],y[6][1],y[6][2],y[6][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
  O4(y[7][0],y[7][1],y[7][2],y[7][3],c0,c1,c2,c3,e0,e1,e2,e3,m1,s1,z);
 }
 float sum=0.f;
 #pragma unroll
 for(int j=0;j<8;++j)for(int q=0;q<4;++q)sum+=x[j][q]+y[j][q];
 if(threadIdx.x==0)sink[blockIdx.x]=sum;
}
