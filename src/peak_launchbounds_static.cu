#include <cstdint>
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))
extern "C" __global__ __launch_bounds__(128,1) void multibank_2x8_mb1(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,2) void multibank_2x8_mb2(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,3) void multibank_2x8_mb3(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,4) void multibank_2x8_mb4(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,5) void multibank_2x8_mb5(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,6) void multibank_2x8_mb6(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,8) void multibank_2x8_mb8(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[8][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[8][4];
  const uint16_t z=0;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<8;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<8;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
  }
  float sum=0.f;
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<8;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,1) void multibank_4x4_mb1(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,2) void multibank_4x4_mb2(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,3) void multibank_4x4_mb3(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,4) void multibank_4x4_mb4(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,5) void multibank_4x4_mb5(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,6) void multibank_4x4_mb6(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,8) void multibank_4x4_mb8(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[4][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[4][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[4][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[4][4];
  const uint16_t z=0;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<4;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<4;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<4;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<4;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
  }
  float sum=0.f;
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<4;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}
