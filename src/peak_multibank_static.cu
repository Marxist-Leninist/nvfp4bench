#include <cstdint>
#define O4(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,M,S,Z) asm volatile( \
"mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
: "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3):"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3),"f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(M),"r"(S),"h"(Z),"h"(Z),"r"(S),"h"(Z),"h"(Z))
extern "C" __global__ __launch_bounds__(128,1) void multibank_1x16(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[16][4];
  const uint16_t z=0;
  for(int j=0;j<16;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<16;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
  }
  float sum=0.f;
  for(int j=0;j<16;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,1) void multibank_2x8(int iters,float* sink){
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

extern "C" __global__ __launch_bounds__(128,1) void multibank_4x4(int iters,float* sink){
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

extern "C" __global__ __launch_bounds__(128,1) void multibank_8x2(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[2][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[2][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[2][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[2][4];
  uint32_t a40=0x55555555u,a41=a40,a42=a40,a43=a40,b40=0x5a5a5a5au,b41=b40,b42=b40,b43=b40,m4=0xaaaaaaaau,s4=0x3c3c3c3cu;
  float d4[2][4];
  uint32_t a50=0x66666666u,a51=a50,a52=a50,a53=a50,b50=0x69696969u,b51=b50,b52=b50,b53=b50,m5=0x99999999u,s5=0x3d3d3d3du;
  float d5[2][4];
  uint32_t a60=0x77777777u,a61=a60,a62=a60,a63=a60,b60=0x78787878u,b61=b60,b62=b60,b63=b60,m6=0x88888888u,s6=0x3e3e3e3eu;
  float d6[2][4];
  uint32_t a70=0x88888888u,a71=a70,a72=a70,a73=a70,b70=0x87878787u,b71=b70,b72=b70,b73=b70,m7=0x77777777u,s7=0x3f3f3f3fu;
  float d7[2][4];
  const uint16_t z=0;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d4[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d5[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d6[j][q]=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) d7[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<2;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<2;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<2;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<2;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
    for(int j=0;j<2;++j) O4(d4[j][0],d4[j][1],d4[j][2],d4[j][3],a40,a41,a42,a43,b40,b41,b42,b43,m4,s4,z);
    for(int j=0;j<2;++j) O4(d5[j][0],d5[j][1],d5[j][2],d5[j][3],a50,a51,a52,a53,b50,b51,b52,b53,m5,s5,z);
    for(int j=0;j<2;++j) O4(d6[j][0],d6[j][1],d6[j][2],d6[j][3],a60,a61,a62,a63,b60,b61,b62,b63,m6,s6,z);
    for(int j=0;j<2;++j) O4(d7[j][0],d7[j][1],d7[j][2],d7[j][3],a70,a71,a72,a73,b70,b71,b72,b73,m7,s7,z);
  }
  float sum=0.f;
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d4[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d5[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d6[j][q];
  for(int j=0;j<2;++j) for(int q=0;q<4;++q) sum += d7[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}

extern "C" __global__ __launch_bounds__(128,1) void multibank_16x1(int iters,float* sink){
  uint32_t a00=0x11111111u,a01=a00,a02=a00,a03=a00,b00=0x1e1e1e1eu,b01=b00,b02=b00,b03=b00,m0=0xeeeeeeeeu,s0=0x38383838u;
  float d0[1][4];
  uint32_t a10=0x22222222u,a11=a10,a12=a10,a13=a10,b10=0x2d2d2d2du,b11=b10,b12=b10,b13=b10,m1=0xddddddddu,s1=0x39393939u;
  float d1[1][4];
  uint32_t a20=0x33333333u,a21=a20,a22=a20,a23=a20,b20=0x3c3c3c3cu,b21=b20,b22=b20,b23=b20,m2=0xccccccccu,s2=0x3a3a3a3au;
  float d2[1][4];
  uint32_t a30=0x44444444u,a31=a30,a32=a30,a33=a30,b30=0x4b4b4b4bu,b31=b30,b32=b30,b33=b30,m3=0xbbbbbbbbu,s3=0x3b3b3b3bu;
  float d3[1][4];
  uint32_t a40=0x55555555u,a41=a40,a42=a40,a43=a40,b40=0x5a5a5a5au,b41=b40,b42=b40,b43=b40,m4=0xaaaaaaaau,s4=0x3c3c3c3cu;
  float d4[1][4];
  uint32_t a50=0x66666666u,a51=a50,a52=a50,a53=a50,b50=0x69696969u,b51=b50,b52=b50,b53=b50,m5=0x99999999u,s5=0x3d3d3d3du;
  float d5[1][4];
  uint32_t a60=0x77777777u,a61=a60,a62=a60,a63=a60,b60=0x78787878u,b61=b60,b62=b60,b63=b60,m6=0x88888888u,s6=0x3e3e3e3eu;
  float d6[1][4];
  uint32_t a70=0x88888888u,a71=a70,a72=a70,a73=a70,b70=0x87878787u,b71=b70,b72=b70,b73=b70,m7=0x77777777u,s7=0x3f3f3f3fu;
  float d7[1][4];
  uint32_t a80=0x99999999u,a81=a80,a82=a80,a83=a80,b80=0x96969696u,b81=b80,b82=b80,b83=b80,m8=0x66666666u,s8=0x40404040u;
  float d8[1][4];
  uint32_t a90=0xaaaaaaaau,a91=a90,a92=a90,a93=a90,b90=0xa5a5a5a5u,b91=b90,b92=b90,b93=b90,m9=0x55555555u,s9=0x41414141u;
  float d9[1][4];
  uint32_t a100=0xbbbbbbbbu,a101=a100,a102=a100,a103=a100,b100=0xb4b4b4b4u,b101=b100,b102=b100,b103=b100,m10=0x44444444u,s10=0x42424242u;
  float d10[1][4];
  uint32_t a110=0xccccccccu,a111=a110,a112=a110,a113=a110,b110=0xc3c3c3c3u,b111=b110,b112=b110,b113=b110,m11=0x33333333u,s11=0x43434343u;
  float d11[1][4];
  uint32_t a120=0xddddddddu,a121=a120,a122=a120,a123=a120,b120=0xd2d2d2d2u,b121=b120,b122=b120,b123=b120,m12=0x22222222u,s12=0x44444444u;
  float d12[1][4];
  uint32_t a130=0xeeeeeeeeu,a131=a130,a132=a130,a133=a130,b130=0xe1e1e1e1u,b131=b130,b132=b130,b133=b130,m13=0x11111111u,s13=0x45454545u;
  float d13[1][4];
  uint32_t a140=0xffffffffu,a141=a140,a142=a140,a143=a140,b140=0xf0f0f0f0u,b141=b140,b142=b140,b143=b140,m14=0x00000000u,s14=0x46464646u;
  float d14[1][4];
  uint32_t a150=0x111111110u,a151=a150,a152=a150,a153=a150,b150=0x1e1e1e1fu,b151=b150,b152=b150,b153=b150,m15=0xeeeeeeefu,s15=0x47474747u;
  float d15[1][4];
  const uint16_t z=0;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d0[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d1[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d2[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d3[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d4[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d5[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d6[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d7[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d8[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d9[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d10[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d11[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d12[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d13[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d14[j][q]=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) d15[j][q]=0.f;
  for(int i=0;i<iters;++i){
    for(int j=0;j<1;++j) O4(d0[j][0],d0[j][1],d0[j][2],d0[j][3],a00,a01,a02,a03,b00,b01,b02,b03,m0,s0,z);
    for(int j=0;j<1;++j) O4(d1[j][0],d1[j][1],d1[j][2],d1[j][3],a10,a11,a12,a13,b10,b11,b12,b13,m1,s1,z);
    for(int j=0;j<1;++j) O4(d2[j][0],d2[j][1],d2[j][2],d2[j][3],a20,a21,a22,a23,b20,b21,b22,b23,m2,s2,z);
    for(int j=0;j<1;++j) O4(d3[j][0],d3[j][1],d3[j][2],d3[j][3],a30,a31,a32,a33,b30,b31,b32,b33,m3,s3,z);
    for(int j=0;j<1;++j) O4(d4[j][0],d4[j][1],d4[j][2],d4[j][3],a40,a41,a42,a43,b40,b41,b42,b43,m4,s4,z);
    for(int j=0;j<1;++j) O4(d5[j][0],d5[j][1],d5[j][2],d5[j][3],a50,a51,a52,a53,b50,b51,b52,b53,m5,s5,z);
    for(int j=0;j<1;++j) O4(d6[j][0],d6[j][1],d6[j][2],d6[j][3],a60,a61,a62,a63,b60,b61,b62,b63,m6,s6,z);
    for(int j=0;j<1;++j) O4(d7[j][0],d7[j][1],d7[j][2],d7[j][3],a70,a71,a72,a73,b70,b71,b72,b73,m7,s7,z);
    for(int j=0;j<1;++j) O4(d8[j][0],d8[j][1],d8[j][2],d8[j][3],a80,a81,a82,a83,b80,b81,b82,b83,m8,s8,z);
    for(int j=0;j<1;++j) O4(d9[j][0],d9[j][1],d9[j][2],d9[j][3],a90,a91,a92,a93,b90,b91,b92,b93,m9,s9,z);
    for(int j=0;j<1;++j) O4(d10[j][0],d10[j][1],d10[j][2],d10[j][3],a100,a101,a102,a103,b100,b101,b102,b103,m10,s10,z);
    for(int j=0;j<1;++j) O4(d11[j][0],d11[j][1],d11[j][2],d11[j][3],a110,a111,a112,a113,b110,b111,b112,b113,m11,s11,z);
    for(int j=0;j<1;++j) O4(d12[j][0],d12[j][1],d12[j][2],d12[j][3],a120,a121,a122,a123,b120,b121,b122,b123,m12,s12,z);
    for(int j=0;j<1;++j) O4(d13[j][0],d13[j][1],d13[j][2],d13[j][3],a130,a131,a132,a133,b130,b131,b132,b133,m13,s13,z);
    for(int j=0;j<1;++j) O4(d14[j][0],d14[j][1],d14[j][2],d14[j][3],a140,a141,a142,a143,b140,b141,b142,b143,m14,s14,z);
    for(int j=0;j<1;++j) O4(d15[j][0],d15[j][1],d15[j][2],d15[j][3],a150,a151,a152,a153,b150,b151,b152,b153,m15,s15,z);
  }
  float sum=0.f;
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d0[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d1[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d2[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d3[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d4[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d5[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d6[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d7[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d8[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d9[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d10[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d11[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d12[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d13[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d14[j][q];
  for(int j=0;j<1;++j) for(int q=0;q<4;++q) sum += d15[j][q];
  if(threadIdx.x==0) sink[blockIdx.x]=sum;
}
