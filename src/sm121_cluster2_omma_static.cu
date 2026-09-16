#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define OMMA_SP(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,B2,B3,META,SF,Z) \
  asm volatile( \
    "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
    "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15}," \
    "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n" \
    : "=f"(D0),"=f"(D1),"=f"(D2),"=f"(D3) \
    : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(B2),"r"(B3), \
      "f"(D0),"f"(D1),"f"(D2),"f"(D3),"r"(META),"r"(SF),"h"(Z),"h"(Z),"r"(SF),"h"(Z),"h"(Z))

// Static reconstruction of SM100 cta_group::2 semantics using an explicit
// SM121 2-CTA cluster. This is NOT claimed to be hardware 2CTA MMA: each CTA
// owns disjoint register accumulators and issues ordinary SM121 OMMA.
extern "C" __global__ __cluster_dims__(2,1,1) __launch_bounds__(128,1)
void sm121_cluster2_omma_static(int iters, float* sink) {
  auto cluster = cg::this_cluster();
  unsigned cr = cluster.block_rank();
  cluster.sync();

  uint32_t seed = cr ? 0x33333333u : 0x22222222u;
  uint32_t a0=seed,a1=seed,a2=seed,a3=seed,b0=seed,b1=seed,b2=seed,b3=seed;
  uint32_t meta=0xEEEEEEEEu, sf=0x38383838u; const uint16_t z=0;
  float d[16][4];
  #pragma unroll
  for (int j=0;j<16;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i) {
    #pragma unroll
    for (int j=0;j<16;++j)
      OMMA_SP(d[j][0],d[j][1],d[j][2],d[j][3],a0,a1,a2,a3,b0,b1,b2,b3,meta,sf,z);
  }
#endif
  float s=0.f;
  #pragma unroll
  for(int j=0;j<16;++j) s += d[j][0]+d[j][1]+d[j][2]+d[j][3];
  cluster.sync();
  if (threadIdx.x==0) sink[blockIdx.x]=s;
}
