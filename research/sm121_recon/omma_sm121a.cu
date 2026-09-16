#include <cstdint>
extern "C" __global__ void omma_probe(float* out) {
  uint32_t a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
  uint32_t meta=0xEEEEEEEEu,sf=0x38383838u; const uint16_t z=0;
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
    "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},"
    "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(b2),"r"(b3),
      "f"(d0),"f"(d1),"f"(d2),"f"(d3),"r"(meta),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  if (threadIdx.x==0) out[0]=d0+d1+d2+d3;
}
