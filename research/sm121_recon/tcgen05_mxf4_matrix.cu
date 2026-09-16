#include <cstdint>
extern "C" __global__ void tc_mxf4_1cta(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t scale, uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n}" ::
    "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(scale),"r"(sfa),"r"(sfb));
}
extern "C" __global__ void tc_mxf4_2cta(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t scale, uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n}" ::
    "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(scale),"r"(sfa),"r"(sfb));
}
extern "C" __global__ void tc_mxf4_sp_1cta(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t tmem_e, uint32_t idesc, uint32_t scale, uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\ttcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%3], %4, [%6], [%7], p;\n}" ::
    "r"(tmem_c),"l"(da),"l"(db),"r"(tmem_e),"r"(idesc),"r"(scale),"r"(sfa),"r"(sfb));
}
extern "C" __global__ void tc_mxf4_sp_2cta(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t tmem_e, uint32_t idesc, uint32_t scale, uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\ttcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%3], %4, [%6], [%7], p;\n}" ::
    "r"(tmem_c),"l"(da),"l"(db),"r"(tmem_e),"r"(idesc),"r"(scale),"r"(sfa),"r"(sfb));
}
