#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Out{float x,y;};
static void ck(CUresult r,const char* w){if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);std::fprintf(stderr,"%s: %s\n",w,s?s:"CUDA error");std::exit(2);}}
struct Mod{CUmodule m{};CUfunction f{};};
static Mod load(const char* p){Mod z;ck(cuModuleLoad(&z.m,p),"cuModuleLoad");ck(cuModuleGetFunction(&z.f,z.m,"sass_patch_target_v2"),"cuModuleGetFunction");return z;}
static Out witness(CUfunction f,int iters){CUdeviceptr d;ck(cuMemAlloc(&d,2*sizeof(float)),"alloc witness");ck(cuMemsetD8(d,0,2*sizeof(float)),"memset witness");void* a[]={&iters,&d};ck(cuLaunchKernel(f,1,1,1,32,1,1,0,0,a,nullptr),"witness launch");ck(cuCtxSynchronize(),"witness sync");float h[2]={};ck(cuMemcpyDtoH(h,d,sizeof(h)),"witness copy");cuMemFree(d);return{h[0],h[1]};}
static bool close_rel(float got,float exp){float tol=std::max(1e-4f,std::fabs(exp)*2e-5f);return std::fabs(got-exp)<=tol;}
static double timed(CUfunction f,int blocks,int block,int iters){CUdeviceptr d;ck(cuMemAlloc(&d,2ull*blocks*sizeof(float)),"alloc timed");void* a[]={&iters,&d};auto launch=[&](){ck(cuLaunchKernel(f,blocks,1,1,block,1,1,0,0,a,nullptr),"timed launch");};launch();ck(cuCtxSynchronize(),"warmup");CUevent s,e;ck(cuEventCreate(&s,0),"ev s");ck(cuEventCreate(&e,0),"ev e");std::vector<float> v;for(int r=0;r<17;++r){ck(cuEventRecord(s,0),"rec s");launch();ck(cuEventRecord(e,0),"rec e");ck(cuEventSynchronize(e),"sync e");float ms=0;ck(cuEventElapsedTime(&ms,s,e),"elapsed");v.push_back(ms);}std::sort(v.begin(),v.end());double med=v[v.size()/2];cuEventDestroy(s);cuEventDestroy(e);cuMemFree(d);return med;}
int main(int ac,char**av){
 if(ac<7){std::fprintf(stderr,"usage: %s base.cubin candidate.cubin candidate_omma_per_iter add_bank0 add_bank1 block [blocks_per_sm=8] [iters=700]\n",av[0]);return 2;}
 const char* bp=av[1];const char* cp=av[2];int cop=std::atoi(av[3]),a0=std::atoi(av[4]),a1=std::atoi(av[5]),block=std::atoi(av[6]);int bpsm=ac>7?std::atoi(av[7]):8,iters=ac>8?std::atoi(av[8]):700;if(cop<16||block%32){std::fprintf(stderr,"bad args\n");return 2;}
 ck(cuInit(0),"cuInit");CUdevice dev;ck(cuDeviceGet(&dev,0),"device");CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev),"retain");ck(cuCtxSetCurrent(ctx),"setctx");int sms=0,clk=0;ck(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev),"sms");ck(cuDeviceGetAttribute(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev),"clk");Mod b=load(bp),c=load(cp);
 // Mandatory correctness gate before performance.
 constexpr int wi=8;Out wb=witness(b.f,wi),wc=witness(c.f,wi);double rx=(8.0+a0)/8.0,ry=(8.0+a1)/8.0;float ex=float(wb.x*rx),ey=float(wb.y*ry);bool ok=std::isfinite(wb.x)&&std::isfinite(wb.y)&&std::isfinite(wc.x)&&std::isfinite(wc.y)&&close_rel(wc.x,ex)&&close_rel(wc.y,ey);std::printf("WITNESS base=(%.9g,%.9g) cand=(%.9g,%.9g) expect=(%.9g,%.9g) ratio=(%.6f,%.6f) %s\n",wb.x,wb.y,wc.x,wc.y,ex,ey,rx,ry,ok?"PASS":"FAIL");if(!ok){cuModuleUnload(b.m);cuModuleUnload(c.m);cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return 3;}
 int blocks=sms*bpsm;double bm=timed(b.f,blocks,block,iters),cm=timed(c.f,blocks,block,iters);long long warps=(long long)blocks*(block/32);constexpr double F=32768.0;double bo=(double)warps*iters*16,co=(double)warps*iters*cop;double bops=bo/(bm*1e-3),cops=co/(cm*1e-3);double btf=bops*F/1e12,ctf=cops*F/1e12;double bfix=bops/(sms*2.5e9),cfix=cops/(sms*2.5e9),battr=bops/(sms*clk*1e3),cattr=cops/(sms*clk*1e3);double target=2e15/(sms*2.5e9*F);
 std::printf("BENCH SM=%d clock_attr_MHz=%.1f block=%d bpsm=%d blocks=%d iters=%d base_omma=16 cand_omma=%d base_ms=%.6f cand_ms=%.6f base_TF=%.3f cand_TF=%.3f speedup_TF=%.4f base_OMMA_SM_cyc_2p5=%.6f cand_OMMA_SM_cyc_2p5=%.6f base_OMMA_SM_cyc_attr=%.6f cand_OMMA_SM_cyc_attr=%.6f target2PF=%.6f\n",sms,clk/1000.0,block,bpsm,blocks,iters,cop,bm,cm,btf,ctf,ctf/btf,bfix,cfix,battr,cattr,target);
 cuModuleUnload(b.m);cuModuleUnload(c.m);cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return 0;
}
