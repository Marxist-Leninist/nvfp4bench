#include <cuda.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static void ck(CUresult r,const char* what){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);std::fprintf(stderr,"%s: %s\n",what,s?s:"CUDA error");std::exit(2);}}
int main(int argc,char**argv){
 if(argc<3){std::fprintf(stderr,"usage: %s cubin omma_per_warp_iter [iters=1000] [block=128] [blocks_per_sm=8]\n",argv[0]);return 2;}
 const char* path=argv[1]; int opi=std::atoi(argv[2]); int iters=argc>3?std::atoi(argv[3]):1000; int block=argc>4?std::atoi(argv[4]):128; int bpsm=argc>5?std::atoi(argv[5]):8;
 ck(cuInit(0),"cuInit"); CUdevice dev; ck(cuDeviceGet(&dev,0),"cuDeviceGet"); CUcontext ctx; ck(cuDevicePrimaryCtxRetain(&ctx,dev),"cuDevicePrimaryCtxRetain"); ck(cuCtxSetCurrent(ctx),"cuCtxSetCurrent");
 int sms=0,clock_khz=0; ck(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev),"SM count"); ck(cuDeviceGetAttribute(&clock_khz,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev),"clock");
 CUmodule mod; ck(cuModuleLoad(&mod,path),"cuModuleLoad"); CUfunction fun; ck(cuModuleGetFunction(&fun,mod,"sass_patch_target"),"cuModuleGetFunction");
 int blocks=sms*bpsm; CUdeviceptr sink; ck(cuMemAlloc(&sink,sizeof(float)*blocks),"cuMemAlloc"); void* args[]={&iters,&sink};
 auto launch=[&](){ck(cuLaunchKernel(fun,blocks,1,1,block,1,1,0,0,args,nullptr),"cuLaunchKernel");}; launch(); ck(cuCtxSynchronize(),"warm sync");
 CUevent a,b; ck(cuEventCreate(&a,CU_EVENT_DEFAULT),"event a");ck(cuEventCreate(&b,CU_EVENT_DEFAULT),"event b");std::vector<float> ms;
 for(int r=0;r<15;++r){ck(cuEventRecord(a,0),"record a");launch();ck(cuEventRecord(b,0),"record b");ck(cuEventSynchronize(b),"event sync");float t=0;ck(cuEventElapsedTime(&t,a,b),"elapsed");ms.push_back(t);}std::sort(ms.begin(),ms.end());double med=ms[7];
 long long warps=(long long)blocks*(block/32); double total_omma=(double)warps*iters*opi; constexpr double FPO=32768.0; double omma_s=total_omma/(med*1e-3); double tf=omma_s*FPO/1e12; double fixed=omma_s/(sms*2.5e9); double observed=omma_s/(sms*(clock_khz*1e3));
 std::printf("cubin=%s SM=%d clock_attr_MHz=%.1f blocks=%d block=%d bpsm=%d iters=%d omma/warp/iter=%d median_ms=%.6f TFLOPS=%.3f OMMA/s=%.6e OMMA/SM/cyc@2.5GHz=%.6f OMMA/SM/cyc@attr=%.6f target2PF=%.6f\n",path,sms,clock_khz/1000.0,blocks,block,bpsm,iters,opi,med,tf,omma_s,fixed,observed,2e15/(sms*2.5e9*FPO));
 cuEventDestroy(a);cuEventDestroy(b);cuMemFree(sink);cuModuleUnload(mod);cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return 0;
}
