#include <cuda.h>
#include <cstdio>
#include <cstdlib>
static void ck(CUresult r,const char* w){if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);std::fprintf(stderr,"%s: %s\n",w,s?s:"CUDA error");std::exit(2);}}
int main(int ac,char**av){
 if(ac<2){std::fprintf(stderr,"usage: %s candidate.cubin [blocks_per_sm=8] [block=128] [iters=700]\n",av[0]);return 2;}
 int bpsm=ac>2?std::atoi(av[2]):8, block=ac>3?std::atoi(av[3]):128, iters=ac>4?std::atoi(av[4]):700;
 if(bpsm<1||block<32||block%32){std::fprintf(stderr,"bad geometry\n");return 2;}
 ck(cuInit(0),"cuInit"); CUdevice dev; ck(cuDeviceGet(&dev,0),"device"); CUcontext ctx; ck(cuDevicePrimaryCtxRetain(&ctx,dev),"retain"); ck(cuCtxSetCurrent(ctx),"ctx");
 int sms=0; ck(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev),"sms"); int blocks=sms*bpsm;
 CUmodule m; ck(cuModuleLoad(&m,av[1]),"module"); CUfunction f; ck(cuModuleGetFunction(&f,m,"pure_issue_witness"),"function");
 CUdeviceptr d; ck(cuMemAlloc(&d,(size_t)blocks*64*sizeof(float)),"alloc"); ck(cuMemsetD8(d,0,(size_t)blocks*64*sizeof(float)),"memset");
 void* args[]={&iters,&d}; ck(cuLaunchKernel(f,blocks,1,1,block,1,1,0,0,args,nullptr),"launch"); ck(cuCtxSynchronize(),"sync");
 std::printf("profile_launch SM=%d blocks=%d block=%d bpsm=%d iters=%d\n",sms,blocks,block,bpsm,iters);
 cuMemFree(d);cuModuleUnload(m);cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return 0;
}
