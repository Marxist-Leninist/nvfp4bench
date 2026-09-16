#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ck(CUresult r,const char* w){if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);std::fprintf(stderr,"%s: %s\n",w,s?s:"CUDA error");std::exit(2);}}
struct Out{float x,y;};
static Out run(const char* cubin,int iters){
 CUmodule m; ck(cuModuleLoad(&m,cubin),"cuModuleLoad"); CUfunction f; ck(cuModuleGetFunction(&f,m,"sass_patch_target_v2"),"cuModuleGetFunction");
 CUdeviceptr d; ck(cuMemAlloc(&d,2*sizeof(float)),"cuMemAlloc"); ck(cuMemsetD8(d,0,2*sizeof(float)),"memset");
 void* args[]={&iters,&d}; ck(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,nullptr),"launch"); ck(cuCtxSynchronize(),"sync");
 float h[2]={}; ck(cuMemcpyDtoH(h,d,sizeof(h)),"copy"); cuMemFree(d); cuModuleUnload(m); return {h[0],h[1]};
}
static bool close_rel(float got,float exp){float tol=std::max(1e-4f, std::fabs(exp)*2e-5f); return std::fabs(got-exp)<=tol;}
int main(int ac,char**av){
 if(ac<5){std::fprintf(stderr,"usage: %s base.cubin patched.cubin added_bank0 added_bank1 [iters=8]\n",av[0]);return 2;}
 int add0=std::atoi(av[3]),add1=std::atoi(av[4]),iters=ac>5?std::atoi(av[5]):8;
 ck(cuInit(0),"cuInit"); CUdevice dev;ck(cuDeviceGet(&dev,0),"dev");CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev),"retain");ck(cuCtxSetCurrent(ctx),"setctx");
 Out b=run(av[1],iters),p=run(av[2],iters);
 double rx=(8.0+add0)/8.0, ry=(8.0+add1)/8.0; float ex=float(b.x*rx),ey=float(b.y*ry);
 bool ok=std::isfinite(b.x)&&std::isfinite(b.y)&&std::isfinite(p.x)&&std::isfinite(p.y)&&close_rel(p.x,ex)&&close_rel(p.y,ey);
 std::printf("base_x=%.9g base_y=%.9g patched_x=%.9g patched_y=%.9g expected_x=%.9g expected_y=%.9g ratio_x=%.6f ratio_y=%.6f iters=%d result=%s\n",b.x,b.y,p.x,p.y,ex,ey,rx,ry,iters,ok?"PASS":"FAIL");
 cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return ok?0:3;
}
