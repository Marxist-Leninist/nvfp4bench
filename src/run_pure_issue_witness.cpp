#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ck(CUresult r,const char* w){if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);std::fprintf(stderr,"%s: %s\n",w,s?s:"CUDA error");std::exit(2);}}
static std::vector<float> run(const char* cubin,int iters){
 CUmodule m; ck(cuModuleLoad(&m,cubin),"cuModuleLoad"); CUfunction f; ck(cuModuleGetFunction(&f,m,"pure_issue_witness"),"getFunction");
 CUdeviceptr d; ck(cuMemAlloc(&d,64*sizeof(float)),"alloc"); ck(cuMemsetD8(d,0,64*sizeof(float)),"memset");
 void* args[]={&iters,&d}; ck(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,nullptr),"launch"); ck(cuCtxSynchronize(),"sync");
 std::vector<float> h(64); ck(cuMemcpyDtoH(h.data(),d,64*sizeof(float)),"copy"); cuMemFree(d);cuModuleUnload(m);return h;
}
static bool near(float got,float exp){float tol=std::max(1e-5f,std::fabs(exp)*5e-5f);return std::isfinite(got)&&std::isfinite(exp)&&std::fabs(got-exp)<=tol;}
int main(int ac,char**av){
 if(ac<20){std::fprintf(stderr,"usage: %s base.cubin candidate.cubin iters m0 m1 ... m15\n",av[0]);return 2;}
 int iters=std::atoi(av[3]); int mult[16]; for(int j=0;j<16;++j)mult[j]=std::atoi(av[4+j]);
 ck(cuInit(0),"cuInit");CUdevice dev;ck(cuDeviceGet(&dev,0),"dev");CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev),"retain");ck(cuCtxSetCurrent(ctx),"ctx");
 auto b=run(av[1],iters), c=run(av[2],iters); int bad=0;
 for(int j=0;j<16;++j)for(int q=0;q<4;++q){int k=4*j+q;float exp=b[k]*mult[j]; if(!near(c[k],exp)){if(bad<12)std::printf("BAD acc=%d comp=%d base=%.9g got=%.9g exp=%.9g mult=%d\n",j,q,b[k],c[k],exp,mult[j]);++bad;}}
 std::printf("iters=%d components=64 bad=%d result=%s\n",iters,bad,bad?"FAIL":"PASS");cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return bad?3:0;
}
