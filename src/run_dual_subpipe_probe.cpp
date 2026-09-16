#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void ck(CUresult r,const char* w){ if(r!=CUDA_SUCCESS){ const char*s=nullptr; cuGetErrorString(r,&s); std::fprintf(stderr,"%s: %s\n",w,s?s:"CUDA error"); std::exit(2);} }

struct Stats { double median_ms=0, min_ms=0; };

static Stats time_kernel(CUfunction f,int iters,int blocks,int block,CUdeviceptr fs,CUdeviceptr is,int reps){
  void* args[]={&iters,&fs,&is};
  ck(cuLaunchKernel(f,blocks,1,1,block,1,1,0,0,args,nullptr),"warmup launch"); ck(cuCtxSynchronize(),"warmup sync");
  CUevent a,b; ck(cuEventCreate(&a,CU_EVENT_DEFAULT),"event a"); ck(cuEventCreate(&b,CU_EVENT_DEFAULT),"event b");
  std::vector<float> v; v.reserve(reps);
  for(int r=0;r<reps;++r){
    ck(cuEventRecord(a,0),"record a");
    ck(cuLaunchKernel(f,blocks,1,1,block,1,1,0,0,args,nullptr),"timed launch");
    ck(cuEventRecord(b,0),"record b"); ck(cuEventSynchronize(b),"sync b");
    float ms=0; ck(cuEventElapsedTime(&ms,a,b),"elapsed"); v.push_back(ms);
  }
  std::sort(v.begin(),v.end()); Stats s{v[v.size()/2],v.front()}; cuEventDestroy(a);cuEventDestroy(b); return s;
}

static void launch_copy(CUfunction f,int iters,int blocks,int block,CUdeviceptr fs,CUdeviceptr is,std::vector<float>& hf,std::vector<int>& hi){
  ck(cuMemsetD8(fs,0,hf.size()*sizeof(float)),"memset f"); ck(cuMemsetD8(is,0,hi.size()*sizeof(int)),"memset i");
  void* args[]={&iters,&fs,&is}; ck(cuLaunchKernel(f,blocks,1,1,block,1,1,0,0,args,nullptr),"check launch"); ck(cuCtxSynchronize(),"check sync");
  ck(cuMemcpyDtoH(hf.data(),fs,hf.size()*sizeof(float)),"copy f"); ck(cuMemcpyDtoH(hi.data(),is,hi.size()*sizeof(int)),"copy i");
}

int main(int ac,char**av){
  if(ac<2){std::fprintf(stderr,"usage: %s dual_subpipe_warp.cubin [iters=700] [blocks_per_sm=8] [reps=17]\n",av[0]);return 2;}
  const char* cub=av[1]; int iters=ac>2?std::atoi(av[2]):700; int bpsm=ac>3?std::atoi(av[3]):8; int reps=ac>4?std::atoi(av[4]):17; const int block=128;
  if(iters<1||bpsm<1||reps<3)return 2;
  ck(cuInit(0),"init"); CUdevice dev;ck(cuDeviceGet(&dev,0),"dev");CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev),"retain");ck(cuCtxSetCurrent(ctx),"ctx");
  int sms=0,clock_khz=0;ck(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev),"sms");ck(cuDeviceGetAttribute(&clock_khz,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev),"clock");
  CUmodule m;ck(cuModuleLoad(&m,cub),"load"); CUfunction fo,fi,fb;ck(cuModuleGetFunction(&fo,m,"dual_omma_even"),"fo");ck(cuModuleGetFunction(&fi,m,"dual_imma_odd"),"fi");ck(cuModuleGetFunction(&fb,m,"dual_both"),"fb");
  int occ_o=0,occ_i=0,occ_b=0; ck(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ_o,fo,block,0),"occ o");ck(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ_i,fi,block,0),"occ i");ck(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ_b,fb,block,0),"occ b");
  int blocks=sms*bpsm; size_t n=(size_t)blocks*2; CUdeviceptr fs,is;ck(cuMemAlloc(&fs,n*sizeof(float)),"fs");ck(cuMemAlloc(&is,n*sizeof(int)),"is");
  std::vector<float> of(n),df(n);std::vector<int> ii(n),di(n),zint(n);std::vector<float> zfloat(n);
  launch_copy(fo,iters,blocks,block,fs,is,of,zint); launch_copy(fi,iters,blocks,block,fs,is,zfloat,ii); launch_copy(fb,iters,blocks,block,fs,is,df,di);
  int badf=0,badi=0; for(size_t k=0;k<n;++k){double tol=std::max(1e-5,std::fabs((double)of[k])*5e-5); if(!std::isfinite(df[k])||std::fabs((double)df[k]-of[k])>tol)++badf; if(di[k]!=ii[k])++badi;}
  if(badf||badi){std::printf("{\"result\":\"FAIL\",\"bad_float\":%d,\"bad_int\":%d}\n",badf,badi);return 3;}
  Stats so=time_kernel(fo,iters,blocks,block,fs,is,reps), si=time_kernel(fi,iters,blocks,block,fs,is,reps), sb=time_kernel(fb,iters,blocks,block,fs,is,reps);
  const double omma_inst=(double)blocks*2*iters*16; const double imma_inst=omma_inst; const double O_FLOP=32768.0,I_OP=16384.0;
  auto rate=[](double work,double ms){return work/(ms*1e-3)/1e12;};
  double serial=so.median_ms+si.median_ms; double slow=std::max(so.median_ms,si.median_ms); double speed=serial/sb.median_ms; double ideal_ratio=sb.median_ms/slow;
  std::printf("{\"schema\":\"agillm.sm121.dual-subpipe-runtime.v1\",\"result\":\"PASS\",\"sms\":%d,\"clock_attr_mhz\":%.3f,\"blocks\":%d,\"blocks_per_sm\":%d,\"block\":128,\"iters\":%d,\"reps\":%d,\"occupancy_blocks_per_sm\":{\"omma\":%d,\"imma\":%d,\"dual\":%d},\"omma_median_ms\":%.9f,\"imma_median_ms\":%.9f,\"dual_median_ms\":%.9f,\"omma_best_ms\":%.9f,\"imma_best_ms\":%.9f,\"dual_best_ms\":%.9f,\"omma_only_tflops_dense_equiv\":%.6f,\"imma_only_tops_dense_equiv\":%.6f,\"dual_combined_tops_dense_equiv\":%.6f,\"concurrency_speedup_vs_serial\":%.6f,\"dual_time_over_slowest_standalone\":%.6f,\"omma_instructions_per_launch\":%.0f,\"imma_instructions_per_launch\":%.0f}\n",sms,clock_khz/1000.0,blocks,bpsm,iters,reps,occ_o,occ_i,occ_b,so.median_ms,si.median_ms,sb.median_ms,so.min_ms,si.min_ms,sb.min_ms,rate(omma_inst*O_FLOP,so.median_ms),rate(imma_inst*I_OP,si.median_ms),rate(omma_inst*O_FLOP+imma_inst*I_OP,sb.median_ms),speed,ideal_ratio,omma_inst,imma_inst);
  cuMemFree(fs);cuMemFree(is);cuModuleUnload(m);cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(dev);return 0;
}
