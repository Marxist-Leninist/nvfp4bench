#include <cuda.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "%s: %s\n", what, s ? s : "CUDA error");
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
      "usage: %s candidate.cubin omma_per_warp_iter [iters=700] [block=128] [blocks_per_sm=8] [reps=17]\n",
      argv[0]);
    return 2;
  }
  const char* path = argv[1];
  const int opi = std::atoi(argv[2]);
  const int iters = argc > 3 ? std::atoi(argv[3]) : 700;
  const int block = argc > 4 ? std::atoi(argv[4]) : 128;
  const int bpsm = argc > 5 ? std::atoi(argv[5]) : 8;
  const int reps = argc > 6 ? std::atoi(argv[6]) : 17;
  if (opi < 1 || iters < 1 || block < 32 || block % 32 || bpsm < 1 || reps < 3) {
    std::fprintf(stderr, "invalid geometry/iteration arguments\n");
    return 2;
  }

  ck(cuInit(0), "cuInit");
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  ck(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  ck(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  int sms = 0, clock_khz = 0, cc_major = 0, cc_minor = 0;
  ck(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SM count");
  ck(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clock rate");
  ck(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc major");
  ck(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc minor");

  CUmodule mod;
  ck(cuModuleLoad(&mod, path), "cuModuleLoad");
  CUfunction fun;
  ck(cuModuleGetFunction(&fun, mod, "pure_issue_witness"), "cuModuleGetFunction");

  const int blocks = sms * bpsm;
  const long long warps = (long long)blocks * (block / 32);
  const double omma_per_launch = (double)warps * (double)iters * (double)opi;
  const size_t sink_bytes = (size_t)blocks * 64u * sizeof(float);
  CUdeviceptr sink;
  ck(cuMemAlloc(&sink, sink_bytes), "cuMemAlloc");
  ck(cuMemsetD8(sink, 0, sink_bytes), "cuMemsetD8");
  void* args[] = {(void*)&iters, (void*)&sink};
  auto launch = [&]() {
    ck(cuLaunchKernel(fun, blocks, 1, 1, block, 1, 1, 0, 0, args, nullptr), "cuLaunchKernel");
  };

  launch();
  ck(cuCtxSynchronize(), "warmup sync");
  CUevent a, b;
  ck(cuEventCreate(&a, CU_EVENT_DEFAULT), "event create a");
  ck(cuEventCreate(&b, CU_EVENT_DEFAULT), "event create b");
  std::vector<float> samples;
  samples.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    ck(cuEventRecord(a, 0), "event record a");
    launch();
    ck(cuEventRecord(b, 0), "event record b");
    ck(cuEventSynchronize(b), "event sync b");
    float ms = 0.0f;
    ck(cuEventElapsedTime(&ms, a, b), "event elapsed");
    samples.push_back(ms);
  }
  std::sort(samples.begin(), samples.end());
  const double min_ms = samples.front();
  const double median_ms = samples[samples.size() / 2];
  constexpr double FLOP_PER_OMMA = 32768.0;
  auto tf = [&](double ms) { return omma_per_launch * FLOP_PER_OMMA / (ms * 1e-3) / 1e12; };
  auto omma_rate = [&](double ms) { return omma_per_launch / (ms * 1e-3); };
  const double median_omma_s = omma_rate(median_ms);
  const double best_omma_s = omma_rate(min_ms);
  const double fixed_median = median_omma_s / (sms * 2.5e9);
  const double fixed_best = best_omma_s / (sms * 2.5e9);
  const double attr_hz = clock_khz * 1e3;
  const double target_rate = 2e15 / (sms * 2.5e9 * FLOP_PER_OMMA);

  std::printf(
    "{\"schema\":\"agillm.sm121.pure-issue-timing.v1\","
    "\"cubin\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"clock_attr_mhz\":%.3f,"
    "\"blocks\":%d,\"block\":%d,\"blocks_per_sm\":%d,\"warps\":%lld,"
    "\"iters\":%d,\"omma_per_warp_iter\":%d,\"omma_per_launch\":%.0f,\"reps\":%d,"
    "\"median_ms\":%.9f,\"min_ms\":%.9f,\"median_tflops\":%.6f,\"best_tflops\":%.6f,"
    "\"median_omma_per_sm_cycle_at_2p5ghz\":%.9f,\"best_omma_per_sm_cycle_at_2p5ghz\":%.9f,"
    "\"median_omma_per_sm_cycle_at_clock_attr\":%.9f,\"best_omma_per_sm_cycle_at_clock_attr\":%.9f,"
    "\"target_2pf_omma_per_sm_cycle_at_2p5ghz\":%.9f}\n",
    path, cc_major, cc_minor, sms, clock_khz / 1000.0, blocks, block, bpsm, warps,
    iters, opi, omma_per_launch, reps, median_ms, min_ms, tf(median_ms), tf(min_ms),
    fixed_median, fixed_best,
    median_omma_s / (sms * attr_hz), best_omma_s / (sms * attr_hz), target_rate);

  cuEventDestroy(a);
  cuEventDestroy(b);
  cuMemFree(sink);
  cuModuleUnload(mod);
  cuCtxSetCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
