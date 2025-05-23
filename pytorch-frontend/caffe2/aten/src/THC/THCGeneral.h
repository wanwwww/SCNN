#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include <TH/THGeneral.h>
#include <TH/THAllocator.h>

#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cusparse.h>

/* #undef USE_MAGMA */

// TH & THC are now part of the same library as ATen and Caffe2
// NB: However, we are planning to split it out to a torch_cuda library
#define THC_API TORCH_CUDA_API
#define THC_CLASS TORCH_CUDA_API

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      _THError(__FILE__, __LINE__, "assert(%s) failed", #exp);          \
    }                                                                   \
  } while(0)
#endif

typedef struct THCState THCState;
struct THCState;

typedef struct _THCCudaResourcesPerDevice {
  /* Size of scratch space per each stream on this device available */
  size_t scratchSpacePerStream;
} THCCudaResourcesPerDevice;

THC_API THCState* THCState_alloc(void);
THC_API void THCState_free(THCState* state);

THC_API void THCudaInit(THCState* state);
THC_API void THCudaShutdown(THCState* state);

/* If device `dev` can access allocations on device `devToAccess`, this will return */
/* 1; otherwise, 0. */
THC_API int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess);

THC_API c10::Allocator* THCState_getCudaHostAllocator(THCState* state);

THC_API void THCMagma_init(THCState *state);

/* For the current device and stream, returns the allocated scratch space */
THC_API size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state);

#define THCAssertSameGPU(expr) if (!expr) THError("arguments are located on different GPUs")
#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCudaCheckWarn(err)  __THCudaCheckWarn(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)
#define THCusparseCheck(err)  __THCusparseCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCudaCheckWarn(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);
THC_API void __THCusparseCheck(cusparseStatus_t status, const char *file, const int line);

THC_API void* THCudaMalloc(THCState *state, size_t size);
THC_API void THCudaFree(THCState *state, void* ptr);

at::DataPtr THCudaHostAlloc(THCState *state, size_t size);

THC_API void THCudaHostRecord(THCState *state, void *ptr);

#endif
