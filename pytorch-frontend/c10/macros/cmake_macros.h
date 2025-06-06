#ifndef C10_MACROS_CMAKE_MACROS_H_
#define C10_MACROS_CMAKE_MACROS_H_

// Automatically generated header file for the C10 library.
// Do not include this file directly. Instead, include c10/macros/Macros.h.

/* #undef C10_BUILD_SHARED_LIBS */
/* #undef C10_USE_GLOG */
/* #undef C10_USE_GFLAGS */
#define C10_USE_NUMA

// Used by libtorch mobile build to enable features that are not enabled by
// caffe2 mobile build. Should only use it when necessary as we are committed
// to converging libtorch and caffe2 mobile builds and removing it eventually.
/* #undef FEATURE_TORCH_MOBILE */

#endif // C10_MACROS_CMAKE_MACROS_H_
