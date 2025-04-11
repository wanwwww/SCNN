# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/zww/stonnesim/stonne/pytorch-frontend/third_party/nccl/nccl")
  file(MAKE_DIRECTORY "/home/zww/stonnesim/stonne/pytorch-frontend/third_party/nccl/nccl")
endif()
file(MAKE_DIRECTORY
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src/nccl_external-build"
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix"
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/tmp"
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src/nccl_external-stamp"
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src"
  "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src/nccl_external-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src/nccl_external-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/zww/stonnesim/stonne/pytorch-frontend/nccl_external-prefix/src/nccl_external-stamp${cfgdir}") # cfgdir has leading slash
endif()
