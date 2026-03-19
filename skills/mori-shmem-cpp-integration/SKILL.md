---
name: mori-shmem-cpp-integration
description: How to integrate mori shmem as a C++ user. Covers globalGpuStates initialization across all user pathways (Python, C++ hipcc, C++ host/device split, Triton/IR), CMake integration with mori_detect_device_config(), and the three main usage scenarios. Use when the user asks about C++ shmem integration, globalGpuStates, device code compilation, or CMake find_package(mori).
---

# Mori SHMEM C++ Integration Guide

## globalGpuStates: Core Concept

`globalGpuStates` is a `__device__` variable (type `GpuStates`) that every
GPU kernel needs to access shmem runtime state (rank, worldSize, RDMA
endpoints, heap info). Because HIP `__device__` variables are
**module-scoped** (each `.hsaco` has its own copy), mori uses a
"host initializes once, copies to each module" pattern.

### GpuStates struct (`include/mori/shmem/internal.hpp`)

```cpp
struct GpuStates {
  int rank{-1};
  int worldSize{-1};
  int numQpPerPe{4};
  application::TransportType* transportTypes{nullptr};
  application::RdmaEndpoint* rdmaEndpoints{nullptr};
  uint32_t* endpointLock{nullptr};
  bool useVMMHeap{false};
  uint8_t vmmChunkSizeShift{0};
  uintptr_t heapBaseAddr{0};
  uintptr_t heapEndAddr{0};
  application::SymmMemObj* heapObj{nullptr};
  uint64_t* internalSyncPtr{nullptr};
};
```

### Initialization flow (shared by all pathways)

```
ShmemInit()
  └─ GpuStateInit()          ← builds GpuStates on host (src/shmem/init.cpp)
       └─ CopyGpuStatesToDevice()  ← copies to device (src/shmem/runtime.cpp)
            ├─ s_deviceGpuStatesAddr   (JIT/LoadShmemModule path)
            └─ s_gpuStatesAddrProvider (hipcc static link path)
```

Host-side copy is stored in `s_hostGpuStatesCopy`. Additional modules loaded
later use `ShmemModuleInit(module)` to copy from this host cache.

## Four User Pathways

### 1. Python user (`mori.shmem`)

```
shmem_mpi_init() / shmem_init_attr()
  └─ _ensure_shmem_module()
  │    └─ JIT compile shmem_kernels.hip → .hsaco
  │    └─ mori_cpp.load_shmem_module(hsaco)  ← stores device symbol addr
  └─ ShmemInit() → GpuStateInit() → CopyGpuStatesToDevice()
```

Extra modules (Triton kernels, EP kernels) get initialized via
`shmem_module_init(module)` automatically (Triton: via `install_hook()`,
EP: during `_load_hip_modules()`).

### 2. C++ hipcc — full hipcc compilation (Scenario 3)

```cpp
#include "mori/shmem/shmem.hpp"
// shmem.hpp auto-defines:
//   __device__ weak globalGpuStates
//   _Registrar → RegisterGpuStatesAddrProvider() before main()
//   _barrier_kernel → RegisterBarrierLauncher()

ShmemMpiInit(MPI_COMM_WORLD);
// → ShmemInit() → CopyGpuStatesToDevice()
//   → calls s_gpuStatesAddrProvider() to get static symbol addr
//   → hipMemcpy to device
```

**User does nothing manually.** The `_Registrar` static object runs before
`main()`, registering the provider. Multi-file builds work because the
`weak` attribute ensures only one copy survives linking.

### 3. C++ host/device split (Scenario 2)

Device code compiled separately to `.hsaco`, host code with clang/gcc.

```
Host: LoadShmemModule("shmem_kernels.hsaco")
        └─ resolves globalGpuStates symbol, stores in s_deviceGpuStatesAddr
      ShmemInit() → CopyGpuStatesToDevice()
        └─ copies to s_deviceGpuStatesAddr

      hipModuleLoad(&my_module, "my_kernel.hsaco")
      ShmemModuleInit(my_module)   ← copies s_hostGpuStatesCopy into module
      hipModuleLaunchKernel(...)
```

Each independently loaded `.hsaco` needs `ShmemModuleInit()`.

### 4. Triton / IR user

```
mori.ir.triton.install_hook()
  └─ jit_post_compile_hook
       └─ shmem_module_init(kernel.module)  ← auto after each Triton compile
```

### Summary table

| Pathway | globalGpuStates defined by | Init mechanism | Manual ShmemModuleInit? |
|---------|---------------------------|----------------|------------------------|
| Python shmem | `shmem_kernels.hip` (JIT) | `LoadShmemModule` + `ShmemInit` | No (automatic) |
| C++ full hipcc | `shmem.hpp` weak symbol | `RegisterGpuStatesAddrProvider` + `ShmemInit` | No (automatic) |
| C++ host/device split | `.hip` explicit definition | `LoadShmemModule` + `ShmemInit` | Yes, per extra module |
| Triton / IR | bitcode symbol | `install_hook()` → `shmem_module_init` | No (automatic via hook) |

## C++ Integration Scenarios

### Scenario 1: EP kernel user (no device code)

Simplest — no hipcc needed, no hardware detection needed.

```cpp
#include "mori/ops/dispatch_combine/launch.hpp"

// KernelRegistry::AutoLoad() detects arch+NIC at runtime, finds .hsaco
mori::moe::LaunchDispatch(handle, input, weights, scales, indices,
                          num_tokens, dtype);
```

```cmake
find_package(mori REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app mori::ops)
```

### Scenario 2: shmem device API — host/device split

Host code with clang/gcc, device code with hipcc.

**Device** (`my_kernel.hip`):

```cpp
#define MORI_SHMEM_NO_STATIC_INIT
#include "mori/shmem/shmem.hpp"

namespace mori { namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}}

using namespace mori::shmem;

__global__ void my_put_kernel(void* buf, size_t size, int dest_pe) {
    ShmemPutMemNbiThread(buf, buf, size, dest_pe);
    ShmemQuietThread(dest_pe);
    ShmemBarrierAllBlock();
}
```

**Host** (`main.cpp`):

```cpp
#include <mpi.h>
#include "mori/shmem/shmem_api.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);

    void* buf = mori::shmem::ShmemMalloc(1024 * 1024);

    hipModule_t my_module;
    hipModuleLoad(&my_module, "my_kernel.hsaco");
    mori::shmem::ShmemModuleInit(my_module);

    hipFunction_t func;
    hipModuleGetFunction(&func, my_module, "my_put_kernel");
    hipModuleLaunchKernel(func, ...);

    mori::shmem::ShmemFree(buf);
    mori::shmem::ShmemFinalize();
    MPI_Finalize();
}
```

### Scenario 3: shmem device API — full hipcc

All files compiled with hipcc, linked into one executable.

```cpp
// kernels.hip
#include "mori/shmem/shmem.hpp"
using namespace mori::shmem;

__global__ void my_put_kernel(void* buf, size_t size, int dest_pe) {
    ShmemPutMemNbiThread(buf, buf, size, dest_pe);
    ShmemQuietThread(dest_pe);
    ShmemBarrierAllBlock();
}
```

```cpp
// main.hip
#include <mpi.h>
#include "mori/shmem/shmem.hpp"
__global__ void my_put_kernel(void* buf, size_t size, int dest_pe);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);

    void* buf = mori::shmem::ShmemMalloc(1024 * 1024);
    int dest = (mori::shmem::ShmemMyPe() + 1) % mori::shmem::ShmemNPes();
    my_put_kernel<<<1, 64>>>(buf, 1024, dest);

    mori::shmem::ShmemBarrierAll();
    mori::shmem::ShmemFree(buf);
    mori::shmem::ShmemFinalize();
    MPI_Finalize();
}
```

No `ShmemModuleInit` needed — `shmem.hpp`'s `_Registrar` handles everything.

### Scenario comparison

| | Scenario 1: EP kernel | Scenario 2: host/device split | Scenario 3: full hipcc |
|---|---|---|---|
| Compiler | clang/gcc | host: clang/gcc, device: hipcc | hipcc |
| `mori_detect_device_config()`? | Not needed | Needed | Needed |
| `globalGpuStates` | AutoLoad handles it | Explicit in .hip + `ShmemModuleInit` | `shmem.hpp` weak symbol (auto) |
| Kernel launch | `LaunchDispatch/Combine` | `hipModuleLaunchKernel` | `<<<>>>` syntax |

## CMake Integration

### mori_detect_device_config()

Exported via `cmake/MoriDetectDevice.cmake`. Detects GPU arch and NIC type
on the current machine, sets three variables:

| Variable | Example | Purpose |
|----------|---------|---------|
| `MORI_GPU_ARCH` | `gfx942` | `--offload-arch` for hipcc |
| `MORI_DEVICE_NIC` | `bnxt` | NIC type string |
| `MORI_DEVICE_NIC_DEFINE` | `MORI_DEVICE_NIC_BNXT` | `-D` macro for device compilation |

Detection priority (same as Python `detect_nic_type()` and CMake
`detect_device_nic()`):

1. `MORI_DEVICE_NIC` / `MORI_GPU_ARCHS` env override
2. `/sys/class/infiniband/` sysfs enumeration
3. `lspci` PCI vendor ID matching
4. Userspace library fallback (`libmlx5.so`, `libbnxt_re.so`, `libionic.so`)

For GPU arch: `GPU_TARGETS` variable → `rocm_agent_enumerator` → `rocminfo`
→ `AMDGPU_TARGETS` env.

### mori_add_device_target(target)

Convenience function — applies `MORI_DEVICE_NIC_DEFINE` and arch to a HIP target:

```cmake
mori_add_device_target(my_app)
# Equivalent to:
#   target_compile_definitions(my_app PRIVATE ${MORI_DEVICE_NIC_DEFINE})
#   target_compile_definitions(my_app PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS)
#   set_target_properties(my_app PROPERTIES HIP_ARCHITECTURES "${MORI_GPU_ARCH}")
```

### Full CMakeLists.txt example (Scenario 2 or 3)

```cmake
cmake_minimum_required(VERSION 3.19)
project(my_shmem_app LANGUAGES HIP CXX)

find_package(hip REQUIRED)
find_package(mori REQUIRED)
mori_detect_device_config()

add_executable(my_app main.hip kernels.hip)
set_source_files_properties(main.hip kernels.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(my_app mori::shmem hip::host hip::device)
mori_add_device_target(my_app)
```

### find_package(mori) provides

| Target | Library | Purpose |
|--------|---------|---------|
| `mori::application` | `libmori_application.so` | Bootstrap, RDMA, topology, memory |
| `mori::shmem` | `libmori_shmem.so` | Symmetric memory runtime |
| `mori::ops` | `libmori_ops.so` | EP dispatch/combine + C++ launch |
| `mori::io` | `libmori_io.so` | P2P IO engine |

Variables: `MORI_INCLUDE_DIR`, `MORI_LIB_DIR`.

## Key Files Reference

| File | Role |
|------|------|
| `include/mori/shmem/shmem.hpp` | All-in-one header (host + device), auto `_Registrar` under hipcc |
| `include/mori/shmem/shmem_api.hpp` | Host-only API (init, malloc, barrier, finalize) |
| `include/mori/shmem/shmem_device_api.hpp` | Device API (PUT/GET/atomic/wait/barrier) |
| `include/mori/shmem/internal.hpp` | `GpuStates` struct, `globalGpuStates` declaration |
| `src/shmem/runtime.cpp` | `CopyGpuStatesToDevice`, `ShmemModuleInit`, `LoadShmemModule` |
| `src/shmem/init.cpp` | `GpuStateInit`, `ShmemInit` |
| `cmake/MoriDetectDevice.cmake` | `mori_detect_device_config()`, `mori_add_device_target()` |
| `cmake/mori-config.cmake` | `find_package(mori)` config, imported targets |

## Distribution Model

Mori ships as a **universal wheel** (pip package):

```
wheel contents:
  ✅ Pre-compiled host libraries (.so)
  ✅ C++ headers (include/mori/)
  ✅ Device bitcode (libmori_shmem_device.bc for Triton/IR)
  ✅ JIT infrastructure (auto-detect arch+NIC, compile, cache)
  ✅ CMake config files (MoriDetectDevice.cmake, mori-config.cmake)
  ❌ No pre-compiled .hsaco (user JIT or AOT compiles on their machine)
```

Device code is not pre-compiled because it depends on user hardware
(GPU arch × NIC type). Users either:
- **JIT**: automatic at runtime (Python path, or C++ via JIT cache)
- **AOT**: `BUILD_OPS_DEVICE=ON` or manual `hipcc --genco`
