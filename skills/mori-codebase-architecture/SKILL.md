---
name: mori-codebase-architecture
description: Understand the MORI codebase architecture, module structure, build system, and key concepts. Use when exploring the codebase, adding features, debugging, reviewing PRs, or answering questions about how mori is structured.
---

# MORI Codebase Architecture

MORI (Modular RDMA Interface) is a C++ framework for RDMA + GPU communication,
with Python bindings. Host code compiles with standard C++; GPU kernels are
JIT-compiled at runtime or AOT-compiled at build time.

## Directory Layout

```
mori/
├── include/mori/           # Public C++ headers
├── src/                    # C++ source (5 libraries)
│   ├── application/        # mori_application: bootstrap, RDMA, topology, memory
│   ├── shmem/              # mori_shmem: symmetric memory runtime
│   ├── ops/                # mori_ops: EP dispatch/combine + C++ launch API
│   │   ├── dispatch_combine/  # Host logic (pure CXX)
│   │   └── kernels/           # .hip GPU kernel sources (JIT/AOT compiled)
│   ├── io/                 # mori_io: P2P communication engine
│   └── pybind/             # mori_pybinds: Python bindings
├── python/mori/            # Python package (lazy imports via __getattr__)
│   ├── ops/                # EP dispatch/combine Python API
│   ├── shmem/              # Shmem Python API
│   ├── io/                 # IO engine Python API
│   ├── ir/                 # Device bitcode for Triton integration
│   ├── jit/                # JIT kernel compilation + caching
│   └── kernel_profiler/    # Warp-level profiler (Perfetto)
├── cmake/
│   ├── MoriDetectDevice.cmake  # Reusable GPU arch + NIC detection module
│   └── mori-config.cmake       # find_package(mori) support
├── tests/
│   ├── python/             # pytest (ops, shmem, io)
│   └── cpp/                # C++ tests (transport, io, AOT launch)
├── examples/               # C++ examples (RDMA, shmem, SDMA)
├── docs/                   # Sphinx documentation website
├── docker/                 # Dockerfile.dev
├── CMakeLists.txt          # Main CMake
└── setup.py                # Python install (calls CMake internally)
```

## C++ Libraries

| Library | CMake Target | Purpose | Dependencies |
|---------|-------------|---------|--------------|
| `mori_application` | `src/application/` | Bootstrap (socket/torch, MPI optional), RDMA transport (mlx5/bnxt/ionic), topology, symmetric memory | ibverbs, hip, MPI (optional) |
| `mori_shmem` | `src/shmem/` | OpenSHMEM-style APIs: init, malloc, barrier, module init | mori_application |
| `mori_ops` | `src/ops/` | EP dispatch/combine handle + C++ launch wrappers | mori_shmem, mori_application |
| `mori_io` | `src/io/` | P2P IO engine (RDMA/XGMI backends) | mori_application |
| `mori_pybinds` | `src/pybind/` | Python bindings (pybind11) | all above |

All libraries are pure CXX (no hipcc). GPU kernel code lives in `src/ops/kernels/*.hip`
and is compiled separately (JIT or AOT).

## Kernel Compilation: Two Paths

### Python JIT (default for `pip install .`)

```
pip install .  →  setup.py  →  CMake builds host .so only (no hipcc)
First import   →  detect GPU arch + NIC  →  hipcc --genco  →  .hsaco
Cached to ~/.mori/jit/<arch>_<nic>/<content_hash>/
Latest symlinks: ~/.mori/jit/<arch>_<nic>/latest/*.hsaco
```

### C++ AOT (`BUILD_OPS_DEVICE=ON`)

```
cmake .. -DBUILD_OPS_DEVICE=ON  →  hipcc --genco at build time
Output: build/lib/<arch>_<nic>/*.hsaco
C++ code: LaunchDispatch(handle, ...)  →  hipModuleLaunchKernel
```

## CMake Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_EXAMPLES` | ON | Build C++ examples (implies `WITH_MPI=ON`) |
| `WITH_MPI` | `${BUILD_EXAMPLES}` | Enable MPI bootstrap + `ShmemMpiInit`. Also controllable via `MORI_WITH_MPI` env var |
| `BUILD_OPS` | ON | Build mori_ops (host + launch.cpp) |
| `BUILD_OPS_DEVICE` | OFF | AOT compile .hip → .hsaco (needs hipcc) |
| `BUILD_SHMEM` | ON | Build mori_shmem |
| `BUILD_IO` | ON | Build mori_io |
| `BUILD_PYBINDS` | ON | Build Python bindings |
| `BUILD_TESTS` | ON | Build C++ tests |
| `ENABLE_PROFILER` | OFF | Kernel profiling |
| `ENABLE_STANDARD_MOE_ADAPT` | OFF | DeepEP-compatible APIs |

`pip install .` passes `BUILD_EXAMPLES=OFF` by default → no MPI required.
`BUILD_EXAMPLES=ON pip install .` or `MORI_WITH_MPI=ON pip install .` enables MPI.

`find_package(mori)` is supported via `cmake/mori-config.cmake` (installed with `cmake --install`).
Available imported targets: `mori::shmem`, `mori::application`, `mori::ops`, `mori::io`.

## NIC Detection (3 implementations, same logic)

Detected at CMake configure, Python JIT runtime, and C++ AutoLoad runtime.

Priority: `MORI_DEVICE_NIC` env → `/sys/class/infiniband/` device names + drivers →
lspci PCI vendor IDs → userspace library fallback → default mlx5.

| NIC | Define | Library |
|-----|--------|---------|
| Mellanox CX7 | (default, no define) | libmlx5.so |
| Broadcom Thor2 | `MORI_DEVICE_NIC_BNXT` | libbnxt_re.so |
| AMD Pollara | `MORI_DEVICE_NIC_IONIC` | libionic.so |

## Key Concepts

### globalGpuStates

Device symbol containing rank, worldSize, RDMA endpoints, heap addresses.
Initialized by `ShmemInit` → `CopyGpuStatesToDevice`. For dynamically loaded
modules (.hsaco), `ShmemModuleInit(hipModule)` copies host-side state into the module.

### KernelRegistry (C++ launch API)

Singleton managing .hsaco loading. `AutoLoad()` detects GPU arch + NIC and searches:
1. `MORI_KERNEL_DIR` env
2. `<libmori_ops.so>/../lib/<arch>_<nic>/` (build)
3. `<libmori_ops.so>/<arch>_<nic>/` (install)
4. `~/.mori/jit/<arch>_<nic>/latest/` (JIT cache)

### EP Kernel Types

| Type | Value | Use Case |
|------|-------|----------|
| IntraNode | 0 | Single-node EP via XGMI |
| InterNode | 1 | Multi-node baseline |
| InterNodeV1 | 2 | Multi-node optimized BW |
| InterNodeV1LL | 3 | Multi-node low-latency |
| AsyncLL | 4 | Async pipelined |

## C++ User Quick Start

MPI bootstrap (requires `WITH_MPI=ON`):
```cpp
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/shmem/shmem_api.hpp"

mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);  // needs WITH_MPI=ON
mori::moe::EpDispatchCombineHandle handle(config);
mori::moe::LaunchDispatch(handle, input, weights, scales, indices, ...);
```

Socket bootstrap (no MPI needed):
```cpp
mori_shmem_uniqueid_t uid;
mori::shmem::ShmemGetUniqueId(&uid);         // rank 0 broadcasts uid
mori_shmem_init_attr_t attr;
mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &attr);
mori::shmem::ShmemInitAttr(MORI_SHMEM_INIT_WITH_UNIQUEID, &attr);
```

With `find_package(mori)`:
```cmake
find_package(mori REQUIRED)
mori_detect_device_config()
target_link_libraries(my_app mori::shmem hip::device)
mori_add_device_target(my_app)
```

## Python User Quick Start

```bash
pip install .
```

```python
import mori
mori.shmem.shmem_torch_process_group_init("default")
op = mori.ops.EpDispatchCombineOp(config)
out = op.dispatch(input, weights, scales, indices)
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MORI_SHMEM_HEAP_SIZE` | Symmetric heap size (e.g. "6G") |
| `MORI_DEVICE_NIC` | Override NIC detection (bnxt/ionic/mlx5) |
| `MORI_KERNEL_DIR` | Override kernel search path for C++ AutoLoad |
| `MORI_GLOBAL_LOG_LEVEL` | Log verbosity (TRACE/DEBUG/INFO/WARN/ERROR) |
| `MORI_PRECOMPILE` | Precompile all JIT kernels on import |
| `MORI_DISABLE_JIT` | Disable JIT compilation |
| `MORI_DISABLE_P2P` | Force RDMA transport (disable XGMI P2P) |
| `MORI_EP_LAUNCH_CONFIG_MODE` | MANUAL or AUTO launch params |
| `MORI_RDMA_DEVICES` | RDMA NIC selection (include/exclude) |
| `MORI_SOCKET_IFNAME` | Network interface for bootstrap TCP |
| `MORI_WITH_MPI` | Enable MPI support at build time (same as `-DWITH_MPI=ON`) |
