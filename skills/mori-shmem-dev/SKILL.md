---
name: mori-shmem-dev
description: Develop new shmem device APIs (GET, PUT, atomics, etc.) for the mori project. Covers the full workflow from kernel implementation to testing across all three NIC vendors (MLX5, BNXT, AINIC). Use when the user asks to add new shmem operations, implement OpenSHMEM APIs, or modify RDMA/P2P device primitives.
---

# Mori SHMEM Device API Development

## Architecture Overview

Mori shmem follows a layered architecture. When adding a new operation
(e.g. GET, PUT, atomics), changes propagate through all layers:

```
shmem_device_api.hpp          ← User-facing API (macros generate typed variants)
  ↓ dispatches by transport
shmem_p2p_kernels.hpp         ← P2P: direct GPU memory copy via peerPtrs
shmem_ibgda_kernels.hpp       ← IBGDA: RDMA WQEs via PostRead/PostWrite/PostAtomic
shmem_sdma_kernels.hpp        ← SDMA: DMA engine (often delegates to P2P)
  ↓ IBGDA dispatches by provider
mlx5_device_primitives.hpp    ← Mellanox ConnectX WQE format
bnxt_device_primitives.hpp    ← Broadcom Thor2 WQE format
ionic_device_primitives.hpp   ← AMD AINIC Pollara WQE format
```

### Key files

| Layer | File | Purpose |
|-------|------|---------|
| Kernel declarations | `include/mori/shmem/shmem_device_kernels.hpp` | Forward-declare all kernel templates |
| P2P kernels | `include/mori/shmem/shmem_p2p_kernels.hpp` | P2P specializations (direct memory copy) |
| IBGDA kernels | `include/mori/shmem/shmem_ibgda_kernels.hpp` | RDMA specializations (PostRead/PostWrite) |
| SDMA kernels | `include/mori/shmem/shmem_sdma_kernels.hpp` | SDMA specializations (or P2P delegation) |
| Device API | `include/mori/shmem/shmem_device_api.hpp` | User-facing macros + dispatch |
| C wrapper header | `include/mori/shmem/shmem_device_api_wrapper.hpp` | `extern "C"` declarations for bitcode |
| C wrapper impl | `src/shmem/shmem_device_api_wrapper.cpp` | Wrapper implementations |
| Python IR ops | `python/mori/ir/ops.py` | ABI metadata for Triton `@core.extern` |
| RDMA primitives | `include/mori/core/transport/rdma/device_primitives.hpp` | PostReadWrite/PostAtomic declarations |
| Provider impls | `include/mori/core/transport/rdma/providers/{mlx5,bnxt,ionic}/` | Vendor-specific WQE construction |

## Step 1: Implement Kernel Layer

### 1a. Declare kernel templates in `shmem_device_kernels.hpp`

Add forward declarations for both API styles:

```cpp
// Pure address-based (new style)
template <application::TransportType TsptType>
inline __device__ void ShmemNewOpThreadKernel(void* dest, const void* source,
                                              size_t bytes, int pe, int qpId = 0);
// SymmMemObjPtr-based (legacy style)
template <application::TransportType TsptType>
inline __device__ void ShmemNewOpThreadKernel(const application::SymmMemObjPtr dest,
                                              size_t destOffset, ...);
```

### 1b. Implement P2P transport in `shmem_p2p_kernels.hpp`

P2P uses direct GPU memory copy via `peerPtrs`:

- **PUT** direction: `localPtr → peerPtrs[pe]` (write to remote)
- **GET** direction: `peerPtrs[pe] → localPtr` (read from remote)
- Use `core::ThreadCopy`, `core::WarpCopy`, `core::BlockCopy`
- Address-based version: compute offset via `addr - heapBaseAddr`,
  then use `heapObj->peerPtrs[pe] + offset`

### 1c. Implement IBGDA transport in `shmem_ibgda_kernels.hpp`

IBGDA posts RDMA WQEs to the NIC. Key patterns:

- **PUT**: `core::PostWrite` (RDMA WRITE) — `laddr` = local source, `raddr` = remote dest
- **GET**: `core::PostRead` (RDMA READ) — `laddr` = local dest, `raddr` = remote source
- **Atomic**: `core::PostAtomic`

The standard IBGDA kernel pattern (see `ShmemPutMemNbiThreadKernelImpl`):

1. Get endpoint: `ep[pe * numQpPerPe + qpId]`
2. Resolve keys: Static Heap (`lkey`/`rkey` direct) or VMM Heap (chunk query)
3. Warp-level WQE index allocation (`__ballot`, leader atomicAdd)
4. Wait for free SQ entries (poll CQ if full)
5. Post WQE (`PostReadWrite<PrvdType, IsRead>`)
6. Leader: update DBR, ring doorbell, advance `dbTouchIdx`

Important: wrap the kernel in a PE-serialization loop (`__ballot`/`__shfl`)
for thread-scope to avoid same-PE conflicts:

```cpp
bool need_turn{true};
uint64_t turns = __ballot(need_turn);
while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
        DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ImplFunc, ...);
        need_turn = false;
    }
    turns = __ballot(need_turn);
}
```

### 1d. Add SDMA stubs in `shmem_sdma_kernels.hpp`

If SDMA-native support is not needed yet, delegate to P2P:

```cpp
template <>
inline __device__ void ShmemNewOpThreadKernel<application::TransportType::SDMA>(...) {
  ShmemNewOpThreadKernel<application::TransportType::P2P>(...);
}
```

## Step 2: Add Device API Layer

In `shmem_device_api.hpp`, use macros to generate typed variants.
Follow the existing pattern (see `DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE`):

1. **Mem API** (byte-level): `ShmemNewOpMem{Thread,Warp,Block}`
2. **Type template API**: `ShmemNewOpType{Thread,Warp,Block}<T>`
3. **Concrete typed API**: `ShmemNewOp{Uint8,Int32,Float,...}{Thread,Warp,Block}`

Each macro body dispatches via `DISPATCH_TRANSPORT_TYPE(KernelName, pe, ...)`.

For **blocking** variants (e.g. `shmem_get` vs `shmem_get_nbi`):
wrap the Nbi call + `ShmemQuietThread`:

```cpp
inline __device__ void ShmemGetMemThread(..., int pe, int qpId = 0) {
    ShmemGetMemNbiThread(..., pe, qpId);
    ShmemQuietThread(pe, qpId);
}
```

## Step 3: Add C Wrapper (Bitcode)

### 3a. Declarations in `shmem_device_api_wrapper.hpp`

```cpp
extern "C" {
__device__ __attribute__((visibility("default"))) int mori_shmem_newop_thread(
    void* dest, const void* source, size_t bytes, int pe, int qpId);
}
```

### 3b. Implementations in `shmem_device_api_wrapper.cpp`

```cpp
__device__ __attribute__((visibility("default"))) int mori_shmem_newop_thread(...) {
  mori::shmem::ShmemNewOpThread(...);
  return 0;
}
```

## Step 4: Add Python IR Metadata

In `python/mori/ir/ops.py`, add entries to `MORI_DEVICE_FUNCTIONS`:

```python
"newop_thread": {
    "symbol": "mori_shmem_newop_thread",
    "args": ["uint64", "uint64", "uint64", "int32", "int32"],
    "ret": "int32",
},
```

Pointers are passed as `"uint64"` (intptr cast). The Triton `@core.extern`
wrappers are auto-generated from this metadata.

## Step 5: Write Tests

### 5a. C++ test (`examples/shmem/`)

Create a test file following `concurrent_put_thread.cpp` /
`concurrent_get_thread.cpp` patterns:

- Add to `examples/CMakeLists.txt`:
  `add_shmem_example(new_test SOURCES shmem/new_test.cpp)`
- Use `mpirun --allow-run-as-root -np 2` to run
- Test both SymmMemObjPtr and address-based APIs
- Test Thread and Block scopes
- Test large data (>200MB for VMM chunk boundary crossing)

### 5b. Triton test (`examples/shmem/ir/`)

Add test kernels and functions to `test_triton_shmem.py`:

```python
@triton.jit
def shmem_newop_kernel(buf_ptr, remote_ptr, nbytes):
    mype = mori_shmem_device.my_pe()
    dest_pe = (mype + 1) % mori_shmem_device.n_pes()
    mori_shmem_device.newop_thread(buf_ptr, remote_ptr, nbytes, dest_pe, 0)
```

## Step 6: Build & Test

### Build

Inside Docker container (see `mori-intranode-testing` skill for container setup):

```bash
# Install with C++ examples
BUILD_EXAMPLES=ON pip install .

# Examples are at $TEST_SRC/build/examples/
```

### Test matrix

Run on **all available NIC types** (MLX5, BNXT, AINIC). For each NIC:

```bash
# P2P transport (intra-node direct memory)
timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/new_test

# IBGDA transport (RDMA via NIC)
MORI_DISABLE_P2P=ON timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/new_test

# Triton bitcode (P2P)
PYTHONPATH=$TEST_SRC:$PYTHONPATH torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

# Triton bitcode (IBGDA)
MORI_DISABLE_P2P=ON PYTHONPATH=$TEST_SRC:$PYTHONPATH torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py
```

Optional: test VMM heap mode with `MORI_SHMEM_MODE=VMM_HEAP`.

### NIC-specific machines

SSH to machines with different NICs and run tests inside Docker containers.
Keep containers alive during development (don't auto-cleanup).
Refer to `mori-intranode-testing` skill for Docker image selection per NIC.

## Step 7: Commit & PR

### Commit convention

Use conventional commits. One commit per logical unit:

```
feat(shmem): add <OP> device API with P2P and IBGDA support
refactor(rdma): <description of refactoring>
```

### PR body template

```markdown
## Motivation
<Why this operation is needed, OpenSHMEM standard reference>

## Technical Details
- Transport implementations (P2P, IBGDA, SDMA)
- Any RDMA primitive changes
- Triton IR integration

## Test Plan
<Test matrix table: NIC × transport × test type, all PASSED>
- **Hardware**: MLX5 + MI300X, BNXT Thor2 + MI325X, AINIC Pollara + MI350X
- **Modes tested**: Static Heap, VMM Heap
```

## RDMA Operation Reference

| Operation | WQE Type | Direction | Use Case |
|-----------|----------|-----------|----------|
| PUT (write) | `PostReadWrite<P, false>` | local → remote | Send data to remote PE |
| GET (read) | `PostReadWrite<P, true>` | remote → local | Fetch data from remote PE |
| Atomic | `PostAtomic` | remote (in-place) | Remote atomic operations |
| Write Inline | `PostWriteInline` | inline → remote | Small immediate values (≤16B) |

### GET vs PUT key differences

| | PUT | GET |
|---|---|---|
| RDMA opcode | RDMA WRITE | RDMA READ |
| `laddr`/`lkey` | local source | local destination |
| `raddr`/`rkey` | remote destination | remote source |
| Data ready after post? | Yes (for sender) | No — must `quiet` first |
| `shmem_g` (single value) | `shmem_p`: write inline | Needs ibuf for IBGDA |

## Common Pitfalls

1. **IBGDA GET requires quiet before data access** — RDMA READ is async;
   data is NOT in local memory until CQ completion is polled.

2. **Large data + block-scoped quiet** — GPU doesn't run all blocks
   simultaneously. If only block 0 calls quiet, later blocks' WQEs may
   not be posted yet. Each block must quiet its own operations.

3. **VMM Heap chunking** — transfers >64MB may cross chunk boundaries.
   Each chunk has its own lkey/rkey. The kernel must loop, querying
   `VmmQueryLocalKey`/`VmmQueryRemoteAddr` per chunk.

4. **SDMA template instantiation** — `DISPATCH_TRANSPORT_TYPE` macro
   dispatches to all three transport types at compile time. Even if SDMA
   is unused, the template specialization must exist or compilation fails.

5. **`_jit_sources` permission** — `pip install` copies headers to
   `python/mori/_jit_sources/` as root. When re-syncing source, exclude
   this directory: `rsync --exclude '_jit_sources/'`.

6. **Single-value operations needing ibuf** — GPU stack is not NIC-registered.
   For `shmem_g` (returns T) over IBGDA, use `ShmemGetAtomicIbufSlot`
   to allocate a registered intermediate buffer, PostRead into it,
   quiet, then copy to return value.
