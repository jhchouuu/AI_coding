---
name: mori-intranode-testing
description: Run intra-node (single node) tests for mori project. Automatically detects the NIC type (BNXT Thor2, Mellanox CX7, or AINIC Pollara) and uses the appropriate Docker image and library verification. Use when the user asks to run tests, debug test failures, benchmark, or asks about testing mori components (EP, IO, shmem, IR) on a single node.
---

# Mori Intra-Node Testing

## Important: All commands run inside Docker

All install and test commands MUST be executed inside the Docker container via
`sudo docker exec`. Never run mori tests directly on the host.

## Pre-check: Verify no other tests are running

Before starting, check if someone else is using the GPUs:

```bash
# Check GPU usage
rocm-smi

# Check for existing mori test containers
sudo docker ps --filter "name=mori_test_"
```

- If `rocm-smi` shows significant GPU memory usage or active processes,
  **stop and ask the user** before proceeding — someone else may be running tests.
- If there are existing `mori_test_*` containers, **report them to the user**
  and confirm whether to proceed (concurrent tests may cause GPU contention).

## Step 1: Detect IBGDA NIC type

The NIC here refers specifically to the **RDMA NIC used for IBGDA**
(InfiniBand GPU-Direct Async). The detection mirrors mori's own
`detect_nic_type()` in `python/mori/jit/config.py` and `detect_device_nic()`
in `CMakeLists.txt`.

If the user explicitly specifies a NIC type (e.g. "run BNXT tests") or sets
`MORI_DEVICE_NIC`, skip detection and use that directly.

Otherwise, run the following detection chain on the **host** (not inside
Docker). Stop at the first step that yields a result.

### Priority 1: MORI_DEVICE_NIC env override

```bash
echo "MORI_DEVICE_NIC=${MORI_DEVICE_NIC:-<not set>}"
```

If set to `bnxt`, `ionic`, or `mlx5`, use that value directly. Skip the rest.

### Priority 2: /sys/class/infiniband/ (sysfs)

```bash
echo "=== InfiniBand sysfs devices ==="
ls /sys/class/infiniband/ 2>/dev/null || echo "No /sys/class/infiniband/"

echo ""
echo "=== Device drivers ==="
for dev in /sys/class/infiniband/*; do
  name=$(basename "$dev")
  driver=$(readlink -f "$dev/device/driver" 2>/dev/null | xargs basename 2>/dev/null)
  echo "  $name -> driver: ${driver:-unknown}"
done
```

Map device names and drivers to NIC types:

| Device name prefix / Driver | NIC Type |
|-----------------------------|----------|
| `bnxt_re*` or driver `bnxt_re` / `bnxt_en` | **bnxt** |
| `mlx5*` or driver `mlx5_core` / `mlx5_ib` | **mlx5** |
| `ionic*` or driver `ionic_rdma` / `ionic` | **ionic** |

Pick the NIC type with **the most devices**. Tie-break order: mlx5 > bnxt > ionic.

### Priority 3: lspci PCI vendor IDs

```bash
echo "=== PCI Ethernet controllers (class 0200) ==="
lspci -nn -d ::0200
```

Count matches by PCI vendor ID:

| Vendor ID | Vendor | NIC Type |
|-----------|--------|----------|
| `14e4` | Broadcom (BCM576xx/578xx) | **bnxt** |
| `15b3` | Mellanox/NVIDIA (ConnectX) | **mlx5** |
| `1dd8` | AMD/Pensando (Pollara) | **ionic** |

Pick the vendor with the most devices. Tie-break: mlx5 > bnxt > ionic.

### Priority 4: Userspace library fallback

```bash
echo "=== NIC userspace libraries ==="
find /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu \
  -maxdepth 1 \( -name 'libmlx5.so' -o -name 'libbnxt_re.so' -o -name 'libionic.so' \) \
  2>/dev/null
```

First library found wins (search order: mlx5 > bnxt > ionic).

### NIC type → Docker image

| NIC Type | Docker Image | Login Required |
|----------|-------------|----------------|
| **mlx5** (Mellanox CX7) | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` | No |
| **bnxt** (Broadcom Thor2) | `rocm/pytorch-private:mori_bnxt235_rocm711_ubuntu24.04_py3.12` | Yes |
| **ionic** (AMD AINIC Pollara) | `rocm/pytorch-private:sglang-0.5.8-rocm720-mi35x-mori-0216` | Yes |

If no NIC is detected at any priority level, **stop and report** the issue.

## Step 2: Create a fresh Docker container

Each test run uses a **new container from scratch**. Generate a unique name
(e.g. `mori_test_<timestamp>`) to avoid conflicts.

For private images, login first:

```bash
sudo docker login -u rocmshared
```

Copy the source code to an isolated directory so concurrent tests don't conflict:

```bash
MORI_SRC="<path to mori repo>"
CONTAINER_NAME="mori_test_$(date +%s)"
TEST_SRC="/tmp/mori_test_$(date +%s)"
rsync -a --exclude build/ "$MORI_SRC"/ "$TEST_SRC"
```

Launch the container with the image determined in Step 1:

```bash
DOCKER_IMAGE="<image from Step 1 table>"
sudo docker run \
  --group-add video --network=host \
  --ulimit nproc=100000:100000 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  -d --ipc=host --privileged -it \
  -v /home/:/home/ -v /root:/root -v /mnt:/mnt -v "$TEST_SRC":"$TEST_SRC" \
  --name "$CONTAINER_NAME" \
  "$DOCKER_IMAGE"
```

## Step 3: Verify NIC libraries (inside container)

Run the check that matches the detected NIC type:

### BNXT Thor2

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== BNXT shared libraries ===' && \
  ls -l /usr/local/lib/libbnxt_re*.so* && \
  echo '=== BNXT headers ===' && \
  ls -l /usr/include/infiniband/bnxt_re_dv.h /usr/include/infiniband/bnxt_re_hsi.h && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '=== BNXT userlib version ===' && \
  strings /usr/local/lib/libbnxt_re.so | grep -i version || true && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify: `libbnxt_re.so`, `libbnxt_re-rdmav34.so`, `bnxt_re_dv.h`,
`bnxt_re_hsi.h` exist, and libibverbs/libbnxt_re userlib versions match.

### Mellanox CX7

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== libmlx5 ===' && \
  find /usr -name 'libmlx5*' 2>/dev/null && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify: `libmlx5.so` exists and libibverbs is installed.

### AINIC Pollara

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== AINIC (libionic) ===' && \
  find /usr -name 'libionic*' 2>/dev/null && \
  ldconfig -p | grep libionic || echo 'libionic not found in ldconfig' && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify: `libionic.so` exists and `ibv_devinfo` detects AINIC devices.

If the required library is missing, **stop and fix the image** before proceeding.

## Step 4: Install mori (inside container)

To install mori **with** C++ shmem example binaries (for MORI-CPP tests),
set `BUILD_EXAMPLES=ON`:

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "cd $TEST_SRC && BUILD_EXAMPLES=ON pip install ."
```

If you only need Python tests and don't need C++ examples:

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "cd $TEST_SRC && pip install ."
```

Verify:

```bash
sudo docker exec "$CONTAINER_NAME" python -c "import mori; print('OK')"
```

When `BUILD_EXAMPLES=ON`, the example binaries are at
`$TEST_SRC/build/examples/` (e.g. `concurrent_put_thread`,
`concurrent_get_thread`).

## Step 5: Run Tests (inside container)

All test commands are run via `sudo docker exec "$CONTAINER_NAME" bash -c "..."`.

### Timeout & Hang Detection

Each test has an expected timeout. Use `timeout` to enforce it. If a test
exceeds its timeout, `timeout` kills it (exit code 124). Mark that test as
**HANG**, skip it, and continue to the next test.

| Test | Timeout |
|------|---------|
| MORI-EP | 120s |
| MORI-IO | 60s |
| MORI-IR shmem | 60s |
| MORI-IR allreduce | 60s |
| MORI-CCL/shmem | 600s |
| MORI-CPP shmem (per binary, per mode) | 60s |

Run each test with `timeout`:

```bash
sudo docker exec "$CONTAINER_NAME" bash -c \
  "cd $TEST_SRC && PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
   timeout <SECONDS> <test_command>"
```

If exit code is 124, the test hung. Record it and move on to the next test.
If exit code is non-zero but not 124, the test failed. Record it and move on.
**Always run all tests regardless of individual failures.**

### MORI-EP (dispatch / combine, timeout 120s)

```bash
timeout 120 pytest tests/python/ops/test_dispatch_combine.py -q
```

### MORI-IO (single node, timeout 60s)

```bash
timeout 60 pytest tests/python/io/
```

### MORI-IR (Triton + shmem integration, single node)

```bash
# shmem put (2 GPUs, timeout 60s)
timeout 60 torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

# allreduce (8 GPUs, timeout 60s)
timeout 60 torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py
```

### MORI-CPP shmem examples (requires BUILD_EXAMPLES=ON)

C++ shmem example binaries test the low-level device API directly. Run
each binary with `mpirun -np 2` in two modes: **P2P** (default) and
**IBGDA** (`MORI_DISABLE_P2P=ON`).

Available shmem binaries (in `$TEST_SRC/build/examples/`):

| Binary | Description |
|--------|-------------|
| `concurrent_put_thread` | PUT: legacy + address API, thread/block, large data, atomics |
| `concurrent_get_thread` | GET: legacy + address API, thread/block, large data |
| `concurrent_put_imm_thread` | PUT immediate (inline small values) |
| `concurrent_put_signal_thread` | PUT with signal |
| `atomic_nonfetch_thread` | Atomic non-fetch operations |
| `atomic_fetch_thread` | Atomic fetch operations |

```bash
# P2P mode (default transport)
timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/concurrent_put_thread
timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/concurrent_get_thread

# IBGDA mode (RDMA transport via NIC)
MORI_DISABLE_P2P=ON timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/concurrent_put_thread
MORI_DISABLE_P2P=ON timeout 60 mpirun --allow-run-as-root -np 2 ./build/examples/concurrent_get_thread
```

Optional: test with VMM heap mode by adding `MORI_SHMEM_MODE=VMM_HEAP`.

### MORI-CCL / shmem (timeout 600s)

```bash
timeout 600 pytest tests/python/shmem/test_api.py -v
```

### Final Report

After all tests complete, produce a summary table:

```
| Test                       | Result | Details                    |
|----------------------------|--------|----------------------------|
| MORI-EP                    | PASS   | 80 passed, 176 skipped     |
| MORI-IO                    | PASS   | 145 passed                 |
| MORI-IR shmem              | PASS   | 2 PEs                      |
| MORI-IR allreduce          | PASS   | 8 PEs, 100 GB/s            |
| MORI-CPP put (P2P)         | PASS   | 10 tests                   |
| MORI-CPP put (IBGDA)       | PASS   | 9 tests (skip direct P2P)  |
| MORI-CPP get (P2P)         | PASS   | 5 tests                    |
| MORI-CPP get (IBGDA)       | PASS   | 5 tests                    |
| MORI-CCL/shmem             | PASS   | 18 passed                  |
```

Possible result values: **PASS**, **FAIL** (non-zero exit), **HANG** (exit 124 / timeout).

## Step 6: Cleanup

**Important:** During active development or iterative debugging, do NOT
automatically delete the container or source copy. Keep them around so
subsequent test runs can reuse them (skip Steps 2–4, jump straight to
Step 5). Only clean up when the user explicitly asks or when all
development tasks are complete.

When the user explicitly requests cleanup:

```bash
sudo docker rm -f "$CONTAINER_NAME"
sudo rm -rf "$TEST_SRC"
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `MORI_GLOBAL_LOG_LEVEL` | Log verbosity | `TRACE`, `DEBUG`, `INFO` |
| `MORI_SHMEM_HEAP_SIZE` | Shmem heap size | `1073741824` |
| `MORI_DISABLE_AUTO_XGMI` | Disable auto XGMI detection | `1` |
| `MORI_PRECOMPILE` | Precompile all JIT kernels | `1` |
| `PYTHONPATH` | Include mori source for tests | `/path/to/mori:$PYTHONPATH` |

## Test Workflow Checklist

- [ ] Check `rocm-smi` and `docker ps` for existing tests
- [ ] Detect NIC type on host (or use user-specified type)
- [ ] Copy source to isolated temp directory (exclude `build/`)
- [ ] Create fresh Docker container with the matching image
- [ ] Verify NIC-specific libraries exist inside container
- [ ] `BUILD_EXAMPLES=ON pip install .` inside container (or without BUILD_EXAMPLES if no CPP tests needed)
- [ ] `python -c "import mori; print('OK')"` passes inside container
- [ ] Run each test with `timeout`, record PASS/FAIL/HANG
- [ ] If a test hangs (exit 124), kill it and continue to the next
- [ ] After all tests, produce final summary report table
- [ ] Keep container for reuse during development; only remove when user requests

## Debugging Failures

1. Re-run with verbose logging (inside container):
   ```bash
   sudo docker exec "$CONTAINER_NAME" bash -c \
     "cd $TEST_SRC && MORI_GLOBAL_LOG_LEVEL=TRACE PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
      pytest tests/python/ops/test_dispatch_combine.py -v -s"
   ```
2. Common issues:
   - **GPU not available**: `rocm-smi` and `ROCR_VISIBLE_DEVICES`
   - **Port conflicts**: Tests use `get_free_port()` from `tests/python/utils.py`
   - **MPI errors**: Ensure OpenMPI is installed and use `--allow-run-as-root`
   - **Missing system deps (Mellanox image)**: May need `apt-get install -y libopenmpi-dev openmpi-bin libpci-dev libibverbs-dev`
