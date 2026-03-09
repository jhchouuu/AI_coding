---
name: mori-bnxt-intranode-testing
description: Run BNXT intra-node tests for mori project. Use when the user asks to run tests, debug test failures, add new tests, benchmark, or asks about testing mori components (EP, IO, shmem, IR, C++ tests) on a single node with BNXT NIC.
---

# Mori BNXT Intra-Node Testing

## Important: All commands run inside Docker

All install and test commands MUST be executed inside the Docker container via
`sudo docker exec`. Never run mori tests directly on the host.

## Step 1: Create a fresh Docker container

Each test run uses a **new container from scratch**. Generate a unique name
(e.g. `mori_test_<timestamp>`) to avoid conflicts.

The image is private, login first if needed:

```bash
sudo docker login -u rocmshared
```

Copy the source code to an isolated directory so concurrent tests don't conflict:

```bash
CONTAINER_NAME="mori_test_$(date +%s)"
TEST_SRC="/tmp/mori_test_$(date +%s)"
rsync -a --exclude build/ /mnt/nvme1/jiahzhou/mori/ "$TEST_SRC"
```

Launch the container:

```bash
sudo docker run \
  --group-add video --network=host \
  --ulimit nproc=100000:100000 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  -d --ipc=host --privileged -it \
  -v /home/:/home/ -v /root:/root -v /mnt:/mnt -v "$TEST_SRC":"$TEST_SRC" \
  --name "$CONTAINER_NAME" \
  rocm/pytorch-private:mori_bnxt235_rocm711_ubuntu24.04_py3.12
```

## Step 2: Verify BNXT libraries (inside container)

Before installing mori, check that the required BNXT dynamic libraries and
headers exist inside the container, and that the userlib version matches libibverbs:

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== BNXT shared libraries ===' && \
  ls -l /usr/local/lib/libbnxt_re*.so* && \
  echo '' && \
  echo '=== BNXT headers ===' && \
  ls -l /usr/include/infiniband/bnxt_re_dv.h /usr/include/infiniband/bnxt_re_hsi.h && \
  echo '' && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '' && \
  echo '=== BNXT userlib version ===' && \
  strings /usr/local/lib/libbnxt_re.so | grep -i version || true && \
  echo '' && \
  echo '=== ibv_devinfo (BNXT device check) ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify:
- `libbnxt_re.so`, `libbnxt_re-rdmav34.so` exist
- `bnxt_re_dv.h`, `bnxt_re_hsi.h` exist
- libibverbs and libbnxt_re userlib are built from the same BNXT driver version
  (version mismatch causes silent IBGDA failures)

If any library is missing or versions don't match, **stop and fix the image**
before proceeding.

## Step 3: Install mori (inside container)

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "cd $TEST_SRC && pip install ."
```

Verify:

```bash
sudo docker exec "$CONTAINER_NAME" python -c "import mori; print('OK')"
```

## Step 4: Run Tests (inside container)

All test commands are run via `sudo docker exec "$CONTAINER_NAME" bash -c "..."`.

### Timeout & Hang Detection

Each test has an expected timeout. Use `timeout` to enforce it. If a test
exceeds its timeout, `timeout` kills it (exit code 124). Mark that test as
**HANG**, skip it, and continue to the next test.

| Test | Timeout |
|------|---------|
| MORI-EP | 120s |
| MORI-IO | 60s |
| MORI-IR shmem put | 60s |
| MORI-IR allreduce | 60s |
| MORI-CCL/shmem | 600s |

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

### MORI-CCL / shmem (timeout 600s)

```bash
timeout 600 pytest tests/python/shmem/test_api.py -v
```

### Final Report

After all tests complete, produce a summary table:

```
| Test              | Result | Details               |
|-------------------|--------|-----------------------|
| MORI-EP           | PASS   | 80 passed, 176 skipped |
| MORI-IO           | PASS   | 145 passed            |
| MORI-IR shmem put | PASS   | 2 PEs                 |
| MORI-IR allreduce | PASS   | 8 PEs, 100 GB/s       |
| MORI-CCL/shmem    | HANG   | killed after 600s     |
```

Possible result values: **PASS**, **FAIL** (non-zero exit), **HANG** (exit 124 / timeout).

## Step 5: Cleanup

After all tests complete (pass or fail), remove the container and the copied source:

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

- [ ] Copy source to isolated temp directory (exclude `build/`)
- [ ] Create fresh Docker container with unique name
- [ ] Verify BNXT libs (`libbnxt_re*.so`) and headers (`bnxt_re_dv.h`, `bnxt_re_hsi.h`) exist
- [ ] Verify libibverbs and libbnxt_re userlib versions match
- [ ] `pip install .` inside container
- [ ] `python -c "import mori; print('OK')"` passes inside container
- [ ] Run each test with `timeout`, record PASS/FAIL/HANG
- [ ] If a test hangs (exit 124), kill it and continue to the next
- [ ] After all tests, produce final summary report table
- [ ] Remove container (`sudo docker rm -f`) and temp source (`sudo rm -rf`)

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
