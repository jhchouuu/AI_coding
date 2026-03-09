---
name: mori-ainic-intranode-testing
description: Run AINIC (Pollara) intra-node tests for mori project. Use when the user asks to run tests, debug test failures, add new tests, benchmark, or asks about testing mori components (EP, IO, shmem, IR) on a single node with AMD AINIC (Pollara) NIC.
---

# Mori AINIC (Pollara) Intra-Node Testing

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

## Step 1: Create a fresh Docker container

Each test run uses a **new container from scratch**. Generate a unique name
(e.g. `mori_test_<timestamp>`) to avoid conflicts.

The image is private, login first if needed:

```bash
sudo docker login -u rocmshared
```

Copy the source code to an isolated directory so concurrent tests don't conflict:

Determine the mori source path (the root of the mori git repo in the current
workspace), then copy it:

```bash
MORI_SRC="<path to mori repo>"  # e.g. /mnt/nvme1/jiahzhou/mori
CONTAINER_NAME="mori_test_$(date +%s)"
TEST_SRC="/tmp/mori_test_$(date +%s)"
rsync -a --exclude build/ "$MORI_SRC"/ "$TEST_SRC"
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
  rocm/pytorch-private:sglang-0.5.8-rocm720-mi35x-mori-0216
```

## Step 2: Verify AINIC libraries (inside container)

Before installing mori, check that the AINIC (Pollara) userspace library exists
inside the container:

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== AINIC (libionic) ===' && \
  find /usr -name 'libionic*' 2>/dev/null && \
  ldconfig -p | grep libionic || echo 'libionic not found in ldconfig' && \
  echo '' && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '' && \
  echo '=== ibv_devinfo (AINIC device check) ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify:
- `libionic.so` exists (required for IBGDA with AMD Pollara NICs)
- libibverbs is installed
- `ibv_devinfo` detects AINIC devices

If `libionic` is missing, **stop and fix the image** before proceeding.

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
| MORI-CCL/shmem    | PASS   | 18 passed             |
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

- [ ] Check `rocm-smi` and `docker ps` for existing tests
- [ ] Copy source to isolated temp directory (exclude `build/`)
- [ ] Create fresh Docker container with unique name
- [ ] Verify `libionic.so` and libibverbs exist inside container
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
