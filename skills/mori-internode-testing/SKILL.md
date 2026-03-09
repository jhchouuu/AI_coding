---
name: mori-internode-testing
description: Run inter-node (2-node) EP dispatch/combine benchmark and stress tests for mori project. Use when the user asks to run internode tests, 2-node tests, EP16 tests, or multi-node benchmarks/stress tests.
---

# Mori Inter-Node Testing (2-Node EP16)

## Important: All commands run inside Docker

All install and test commands MUST be executed inside the Docker container via
`sudo docker exec`. Never run mori tests directly on the host.

## Prerequisites

- **Two servers** with RDMA NICs used for **IBGDA** (InfiniBand GPU-Direct
  Async), reachable from each other via `eno1` network interface.
- SSH access to both servers from the machine running this skill.
- A Docker image with the matching NIC userspace libraries available on both
  servers.

### Detect IBGDA NIC type

The detection mirrors mori's `detect_nic_type()` (`python/mori/jit/config.py`)
and `detect_device_nic()` (`CMakeLists.txt`). Run on **either server** (both
must have the same NIC). Stop at the first step that yields a result.

If the user explicitly specifies a NIC type or sets `MORI_DEVICE_NIC`, use
that directly and skip detection.

**Priority 1 — MORI_DEVICE_NIC env override:**
```bash
echo "MORI_DEVICE_NIC=${MORI_DEVICE_NIC:-<not set>}"
```

**Priority 2 — /sys/class/infiniband/ (sysfs):**
```bash
for dev in /sys/class/infiniband/*; do
  name=$(basename "$dev")
  driver=$(readlink -f "$dev/device/driver" 2>/dev/null | xargs basename 2>/dev/null)
  echo "  $name -> driver: ${driver:-unknown}"
done
```

Map device name / driver to NIC type:

| Device prefix / Driver | NIC Type |
|------------------------|----------|
| `bnxt_re*` or `bnxt_re` / `bnxt_en` | **bnxt** |
| `mlx5*` or `mlx5_core` / `mlx5_ib` | **mlx5** |
| `ionic*` or `ionic_rdma` / `ionic` | **ionic** |

Pick the type with the most devices. Tie-break: mlx5 > bnxt > ionic.

**Priority 3 — lspci PCI vendor IDs (class 0200):**
```bash
lspci -nn -d ::0200
```

| Vendor ID | NIC Type |
|-----------|----------|
| `14e4` (Broadcom) | **bnxt** |
| `15b3` (Mellanox/NVIDIA) | **mlx5** |
| `1dd8` (AMD/Pensando) | **ionic** |

**Priority 4 — Userspace library fallback:**
```bash
find /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu \
  -maxdepth 1 \( -name 'libmlx5.so' -o -name 'libbnxt_re.so' -o -name 'libionic.so' \) \
  2>/dev/null
```

### NIC type → Docker image

| NIC Type | Docker Image | Login Required |
|----------|-------------|----------------|
| **mlx5** (Mellanox CX7) | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` | No |
| **bnxt** (Broadcom Thor2) | `rocm/pytorch-private:mori_bnxt235_rocm711_ubuntu24.04_py3.12` | Yes (`sudo docker login -u rocmshared`) |
| **ionic** (AMD AINIC Pollara) | `rocm/pytorch-private:sglang-0.5.8-rocm720-mi35x-mori-0216` | Yes (`sudo docker login -u rocmshared`) |

If the image is missing on a server, login (if needed) and pull first:
```bash
sudo docker login -u rocmshared  # only for private images
sudo docker pull <image>
```

## Step 1: Identify nodes and determine master

Determine the `eno1` IP addresses on both servers to identify which is the
master node:

```bash
ssh <node0> "ip addr show eno1 | grep 'inet '"
ssh <node1> "ip addr show eno1 | grep 'inet '"
```

The `master_addr` in the torchrun command must be the `eno1` IP of **node 0**
(node_rank=0). Record which server is node 0 and which is node 1.

## Step 2: Pre-check on both nodes

Run on **both** servers (in parallel if possible):

```bash
# Check GPU usage
rocm-smi

# Check for existing mori test containers
sudo docker ps --filter "name=mori_ep16_"
```

- If `rocm-smi` shows significant GPU memory usage or active processes,
  **stop and ask the user** before proceeding.
- If there are existing `mori_ep16_*` containers, **report them to the user**
  and confirm whether to proceed.

## Step 3: Sync source and create containers on both nodes

Generate a shared timestamp for unique naming:

```bash
TIMESTAMP=$(date +%s)
CONTAINER_NAME="mori_ep16_${TIMESTAMP}"
TEST_SRC="/tmp/mori_ep16_${TIMESTAMP}"
MASTER_ADDR="<eno1 IP of node 0>"
MASTER_PORT=1236
```

Sync the mori source to both servers (in parallel):

```bash
MORI_SRC="<path to mori repo>"
rsync -az --exclude build/ --exclude '.git/objects' "$MORI_SRC"/ <node0>:$TEST_SRC/
rsync -az --exclude build/ --exclude '.git/objects' "$MORI_SRC"/ <node1>:$TEST_SRC/
```

Create Docker containers on both servers (in parallel), using the image
selected from the Prerequisites table:

```bash
# On each node:
DOCKER_IMAGE="<image from Prerequisites table>"
sudo docker run \
  --group-add video --network=host \
  --ulimit nproc=100000:100000 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  -d --ipc=host --privileged -it \
  -v /home/:/home/ -v /root:/root -v /mnt:/mnt -v "$TEST_SRC":"$TEST_SRC" \
  --name "$CONTAINER_NAME" \
  "$DOCKER_IMAGE"
```

## Step 4: Verify NIC libraries (inside containers)

Run on **both** containers. The check depends on the NIC type:

**BNXT Thor2:**
```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== BNXT shared libraries ===' && \
  ls -l /usr/local/lib/libbnxt_re*.so* && \
  echo '=== BNXT headers ===' && \
  ls -l /usr/include/infiniband/bnxt_re_dv.h /usr/include/infiniband/bnxt_re_hsi.h && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

**Mellanox CX7:**
```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== libmlx5 ===' && \
  find /usr -name 'libmlx5*' 2>/dev/null && \
  echo '=== libibverbs ===' && \
  dpkg -l | grep libibverbs || rpm -qa | grep libibverbs && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

**AINIC Pollara:**
```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  echo '=== AINIC (libionic) ===' && \
  find /usr -name 'libionic*' 2>/dev/null && \
  ldconfig -p | grep libionic || echo 'libionic not found in ldconfig' && \
  echo '=== ibv_devinfo ===' && \
  ibv_devinfo 2>/dev/null | head -20 || echo 'ibv_devinfo not available'"
```

Verify the NIC-specific library, libibverbs, and `ibv_devinfo` detect devices on both nodes.

## Step 5: Install mori (inside containers)

Run on **both** containers (in parallel):

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "cd $TEST_SRC && pip install . && pip install prettytable"
```

Verify on both:

```bash
sudo docker exec "$CONTAINER_NAME" python -c "import mori; print('OK')"
```

## Step 6: Run EP16 Internode Benchmark

The test uses `torchrun` with `--nnodes=2` and `--nproc_per_node=1` (each
node uses 8 GPUs internally via `torch.multiprocessing.spawn`), giving 16 PEs
total.

**Both commands must be launched simultaneously** (in parallel). If one side
starts too late, the Gloo rendezvous may time out.

### On node 0 (master, node_rank=0):

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  cd $TEST_SRC && PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
  GLOO_SOCKET_IFNAME=eno1 MORI_SOCKET_IFNAME=eno1 \
  timeout 300 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --max-tokens 4096 --cmd bench --kernel-type v1 --num-qp 2"
```

### On node 1 (worker, node_rank=1):

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  cd $TEST_SRC && PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
  GLOO_SOCKET_IFNAME=eno1 MORI_SOCKET_IFNAME=eno1 \
  timeout 300 torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --max-tokens 4096 --cmd bench --kernel-type v1 --num-qp 2"
```

### Timeout & Hang Detection

| Test | Timeout |
|------|---------|
| EP16 bench | 300s |
| EP16 stress | 600s |

If exit code is 124 on either node, the test hung. Record it as **HANG**.

### Final Report (Benchmark)

After the benchmark completes, the master node (node_rank=0) prints a summary
table with Dispatch and Combine performance. Extract and report:

```
| Test          | Result | RDMA BW (GB/s) | XGMI BW (GB/s) | LL BW (GB/s) | Latency (us) |
|---------------|--------|----------------|-----------------|--------------|--------------|
| EP16 Dispatch | PASS   | 56 avg         | 185 avg         | 229 avg      | 2094 avg     |
| EP16 Combine  | PASS   | 60 avg         | 199 avg         | 246 avg      | 1946 avg     |
```

Possible result values: **PASS**, **FAIL** (non-zero exit), **HANG** (exit 124 / timeout).

## Step 6b: Run EP16 Internode Stress Test (optional)

The stress test runs 5000 iterations of dispatch/combine to verify stability
under sustained load. Use `--cmd stress` instead of `--cmd bench`.

**Both commands must be launched simultaneously** (in parallel), same as the
benchmark.

### On node 0 (master, node_rank=0):

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  cd $TEST_SRC && PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
  GLOO_SOCKET_IFNAME=eno1 MORI_SOCKET_IFNAME=eno1 \
  timeout 600 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --max-tokens 4096 --cmd stress --kernel-type v1 --num-qp 2"
```

### On node 1 (worker, node_rank=1):

```bash
sudo docker exec "$CONTAINER_NAME" bash -c "\
  cd $TEST_SRC && PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
  GLOO_SOCKET_IFNAME=eno1 MORI_SOCKET_IFNAME=eno1 \
  timeout 600 torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --max-tokens 4096 --cmd stress --kernel-type v1 --num-qp 2"
```

### Final Report (Stress)

The stress test prints iteration progress every 500 iterations and a final
summary. Report:

```
| Test         | Result | Iterations | Throughput (it/s) |
|--------------|--------|------------|-------------------|
| EP16 Stress  | PASS   | 5000       | ~305              |
```

Possible result values: **PASS**, **FAIL** (non-zero exit), **HANG** (exit 124 / timeout).

## Step 7: Cleanup

After the test completes (pass or fail), remove containers and temp source on
**both** nodes:

```bash
sudo docker rm -f "$CONTAINER_NAME"
sudo rm -rf "$TEST_SRC"
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `GLOO_SOCKET_IFNAME` | Network interface for Gloo rendezvous | `eno1` |
| `MORI_SOCKET_IFNAME` | Network interface for mori RDMA transport | `eno1` |
| `MORI_GLOBAL_LOG_LEVEL` | Log verbosity | `TRACE`, `DEBUG`, `INFO` |
| `PYTHONPATH` | Include mori source for tests | `/path/to/mori:$PYTHONPATH` |

## Test Workflow Checklist

- [ ] Identify both nodes' `eno1` IPs, determine master (node_rank=0)
- [ ] Pre-check GPU usage and existing containers on both nodes
- [ ] Sync source to both nodes (exclude `build/`)
- [ ] Create fresh Docker containers on both nodes with unique name
- [ ] Verify NIC libs inside both containers (BNXT/Mellanox/AINIC)
- [ ] `pip install .` and `pip install prettytable` inside both containers
- [ ] `python -c "import mori; print('OK')"` passes on both containers
- [ ] Launch benchmark torchrun on both nodes **simultaneously** (`--cmd bench`)
- [ ] Wait for both to complete, record PASS/FAIL/HANG
- [ ] Extract benchmark performance summary from master node output
- [ ] (Optional) Launch stress test on both nodes (`--cmd stress`)
- [ ] Remove containers and temp source on both nodes

## Debugging Failures

1. **Gloo rendezvous timeout**: Ensure both nodes can reach `master_addr` on
   `master_port`. Check firewall rules and that `eno1` interface is up.
2. **RDMA errors**: Verify the NIC driver is loaded (`lsmod | grep -iE "bnxt|mlx5|ionic"`)
   and `ibv_devinfo` shows active ports on both nodes.
3. **prettytable missing**: The benchmark requires `prettytable` for output
   formatting. Install it via `pip install prettytable` inside the container.
4. Re-run with verbose logging:
   ```bash
   sudo docker exec "$CONTAINER_NAME" bash -c \
     "cd $TEST_SRC && MORI_GLOBAL_LOG_LEVEL=TRACE PYTHONPATH=$TEST_SRC:\$PYTHONPATH \
      GLOO_SOCKET_IFNAME=eno1 MORI_SOCKET_IFNAME=eno1 \
      torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
      --max-tokens 4096 --cmd bench --kernel-type v1 --num-qp 2"
   ```
