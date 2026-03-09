---
name: mori-testing
description: Run mori intra-node tests. Automatically detects the NIC type on the server (BNXT Thor2, Mellanox CX7, or AMD AINIC Pollara) and delegates to the appropriate testing skill. Use when the user asks to run tests, debug test failures, benchmark, or asks about testing mori.
---

# Mori Testing (Auto-Detect NIC)

This is the top-level testing skill. It detects the server's NIC type and
delegates to the correct NIC-specific skill.

## Step 1: Detect NIC type

Run on the **host** (not inside Docker):

```bash
echo "=== PCI devices (RDMA NICs) ==="
lspci | grep -iE "mellanox|broadcom|pensando|ionic" || echo "No RDMA NICs found via lspci"

echo ""
echo "=== ibv_devinfo ==="
ibv_devinfo 2>/dev/null | grep -E "hca_id|vendor_id|fw_ver" || echo "ibv_devinfo not available"

echo ""
echo "=== NIC driver modules ==="
lsmod | grep -iE "mlx5|bnxt|ionic" || echo "No RDMA driver modules loaded"
```

## Step 2: Select the correct skill

Based on the detection results:

| Detection | NIC Type | Skill to use |
|-----------|----------|--------------|
| `mlx5` in lsmod, or `Mellanox` in lspci | **Mellanox CX7** | Read and follow `mori-mlnx-intranode-testing/SKILL.md` |
| `bnxt` in lsmod, or `Broadcom` in lspci | **BNXT Thor2** | Read and follow `mori-bnxt-intranode-testing/SKILL.md` |
| `ionic` in lsmod, or `Pensando` in lspci | **AINIC Pollara** | Read and follow `mori-ainic-intranode-testing/SKILL.md` |
| Multiple NICs detected | **Ask user** | Report which NICs are found and ask which one to test with |
| No RDMA NIC detected | **Stop** | Report the issue and do not proceed |

## Step 3: Delegate

Once the NIC type is determined, **read the corresponding skill file** and
follow it from the beginning (Pre-check → Docker → Verify libs → Install →
Test → Cleanup).

The sub-skills are located at:

```
.cursor/skills/mori-bnxt-intranode-testing/SKILL.md
.cursor/skills/mori-mlnx-intranode-testing/SKILL.md
.cursor/skills/mori-ainic-intranode-testing/SKILL.md
```

## Notes

- Always run NIC detection on the **host**, not inside Docker.
- If the user explicitly specifies a NIC type (e.g. "run BNXT tests"), skip
  detection and go directly to the corresponding skill.
- If the server has multiple NIC types, ask the user which one to test with
  unless they already specified.
