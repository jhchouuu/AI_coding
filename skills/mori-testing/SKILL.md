---
name: mori-testing
description: Run mori tests. Supports both intra-node (single node) and inter-node (multi-node) testing. Automatically detects the IBGDA NIC type on the server (bnxt/mlx5/ionic) and delegates to the appropriate testing skill. Use when the user asks to run tests, debug test failures, benchmark, or asks about testing mori.
---

# Mori Testing (Auto-Detect)

This is the top-level testing skill. It determines the test scope
(intra-node vs inter-node) and delegates to the correct sub-skill.
Both sub-skills detect the IBGDA NIC type internally using the same
multi-priority logic as mori's `detect_nic_type()` (sysfs → lspci vendor
IDs → library fallback).

## Step 1: Determine test scope

| User request | Skill to use |
|--------------|--------------|
| General testing, intra-node, single node | `mori-intranode-testing/SKILL.md` |
| Inter-node, multi-node, 2-node, EP16, bench, stress | `mori-internode-testing/SKILL.md` |

If the user asks for **inter-node**, **multi-node**, **2-node**, **EP16**,
**bench**, or **stress** testing, go directly to the inter-node skill.

Otherwise, go to the intra-node skill. Both sub-skills handle NIC detection
internally.

## Step 2: Delegate

Read the corresponding skill file and follow it from the beginning.

The sub-skills are located at:

```
skills/mori-intranode-testing/SKILL.md
skills/mori-internode-testing/SKILL.md
```

## Notes

- If the user explicitly specifies a NIC type (e.g. "run BNXT tests") or sets
  `MORI_DEVICE_NIC` env var, pass that to the sub-skill to skip auto-detection.
- For inter-node tests, the user must provide or confirm two server hostnames
  and ensure SSH access is available to both.
- The NIC here refers specifically to the **IBGDA NIC** (InfiniBand GPU-Direct
  Async), not general network NICs. Detection follows the same priority chain
  as mori's build system: `MORI_DEVICE_NIC` env → `/sys/class/infiniband/`
  sysfs → `lspci -nn -d ::0200` vendor IDs (`14e4`=bnxt, `15b3`=mlx5,
  `1dd8`=ionic) → userspace library fallback (`libbnxt_re.so` / `libmlx5.so` /
  `libionic.so`).
