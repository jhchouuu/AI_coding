# AI Coding

A collection of AI agent skills and rules for automated coding workflows.

## Skills

| Skill | Description |
|-------|-------------|
| [mori-testing](skills/mori-testing/SKILL.md) | **Top-level** — auto-detects IBGDA NIC type and delegates to intranode or internode |
| [mori-intranode-testing](skills/mori-intranode-testing/SKILL.md) | Single-node tests (EP, IO, IR, CPP, CCL/shmem) — supports bnxt/mlx5/ionic |
| [mori-internode-testing](skills/mori-internode-testing/SKILL.md) | 2-node EP16 benchmark and stress tests — supports bnxt/mlx5/ionic |
| [mori-shmem-dev](skills/mori-shmem-dev/SKILL.md) | SHMEM device API development workflow — kernel impl, testing across all NIC vendors |
| [pytorch-rocm-build](skills/pytorch-rocm-build/SKILL.md) | Build PyTorch from source on AMD ROCm — Docker setup, hipify, editable install, dev workflow |
| [mori-codebase-architecture](skills/mori-codebase-architecture/SKILL.md) | MORI codebase architecture — module structure, build system, key concepts |

## Usage

### Cursor

Symlink or copy skills into `~/.cursor/skills/`:

```bash
ln -s /path/to/AI_coding/skills/mori-intranode-testing ~/.cursor/skills/mori-intranode-testing
```

### Other AI Agents

The skills are standard markdown files. You can feed them as context/system prompts to any LLM-based agent (Copilot, Cline, Aider, etc.).
