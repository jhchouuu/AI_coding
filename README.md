# AI Coding

A collection of AI agent skills and rules for automated coding workflows.

## Skills

| Skill | Description |
|-------|-------------|
| [mori-testing](skills/mori-testing/SKILL.md) | **Top-level** — auto-detects NIC type and delegates to the correct skill |
| [mori-bnxt-intranode-testing](skills/mori-bnxt-intranode-testing/SKILL.md) | Run BNXT (Thor2) intra-node tests |
| [mori-mlnx-intranode-testing](skills/mori-mlnx-intranode-testing/SKILL.md) | Run Mellanox (CX7) intra-node tests |
| [mori-ainic-intranode-testing](skills/mori-ainic-intranode-testing/SKILL.md) | Run AINIC (Pollara) intra-node tests |

## Usage

### Cursor

Symlink or copy skills into `~/.cursor/skills/`:

```bash
ln -s /path/to/AI_coding/skills/mori-bnxt-intranode-testing ~/.cursor/skills/mori-bnxt-intranode-testing
```

### Other AI Agents

The skills are standard markdown files. You can feed them as context/system prompts to any LLM-based agent (Copilot, Cline, Aider, etc.).
