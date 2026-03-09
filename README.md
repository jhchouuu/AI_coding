# AI Coding

A collection of AI agent skills and rules for automated coding workflows.

## Skills

| Skill | Description |
|-------|-------------|
| [mori-bnxt-intranode-testing](skills/mori-bnxt-intranode-testing/SKILL.md) | Run BNXT (Thor2) intra-node tests for the mori project |
| [mori-mlnx-intranode-testing](skills/mori-mlnx-intranode-testing/SKILL.md) | Run Mellanox (CX7) intra-node tests for the mori project |

## Usage

### Cursor

Symlink or copy skills into `~/.cursor/skills/`:

```bash
ln -s /path/to/AI_coding/skills/mori-bnxt-intranode-testing ~/.cursor/skills/mori-bnxt-intranode-testing
```

### Other AI Agents

The skills are standard markdown files. You can feed them as context/system prompts to any LLM-based agent (Copilot, Cline, Aider, etc.).
