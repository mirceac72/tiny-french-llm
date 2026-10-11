# AGENTS.md — working rules for this repository

Project: tiny French tutor LLM (see `intent/intent.md`).

## How development is organised

Development follows the [AI-native SDLC playbook](https://claude.com/resources/articles/the-ai-native-sdlc-playbook).

Every stage commits an artifact (`intent/intent.md` -> `intent/spec.md` -> `plan.md` -> code + evals -> deploy runbook -> feedback loop), and each accepted artifact triggers the next stage.

## Remote policy

- Single commit per push: each push to the remote contains exactly one commit. Squash all local work into a single commit before pushing — never push a run of small per-edit commits.
- Commits and pushes happen only on the product owner's explicit instruction.
