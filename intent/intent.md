# Intent: tiny French tutor LLM

Status: draft. Date: 2026-10-10.

## Problem

A French learner at an intermediate level (A2-B1) has no always-available, private, offline tutor. Existing options are generic chatbots that are not specialized for language learning, paid APIs, or tools that require an internet connection. What is wanted is a small assistant that teaches in French — instructions, dialogs and explanations all in French — where the learner writes short instructions in French to say what to generate. It should explain French grammar and vocabulary, correct learner mistakes with French explanations, generate sample dialogues, and run locally on a laptop.

French-only is a deliberate immersion choice, accepted in writing: the target user is an intermediate learner (A2-B1) who can write short instructions and read explanations in French. A beginner (A1 and below) is out of scope for v1 — they would struggle to write instructions in French or to read grammar explanations written in French.

The project also has an educational purpose: learning to create and train an LLM from scratch on free infrastructure is a goal in itself, for the product owner and for anyone following the repository. The documented, reproducible training process is a deliverable alongside the model — the repository is written as a teaching artifact, recording decisions, estimates and failure modes, not just results.

What to expect from 125M x 2B tokens — a model of 125 million parameters trained over 2 billion tokens of French. 125M parameters is the model's size: its learnable weights, tiny by LLM standards (GPT-3 has 175 billion; SmolLM2-135M, the shape this project follows, was trained on 11 trillion tokens). 2B tokens is the size of the pretraining corpus: a token is a word or a subword piece, so this is roughly 1.5 billion words of French — the two numbers are linked by [Chinchilla's rule](https://arxiv.org/abs/2203.15556), which sizes ~125M parameters as optimal for ~2B tokens (~16 tokens per parameter) at this compute budget. At that scale, expect fluent French sentences, thin lexical knowledge beyond frequent words, and almost no metalinguistic knowledge from pretraining (web text rarely explains grammar) — nearly every tutoring behavior will come from imitating the SFT data. Quality therefore comes from the scope and consistency of the data, not its volume ([TinyStories](https://arxiv.org/abs/2305.07759), Eldan & Li 2023; phi-1, ["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644), Gunasekar et al. 2023).

Training a capable model is normally expensive. This project must prove that a useful, specialized tiny LLM can be created and trained from scratch using only free infrastructure.

## Proposed outcome

A tiny, open, French-only instruction-tuned language model ("French tutor") that:

- takes short instructions in French and produces French output — the model supports only French;
- has dialogs and explanations in French;
- explains French grammar rules and vocabulary in French, from a closed, canonical curriculum;
- corrects learner mistakes with explanations in French;
- answers simple questions about French usage whose answer is in the curriculum, in French;
- generates short, level-appropriate sample dialogues in French (single-turn generation; multi-turn chat is a non-goal);
- when the user writes in English (which will happen), replies politely in French, asking the user to continue in French — never produces English output; French input containing anglicisms or loanwords is answered normally;
- abstains politely, in French, when the request is outside its scope (translation requests, off-topic questions, beyond-curriculum grammar, rare vocabulary), with a pointer to what it can do;
- is safe to learn from: its own French is grammatical ([LanguageTool](https://languagetool.org)-checked, [GitHub](https://github.com/languagetool-org/languagetool)), it does not "correct" correct sentences, and its corrections are themselves valid French;
- runs locally on a MacBook Air (Apple Silicon, 8 GB RAM or more) via Ollama or MLX, fully offline;
- is created and trained from scratch (own tokenizer, own weights — no fine-tuning of an existing model) on free GPU infrastructure (Google Colab free tier T4 by default; Kaggle free tier as an option), with training split across multiple sessions using checkpoint/resume;
- is released with open data, code, and weights under a permissive license (target Apache 2.0 — a known, unsettled legal risk pending data-provenance review, see spec section 7), so anyone can retrain, reuse, or improve it.

There is no fallback to fine-tuning an existing model. Training from scratch is a goal in itself (educational purpose): if the model misses the quality bars, the response is to iterate on data and training, and the documented attempt is itself a deliverable. French-capable baselines (Qwen2.5-0.5B-Instruct, CroissantLLMChat-v0.1 — both Apache 2.0) are evaluated on the same suite for context only; the comparison never triggers a change of approach.

## Success criteria (v1)

Two tiers, because with no fallback the project needs a defined point at which the model is worth deploying to the owner's laptop even while iteration continues.

- Harm checks (floor for both tiers): >= 95% of eval responses LanguageTool-clean (grammar and spelling); 100% of `Correction : …` lines valid French; >= 90% of correct sentences correctly identified as correct (no false corrections).
- Usable v1 tier (deployable while iteration continues): harm checks + >= 70% correction accuracy + >= 65% explanation correctness.
- Target tier: >= 80% of grammar explanations factually correct (human rubric against curriculum answer keys); >= 80% of mistake corrections match reference corrections (automatic check, multiple references, reported per error type); sample dialogues rated >= 3.5/5 on a human rubric.
- Language policy: >= 99% of tutor-eval responses in French (the model is French-only); 100% of ~50 English-input probes answered with a polite French reply; 100% of ~30 out-of-scope probes answered with the abstention reply; French input containing anglicisms or loanwords answered normally (negative probes).
- No fallback: training from scratch is a goal in itself (educational purpose). The French-capable baselines are evaluated for context only. If the model misses the bars, the response is to iterate on synthetic data and training — never to switch to fine-tuning an existing model.

The project follows the AI-native SDLC playbook (https://claude.com/resources/articles/the-ai-native-sdlc-playbook) adapted to LLM development: every stage commits an artifact (`intent.md` -> `spec.md` -> `plan.md` -> code + evals -> deploy runbook -> feedback loop), and each accepted artifact triggers the next stage.

## Affected users and systems

- Primary user: a French learner at an intermediate level (A2-B1).
- Secondary audience: anyone learning to train an LLM from scratch on free infrastructure (the educational purpose — they follow the artifacts and code in this repository).
- Systems: Google Colab notebook (training, multiple sessions; CPU runtime for data prep) and/or Kaggle notebook (free GPU tier), Hugging Face Hub (datasets, model hosting, private checkpoint repo), Google Drive (scratch storage), local MacBook Air (Ollama / MLX inference), this git repository (artifacts and code).

## Constraints

- Training compute: free GPU tier only — Google Colab free tier (T4 GPU, 15 GB VRAM, no native bf16; fp16 with GradScaler, ~12 GB RAM, GPU availability not guaranteed) by default, with Kaggle free tier (30 GPU-hours/week) as an option to halve calendar time. Training is split into multiple sessions of ~3-4 h each; every session ends by saving a checkpoint (model + optimizer + scheduler + GradScaler state + data position) and uploading it to a private Hugging Face repo, and the next session resumes from the latest checkpoint. No training run may depend on a single session longer than ~4 h.
- Inference: MacBook Air (Apple Silicon, 8-16 GB unified memory), offline. Target: model well under 2 GB (F16 GGUF ~250 MB or Q8_0 ~135 MB), interactive response times (tokens/s checked in the runbook smoke test). Context is capped at 1024 tokens (the Modelfile sets `num_ctx 1024` and caps `num_predict`).
- Data: every training dataset must be open source with a permissive, non-share-alike license that permits training other models (Apache 2.0, MIT, CC-BY, CC0, ODC-By, or equivalent). No proprietary, non-commercial-only, or share-alike (e.g., CC-BY-SA) data. Public-domain text (expired copyright) is acceptable. Conjugation facts are not copyrightable, but the license of any lexicon used is checked.
- Budget: zero cost.
- Scope (v1): text-only; French only — instructions and output in French; no English output and no translation; short tutoring turns and single-turn generated sample dialogues (no long multi-turn conversations, no speech); always trained from scratch — no fine-tuning of an existing model, no fallback. Target variety: metropolitan French. Vocabulary scope: the top ~5k lemmas by corpus frequency; rarer words are routed to abstention.
