# Spec: tiny French tutor LLM

Status: draft, for product owner review. From: `intent/intent.md` (2026-10-10).

## 1. Requirements (traced from intent.md)

- R1: explain French grammar rules and vocabulary in French, from a closed, canonical curriculum (2.6).
- R2: correct learner mistakes with explanations in French.
- R3: answer simple questions about French usage whose answer is in the curriculum, in French; abstain politely otherwise (R10). "Answer simple questions about French usage" is deliberately scoped: it is the door through which hallucinations walk in.
- R4: generate short, level-appropriate sample dialogues in French (single-turn generation; multi-turn chat is a non-goal).
- R5 (English-input handling): when the user writes in English, reply politely in French, asking the user to continue in French; never produce English output. French input containing anglicisms or loanwords (`le week-end`, `un e-mail`, `le parking`) is legitimate French and must be answered normally (negative probes, section 5).
- R6: run offline on a MacBook Air (Apple Silicon, 8-16 GB) via Ollama or MLX; model well under 2 GB.
- R7: model created and trained from scratch (own tokenizer, own weights) on free GPU tier (Google Colab free T4 by default; Kaggle free tier as an option), split across multiple sessions with checkpoint/resume.
- R8: all training data open with permissive, non-share-alike licenses; model released under a permissive license (target Apache 2.0 — known, unsettled legal risk, see section 7).
- R9 (language policy): French only — the user writes short instructions in French; all model generations (dialogs, explanations, corrections) are in French. The model supports only French: no English output, no translation.
- R10 (abstention): when the request is outside the tutor's scope — a French instruction asking for English or translation, an off-topic question (history, maths, recipes), beyond-curriculum grammar, or rare vocabulary — reply politely in French that this is outside what the tutor covers, with a pointer to what it can do.
- Non-goals (v1): speech, long multi-turn conversations, any language other than French, English output, translation (EN<->FR), fine-tuning an existing model, beginner (A1) support.

## 2. Data spec

### 2.1 Tokenizer

- SentencePiece BPE, vocab 32,000, character coverage 0.9995. All special tokens are defined inside the 32,000 vocabulary (trainer `vocab_size` includes them).
- `normalization_rule_name="identity"`.
- `byte_fallback=True`, `remove_extra_whitespaces=False`, `add_dummy_prefix=False`.
- `user_defined_symbols=["\n"]` (newline is an ordinary piece, not a special token).
- Chat special tokens `<|bos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>` are `control_symbols`.
- BOS policy: every training sequence (pretraining document, SFT example) starts with `<|bos|>`; EOS is `<|end|>`.
- Trained once on a ~50-100M token sample of the French corpus (seeded, reproducible).

### 2.2 Pretraining corpus (~2B tokens, 100% French)

French-only by construction: no English data, no parallel data. Documents are packed into 1024-token sequences as `BOS … EOS` — BOS starts each document, EOS terminates it and separates documents (no length-drop filtering).

| Source | License | Est. tokens | Processing |
|---|---|---|---|
| FineWeb-2 French subset (`HuggingFaceFW/fineweb-2`) | ODC-By 1.0 | ~1.99B | plain text; packed as BOS … EOS |
| FineWeb-2 French, quality-filtered subset (`epfml/FineWeb2-HQ`) | ODC-By 1.0 | decay-phase share (50%) | reserved for the WSD decay-phase mix (4.1) |
| Conjugation tables (programmatic) | Facts not copyrightable; any lexicon license checked | decay-phase share (5%) | ~1-2k most frequent verbs x all persons x the curriculum's tenses, rendered as prose and tables; generated from a verb-conjugation library or the teacher with verification |
| Teacher-written textbook-style lessons + graded dialogues | follows teacher license (Apache 2.0) | decay-phase share (25%) | phi-1-style synthetic lessons; budgeted alongside synthetic generation (2.6) |
| Public-domain French (Project Gutenberg / Wikisource, expired copyright) | Public domain | decay-phase share (5%) | clean, correct, older-register French; small share only |
| `CohereLabs/aya_dataset` (fr subset: 1,422 examples) | Apache 2.0 | ~1M | instruction + response as plain text |
| `antoinelb7/alloprof` (French-language subject subset, see 2.3) | MIT | ~1-2M | question + explanation as plain text; also in the decay-phase mix (8%) |

### 2.3 Supervised Fine Tuning (SFT) corpus (~36-61k examples, French instruction -> French response)

| Source | License | Est. examples | Task |
|---|---|---|---|
| Synthetic tutor set (REQUIRED) | follows teacher license (Apache 2.0 teachers) | 30-50k default (volume is a go/no-go ablation; up to 100-150k if budget allows) | curriculum-conditioned French instruction -> French response: explanations, error-injection corrections, vocabulary drills, sample dialogues, refusals (1-2k), English-input handling, abstention (3-5k), two-turn share (10-20%) (R1-R5, R10) — primary source; full spec in 2.6 |
| `CohereLabs/aya_dataset` (fr subset) | Apache 2.0 | ~1.4k | French instruction -> French response (French dialog register) |
| `antoinelb7/alloprof` (French-language subject only) | MIT | ~5-10k | French question -> French explanation (educational dialogs). Filtered to the French-language subject category: the full 29k set covers maths, science, history, etc., which would teach a 125M model to answer out-of-scope subjects it will get wrong |

Chat template (special tokens from 2.1), including a system turn so the deployed system prompt is trained, not improvised. Every sequence starts with BOS (2.1):

```
<|bos|><|system|>\n{system}<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>\n
```

Fixed system prompt, in French (identical in training and in the Ollama Modelfile): « Tu es un tuteur de français patient. L'utilisateur écrit de courtes consignes en français ; réponds toujours en français. »

SFT data building and packing:

- Build each example by tokenizing the full rendered template string once — never tokenize segments separately and concatenate them (SentencePiece adds a leading-space marker to each segment it tokenizes, so segment-wise tokenization would disagree with deployment). Set the loss mask via character offsets.
- Pack whole examples only: an example is never split across the 1024-token boundary (a split would leave the second half without its instruction).
- Packed examples are isolated with a block-diagonal attention mask (the memory-efficient SDPA kernel accepts `attn_mask`) and `position_ids` reset per example, so a packed example cannot attend to unrelated earlier conversations (including repeated system prompts) — at inference every conversation starts in an empty context. Simpler alternative: no packing, examples grouped by length.
- Loss is computed on assistant tokens only, **including the `<|end|>` that terminates the response** — without it the model never learns to stop and runs to the context limit.
- 1-2 epochs (a small model trained repeatedly on synthetic data starts memorising it).

### 2.4 Filtering and dedup

- SFT: drop examples > 1024 tokens after tokenization. Pretraining: no length-drop — packing (2.2).
- Exact dedup by hash; near-dup via normalized-text hash (whitespace/punctuation-insensitive).
- Language sanity check (fastText lid: text must be French, threshold 0.9). fastText lid is unreliable on very short texts, so the check applies only to texts above ~20 tokens (or to the explanation part of a short correction).
- Strip boilerplate/navigation artifacts from web-derived text.
- Input noise: ~30% of SFT user turns are programmatically noised — accents stripped, ASCII `'`, non-breaking spaces removed, lowercased, occasional missing final punctuation. A2-B1 learners typing on an English keyboard write this way. The same noise is mirrored in a share of the tutor eval prompts (section 5). Accent policy: missing accents in the *sentence to correct* are errors to fix; missing accents in the *instruction* are tolerated silently.

### 2.5 Contamination control

- Monolingual data: 13-gram overlap — exclude documents sharing any 13-gram with the held-out FineWeb-2 fr eval slice (exact containment would drop every document containing a short eval sentence such as "Merci.").
- Tutor eval items: a 13-gram check cannot catch correction sentences, which are usually shorter than 13 words. Run a normalized exact-sentence match of every tutor eval item (prompts and reference answers) against the SFT data and Alloprof. On a hit, remove the *training* example when the hit is in synthetic data (the SFT set is large and synthetic — it gives way); remove the *eval item* only when the hit is in Aya or Alloprof (human data that should not be touched).
- Verified in data prep with a report. Eval sets are never mixed into training data.

### 2.6 Synthetic tutor data spec (REQUIRED — primary source of R1-R5, R10)

The synthetic set is the only source for the core tutoring behaviors (R1-R3), sample dialogues (R4), refusals (safety eval), English-input handling (R5) and abstention (R10). Quality comes from the scope and consistency of the data, not its volume (TinyStories, Eldan & Li 2023; phi-1, Gunasekar et al. 2023): at 125M x 2B tokens, nearly all tutoring ability comes from imitating this set.

**Curriculum (the single biggest quality lever).** A curriculum file, `curriculum/grammar.yaml`: ~80-120 topics, each with a CEFR level (A1-B2), a canonical rule in simple French (2-4 sentences, in the project's own words, with the authoritative source cited for verification), 3-5 canonical examples, and the typical English-speaker error. The teacher is conditioned on the rule — generation is "apply or rephrase this rule for this example", not "recall this rule" — which raises the teacher ceiling (applying a given rule is far easier than recalling it) and makes all examples agree with each other (a 125M student does not reconcile contradicting paraphrases; it learns a blurred average and produces confident, slightly wrong rules). The eval answer keys are derived from the same file: the 100 explanation eval items are held-out *phrasings* of curriculum topics, judged against the canonical rule (section 5). Copyright: published grammars are reference-only (fair use for private evaluation); the rules that enter training are in the project's own words.

**Teacher and serving.** Served with vLLM (AWQ or GPTQ checkpoint) on the T4 — bitsandbytes `generate` throughput is a few hundred tokens/s at best, which would make the budget 30-150 hours. The teacher is Mistral-Nemo-Instruct-2407 (12B, Apache 2.0, fits the T4 at 4-bit, French is its home language), used for all formats — drills, dialogues, explanations, correction explanations, English-input handling, refusals. The teacher's score is reported as context; the student's bars are the two tiers of section 5.

**Volume.** Default 30-50k curriculum-conditioned examples. Volume is a go/no-go ablation (30-50k vs 100-150k), not an assumption: at 125M parameters, 30-50k consistent examples may train as well as 150k free-form ones. Generation budget: 3-4 GPU sessions (realistic with vLLM). Provenance recorded in the dataset card.

**Formats.**

- (a) Explanation — fixed structure: rule, two examples, one contrast with the English-speaker error. A fixed shape is learnable at 125M and makes the examples LanguageTool-checkable.
- (b) Correction — built by construction, not by asking the teacher to invent mistakes (LLMs produce stereotyped, unrealistic learner errors and sometimes a "correction" that is itself wrong). Start from correct French sentences (corpus sentences that pass LanguageTool, or teacher-written level-appropriate sentences); inject errors by rule, per error type; the reference correction is then correct by construction and each item carries an error-type label (accuracy is reported per type). The teacher writes only the explanation, conditioned on the rule and the label. A teacher-generated share is kept for naturalness, verified with LanguageTool. The set includes correct sentences with no error, with the fixed output form `Correction : aucune. La phrase est correcte.` — the model must learn not to "correct" correct sentences. Error taxonomy (English-speaker mistakes prioritized): gender agreement, past-participle agreement, `avoir`/`être` auxiliary swap, negation (drop or mis-placement), preposition substitution, anglicism/faux-ami substitution, accents and spelling, homophones (`a`/`à`, `et`/`est`, `ou`/`où`, `ce`/`se`, `son`/`sont`), elision and contraction (`le` -> `l'`, `de le` -> `du`, `à les` -> `aux`).
- (c) Vocabulary drill — items constrained to the top ~5k lemmas by corpus frequency (A2-B1 productive vocabulary is roughly 2-4k words); rarer words are routed to the abstention behavior (format (g)).
- (d) Sample dialogue — short, single-turn, level-tagged, filtered by the level proxy (below). Fixed typographic convention: one turn per line, French dash `—` introducing each turn, speaker labels (`A :`, `B :`), level tag on the first line.
- (e) Refusal — 1-2k examples of polite French refusals for disallowed requests (matches the safety eval).
- (f) English-input handling — user writes in English, model replies politely in French asking to continue in French.
- (g) Abstention — 3-5k examples: French instructions asking for English or translation, off-topic questions, beyond-curriculum grammar, rare vocabulary -> a polite French reply that this is outside what the tutor covers, with a pointer to what it can do.
- (h) Two-turn examples — 10-20% of the set: a follow-up turn ("Donne un autre exemple", "Et au pluriel ?") with its answer. The deployed chat keeps conversation history (Ollama renders it into the template), so the model must see a second turn; this does not reopen the multi-turn non-goal, which is about long conversations.

**Level proxy.** Lexical coverage against a frequency list derived from the corpus (>= 95% of tokens within the top 2k lemmas for A2, top 4k for B1) plus mean sentence length. Used to filter generated dialogues and reported as a metric in the dialogue eval. Target variety: metropolitan French — Alloprof is Québec French, so the dialogue rubric flags Québec-specific vocabulary.

**Input noise.** ~30% of user turns are programmatically noised (2.4); the same noise is mirrored in a share of the tutor eval prompts.

**Diversity and dedup.** Prompts are varied through a template registry (seeded sampling over topics, CEFR levels, registers, sentence frames), conditioned on the curriculum rule; exact dedup by hash plus normalized-text dedup (2.4). Eval prompts are disjoint from the generation templates (hash dedupe + template registry, section 5).

**Quality filtering.** LanguageTool (grammar and spelling) on every generated response and on every correction; language-ID filter drops non-French generations (catches stray non-French text the teacher occasionally emits); level proxy on dialogues; examples failing filters are dropped, not repaired.

## 3. Model architecture

Llama-style decoder-only transformer in the SmolLM2-135M shape (deep and narrow, per MobileLLM/SmolLM2):

| Parameter | Value |
|---|---|
| Parameters | ~125M (124.6M recomputed: 30 layers x 3.54M + 18.4M tied embeddings) |
| d_model | 576 |
| Layers | 30 |
| Heads | 9 (head_dim 64) |
| KV heads | 3 (GQA) |
| FFN | SwiGLU, d_ff 1536 |
| Vocab | 32,000 (tied embeddings; a multiple of 128, suits tensor cores; 24k is the fallback if rare-token glitches show up at 2B tokens) |
| Context | 1024 |
| Norm | RMSNorm (no QK-norm — it would take the model out of `LlamaForCausalLM` and break the no-custom-code export) |
| Position | RoPE, base 10000 (context is 1024; a larger base is only needed for context extension) |
| Training precision | fp16 compute via AMP autocast with fp32 master weights (never `model.half()`) + GradScaler (the T4 has no native bf16); loss softmax in fp32 (the attention kernel already accumulates in fp32) |
| Attention | memory-efficient SDPA pinned with `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` so a fallback fails loudly instead of silently using the math kernel (no flash on the T4's Turing architecture); K/V heads expanded explicitly (`repeat_kv`) before the call — `enable_gqa` is not relied on |
| Loss | chunked cross-entropy (or a fused linear + cross-entropy kernel): the full fp32 logits over the 32k vocabulary are never materialized; z-loss ~1e-4 on the output logits (controls logit drift; adds nothing to the exported model) |
| Gradient checkpointing | off (activations ~6 GB at micro-batch 8 x 1024 — fits in VRAM) |
| Compilation | `torch.compile` for throughput (30 narrow layers mean small matmuls and many kernel launches) |

Memory budget at micro-batch 8 x 1024 (estimates from arithmetic, corrected by measurement in the calibration session): weights + gradients + Adam state ~2 GB; activations ~6 GB (about 0.7 MB per token over 30 SwiGLU layers); logits small with the chunked loss. Total ~10-11 GB — fits the T4's 15 GB with headroom. Micro-batch 16 would need ~18-19 GB and does not fit; micro-batch 8 x accumulation 4 is the default.

Rationale: depth over width is proven at this scale (SmolLM2-135M: 30 layers x width 576, 9/3 heads); Chinchilla sizing for the compute budget (~125M params x ~2B tokens, ~16 tokens per parameter; training needs ~1.5-1.7e18 FLOPs); a smaller model also means faster MacBook Air inference.

## 4. Training spec

### 4.1 Phases

1. Pretraining: causal LM on 2.2 corpus, ~2B tokens, ~61k steps.
2. SFT: chat-format fine-tuning on 2.3 corpus (French instruction -> French response), 1-2 epochs. Estimate: ~36-61k examples x ~350 tokens = ~13-21M tokens per epoch; 1 epoch ≈ 0.4-0.65k steps, 2 epochs ≈ 0.8-1.3k steps (~0.5-2 h). Checkpoint selection for SFT: select by eval score, not training loss; keep both the decayed and the last stable checkpoint.

WSD decay-phase data mix (fixed at 20% of pretraining — quality, not throughput, is the goal here): quality-filtered FineWeb-2 fr 50%; teacher-written textbook-style lessons and graded dialogues 25%; programmatically generated conjugation tables 5%; Alloprof (French-subject) 8%; public-domain French 5%; tutor-style synthetic text 7%. The curated (non-web) share is a ~100-200M-token mix repeated within the 400M-token decay phase — repeating a smaller curated set costs little (Muennighoff et al., 2023).

### 4.2 Hyperparameters

| | Pretraining | SFT |
|---|---|---|
| Optimizer | AdamW (fused) | AdamW (fused) |
| Schedule | WSD: warmup 2000 steps, stable at the peak LR (default 6e-4, set by the calibration sweep), decay over the final 20% when stopping | cosine 1e-4 -> 1e-5 |
| Weight decay | 0.1 (excludes RMSNorm weights and the tied embedding — decay applies to the linear weights only) | 0.1 (same exclusions) |
| Betas | (0.9, 0.95); the calibration sweep considers raising beta2 toward 0.99 at this batch size | (0.9, 0.95) |
| Grad clip | 1.0 | 1.0 |
| Batch | 32 seqs x 1024 = 32,768 tokens/step (micro 8 x accum 4 — micro-batch 16 does not fit the T4) | same |
| Precision | fp16 compute (AMP autocast, fp32 master weights) + GradScaler | same |
| Gradient checkpointing | off | off |
| Dropout | 0 | 0 |

WSD note: the stable phase can be extended or cut at any session boundary; the decay phase runs only when training is stopped (or at the 2B-token target). This replaces cosine decay, which requires the total step count up front. Decay shape: linear by default; a 1−√ shape is an option (Hägele et al., 2024).

fp16 stability (training from scratch in fp16 with a 32k-token batch is spike-prone):

- AMP autocast with fp32 master weights — never `model.half()`.
- Initialization: normal init, std 0.02 for embeddings and linear weights; output projections scaled down (std / √(2 × layers)) — this matters for a 30-layer stack in fp16.
- z-loss ~1e-4 on the output logits.
- The `GradScaler` state is saved in every checkpoint (alongside model + optimizer + scheduler + data position).
- The loss scale and the number of skipped (overflow) steps are logged every 50 steps.
- The loss softmax runs in fp32 (the attention kernel already accumulates in fp32).
- The calibration session (first pretraining session) runs a short learning-rate sweep — 3e-4, 6e-4 and 1e-3 at equal token counts, with beta2 toward 0.99 included — instead of a single check (6e-4 is GPT-3-small's peak LR, used there with a batch ~16x larger), and sets the peak LR. It also records `torch.cuda.max_memory_allocated()` and the measured tokens/s, correcting the estimates in sections 3 and 4.4.

### 4.3 Multi-session mechanics (free GPU tier)

- Checkpoint every 500 steps: model + optimizer + scheduler + GradScaler state + explicit data position (step x batch), ~1.5 GB each (fp32 weights + two Adam states). Written to local disk first (copy + rename, atomic locally), then uploaded to a private Hugging Face Hub repo. Keep only the last 3.
- Hub storage: HF repos are git repositories — deleting old checkpoints does not free history (~120 checkpoints x ~1.5 GB = ~180 GB). After each upload, squash the repo history (`HfApi.super_squash_history`); alternative: one throwaway branch or repo per checkpoint, deleted after upload. A Hub commit is already atomic, so no copy+rename is needed for the Hub step.
- Uploads run in the background (`run_as_future=True`) so a 1.5 GB upload never pauses training.
- On load, validate the latest checkpoint and fall back to the previous one (a session can be killed mid-write).
- Custom training loop: HF Trainer's `resume_from_checkpoint` is not used with a streaming dataset (on resume it skips through every batch already seen); the data position is stored explicitly in the checkpoint instead of relying on RNG state.
- The pre-tokenized dataset (~4-6 GB) lives on Drive and is copied to local disk at the start of each session.
- Data order is deterministic (fixed seed); metrics (loss, LR, tok/s, loss scale, skipped steps, ETA) appended to a JSONL log every 50 steps.
- The notebook takes `TARGET_STEP` (or `SESSION_HOURS`); it resumes from the latest checkpoint and stops cleanly at the target, saving a final checkpoint.
- Early go/no-go check: at ~300-500M pretraining tokens, copy the checkpoint, run a short WSD decay and a quick SFT on the copy, and run the full eval suite — while the main run continues from the stable checkpoint. A second checkpoint eval at ~1B tokens gives a trend instead of a single point (the quality gap between a 300M-token and a 2B-token checkpoint is large, so 55% at 400M may be exactly on track for 80% at 2B — the threshold is interpreted against the trend). An SFT-only floor — the same quick SFT on a randomly initialized model, no pretraining — shows how much pretraining contributes versus pure imitation, one of the most instructive numbers the educational deliverable can record. The go/no-go also runs the synthetic-volume ablation (30-50k vs 100-150k examples, 2.6). Decision rule: if the early checkpoint scores below 60% on correction accuracy or explanation correctness (well under the v1 bars — at 125M x 2B tokens, most tutoring ability comes from imitating the synthetic SFT data, so the 80% bar is the main capacity risk), intervene in week 1: improve the synthetic data (curriculum, generation prompts), extend pretraining, and re-run the check. The 60% threshold is a heuristic, revisited after the first go/no-go run.
- Session plan: data prep 1 CPU session (no GPU quota); synthetic data generation 3-4 GPU sessions (vLLM, the teacher, 2.6); pretraining ~50-80 GPU-hours (est. 7-12k tok/s, confirmed in the calibration session) = 15-22 sessions of 3-4 h; SFT ~0.5-2 h, 1 session; baseline evals ~1 session; early go/no-go ~1-2 sessions. Total ~20-30 GPU sessions. Calendar estimate 4-6 weeks with free-tier cooldowns on Colab alone; Kaggle's free 30 GPU-hours/week roughly halves the calendar time at zero cost. WSD makes an early stop (e.g., 1.5B tokens) safe at any session boundary.

### 4.4 Hardware

- Training: Colab free T4 (15 GB VRAM, ~12 GB RAM) by default; Kaggle free tier (30 GPU-hours/week) as an option. Model+optimizer+activations ~10-11 GB at micro-batch 8 — fits with headroom (estimate; the calibration session records `torch.cuda.max_memory_allocated()` and corrects this figure).
- Data prep: Colab CPU runtime or the MacBook (no GPU quota).
- Inference: MacBook Air, Ollama (GGUF) or MLX-LM.

## 5. Evaluation spec

| Eval | Set | Metric | Bar |
|---|---|---|---|
| Output grammaticality (harm) | every eval response | LanguageTool (grammar and spelling, not style): zero errors. LanguageTool misses some agreement errors, so "zero errors" is necessary, not sufficient; the human rubric still gates | >= 95% of responses clean |
| Correction-line validity (harm) | every correction response | LanguageTool on the `Correction : …` line — a correction must not introduce an error | 100% valid |
| No false corrections (harm) | ~50 correct sentences in the correction set | fixed form `Correction : aucune. La phrase est correcte.` | >= 90% |
| Explanation correctness | 100 prompts = held-out phrasings of curriculum topics; answer key = the canonical rule from `curriculum/grammar.yaml` | human rubric against the key, split in two: correctness (vs the key) and comprehensibility for a B1 reader (the owner — exactly the target reader, fully qualified for that half) | target >= 80% correct; usable tier >= 65% |
| Correction accuracy | 500-1000 items: synthetic error-injection items (labeled per error type) + ~50 correct sentences + 100-200 of the owner's own real mistakes (B1), corrected by a fluent speaker or a strong model plus LanguageTool — an out-of-distribution test the synthetic pipeline cannot game | automatic normalized match of the `Correction : …` line, multiple references allowed; per-error-type reporting; the LanguageTool-alone baseline must be beaten on the same items | target >= 80%; usable tier >= 70% |
| Sample dialogues | ~50 prompts (situation + level) | human rubric, mean score; level proxy reported; fixed dialogue format (2.6); Québec-specific vocabulary flagged (target: metropolitan French) | >= 3.5/5 |
| Vocabulary | ~40 items: 30 inside the top ~5k lemmas, 10 outside | correct meaning for in-scope items; out-of-scope items must abstain or hedge | reported; out-of-scope abstain/hedge |
| Two-turn follow-up | ~20 two-turn items | human rubric, mean score | >= 3.5/5 |
| English-input handling | ~50 probes written in English + ~20 negative probes (French sentences containing anglicisms or loanwords: `le week-end`, `un e-mail`, `le parking`) | language ID + behavior: polite French reply asking to continue in French; negative probes must be answered normally (R5 must not over-fire) | 100% |
| Abstention (out-of-scope) | ~30 probes: translation requests, off-topic questions, beyond-curriculum grammar, rare vocabulary | polite French out-of-scope reply with a pointer to what the tutor can do (R10) | 100% |
| Language policy | tutor eval set responses | language ID (fastText lid), applied to responses above ~20 tokens (or the explanation part) | >= 99% of responses in French |
| Language modeling | held-out FineWeb-2 fr slice + a domain held-out slice (synthetic tutor text + Alloprof) | perplexity, track trend; domain loss is the earliest predictor of how well SFT will take; baseline comparison in bits per byte (perplexities over different tokenizers are not comparable) | no absolute bar; track trend |
| Safety | ~50 hand-written disallowed probes + ~20 benign-but-sensitive probes (a sentence to correct that mentions `tuer le temps`, `une attaque de panique`, `la guerre de 1870`) | human review: polite refusal in French, no compliance; benign probes must be answered normally (over-refusal is measured) | no compliance; benign probes answered normally |
| Automatic failure metrics | all eval generations | degenerate-repetition rate (any 4-gram repeated 3+ times); truncation rate (hit `num_predict` without `<|end|>`) — both common tiny-model failures, neither otherwise measured | reported each run; truncation < 5% |
| LLM judge (iteration signal) | explanation eval items | the teacher judges explanations against the answer key in the prompt; calibrated once against the owner's ratings on ~30 items, agreement reported | the human rating remains the gate for the final v1 number |
| Contamination | data prep report | 13-gram overlap vs held-out fr slice; normalized exact-sentence match of every tutor eval item vs SFT data and Alloprof | 0 overlaps; removal rule per 2.5 |
| Baseline (informational) | Qwen2.5-0.5B-Instruct + CroissantLLMChat-v0.1 (both Apache 2.0, French-capable), same suite, same deployed stack | same metrics; LM comparison in bits per byte | reported for context only |

- Two tiers: the harm checks are the floor for both tiers. Usable v1 tier = harm checks + >= 70% correction accuracy + >= 65% explanation correctness — the point at which the model is worth deploying to the owner's laptop even while iteration continues. Target tier = 80% explanation correctness / 80% correction accuracy / 3.5 dialogues.
- The eval suite runs through the deployed stack — the GGUF with the Modelfile's decoding settings (section 6) — not through the HF model in fp32 with greedy decoding, so the reported numbers describe the model the owner uses.
- 95% intervals are reported next to point estimates. At n=100 and p≈0.8 the interval is about ±8 points; two data mixes at 76% and 82% are not distinguishable — iteration comparisons lean on the automatic sets (the correction set is grown to 500-1000 items because error injection makes it nearly free).
- Per-type and per-topic reporting: correction accuracy per error type, explanation correctness per curriculum topic — a "what this model is good and bad at" table, both a quality tool and part of the educational deliverable.
- Correction-accuracy normalization handles French typographic spaces: the special spaces before `; : ! ?` and inside `« »` are normalized before matching, or correct answers will fail to match.
- Tutor eval prompts are disjoint from the synthetic generation templates (hash dedupe + template registry).
- No translation eval: translation is out of scope for v1.
- Cadence: early go/no-go at ~300-500M pretraining tokens (decay + quick SFT on a checkpoint copy, full suite; second checkpoint eval at ~1B tokens for trend; SFT-only floor on a randomly initialized model; main run continues); perplexity after pretraining; full suite after SFT. Results committed under `evals/` in the repo.

## 6. Deployment spec

- Export as Hugging Face `LlamaForCausalLM` with a SentencePiece `tokenizer.model`, so llama.cpp conversion, `ollama create`, and MLX all work without custom code.
- GGUF default F16 (~250 MB) or Q8_0 (~135 MB) for 125M parameters; 4-bit avoided — it hurts small models most. MLX fp16 or 8-bit.
- Special tokens in GGUF: Ollama renders the chat template as text and llama.cpp tokenizes it, so the chat special tokens (`<|bos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`) are marked as added tokens with type CONTROL in the GGUF metadata — llama.cpp then parses them from the rendered template. `\n` is a normal (user-defined) token, so it prints. This slightly relaxes the "literal user text cannot produce them" guarantee at the llama.cpp layer (section 2.1); accepted trade-off.
- `<|end|>` is the EOS token and a stop sequence in the Ollama Modelfile.
- Context: the Modelfile sets `PARAMETER num_ctx 1024` and caps `num_predict` (e.g., 512) — context 1024 is a hard limit (RoPE base 10000, no extension) and Ollama's default `num_ctx` is larger.
- Decoding (Ollama Modelfile): `PARAMETER temperature 0.3` (Ollama's default 0.8 is far too high for a 125M model — tiny models degrade sharply with sampling noise; dialogues tolerate up to ~0.5, so the owner can raise it for dialogue use), `PARAMETER repeat_penalty 1.0` (the default 1.1 penalizes repeating the user's tokens, but a correction is a near-copy of the user's sentence and explanations quote it), `PARAMETER top_p 0.9`. The eval suite runs through the GGUF with these settings (section 5).
- Ollama Modelfile: chat template from 2.3 including the `<|system|>` turn with the identical French system prompt.
- Smoke test (runbook): tokenize the same rendered template with the SFT data builder and with the GGUF tokenizer and compare token IDs — HF and GGUF can agree with each other and both disagree with training if SFT examples were built by tokenizing segments separately (2.3). The test text includes ordinary French with accents, `l'`/`qu'`, `’`, guillemets `« »` and non-breaking spaces — not only the template. Also check generation speed (tokens/s) — at 125M parameters this passes easily, but the number is recorded.
- Runbook: `docs/runbook-macbook-air.md` (Stage 5) covering install, model load, smoke test.

## 7. Areas of concern (flagged)

1. Absolute quality is still modest: 125M x 2B tokens is Chinchilla-optimal for the budget, but a tiny from-scratch model will make grammar mistakes. The eval suite measures factual correctness precisely because confident-but-wrong explanations are the main learner risk — and adds harm-ordered checks (the model's own French must be grammatical; a correction must not introduce an error; no false corrections), because for a learner, errors are asymmetric: a tutor whose own French contains mistakes, or that "corrects" a correct sentence, does damage, while a missing explanation does not. If the bars are missed, the response is to iterate on data and training; the early go/no-go check (4.3) surfaces the risk in week 1, and the documented attempt is itself a deliverable.
2. Language policy: the model is French-only by construction (French-only pretraining and SFT), which lowers the drift risk; the >= 99% French check still gates every eval run (residual risk: English leakage from web data or code-switching). English input is handled by design (R5): a polite French reply, never English output — and R5 must not over-fire on legitimate French containing anglicisms (negative probes).
3. Free-tier availability: ~20-30 GPU sessions; calendar 4-6 weeks with Colab cooldowns, roughly halved with Kaggle. Mitigations: 500-step checkpoint cadence, small per-session targets, resume, WSD early stop, background uploads with squashed Hub history.
4. Throughput, memory and eval numbers are estimates from arithmetic, not measurements; the calibration session records `torch.cuda.max_memory_allocated()` and measured tokens/s and corrects them (the token target is adjusted if needed — WSD makes the adjustment safe).
5. Legal: all data sources are permissive non-share-alike (Apache 2.0, MIT, ODC-By 1.0) or public domain. The Apache 2.0 release of the weights is a target, not settled: data-provenance review is pending; until then the weights' license is a known, unsettled legal risk. Synthetic data: Apache 2.0 teachers only; provenance recorded in the dataset card. The decay-phase sources (quality-filtered FineWeb-2 subset, conjugation lexicon) must pass the same license check before use. Curriculum text is original wording; published grammars are reference-only.
6. Synthetic data quality: the synthetic set is the largest SFT source and carries R1-R5 and R10; teacher-generated explanations can be wrong. Mitigated by the canonical curriculum (2.6 — the teacher applies a given rule rather than recalling it), error-injection corrections (reference corrections correct by construction), LanguageTool filtering of every response, the correctness eval against curriculum answer keys, and mixing with human-annotated data (Aya, Alloprof).
7. The eval measures the deployed model: the suite runs through the GGUF with the Modelfile's decoding settings (section 6), so reported numbers describe the model the owner uses — but decoding settings are themselves a quality lever (temperature 0.3, repeat_penalty 1.0), and the two are tuned together.

## 8. Open questions

1. Synthetic tutor data volume: default 30-50k curriculum-conditioned examples; volume is a go/no-go ablation (30-50k vs 100-150k), not an assumption. Confirm.
2. Resolved: explanation-correctness answer keys are derived from the curriculum file (`curriculum/grammar.yaml`) — the 100 eval items are held-out phrasings of curriculum topics, judged against the canonical rule; nobody has to be a grammarian at rating time.
3. Token-target cut point if GPU sessions become scarce: default cut to ~1.5B tokens + WSD decay. Confirm when needed.
4. Compute platform: Colab free tier (T4) by default; mix in Kaggle free tier (30 GPU-h/week) if Colab cooldowns stretch the calendar. Decide after the first sessions.
5. License checks before use: the quality-filtered FineWeb-2 subset (decay mix) and any conjugation lexicon.
