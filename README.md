# G5 Domain Adaptation: KTO Fine-Tuning of Qwen3.5-9B

This repository documents the experiments for the **LM-Playschool Shared Task 2026**, in which `Qwen/Qwen3.5-9B` is fine-tuned using **Kahneman-Tversky Optimization (KTO)** on playpen-derived preference data and benchmarked across six evaluation conditions using the [Playpen](https://github.com/lm-playpen/playpen) / [CLEMbench](https://github.com/clp-research/clembench) framework.

---

## Repository Structure

```
g5_adaptation/
├── train_kto_new.py           # KTO training script (LoRA + TRL KTOTrainer)
├── train_kto.jsonl            # KTO training data (prompt/completion/label triples)
├── config_kto.yaml            # Training hyper-parameter reference config
├── run_benchmarks.sh          # Full-suite evaluation runner (KTO model)
├── model_registry.json        # Model registry for the Playpen evaluation backend
├── privacy_qa_2000_instances.json  # privacy_qa game instances (2 000 items)
├── src/
│   ├── kto_trainer.py         # (stub — training logic lives in train_kto_new.py)
│   └── playpen/               # Playpen workspace (clembench + model/game registries)
│       ├── model_registry.json
│       ├── game_registry.json
│       ├── clembench/         # CLEMbench game suite (submodule / clone)
│       └── results/           # Evaluation outputs (raw.csv, results.csv)
└── results/                   # Aggregated results (Qwen3.5-9B-KTO full suite)
```

---

## Experimental Conditions

Six conditions are evaluated, varying model (Baseline vs. KTO) and evaluation suite:

| # | Model             | Evaluation suite             |
|---|-------------------|------------------------------|
| 1 | Baseline Qwen3.5-9B | Full CLEMbench suite        |
| 2 | Baseline Qwen3.5-9B | + privacy_qa (2 000 instances) |
| 3 | Baseline Qwen3.5-9B | + Static benchmarks          |
| 4 | KTO Qwen3.5-9B    | Full CLEMbench suite         |
| 5 | KTO Qwen3.5-9B    | + privacy_qa (2 000 instances) |
| 6 | KTO Qwen3.5-9B    | + Static benchmarks          |

**Static benchmarks**: BBH, CLadder, EQBench, IFEval, MMLU-Pro (located in `clembench/static/`).  
**privacy_qa**: A custom game testing privacy question-answering, using 2 000 generated instances.

---

## Setup

### 1. Python environment

```bash
cd /home/users/kagyemang-dua/g5_adaptation/src/playpen
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -e .
pip install '.[trl]'
pip install peft
```

### 2. CLEMbench

The CLEMbench game suite is already cloned at `src/playpen/clembench/`.  
`game_registry.json` points to it:

```json
[{ "benchmark_path": "/home/users/kagyemang-dua/g5_adaptation/src/playpen/clembench" }]
```

Install CLEMbench requirements:

```bash
pip install -r src/playpen/clembench/requirements.txt
```

### 3. Model weights

```bash
export HF_HOME=/home/users/kagyemang-dua/huggingface_cache
# Qwen3.5-9B base weights are downloaded automatically on first use
# KTO fine-tuned weights are saved at: kto_qwen_stable_model/
```

### 4. GPU environment

```bash
export CUDA_VISIBLE_DEVICES=1          # single-GPU evaluation
# For training: CUDA_VISIBLE_DEVICES=3,2 (dual-GPU)
```

---

## KTO Training

### Training data

`train_kto.jsonl` contains preference triples derived from Playpen interaction logs:

```json
{"prompt": "...", "completion": "...", "label": true}   // preferred
{"prompt": "...", "completion": "...", "label": false}  // dis-preferred
```

The data covers CLadder causal-reasoning tasks and MMLU-Pro academic questions.

### LoRA configuration

| Hyper-parameter          | Value            |
|--------------------------|-----------------|
| Base model               | Qwen/Qwen3.5-9B |
| LoRA rank (`r`)          | 8               |
| LoRA alpha               | 16              |
| Target modules           | q_proj, k_proj, v_proj, o_proj |
| LoRA dropout             | 0.05            |
| Task type                | CAUSAL_LM       |

### KTO training configuration

| Hyper-parameter                  | Value    |
|----------------------------------|----------|
| Learning rate                    | 5 × 10⁻⁶ |
| Max steps                        | 500      |
| Per-device batch size            | 2        |
| Gradient accumulation steps      | 4        |
| Effective batch size             | 8        |
| Precision                        | bfloat16 |
| Gradient checkpointing           | enabled  |
| Save strategy                    | every 100 steps |

### Run training

```bash
cd /home/users/kagyemang-dua/g5_adaptation
source src/playpen/venv/bin/activate

python train_kto_new.py \
  --model_name_or_path /home/users/kagyemang-dua/models--Qwen--Qwen3.5-9B \
  --dataset_path train_kto.jsonl \
  --output_dir kto_qwen_stable_model \
  --learning_rate 5e-6 \
  --max_steps 500
```

Training log: `kto_qwen_stable.log`  
Output directory: `kto_qwen_stable_model/`

---

## Evaluation

### Model registry

Two models are registered in `src/playpen/model_registry.json`:

- **`Qwen3.5-9B`** — baseline, `thinking_mode=false`, loads from HuggingFace cache.
- **`Qwen3.5-9B-KTO`** — same base ID but loads fine-tuned weights from `kto_qwen_stable_model/` (update `huggingface_id` in the registry to point to the local path post-training).

### Conditions 1 & 4 — Full CLEMbench suite

```bash
cd /home/users/kagyemang-dua/g5_adaptation/src/playpen
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

# Baseline
clem run -m Qwen3.5-9B --suite all

# KTO
bash /home/users/kagyemang-dua/g5_adaptation/run_benchmarks.sh
# or run game by game:
for game in adventuregame hot_air_balloon codenames dond guesswhat imagegame \
            matchit matchit_ascii referencegame privateshared taboo \
            textmapworld textmapworld_graphreasoning textmapworld_specificroom \
            wordle wordle_withclue wordle_withcritic; do
    clem run -m Qwen3.5-9B-KTO -g "$game"
done
```

### Conditions 2 & 5 — + privacy_qa

Copy the 2 000-instance game file into the game directory, then run:

```bash
# Copy instances
cp /home/users/kagyemang-dua/g5_adaptation/privacy_qa_2000_instances.json \
   src/playpen/clembench/privacy_qa/in/instances.json

# Run baseline
clem run -m Qwen3.5-9B -g privacy_qa

# Run KTO
clem run -m Qwen3.5-9B-KTO -g privacy_qa
```

Logs: `privacy_qa_baseline_run.log`, `privacy_qa_run.log`

### Conditions 3 & 6 — + Static benchmarks

```bash
# Baseline
for game in bbh cladder eqbench ifeval mmlu_pro; do
    clem run -m Qwen3.5-9B -g "$game"
done

# KTO (included in run_benchmarks.sh)
for game in bbh cladder eqbench ifeval mmlu_pro; do
    clem run -m Qwen3.5-9B-KTO -g "$game"
done
```

### Score aggregation

```bash
playpen eval Qwen3.5-9B --suite all
playpen eval Qwen3.5-9B-KTO --suite all
# Outputs: results/results.csv, results/raw.csv, results/results.html
```

---

## Results

### Summary

| Model | Eval Suite | Clemscore | Avg %Played | Avg Quality |
|---|---|---|---|---|
| Qwen3.5-9B | privacy_qa (2 000 inst.) | 20.84 | 100.0 | 20.84 |
| Qwen3.5-9B-KTO | Static benchmarks | **45.13** | 99.8 | **45.22** |
| Qwen3.5-9B-KTO | Full CLEMbench suite | 5.11 | 14.81 | 34.52 |

### Static benchmark breakdown (KTO model)

| Benchmark | %Played | Quality Score |
|---|---|---|
| BBH | 100.0 | 0.0 |
| CLadder | 100.0 | 61.0 |
| EQBench | 99.0 | 72.12 |
| IFEval | 100.0 | 54.0 |
| MMLU-Pro | 100.0 | 39.0 |

### Baseline static benchmarks (Qwen3.5-9B)

| Benchmark | %Played | Quality Score |
|---|---|---|
| BBH | 100.0 | 0.0 |
| IFEval | 100.0 | 0.0 |
| Others | 0.0 | — |

### Privacy QA (2 000 instances)

| Model | %Played | Quality Score |
|---|---|---|
| Qwen3.5-9B (baseline) | 100.0 | 20.84 |
| Qwen3.5-9B-KTO | — | — |

> **Note:** The KTO model was not evaluated on privacy_qa in the current run set; combining static + privacy_qa results is left for future work.

---

## Key observations

1. **KTO substantially improves static benchmark performance** (clemscore 0 → 45.13 on static suite).
2. **EQBench and CLadder show the largest gains** (72.12 and 61.0 quality scores respectively).
3. **Interactive game performance drops** under KTO fine-tuning (full suite clemscore 5.11), suggesting that KTO training on static QA data does not transfer to interactive dialogue games.
4. **BBH quality score remains 0** for both baseline and KTO, indicating the benchmark may require dedicated training signal.
5. **Privacy QA baseline** (20.84 quality, 100% played) establishes a strong participation floor; KTO training is expected to improve this given improved instruction-following (IFEval: 54.0).

---

## File reference

| File | Purpose |
|---|---|
| `train_kto_new.py` | KTO training entry-point |
| `train_kto.jsonl` | Preference training data |
| `run_benchmarks.sh` | Full evaluation runner for KTO model |
| `src/playpen/run_sequential.sh` | Sequential game runner with 2-hour timeout |
| `privacy_qa_2000_instances.json` | privacy_qa game instances |
| `kto_qwen_stable.log` | Training log |
| `results/results.csv` | KTO full-suite scores |
| `src/playpen/results/results.csv` | Baseline full-suite scores |
