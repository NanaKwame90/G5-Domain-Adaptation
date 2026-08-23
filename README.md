# Privacy-Aware KTO Fine-Tuning of Qwen3.5-9B

This repository contains the code, data configurations, and evaluation pipeline for the **G5 Domain Adaptation project**. We investigate **Kahneman-Tversky Optimization (KTO)** applied to `Qwen/Qwen3.5-9B` to improve performance across static reasoning benchmarks, privacy question-answering (`privacy_qa`), and interactive dialogue games within the **Playpen/CLEMbench** framework.

---

## Repository Structure

```
g5-privacy-kto/
├── README.md
├── .gitignore
├── config_kto.yaml
├── run_benchmarks.sh
├── train_kto_new.py
├── aborts/
│   ├── abort_summary.csv
│   └── extract_aborts.py
├── configs/
│   ├── game_registry.json
│   └── model_registry.template.json
├── data/
│   └── train_kto.jsonl
├── privacy_qa/
│   └── privacy_qa_2000_instances.json
├── results/
│   ├── Qwen3.5-9B-KTO/
│   ├── raw.csv
│   ├── results (1).csv
│   ├── results.html
│   ├── results_baseline.xlsx
│   ├── results_baseline_privacy.xlsx
│   ├── results_static.csv
│   └── results_without.csv
└── src/
    ├── playpen/
    ├── .DS_Store
    ├── custom_adapter_backend_api.py
    ├── custom_api.py
    ├── fix_cache.py
    ├── game_master.py
    ├── kto_trainer.py
    ├── playpen_eval.py
    ├── process_data.py
    └── split_instances.py
```

---


## Key Methodology & Problem Formulation

* **The Problem:** Large language models often achieve high performance on static knowledge benchmarks but fail to maintain structured multi-turn dialogue protocols in interactive games, leading to format violations and early aborts (`Played_%`).
* **Kahneman-Tversky Optimization (KTO):** Instead of paired preference comparisons (like DPO), KTO operates on singleton binary feedback $y \in \{+1, -1\}$, leveraging prospect theory's asymmetry between the psychological penalty of undesirable outputs and the reward of desirable outputs.
* **Composite Evaluation Metric (`clemscore`):**
  $\text{clemscore} = \sqrt{\text{Played} \times \text{Quality}}$

---

## Training Setup & Hyperparameters

* **Base Model:** `Qwen/Qwen3.5-9B` (`bfloat16`, `enable_thinking=false`)
* **Parameter-Efficient Fine-Tuning (LoRA):** Rank $r = 8$, $\alpha = 16$, dropout $0.05$, targeting attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Updates $\sim$8.4M parameters out of 9.4B ($\sim$0.09%).
* **Trainer Configuration:** TRL `KTOTrainer`, effective batch size of 8, learning rate $5 \times 10^{-6}$, max 500 steps.

```
---


### Summary Scores Across Evaluation Suites

| Model | Evaluation Suite | Clemscore | Avg % Played | Avg Quality |
| --- | --- | --- | --- | --- |
| Qwen3.5-9B (Baseline) | Full CLEMbench | 3.57 | 24.92 | 14.33 |
| Qwen3.5-9B (Baseline) | + privacy_qa (2,000 inst.) | 4.35 | 28.49 | 15.28 |
| Qwen3.5-9B (Baseline) | + Static benchmarks | 20.84 | 100.0 | 20.84 |
| **Qwen3.5-9B-KTO** | Full CLEMbench | 4.66 | 11.58 | 34.52 |
| **Qwen3.5-9B-KTO** | + privacy_qa | 5.11 | 14.81 | 40.25 |
| **Qwen3.5-9B-KTO** | + Static benchmarks | **45.13** | **99.8** | **45.22** |

### Static Benchmark Breakdown (Qwen3.5-9B-KTO)

| Benchmark | % Played | Quality | Description |
| --- | --- | --- | --- |
| BBH | 100.0 | 0.0 | 23-task reasoning battery |
| CLadder | 100.0 | **61.0** | Causal ladder counterfactuals |
| EQBench | 99.0 | **72.12** | Emotional intelligence QA |
| IFEval | 100.0 | 54.0 | Instruction-following evaluation |
| MMLU-Pro | 100.0 | 39.0 | Academic multi-choice (12 domains) |

---

# Complete Setup, Installation, and Evaluation Guide

This guide provides step-by-step instructions from start to end for setting up the environment, training the KTO model, and executing evaluations for both the **Baseline** and **KTO** models across all **three evaluation variations**:

1. **Full CLEMbench suite** (24 interactive multi-turn dialogue games)
2. **CLEMbench + privacy_qa** (Full suite augmented with 2,000 privacy QA instances)
3. **Static benchmarks** (BBH, CLadder, EQBench, IFEval, MMLU-Pro run as single-turn episodes)

---

## 1. Installation & Environment Setup

##  Evaluation Framework Note (`CLEMbench`)

To keep this repository lightweight, the core **CLEMbench** framework is not bundled directly inside this repo. You must clone or set it up in your workspace prior to running evaluations. 

Quick setup for CLEMbench:
```bash
# Clone the official clembench framework into the src directory
git clone [https://github.com/clemgame/clembench.git](https://github.com/clemgame/clembench.git) src/clembench
pip install -e src/clembench

Clone the repository and set up a Python virtual environment containing PyTorch, TRL, and the Playpen/CLEMbench dependencies.

```bash
# Clone or navigate to your working directory
cd g5_adaptation

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies (PyTorch, TRL, Transformers, Accelerate)
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft trl pyyaml pandas scikit-learn

# Install Playpen and CLEMbench backend requirements
pip install -e src/playpen/clembench

```

---

## 2. Model & Game Registry Configuration

Ensure your local model registry (`model_registry.json`) points to both the baseline model (`Qwen/Qwen3.5-9B`) and your trained KTO checkpoint, with `enable_thinking=false` configured for both.

```json
{
  "models": {
    "qwen3.5-9b-base": {
      "path": "Qwen/Qwen3.5-9B",
      "enable_thinking": false
    },
    "qwen3.5-9b-kto": {
      "path": "./checkpoints/qwen3.5-9b-kto",
      "enable_thinking": false
    }
  }
}

```

---

## 3. Training the KTO Model

To generate the **Qwen3.5-9B-KTO** checkpoint from the baseline model using your singleton preference data (`train_kto.jsonl`) and LoRA configuration (`config_kto.yaml`), run:

```bash
python train_kto_new.py --config config_kto.yaml

```

* **Under the Hood:** This loads `Qwen/Qwen3.5-9B` in `bfloat16`, attaches LoRA adapters ($r=8, \alpha=16$) to attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`), and trains using TRL's `KTOTrainer` (effective batch size 8, learning rate $5 \times 10^{-6}$, max 500 steps).

---

## 4. Running Evaluations Across All 3 Variations

Evaluations are driven through the Playpen orchestration command (`clem run`). You will need to execute separate runs for the **Baseline** and **KTO** models across each of the three suites.

### Variation 1: Full CLEMbench Suite (24 Interactive Games)

* **Baseline Model Evaluation:**
```bash
clem run --model qwen3.5-9b-base --suite full

```


* **KTO Model Evaluation:**
```bash
clem run --model qwen3.5-9b-kto --suite full

```



---

### Variation 2: CLEMbench + privacy_qa Suite (Augmented with 2,000 Instances)

To include the privacy question-answering dataset (`privacy_qa_2000_instances.json`), register it into your Playpen game suite configuration, then execute:

* **Baseline Model Evaluation:**
```bash
clem run --model qwen3.5-9b-base --suite privacy_qa

```


* **KTO Model Evaluation:**
```bash
clem run --model qwen3.5-9b-kto --suite privacy_qa

```



---

### Variation 3: Static Benchmarks Suite (BBH, CLadder, EQBench, IFEval, MMLU-Pro)

To run the single-turn static knowledge benchmarks:

* **Baseline Model Evaluation:**
```bash
clem run --model qwen3.5-9b-base --suite static

```


* **KTO Model Evaluation:**
```bash
clem run --model qwen3.5-9b-kto --suite static

```



*(Note: You can automate full-suite batch evaluations using the provided `run_benchmarks.sh` script after adjusting its model parameters).*

---

## 5. Aggregating and Viewing Results

After evaluations complete, compute the final scores (percentage played, quality score, and composite `clemscore`) using the evaluation parser:

```bash
clem eval --results-dir src/playpen/results/

```

Results will be compiled into summary tables matching the benchmark metrics (such as reaching the peak static clemscore of **45.13** for the KTO model versus **20.84** for the baseline).
