# Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis

> **Accepted as a Long Paper at [Canadian AI 2026](https://www.caiac.ca/en/conferences/canadianai-2026/home)**

This repository contains the code and experimental framework for our paper *"Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis"*. We show that encoding harmful prompts as coherent mathematical problems—using formalisms such as set theory, formal logic, and quantum mechanics—bypasses LLM safety filters at high rates (46–56% average ASR across eight target models and two benchmarks), while rule-based encodings that apply mathematical formatting without deep reformulation perform no better than unencoded baselines.

> **Disclaimer**: This project is intended solely for academic research on AI safety. The goal is to identify and understand vulnerabilities in LLM safety mechanisms so that more robust defenses can be developed.

## Authors

- **Haoyu Zhang** — Northeastern University (zhang.haoyu6@northeastern.edu)
- **Mohammad Zandsalimy** — Ansys
- **Shanu Sushmita** — Northeastern University

## Key Findings

- **LLM-based encodings** (Set Theory, Formal Logic, Quantum Mechanics) achieve 46–56% average ASR, dramatically outperforming **rule-based encodings** (Addition Equation, Conditional Probability, Symbol Injection) at 9–11% ASR, demonstrating that deep reformulation by a helper LLM—not mathematical notation itself—is the primary driver of attack success.
- Our novel **Formal Logic encoding** achieves ASR comparable to Set Theory overall and outperforms it on newer models (GPT-5, GPT-5-Mini).
- Mathematical encoding attacks are **robust to prompt repetition** (85.7% of experiments show |ΔASR| ≤ 5pp).
- Newer models (GPT-5, GPT-5-Mini) show substantially greater robustness but remain vulnerable.

## Attack Pipeline

The framework implements a three-stage pipeline:

1. **Encode** — A processing model (helper LLM or rule-based algorithm) transforms the harmful prompt into a mathematical formulation.
2. **Submit** — The encoded prompt, along with a strategy-specific instruction, is sent to the target model.
3. **Evaluate** — A judge model (GPT-5-Nano with HarmBench classifier) classifies the response as a successful jailbreak or refusal.

## Encoding Strategies

### LLM-Based (require a helper LLM for encoding)

| Strategy | Key | Description |
|---|---|---|
| Set Theory | `llm_set_theory` | Maps prompts to set operations and group compositions |
| Formal Logic | `llm_formal_logic` | Encodes prompts as first-order logic proof problems (novel) |
| Quantum Mechanics | `llm_quantum_mechanics` | Encodes prompts as quantum state evolution problems |

### Rule-Based (deterministic, no LLM needed)

| Strategy | Key | Description |
|---|---|---|
| Addition Equation | `non_llm_addition_equation_split_reassemble` | Splits prompt into segments presented as an addition equation |
| Conditional Probability | `non_llm_conditional_probability` | Frames prompt reconstruction as a conditional probability problem |
| Symbol Injection | `non_llm_symbol_injection` | Injects mathematical symbols at uniform intervals into the prompt |
| Baseline | `non_llm_baseline` | Sends original prompt unmodified (control condition) |
| Repeat | `non_llm_repeat` | Replays processed prompts from a previous experiment |

## Project Structure

```
├── main.py                      # Entry point
├── experiment_config/           # Experiment configuration JSONs
│   ├── config.json              # API experiment batch
│   └── cluster_config.json      # Cluster experiment batch
├── src/
│   ├── experiment/              # Experiment orchestration
│   ├── prompt_processor/        # Prompt encoding strategies
│   │   ├── processors/          # Individual processor implementations
│   │   ├── base_processor.py
│   │   ├── processor_factory.py
│   │   └── processor_type.py
│   ├── evaluation/              # Response evaluation (HarmBench classifier)
│   ├── dataloader/              # Dataset loading (HarmBench, JailbreakBench)
│   ├── llm_utils/               # LLM service layer (OpenAI, Anthropic, Google, vLLM)
│   └── data/                    # Datasets and experiment output
└── pyproject.toml
```

## Installation

### Prerequisites

- Python ≥ 3.12
- API keys for the LLM providers you wish to use

### Option 1: pip (recommended for API-only experiments)

```bash
git clone https://github.com/vacantfury/math_encoding_llm_jailbreaking.git
cd math_encoding_llm_jailbreaking
pip install -e .
```

### Option 2: Conda (recommended for cluster / local GPU experiments)

```bash
conda env create -f environment.yml
conda activate LLM_prompting
pip install -e .
```

### Environment Variables

Create a `.env` file or export the following variables for the providers you need:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export HF_HOME="/path/to/huggingface/cache"   # for local open-weight models
```

## Usage

### Running Experiments

```bash
python main.py experiment_config/config.json
```

### Configuration

Experiments are defined in JSON config files. Each task specifies a target model, dataset, encoding strategy, and evaluation method:

```json
{
    "target_model": "gpt-4o-mini",
    "dataloader": { "dataset_name": "harmbench_standard" },
    "processor": {
        "strategy": "llm_set_theory",
        "rephrase_first": false,
        "is_repeating": false,
        "experiment_dir": null
    },
    "evaluation": { "method": "harmbench" }
}
```

| Field | Options |
|---|---|
| `target_model` | `gpt-4o-mini`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash`, `llama3-8b-cluster`, `vicuna-13b-cluster`, etc. |
| `dataset_name` | `harmbench_standard` (159 prompts), `jailbreakbench` (100 prompts) |
| `strategy` | Any encoding strategy key from the tables above |
| `rephrase_first` | `true` / `false` — rephrase prompt before encoding |
| `is_repeating` | `true` / `false` — apply repeat post-processing (prompt duplication) |
| `method` | `harmbench` |

### Cluster Usage (SLURM + vLLM)

For running open-weight models (Llama-3-8B, Vicuna-13B) on a GPU cluster:

```bash
sbatch cluster_files_and_commands/run_cluster_experiment.sbatch
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026exposing,
  title={Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis},
  author={Zhang, Haoyu and Zandsalimy, Mohammad and Sushmita, Shanu},
  booktitle={Proceedings of the 39th Canadian Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgements

This work used the Northeastern University Research Computing (NURC) cluster for open-source model experiments.
