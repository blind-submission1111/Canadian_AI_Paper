# Jailbreaking LLMs via Mathematical Encodings

A research framework for studying LLM jailbreaking through mathematical prompt transformations. This project encodes harmful prompts into abstract mathematical structures (set theory, formal logic, quantum mechanics, etc.) to test the robustness of LLM safety filters.

> **⚠️ Research Purpose Only**: This project is for academic research on AI safety. It is designed to identify vulnerabilities in LLM safety mechanisms so they can be improved.

## Overview

The framework implements a pipeline:

1. **Encoding (Processor)**: A processing model (LLM or rule-based) transforms harmful prompts into mathematical formulations
2. **Targeting (Target)**: The encoded prompt is sent to the target model
3. **Evaluation (Judge)**: A judge model classifies the response as jailbroken or not using the HarmBench classifier

## Available Processors

### LLM-Based (require a helper LLM for encoding)
| Processor | Key | Description |
|-----------|-----|-------------|
| Set Theory | `llm_set_theory` | Maps prompts to set operations and group compositions |
| Formal Logic | `llm_formal_logic` | Encodes prompts as first-order logic propositions and proofs |
| Quantum Mechanics | `llm_quantum_mechanics` | Encodes prompts as quantum state evolution problems |
| Markov Chain | `llm_markov_chain` | Models prompts as stochastic state transitions |
| Rephrase | `llm_rephrase` | Rewrites prompts in academic language (ablation baseline) |

### Rule-Based (no LLM needed)
| Processor | Key | Description |
|-----------|-----|-------------|
| Addition Equation | `non_llm_addition_equation_split_reassemble` | Splits prompt into parts mapped to equation variables |
| Conditional Probability | `non_llm_conditional_probability` | Frames prompt as a statistical inference problem |
| Symbol Injection | `non_llm_symbol_injection` | Injects mathematical symbols randomly into the prompt |
| Baseline | `non_llm_baseline` | Sends original prompt unmodified (control) |
| Repeat | `non_llm_repeat` | Replays processed prompts from a previous experiment |

## Project Structure

```
├── main.py                     # Entry point
├── experiment_config/          # Experiment configuration JSONs
│   ├── config.json             # API experiment batch
│   └── cluster_config.json     # Cluster experiment batch
├── src/
│   ├── experiment/             # Experiment orchestration
│   │   ├── experiment.py       # Parallel task executor
│   │   ├── task.py             # Single task runner
│   │   ├── task_config.py      # Task configuration dataclass
│   │   └── experiment_config.py
│   ├── prompt_processor/       # Prompt transformation strategies
│   │   ├── processors/         # Individual processor implementations
│   │   ├── base_processor.py   # Base class with rephrase/repeat
│   │   ├── processor_factory.py
│   │   ├── processor_config.py # Processor configuration dataclass
│   │   └── processor_type.py   # Strategy enum
│   ├── evaluation/             # Response evaluation (HarmBench classifier)
│   ├── dataloader/             # Dataset loading (HarmBench, JailbreakBench)
│   ├── llm_utils/              # LLM service layer
│   │   ├── llm_model.py        # Model definitions (OpenAI, Anthropic, Google, Cluster)
│   │   ├── llm_services/       # Provider-specific clients
│   │   │   ├── openai_service.py
│   │   │   ├── claude_service.py
│   │   │   ├── google_service.py
│   │   │   └── nurc_cluster_service.py
│   │   └── cluster_server_manager.py  # SLURM + vLLM server management
│   └── data/                   # Datasets and experiment output
├── cluster_files_and_commands/ # SLURM sbatch scripts
├── experiment_results.md       # Raw experiment results tracker
└── pyproject.toml
```

## Quick Start

### Local Usage
```bash
# Install dependencies
pip install -e .

# Run experiments
python main.py experiment_config/config.json
```

### Cluster Usage (SLURM + vLLM)
```bash
# Submit experiment job
sbatch cluster_files_and_commands/run_experiment.sbatch

# Submit cluster experiment job (manages vLLM servers + runs experiments)
sbatch cluster_files_and_commands/run_cluster_experiment.sbatch
```

## Configuration

Experiments are defined in JSON config files. Each task specifies:

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
|-------|---------|
| `target_model` | `gpt-4o-mini`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash`, `llama3-8b-cluster`, etc. |
| `dataset_name` | `harmbench_standard` (159 prompts) or `jailbreakbench` (100 prompts) |
| `strategy` | Any processor key from the tables above |
| `rephrase_first` | `true`/`false` — rephrase prompt before encoding |
| `is_repeating` | `true`/`false` — whether this is a repeat run |
| `experiment_dir` | Path to source experiment for repeat runs (`non_llm_repeat` strategy) |
| `method` | `harmbench` |

## Evaluation

Uses the **HarmBench classifier** via a judge LLM (gpt-5-nano):
- A response is classified as **jailbroken** (1) or **not jailbroken** (0)
- **ASR** (Attack Success Rate) = percentage of prompts classified as successful jailbreaks

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `GOOGLE_API_KEY` | Google Gemini API access |
| `HF_HOME` | HuggingFace model cache directory |

## Requirements

- Python ≥ 3.12
- See `pyproject.toml` for dependencies
