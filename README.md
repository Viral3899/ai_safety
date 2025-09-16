# AI Safety POC

A practical, modular proof-of-concept for multi-model AI safety in conversational systems. It includes data pipelines, classical and transformer-backed detectors, evaluation utilities, and multiple demos (CLI, chat simulator, and scripts) to showcase end-to-end safety workflows.

## Highlights

- Multi-detector safety layer: abuse, crisis, content filtering, and escalation
- Real-time oriented orchestration with actionable interventions
- Reproducible data pipeline with Kaggle + synthetic data
- Training + evaluation scripts with saved artifacts and metrics
- Simple to run demos (no heavy infra required)

## Repository Layout

```
config/                 # Project settings + per-model configs
 data/                  # Synthetic + processed datasets
 demo/                  # CLI + web + chat simulator demos
 evaluation/            # Metrics, bias, and integration checks
 scripts/               # Entrypoints for training/evaluation/demo
 src/                   # Core library (models, safety system, utils)
 trained_models/        # Saved model artifacts + training metrics
 notebooks/             # Optional exploration + experiments
```

Key modules:
- `src/core/`: `base_model.py`, `preprocessor.py`, `safety_result.py`
- `src/models/`: classical and Hugging Face variants for all detectors
- `src/safety_system/`: `safety_manager.py`, `realtime_processor.py`, `intervention_handler.py`
- `src/utils/`: `bias_mitigation.py`, `metrics.py`, `data_generator.py`

## Theoretical flow (conceptual)

- Problem framing → define unsafe classes and intervention goals
- Data lifecycle → acquire, curate, label, split with leakage prevention
- Preprocessing → deterministic cleaning + vectorization/embeddings
- Modeling → per-task detectors with calibration and thresholds
- Evaluation → performance, robustness, and subgroup fairness
- Decision fusion → aggregate detector risks, resolve conflicts by policy
- Interventions → redact, warn, block, or escalate with concise reasons
- Serving → batch and real-time paths; warm loads and fallbacks
- Monitoring → telemetry, drift detection, human-in-the-loop feedback
- Governance → versioned configs/models; privacy and risk management

## Setup

- Python 3.9+ recommended
- Windows (Git Bash/PowerShell) or macOS/Linux

Create environment and install:
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Git Bash/CMD: source .venv/Scripts/activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

(Optional) Configure Kaggle if downloading datasets:
- Place your `kaggle.json` in the repo root (already present in this project layout)

## Data pipeline

Run the end-to-end sequence (adjust as needed):
```bash
# 1) (Optional) Download public datasets from Kaggle
python download_kaggle_data.py

# 2) Normalize/merge and generate robust processed data
python process_downloaded_data.py
python robust_data_pipeline.py

# 3) Prepare train/test splits
python prepare_training_data.py
```
Outputs:
- `data/processed/...`: train/test JSON files per detector
- `data/synthetic/...`: generated synthetic datasets and metadata
- `data/processed/data_summary.json`: quick overview

## Training

Train all models and store artifacts:
```bash
python scripts/train_models.py
```
Artifacts and results:
- `trained_models/*.pkl`: serialized model files
- `trained_models/*_config.json`: per-model config snapshots
- `trained_models/training_results.json`: aggregate metrics

## Evaluation

Run evaluation utilities:
```bash
python scripts/evaluate_models.py
```
Relevant modules:
- `evaluation/performance_metrics.py`: accuracy, F1, ROC AUC, PR metrics
- `evaluation/bias_evaluation.py`: subgroup fairness checks
- `evaluation/integration_tests.py`: end-to-end sanity validations

## Demos

Quick prediction demos:
```bash
python simple_prediction_demo.py
python comprehensive_prediction_demo.py
```
Interactive demos:
```bash
python demo/cli_demo.py
python demo/chat_simulator.py
python run_demo.py
```
If you want a minimal web interface and it’s supported in your environment:
```bash
python demo/web_interface.py
```

## Configuration

- `config/model_configs.json`: per-model thresholds and parameters
- `config/settings.py`: shared constants (paths, seeds, feature flags)

Example: tune thresholds by editing `config/model_configs.json` and retraining, or implement dynamic threshold logic in `src/safety_system/safety_manager.py`.

## Programmatic usage (example)

```python
from src.safety_system.safety_manager import SafetyManager

safety_manager = SafetyManager()
text = "This looks risky."
result_bundle = safety_manager.analyze(text=text, user_id="u1", session_id="s1")

print(result_bundle["overall_assessment"])  # consolidated risk + intervention
```

## Where to look next

- `trained_models/training_results.json`: your current training summary
- `src/safety_system/safety_manager.py`: fusion + intervention logic
- `src/models/`: swap classical vs. Hugging Face variants
- `evaluation/`: tailor metrics and bias checks to your policies

## Notes

- This is a POC. For production, add rigorous validation, monitoring, canarying, SLAs, privacy reviews, and incident playbooks.
- If GPU/transformers aren’t desired, stick to classical models for low-latency CPU inference.
