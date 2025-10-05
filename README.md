# SLM-MUX: Orchestrating Small Language Models for Reasoning

This repository contains the implementation code for the SLM-MUX method, which effectively orchestrates multiple small language models (SLMs) to achieve superior reasoning performance.

## 📁 Project Structure

```
slm_mux_code/
├── slm_mux_orchestrator/ # Stage 2: SLM-MUX Orchestration
│   ├── math_benchmark.py
│   ├── gpqa_benchmark.py
│   └── gsm8k_benchmark.py
├── single_model_inference/ # Stage 1: Single Model Inference
│   ├── collect_math.py
│   ├── collect_gpqa.py
│   └── collect_gsm8k.py
├── data/
├── evaluation/
├── utils/
├── config/
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/slm-mux/slm-mux.github.io.git
cd slm-mux.github.io/slm_mux_code

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set up your API keys as environment variables:

```bash
export TOGETHER_API_KEY="your_together_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # Optional, for verification
```

### 3. Run SLM-MUX (Main Workflow)

**Example: Run SLM-MUX on MATH with 2 models**
```bash
python slm_mux_orchestrator/math_benchmark.py \
    --data_path data/math_500.json \
    --output results/math_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

**Example: Run SLM-MUX on GPQA**
```bash
python slm_mux_orchestrator/gpqa_benchmark.py \
    --data_path data/gpqa_shuffled.json \
    --output results/gpqa_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

**Example: Run SLM-MUX on GSM8K**
```bash
python slm_mux_orchestrator/gsm8k_benchmark.py \
    --data_path data/gsm8k_500.json \
    --output results/gsm8k_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

This will:
1. Query all specified models for each problem
2. Run each model multiple times (controlled by `--extra_calls`)
3. Apply the SLM-MUX algorithm to select the best answer
4. Save results with accuracy metrics

### 4. Optional: Run Single Model Inference (For Comparison)

If you want baseline performance for a single model:

```bash
python single_model_inference/collect_math.py \
    --dataset data/math_500.json \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --output collected_outputs/math_mistral7b_baseline.json \
    --num_llm_sub_iterations 3
```

**Note:** This is optional and only needed for comparison with single-model baselines.

## 📊 Understanding the Results

Each benchmark script outputs a JSON file containing:
- Individual model responses and extracted answers
- Vote counts for each unique answer
- Selected final answer based on confidence
- Token usage statistics
- Overall accuracy

Example output structure:
```json
{
    "problem_id": "...",
    "problem": "...",
    "reference_answer": "...",
    "models": [
        {
            "model_name": "...",
            "calls": [...],
            "best_answer": "...",
            "confidence": 0.67
        }
    ],
    "final_answer": "...",
    "is_correct": true
}
```

## 🔧 Optional: Single Model Baseline Collection

The `single_model_inference/` directory contains scripts for collecting single-model baseline performance. **This is optional and only needed for comparison purposes.**

```bash
# Example: Collect MATH responses from a single model
python single_model_inference/collect_math.py \
    --dataset data/math_500.json \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --output collected_outputs/math_mistral7b_baseline.json \
    --num_llm_sub_iterations 3
```

This is useful for comparing SLM-MUX results against individual model performance.

## 🧪 Answer Equivalence Checking

For MATH and GSM8K, we provide scripts to verify answer equivalence using GPT-4o:

```bash
# Check MATH answers
python evaluation/check_equivalence_math.py \
    --results results/math_results.json \
    --output results/math_verified.json

# Check GSM8K answers
python evaluation/check_equivalence_gsm8k.py \
    --results results/gsm8k_results.json \
    --output results/gsm8k_verified.json
```

## 📝 Key Components

### SLM-MUX Algorithm (`slm_mux_orchestrator/`)

The core of SLM-MUX involves:

1. **Multiple Sampling**: Each model generates multiple responses (controlled by `--extra_calls`)
2. **Confidence Estimation**: Count the frequency of each unique answer
3. **Model Selection**: Choose the answer with highest confidence across models
4. **Tie Breaking**: Use validation set performance when confidence scores tie

### Single Model Inference (`single_model_inference/`) - Optional

These scripts are **optional** and used for baseline comparison:
- Collect responses from a single model for performance benchmarking
- Generate prompts for each problem
- Call the specified model via the Together API
- Save raw responses and token usage for analysis

### Answer Extraction (`utils/`)

Each dataset has specialized answer extraction logic:
- **MATH**: Extracts content within `\boxed{...}` and normalizes LaTeX
- **GPQA**: Uses multiple choice extraction (A/B/C/D)
- **GSM8K**: Extracts numerical answers with ####

### API Integration

The `api_client.py` supports:
- Together AI API for running open-source models
- OpenAI API for verification (optional)
- Automatic retry with exponential backoff
- Token usage tracking

## 🤝 Contributing

When contributing code, please:
1. Follow the existing code style (PEP 8)
2. Add type hints to function signatures
3. Include docstrings for public functions
4. Update README if adding new features

## 📄 License

This code is released under the MIT License. See LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{slm-mux-2025,
  title={SLM-MUX: Orchestrating Small Language Models for Reasoning},
  author={Wang, Chenyu and Wan, Zishen and Kang, Hao and Chen, Emma and Xie, Zhiqiang and Krishna, Tushar and Reddi, Vijay Janapa and Du, Yilun},
  year={2025}
}
```

## 🔗 Links

- **Project Page**: https://slm-mux.github.io
- **Paper**: [Under Review]
- **GitHub**: https://github.com/slm-mux/SLM-MUX

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: ydu@seas.harvard.edu