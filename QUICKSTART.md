# 🚀 SLM-MUX Quick Start Guide

This guide explains how to run the main SLM-MUX workflow and, optionally, how to run single-model inference for comparison.

## 1. Setup (5 minutes)

```bash
# Navigate to the code directory
cd slm_mux_code

# Install dependencies
pip install -r requirements.txt

# Set up your API key
export TOGETHER_API_KEY="your_api_key_here"
```

---

## 2. How to Run SLM-MUX (Main Workflow)

The orchestrator scripts handle the entire process: calling multiple models, running the SLM-MUX algorithm, and calculating the final accuracy.

### MATH Dataset Example
```bash
python slm_mux_orchestrator/math_benchmark.py \
    --data_path data/math_500.json \
    --output results/math_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

### GPQA Dataset Example
```bash
python slm_mux_orchestrator/gpqa_benchmark.py \
    --data_path data/gpqa_shuffled.json \
    --output results/gpqa_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

### GSM8K Dataset Example
```bash
python slm_mux_orchestrator/gsm8k_benchmark.py \
    --data_path data/gsm8k_500.json \
    --output results/gsm8k_slmmux_results.json \
    --models \
        "mistralai/Mistral-7B-Instruct-v0.3" \
        "Qwen/Qwen2.5-7B-Instruct" \
    --extra_calls 3
```

**What this does**:
1.  For each problem, it queries all specified models (`--models`) in parallel.
2.  Each model is called multiple times (`--extra_calls`) to get a sample of responses.
3.  The **SLM-MUX algorithm** is applied in real-time to select the best answer based on cross-model confidence.
4.  The final accuracy and detailed results are saved to the output file.

---

## 3. Optional: Run Single Model Inference (For Comparison)

If you want to get baseline performance for a single model, use the `single_model_inference` scripts. **This is not required to run SLM-MUX.**

### MATH Dataset Example
```bash
python single_model_inference/collect_math.py \
    --dataset data/math_500.json \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --output collected_outputs/math_mistral7b_baseline.json \
    --num_llm_sub_iterations 3
```

**What this does**:
-   Queries a **single model** (`--model`) for each problem.
-   Saves the raw responses to a JSON file for analysis.

## 4. Understanding the Output

The final results JSON file (`results/*.json`) from the orchestrator contains:
- ✅ The final answer chosen by the SLM-MUX algorithm.
- ✅ A correctness check against the reference answer.
- ✅ The confidence score that led to the decision.
- ✅ A detailed breakdown of each model's individual performance within the run.

## 5. Common Options

### `slm_mux_orchestrator` (Main Workflow)
| Option | Description |
|---|---|
| `--data_path` | Path to dataset with reference answers. |
| `--models` | **List** of model names to orchestrate. |
| `--output` | Path to save final benchmark results. |
| `--extra_calls` | Number of samples to generate per model per problem. |
| `--limit` | Limit number of problems to process. |
| `--max_workers` | Number of concurrent API calls. |

### `single_model_inference` (Optional)
| Option | Description |
|---|---|
| `--dataset` | Path to dataset JSON. |
| `--model` | **Single** model name for baseline inference. |
| `--output` | Path to save collected responses. |
| `--num_llm_sub_iterations` | Number of samples to generate per problem. |


## 6. Troubleshooting

### API Key Error
```bash
# Make sure your API key is set
echo $TOGETHER_API_KEY
```

### Rate Limiting
You can reduce concurrency for either workflow:
```bash
python slm_mux_orchestrator/math_benchmark.py ... --max_workers 4
```

## 7. Next Steps

- 📖 Read the full [README.md](README.md) for more details.
- 🔧 Customize prompts in the scripts.
- 🚀 Experiment with different model combinations in the orchestrator.
