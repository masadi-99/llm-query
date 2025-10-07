# vLLM Report Query System - Setup and Usage

## Overview
This system queries multiple reports using vLLM in batch mode. It uses SLURM for job scheduling and runs inside a Singularity container.

## Files
1. **report_query.py** - Main Python script for querying reports
2. **run_vllm_inference.sh** - SLURM batch script

## Setup Instructions

### 1. Prepare Your Environment

```bash
# Create reports directory in your home directory
mkdir -p ~/reports

# Place your report files (*.txt) in the reports directory
# Example: ~/reports/report1.txt, ~/reports/report2.txt, etc.

# Copy the Python script to your home directory
cp report_query.py ~/
```

### 2. Configure the SLURM Script

Edit `run_vllm_inference.sh` and modify these variables:

```bash
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"  # Your model
REPORTS_DIR="${HOME}/reports"  # Your reports location
QUESTION="What are the key findings in this report?"  # Your question
MAX_MODEL_LEN=4096  # Context length
MAX_TOKENS=512      # Output tokens
```

### 3. Submit the Job

```bash
sbatch run_vllm_inference.sh
```

### 4. Monitor the Job

```bash
# Check job status
squeue -u sunet

# View output log (after job starts)
tail -f slurm-<job_id>.out
```

## Directory Structure

```
/local-scratch/sunet/         # All cached data and temp files
├── hf_cache/                 # HuggingFace cache
├── transformers_cache/       # Transformers cache
├── datasets_cache/           # Datasets cache
├── vllm_cache/              # vLLM cache
├── vllm-openai-latest.sif   # Singularity image
└── report_query.py          # Temporary copy

${HOME}/                      # Your home directory
├── reports/                  # Your report files
│   ├── report1.txt
│   ├── report2.txt
│   └── ...
├── report_query.py          # Main script
└── report_results.json      # Output results
```

## Command Line Options

The Python script supports these arguments:

- `--model`: Model name or path (required)
- `--reports-dir`: Directory with report files (required)
- `--question`: Question to ask about each report
- `--output-file`: Where to save results (default: ~/report_results.json)
- `--max-model-len`: Maximum context length (default: 4096)
- `--max-tokens`: Maximum output tokens (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--tensor-parallel-size`: Number of GPUs (default: 1)
- `--gpu-memory-utilization`: GPU memory usage 0.0-1.0 (default: 0.9)

## Customizing Questions

### Method 1: Edit the SLURM script
Change the `QUESTION` variable in `run_vllm_inference.sh`:

```bash
QUESTION="Summarize the main conclusions of this report."
```

### Method 2: Run directly with custom parameters
```bash
singularity exec --nv vllm-openai-latest.sif \
    python report_query.py \
        --model "meta-llama/Llama-2-7b-chat-hf" \
        --reports-dir ~/reports \
        --question "What methodology was used in this report?" \
        --max-model-len 8192 \
        --max-tokens 1024
```

## Output Format

Results are saved as JSON with this structure:

```json
[
  {
    "report_name": "report1",
    "question": "What are the key findings?",
    "answer": "The report shows...",
    "prompt_tokens": 1234,
    "completion_tokens": 256
  },
  ...
]
```

## Troubleshooting

### Out of Memory
- Reduce `MAX_MODEL_LEN`
- Reduce `--gpu-memory-utilization` (e.g., 0.8)
- Use a smaller model
- Reduce `MAX_TOKENS`

### Slow Processing
- Increase `TENSOR_PARALLEL_SIZE` to match available GPUs
- Ensure all GPUs are allocated (`--gres=gpu:4`)

### Model Download Issues
- Check HuggingFace credentials if using gated models
- Verify network connectivity
- Check disk space in `/local-scratch/sunet/`

### Singularity Image Issues
```bash
# Manually pull the image
singularity pull /local-scratch/sunet/vllm-openai-latest.sif \
    docker://vllm/vllm-openai:latest
```

## Resource Adjustments

To use different resources, modify the SLURM parameters:

```bash
#SBATCH --gres=gpu:2          # Use 2 GPUs
#SBATCH --mem=128G            # Reduce memory
#SBATCH --time=12:00:00       # Shorter time limit
```

Then adjust:
```bash
TENSOR_PARALLEL_SIZE=2  # Match GPU count
```

## Notes

- The first run will download the model and Singularity image (may take time)
- Subsequent runs use cached versions
- All caching happens in `/local-scratch/sunet/` for better performance
- Results are saved in your home directory for persistence
