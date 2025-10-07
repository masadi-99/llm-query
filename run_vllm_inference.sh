#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH --partition=gpu
#SBATCH -C GPU_SKU:A100_PCIE
#SBATCH --ntasks=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
# Define how long the job will run d-hh:mm:ss
#SBATCH --time=24:00:00
# Get email notification when job finishes or fails
#SBATCH --mail-user=sunet@stanford.edu
#SBATCH --mail-type=END,FAIL
# Give your job a name, so you can recognize it in the queue overview
#SBATCH -J vllm-inference
# ----------------Load Modules--------------------
module load singularity/3.8.3

# ----------------Setup Directories--------------------
# Create local-scratch directory for user if it doesn't exist
LOCAL_SCRATCH="/local-scratch/sunet"
mkdir -p ${LOCAL_SCRATCH}

# Set environment variables for HuggingFace and vLLM cache
export HF_HOME="${LOCAL_SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${LOCAL_SCRATCH}/transformers_cache"
export HF_DATASETS_CACHE="${LOCAL_SCRATCH}/datasets_cache"
export VLLM_CACHE="${LOCAL_SCRATCH}/vllm_cache"

# Create cache directories
mkdir -p ${HF_HOME}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${VLLM_CACHE}

# Singularity image location
SINGULARITY_IMAGE="${LOCAL_SCRATCH}/vllm-openai-latest.sif"

# ----------------Download Singularity Image (if not exists)--------------------
if [ ! -f "${SINGULARITY_IMAGE}" ]; then
    echo "Downloading Singularity image..."
    singularity pull ${SINGULARITY_IMAGE} docker://vllm/vllm-openai:latest
else
    echo "Using existing Singularity image: ${SINGULARITY_IMAGE}"
fi

# ----------------Configuration--------------------
# Modify these parameters as needed
MODEL_NAME="openai/gpt-oss-20b"  # Change to your model
REPORTS_DIR="${HOME}/reports"  # Directory containing your report files
OUTPUT_FILE="${HOME}/report_results.json"  # Output file location
QUESTION="What are the key findings in this report?"  # Change your question here

MAX_MODEL_LEN=4096  # Adjustable context length
MAX_TOKENS=512      # Adjustable output tokens
TEMPERATURE=0.7
TENSOR_PARALLEL_SIZE=4  # Should match number of GPUs

# ----------------Commands------------------------
echo "Starting vLLM inference job"
echo "Model: ${MODEL_NAME}"
echo "Reports directory: ${REPORTS_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Max context length: ${MAX_MODEL_LEN}"
echo "Max output tokens: ${MAX_TOKENS}"
echo "Question: ${QUESTION}"

# Copy Python script to local-scratch for better I/O performance
cp ${HOME}/report_query.py ${LOCAL_SCRATCH}/

# Run the Python script inside Singularity container
singularity exec --nv \
    --bind ${LOCAL_SCRATCH}:${LOCAL_SCRATCH} \
    --bind ${HOME}:${HOME} \
    --env HF_HOME=${HF_HOME} \
    --env TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
    --env HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
    --env VLLM_CACHE=${VLLM_CACHE} \
    ${SINGULARITY_IMAGE} \
    python ${LOCAL_SCRATCH}/report_query.py \
        --model "${MODEL_NAME}" \
        --reports-dir "${REPORTS_DIR}" \
        --question "${QUESTION}" \
        --output-file "${OUTPUT_FILE}" \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-tokens ${MAX_TOKENS} \
        --temperature ${TEMPERATURE} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --gpu-memory-utilization 0.9

echo "Job completed!"
