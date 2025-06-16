#!/usr/bin/env bash
set -euo pipefail

# M3-0 CI Reproducible Package - One-click validation
# 6-hour sprint execution: Sanity + Latency + FID + Ablation

echo "🚀 ParScale-EAR Lite-Hybrid CI Pipeline"
echo "Version: v0.3.0-lite-hybrid"
echo "Target: 100% reproducible validation"
echo "=" * 60

# ----------- Environment Setup -----------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Key paths - CONFIGURE THESE FOR YOUR ENVIRONMENT
REF_DIR="${REF_DIR:-data/imagenet_ref_1k}"  # Override with: export REF_DIR=/path/to/your/imagenet
RESULTS_DIR="results"
CHECKPOINT_DIR="checkpoints"

# Quick environment check
echo "🔧 Environment paths:"
echo "   REF_DIR: ${REF_DIR}"
echo "   RESULTS_DIR: ${RESULTS_DIR}"
echo "   CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Environment check
echo "🔧 Environment validation..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# ----------- M2-1: Sanity Check -----------
echo ""
echo "🔍 [M2-1] Sanity Check - Architecture validation"
python ci/sanity.py --batch 4

# ----------- M2-2: Latency Profile -----------
echo ""
echo "⚡ [M2-2] Latency Profile - Performance validation"
python ci/latency.py \
  --batch_list 4 8 16 \
  --num_runs ${CI_NUM_RUNS:-50} \
  --output ${RESULTS_DIR}/ci_latency.json

# ----------- M2-3: FID Validation -----------
echo ""
echo "🎯 [M2-3] FID Smoke Test - Quality validation"

# Generate ImageNet reference if not exists
if [ ! -d "${REF_DIR}" ]; then
  echo "📦 Creating ImageNet reference dataset..."
  python ci/create_imagenet_ref.py --output ${REF_DIR} --num_samples 1000
fi

# Calculate baseline FID
python ci/fid_eval.py \
  --mode baseline \
  --num_samples ${CI_FID_SAMPLES:-500} \
  --ref_dir ${REF_DIR} \
  --output ${RESULTS_DIR}/fid_baseline.json

# Calculate Hybrid FID  
python ci/fid_eval.py \
  --mode hybrid \
  --num_samples ${CI_FID_SAMPLES:-500} \
  --ref_dir ${REF_DIR} \
  --output ${RESULTS_DIR}/fid_hybrid.json

# ----------- M3-1: Ablation Study -----------
echo ""
echo "🔬 [M3-1] Ablation Study - Component validation"

# Coarse branch ablation
python ci/ablate_coarse.py \
  --num_samples ${CI_FID_SAMPLES:-500} \
  --ref_dir ${REF_DIR} \
  --output ${RESULTS_DIR}/fid_no_coarse.json

# Fine branch ablation  
python ci/ablate_fine.py \
  --num_samples ${CI_FID_SAMPLES:-500} \
  --ref_dir ${REF_DIR} \
  --output ${RESULTS_DIR}/fid_no_fine.json

# ----------- Validation Summary -----------
echo ""
echo "📊 Validation Summary"
python ci/validate_results.py --results_dir ${RESULTS_DIR}

# ----------- SHA256 Verification -----------
echo ""
echo "🔐 SHA256 Checksum verification"
sha256sum -c SHA256SUMS || (echo "❌ Checksum mismatch!" && exit 1)

echo ""
echo "🎉 CI Pipeline completed successfully!"
echo "✅ All M2-M3 milestones validated"
echo "📁 Results saved in: ${RESULTS_DIR}/"
echo "🏷️  Ready for v0.3.0-lite-hybrid release"