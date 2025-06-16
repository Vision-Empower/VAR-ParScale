#!/usr/bin/env bash
# ParScale-EAR Environment Setup Script
# Quick start for the 6-hour M3 sprint

set -euo pipefail

echo "🚀 ParScale-EAR v0.3.0-lite-hybrid Environment Setup"
echo "=" * 60

# Environment variables for CI tuning
echo "🔧 Environment configuration:"
echo "   CI_NUM_RUNS: ${CI_NUM_RUNS:-50} (latency measurement runs)"
echo "   CI_FID_SAMPLES: ${CI_FID_SAMPLES:-500} (FID sample count)"
echo "   REF_DIR: ${REF_DIR:-data/imagenet_ref_1k} (ImageNet reference)"

# Check CUDA availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo ""
    echo "⚠️ Warning: nvidia-smi not found - CPU mode only"
fi

# Check Python environment
echo ""
echo "🐍 Python Environment:"
if command -v poetry >/dev/null 2>&1; then
    echo "   Poetry: $(poetry --version)"
    if [ -f "poetry.lock" ]; then
        echo "   Dependencies: locked (poetry.lock exists)"
    else
        echo "   Dependencies: not locked (run: poetry install)"
    fi
else
    echo "   Poetry: not installed"
    echo "   Install: pip install poetry"
fi

# Quick dependency install
if command -v poetry >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then
    echo ""
    echo "📦 Installing dependencies..."
    poetry install --no-root
    echo "✅ Dependencies installed"
else
    echo ""
    echo "⚠️ Skipping dependency install (poetry or pyproject.toml missing)"
fi

# Verify checksums
echo ""
echo "🔐 Verifying file integrity..."
if sha256sum -c SHA256SUMS >/dev/null 2>&1; then
    echo "✅ All files verified (SHA256)"
else
    echo "❌ Checksum verification failed!"
    echo "   Run: ./generate_sha256.sh"
fi

# Environment readiness check
echo ""
echo "📋 Readiness Check:"

# Check key files
key_files=(
    "ci/run.sh"
    "ci/sanity.py"
    "ci/latency.py"
    "ci/fid_eval.py"
    "e2e_lite_hybrid_pipeline_fixed.py"
    "lite_hybrid_h100_final.py"
)

missing_files=()
for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo ""
    echo "🟢 Environment ready for CI execution!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Run full CI: bash ci/run.sh"
    echo "   2. Quick check: python quick_check.py" 
    echo "   3. Run tests: poetry run pytest ci/tests/ -v"
    echo ""
    echo "⚡ Environment variables for faster CI:"
    echo "   export CI_NUM_RUNS=30        # Faster latency tests"
    echo "   export CI_FID_SAMPLES=256    # Faster FID tests"
    echo "   export REF_DIR=/path/to/data # Custom ImageNet path"
else
    echo ""
    echo "🔴 Missing files detected! Please ensure all required files are present."
    exit 1
fi