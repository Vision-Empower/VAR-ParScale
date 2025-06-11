# 🔧 H100 CODEBASE SETUP REQUIREMENTS

## ⚠️ CRITICAL: VAR Codebase Missing on H100

Our implementation assumes the VAR codebase is at `/root/VAR/` on the H100 instance, but this needs to be set up first.

## 🚀 H100 Environment Setup (Required Before Phase 2A)

### 1. VAR Repository Setup
```bash
# On H100 instance
cd /root
git clone https://github.com/FoundationVision/VAR.git
cd VAR

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install huggingface_hub scipy numpy pillow
```

### 2. Download VAR Pretrained Models
```bash
# Download VAR models (these should be available on H100)
wget -O /root/vae_ch160v4096z32.pth [VAR_VAE_MODEL_URL]
wget -O /root/var_d16.pth [VAR_MODEL_URL]
```

### 3. Verify H100 Setup
```bash
# Test VAR installation
cd /root/VAR
python3 -c "from models import build_vae_var; print('VAR models imported successfully')"

# Test GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 🎯 Alternative: Self-Contained Setup

If VAR codebase is not available, we need to modify our implementation to be self-contained.

### Modified CloudExe Execution:
```bash
# Setup VAR environment first
curl -k -X POST https://cloudexe.io/api/execute \
  -H "Authorization: Bearer $CLOUDEXE_APIKEY" \
  -H "Content-Type: application/json" \
  -d '{"command": "cd /root && git clone https://github.com/FoundationVision/VAR.git && cd VAR && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"}'

# Then execute our implementation
python3 cloudexe/execute_phase2a.py
```

## 📋 RESTART CHECKLIST

**Before Phase 2A execution:**
1. ✅ Clone VAR-ParScale repository 
2. ⚠️ **Setup VAR codebase on H100** (REQUIRED)
3. ⚠️ **Download VAR pretrained models** (REQUIRED)
4. ✅ Execute Phase 2A implementation

**This setup is CRITICAL for successful execution!**