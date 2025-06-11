#!/bin/bash
# SSH-based H100 Setup Script
# Usage: ./ssh_setup.sh [SSH_CONNECTION_STRING]

set -e

if [ $# -eq 0 ]; then
    echo "🔑 Using default CloudExe SSH connection"
    SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
else
    SSH_CMD="$1"
fi
echo "🚀 Setting up H100 via SSH"
echo "Connection: $SSH_CMD"
echo "="*60

echo "🔧 Setting up VAR codebase..."
$SSH_CMD << 'EOF'
cd /root
echo "📂 Cloning VAR repository..."
git clone https://github.com/FoundationVision/VAR.git

echo "📂 Cloning VAR-ParScale repository..."
git clone https://github.com/peteryuqin/VAR-ParScale.git

echo "🔄 Installing PyTorch with CUDA..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

echo "📦 Installing dependencies..."
pip3 install huggingface_hub scipy numpy pillow

echo "✅ Testing VAR import..."
cd /root/VAR
python3 -c "from models import build_vae_var; print('VAR models imported successfully')"

echo "🔥 Testing GPU..."
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')"

echo "📊 Checking for VAR models..."
ls -la /root/*.pth || echo "⚠️ VAR model files may need to be downloaded"

echo "🎯 H100 setup complete!"
EOF

echo ""
echo "✅ H100 Environment Setup Complete"
echo "🚀 Ready to execute Phase 2A"
echo ""
echo "Execute Phase 2A:"
echo "$SSH_CMD 'cd /root/VAR-ParScale && python3 phase2a/enhanced_parscale_var.py'"