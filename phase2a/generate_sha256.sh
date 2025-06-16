#!/usr/bin/env bash
# Generate SHA256SUMS for CI reproducibility

set -e

echo "🔐 Generating SHA256 checksums..."

# Remove old checksums
rm -f SHA256SUMS

{
    echo "# ParScale-EAR v0.3.0-lite-hybrid SHA256 Checksums"
    echo "# Generated: $(date -u)"
    echo ""
    
    echo "# Source code files"
    find . -name "*.py" -type f | grep -v __pycache__ | grep -v parscale_fix_env | sort | xargs sha256sum
    
    echo ""
    echo "# CI scripts"
    find ci/ -name "*.py" -o -name "*.sh" | sort | xargs sha256sum
    
    echo ""
    echo "# Configuration files"
    sha256sum pyproject.toml
    
    if [ -f "poetry.lock" ]; then
        sha256sum poetry.lock
    fi
    
    echo ""
    echo "# Documentation"
    sha256sum README.md
    
    # Add model checkpoints if they exist
    if ls checkpoints/*.pth >/dev/null 2>&1; then
        echo ""
        echo "# Model checkpoints"
        sha256sum checkpoints/*.pth
    fi
    
} > SHA256SUMS

echo "✅ SHA256SUMS generated"
echo "📝 Verifying checksums..."

if sha256sum -c SHA256SUMS; then
    echo "🟢 All checksums verified successfully"
else
    echo "🔴 Checksum verification failed!"
    exit 1
fi