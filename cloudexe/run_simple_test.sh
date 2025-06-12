#!/bin/bash
# Execute simplified VAR-ParScale test on H100
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 Executing simplified VAR-ParScale test on H100"
echo "="*60

# Upload and execute simplified test that works with VAR's actual interface
$SSH_CMD << 'EOF'
cd /root

# Create a simplified test that uses VAR's existing methods
cat > /root/test_simple_parscale.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
"""
Simplified VAR-ParScale Test using existing VAR methods
"""

import torch
import torch.nn.functional as F
import os
import sys
import time

def main():
    print("🚀 Simplified VAR-ParScale Test")
    print("=" * 50)
    
    # Setup environment
    os.chdir("/root/VAR")
    sys.path.append("/root/VAR")
    
    from models import build_vae_var
    
    # Load models
    device = "cuda"
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=16, shared_aln=False,
    )
    
    vae.load_state_dict(torch.load("/root/vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    var.load_state_dict(torch.load("/root/var_d16.pth", map_location="cpu"), strict=True)
    
    vae.eval()
    var.eval()
    
    print("✅ Models loaded")
    
    # Test P=1, P=2, P=4 using existing VAR methods
    for P in [1, 2, 4]:
        print(f"\n🔧 Testing P={P}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        # Use multiple calls to simulate P-stream processing
        outputs = []
        with torch.no_grad():
            for i in range(P):
                # Slightly different parameters for diversity
                top_p = 0.95 - i * 0.01
                top_k = 900 - i * 50
                
                output = var.autoregressive_infer_cfg(
                    B=1, label_B=None, cfg=1.0, top_p=top_p, top_k=top_k
                )
                outputs.append(output)
        
        torch.cuda.synchronize()
        latency = (time.time() - start_time) * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Compute diversity between outputs
        if P > 1:
            similarities = []
            for i in range(P):
                for j in range(i+1, P):
                    sim = F.cosine_similarity(
                        outputs[i].flatten(), outputs[j].flatten(), dim=0
                    ).item()
                    similarities.append(abs(sim))
            diversity = 1.0 - sum(similarities) / len(similarities)
        else:
            diversity = 0.0
        
        print(f"P={P}: Latency={latency:.1f}ms, PeakMem={peak_mem:.2f}GB, Diversity={diversity:.3f}")

if __name__ == "__main__":
    main()
SCRIPT_EOF

# Run the simplified test
echo "🚀 Running simplified VAR-ParScale test..."
python3 test_simple_parscale.py
EOF

echo "✅ Simplified test execution complete!"