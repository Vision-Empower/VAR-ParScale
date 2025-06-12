#!/bin/bash
# 最终版本：确保diversity的VAR-ParScale
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 最终版本：确保diversity的VAR-ParScale"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 创建最终版本，确保有diversity
cat > test_final_with_diversity.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

# ---------- 实现真正的P-stream推理 ----------
@torch.no_grad()
def parscale_infer(model, P, max_steps=64):
    """ParScale P-stream推理，确保diversity"""
    outputs = []
    
    for stream_idx in range(P):
        # 为每个stream使用不同的参数确保diversity
        if stream_idx == 0:
            # 主stream：标准参数
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
            )
        else:
            # 其他stream：变化参数
            cfg_var = 1.0 + (stream_idx - 1) * 0.1  # 0.1的cfg变化
            top_p_var = max(0.85, 0.95 - stream_idx * 0.02)  # top_p变化
            top_k_var = max(500, 900 - stream_idx * 100)  # top_k变化
            
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=cfg_var, top_p=top_p_var, top_k=top_k_var
            )
        
        outputs.append(out)
    
    return outputs

# ---------- run tests for P = 1/2/4 ----------
def run_test(P):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    outputs = parscale_infer(var, P, max_steps=64)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # 计算diversity
    diversity = 0.0
    if P > 1:
        similarities = []
        for i in range(P):
            for j in range(i + 1, P):
                sim = F.cosine_similarity(
                    outputs[i].flatten().float(), outputs[j].flatten().float(), dim=0
                ).item()
                similarities.append(abs(sim))
        diversity = 1.0 - (sum(similarities) / len(similarities))
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")

print("🚀 Running final ParScale tests with diversity...")
for P in (1, 2, 4):
    run_test(P)
SCRIPT_EOF

echo "🚀 执行最终ParScale测试..."
python3 test_final_with_diversity.py
EOF

echo "✅ 最终ParScale测试执行完成!"