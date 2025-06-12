#!/bin/bash
# Debug diversity的VAR-ParScale
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 Debug diversity的VAR-ParScale"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 创建debug版本，深入调查diversity问题
cat > test_debug_diversity.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

# ---------- 多次单独调用确保diversity ----------
@torch.no_grad()
def multiple_calls_infer(model, P):
    """确保通过多次独立调用产生diversity"""
    outputs = []
    
    for stream_idx in range(P):
        print(f"  Generating stream {stream_idx + 1}/{P}...")
        
        # 每次清空缓存并重新设置随机种子
        torch.cuda.empty_cache()
        
        # 设置不同的随机种子
        torch.manual_seed(42 + stream_idx * 1000)
        torch.cuda.manual_seed(42 + stream_idx * 1000)
        
        # 使用显著不同的参数
        if stream_idx == 0:
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
            )
        elif stream_idx == 1:
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.2, top_p=0.90, top_k=800
            )
        elif stream_idx == 2:
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=0.8, top_p=0.85, top_k=700
            )
        else:
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.5, top_p=0.80, top_k=600
            )
        
        outputs.append(out)
        print(f"    Output shape: {out.shape}, unique values: {torch.unique(out).size(0)}")
        
        # 打印前几个token用于debug
        if out.numel() >= 10:
            print(f"    First 10 tokens: {out.flatten()[:10].cpu().tolist()}")
    
    return outputs

# ---------- run tests for P = 1/2/4 ----------
def run_test(P):
    print(f"\n=== Testing P={P} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    outputs = multiple_calls_infer(var, P)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # 详细计算diversity
    diversity = 0.0
    if P > 1:
        print(f"  Computing diversity between {P} outputs...")
        similarities = []
        for i in range(P):
            for j in range(i + 1, P):
                # 确保tensor在正确设备上并转换类型
                out_i = outputs[i].flatten().float()
                out_j = outputs[j].flatten().float()
                
                sim = F.cosine_similarity(out_i, out_j, dim=0).item()
                similarities.append(abs(sim))
                print(f"    Similarity between stream {i} and {j}: {sim:.6f}")
        
        avg_sim = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_sim
        print(f"  Average similarity: {avg_sim:.6f}")
        print(f"  Computed diversity: {diversity:.6f}")
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")

print("🚀 Running debug ParScale tests...")
for P in (1, 2, 4):
    run_test(P)
SCRIPT_EOF

echo "🚀 执行debug ParScale测试..."
python3 test_debug_diversity.py
EOF

echo "✅ debug ParScale测试执行完成!"