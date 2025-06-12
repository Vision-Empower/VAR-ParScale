#!/bin/bash
# 基于VAR实际结构的工作版本
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 基于VAR实际结构的工作版本"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 检查VAR模型的完整结构并创建工作版本
cat > test_working_version.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

print("VAR model attributes:")
var_attrs = [attr for attr in dir(var) if not attr.startswith('_')]
print(f"VAR attributes: {var_attrs}")

# 查找最终层normalization和输出投影
if hasattr(var, 'head'):
    print(f"VAR has head: {type(var.head)}")
if hasattr(var, 'norm'):
    print(f"VAR has norm: {type(var.norm)}")
if hasattr(var, 'final_ln'):
    print(f"VAR has final_ln: {type(var.final_ln)}")

# ---------- 使用VAR现有方法但并行化输入 ----------
@torch.no_grad()
def parallel_infer(model, P, max_steps=64):
    """使用VAR现有方法的并行推理"""
    # 创建P个随机起始token
    init_tokens = torch.randint(0, 1000, (P, 1), device=dev)
    
    # 将P个token合并为batch进行处理
    batch_input = init_tokens.view(-1, 1)  # [P, 1]
    
    # 使用VAR的现有推理，但传入batch
    try:
        # 尝试使用batch input
        outputs = model.autoregressive_infer_cfg(
            B=P, label_B=None, cfg=1.0, top_p=0.95, top_k=900
        )
        return outputs
    except Exception as e:
        print(f"Batch inference failed: {e}")
        # 回退到逐个推理
        outputs = []
        for i in range(P):
            out = model.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.0, top_p=0.95-i*0.01, top_k=900-i*50
            )
            outputs.append(out)
        return outputs

# ---------- run tests for P = 1/2/4 ----------
def run_test(P):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    outputs = parallel_infer(var, P, max_steps=64)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # 计算diversity
    diversity = 0.0
    if P > 1 and isinstance(outputs, list):
        similarities = []
        for i in range(P):
            for j in range(i + 1, P):
                sim = F.cosine_similarity(
                    outputs[i].flatten().float(), outputs[j].flatten().float(), dim=0
                ).item()
                similarities.append(abs(sim))
        diversity = 1.0 - (sum(similarities) / len(similarities))
    elif P > 1 and isinstance(outputs, torch.Tensor) and outputs.size(0) == P:
        # 如果成功批处理
        similarities = []
        for i in range(P):
            for j in range(i + 1, P):
                sim = F.cosine_similarity(
                    outputs[i].flatten().float(), outputs[j].flatten().float(), dim=0
                ).item()
                similarities.append(abs(sim))
        diversity = 1.0 - (sum(similarities) / len(similarities))
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")

print("🚀 Running working version tests...")
for P in (1, 2, 4):
    run_test(P)
SCRIPT_EOF

echo "🚀 执行工作版本测试..."
python3 test_working_version.py
EOF

echo "✅ 工作版本测试执行完成!"