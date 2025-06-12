#!/bin/bash
# 最小工作版本：验证真正的批处理概念
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 最小工作版本：验证真正的批处理概念"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 创建最小工作版本，证明真正的批处理概念
cat > minimal_working_test.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

print("✅ VAR model loaded successfully")

# ---------- 测试VAR原生批处理能力 ----------
@torch.no_grad()
def test_var_native_batch(P):
    """测试VAR是否原生支持批处理"""
    print(f"\n🧪 Testing VAR native batch capability with P={P}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    try:
        # 尝试直接用P作为batch size调用VAR
        print(f"📝 Calling autoregressive_infer_cfg with B={P}")
        outputs = var.autoregressive_infer_cfg(
            B=P, label_B=None, cfg=1.0, top_p=0.95, top_k=900
        )
        success = True
        print(f"✅ Success! Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"❌ Native batch failed: {e}")
        print(f"🔄 Falling back to sequential calls...")
        
        # 回退到顺序调用
        outputs = []
        for i in range(P):
            out = var.autoregressive_infer_cfg(
                B=1, label_B=None, cfg=1.0, 
                top_p=max(0.85, 0.95-i*0.02), 
                top_k=max(600, 900-i*50)
            )
            outputs.append(out)
        success = False
    
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # 分析输出
    if success:
        print(f"📊 Native batch output shape: {outputs.shape}")
        if outputs.dim() >= 1 and outputs.size(0) == P:
            print(f"✅ Successfully generated {P} samples in single call")
            
            # 计算diversity
            diversity = 0.0
            if P > 1:
                similarities = []
                for i in range(P):
                    for j in range(i + 1, P):
                        # 安全的similarity计算
                        try:
                            sim = F.cosine_similarity(
                                outputs[i].flatten().float(), 
                                outputs[j].flatten().float(), 
                                dim=0
                            ).item()
                            similarities.append(abs(sim))
                        except:
                            similarities.append(1.0)  # 假设相似
                
                if similarities:
                    diversity = 1.0 - (sum(similarities) / len(similarities))
            
            print(f"🎨 Diversity between samples: {diversity:.3f}")
        else:
            print(f"⚠️ Unexpected output shape for batch size {P}")
            diversity = 0.0
    else:
        print(f"📊 Sequential outputs: {len(outputs)} samples")
        if len(outputs) > 1:
            # 计算diversity for sequential outputs
            similarities = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    try:
                        sim = F.cosine_similarity(
                            outputs[i].flatten().float(), 
                            outputs[j].flatten().float(), 
                            dim=0
                        ).item()
                        similarities.append(abs(sim))
                    except:
                        similarities.append(1.0)
            
            diversity = 1.0 - (sum(similarities) / len(similarities)) if similarities else 0.0
        else:
            diversity = 0.0
    
    implementation = "native_batch" if success else "sequential"
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}  ({implementation})")
    
    return {
        'latency': latency,
        'peak_mem': peak_mem, 
        'diversity': diversity,
        'implementation': implementation,
        'success': success
    }

# ---------- 运行测试并分析 ----------
print("🚀 Running comprehensive VAR batch tests...")
print("="*60)

results = {}
for P in (1, 2, 4):
    results[P] = test_var_native_batch(P)

print(f"\n📊 FINAL ANALYSIS")
print("="*30)

baseline = results[1]
print(f"Baseline (P=1): {baseline['latency']:.1f}ms, {baseline['peak_mem']:.2f}GB")

for P in (2, 4):
    result = results[P]
    
    # 计算效率
    expected_latency = baseline['latency'] * P  # 如果是顺序执行的期望延迟
    efficiency = expected_latency / result['latency'] if result['latency'] > 0 else 0
    
    # 内存比例
    mem_ratio = result['peak_mem'] / baseline['peak_mem']
    
    print(f"P={P}: {result['latency']:.1f}ms, {result['peak_mem']:.2f}GB")
    print(f"  Efficiency: {efficiency:.1%} (expect ≤100% for true batching)")
    print(f"  Memory ratio: {mem_ratio:.2f}x (expect >1.5x for P=2, >2.5x for P=4)")
    print(f"  Implementation: {result['implementation']}")
    print(f"  Diversity: {result['diversity']:.3f}")
    
    # 判断是否为真正的批处理
    if result['implementation'] == 'native_batch':
        if efficiency <= 1.0 and mem_ratio >= 1.5:
            print(f"  ✅ TRUE BATCHING DETECTED!")
        elif efficiency > 1.5:
            print(f"  🚨 SUSPICIOUS: Super-linear efficiency!")
        else:
            print(f"  ⚠️ Partial batching or other effects")
    else:
        print(f"  📝 Sequential implementation (baseline)")
    print()

print("🎯 Summary: Looking for native_batch + efficiency ≤100% + memory scaling")
SCRIPT_EOF

echo "🚀 执行最小工作版本测试..."
python3 minimal_working_test.py
EOF

echo "✅ 最小工作版本测试执行完成!"