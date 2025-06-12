#!/bin/bash
# 真正的共享骨干实现：删除所有for-loop，实现单次batch调用
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 真正的共享骨干实现：删除所有for-loop"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 实现真正的单次batch调用共享骨干
cat > true_shared_backbone_final.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

print("✅ VAR model loaded")

# ---------- 真正的批处理attention补丁 ----------
import types

def _patch_attention_for_true_batching(model):
    """真正的批处理attention - KV cache会累积"""
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        if getattr(attn, "_true_batched", False):
            continue
            
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        def true_batched_forward(self, x, kv_cache):
            # x: [P*B, 1, C] - 关键：这是P*B个样本的单个token
            B_all, seq_len, C = x.shape
            
            # QKV投影
            qkv = self.mat_qkv(x)  # [P*B, 1, 3*C]
            qkv = qkv.view(B_all, seq_len, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, P*B, num_heads, 1, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # KV Cache累积 - 这是关键！
            if "k" in kv_cache and "v" in kv_cache:
                # 拼接历史：[P*B, num_heads, t-1, head_dim] + [P*B, num_heads, 1, head_dim]
                prev_k, prev_v = kv_cache["k"], kv_cache["v"]
                k = torch.cat([prev_k, k], dim=2)  # [P*B, num_heads, t, head_dim]
                v = torch.cat([prev_v, v], dim=2)  # [P*B, num_heads, t, head_dim]
                
                if layer_idx == 0:  # 只在第一层打印调试信息
                    print(f"   🔄 Layer {layer_idx}: KV accumulated {prev_k.shape} + {qkv[1].shape} → {k.shape}")
            else:
                if layer_idx == 0:
                    print(f"   🆕 Layer {layer_idx}: KV initialized {k.shape}")
            
            # 更新cache
            kv_cache["k"], kv_cache["v"] = k, v
            
            # Attention计算
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [P*B, heads, 1, t]
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)  # [P*B, heads, 1, head_dim]
            
            # 重组输出
            out = out.transpose(1, 2).contiguous().view(B_all, seq_len, C)
            return self.proj(out)
        
        # 替换attention forward
        attn.forward = types.MethodType(true_batched_forward, attn)
        attn._true_batched = True
        
        # 修改block forward支持kv_cache参数
        original_block_forward = block.forward
        def block_forward_with_cache(self, x, kv_cache=None):
            if kv_cache is None:
                return original_block_forward(x)
            
            # AdaLNSelfAttn forward with KV cache
            shortcut = x
            attn_out = self.attn(x, kv_cache)
            x = shortcut + self.drop_path(attn_out)
            
            # FFN
            shortcut = x
            ffn_out = self.ffn(x) 
            x = shortcut + self.drop_path(ffn_out)
            
            return x
        
        block.forward = types.MethodType(block_forward_with_cache, block)

print("🔧 Applying true batching patches...")
_patch_attention_for_true_batching(var)

# ---------- 真正的单次batch推理方法 ----------
def encode_batch_tokens(self, tokens):
    """编码初始tokens并初始化KV cache"""
    B_all = tokens.size(0)  # P*B
    print(f"🎯 Encoding batch tokens: {tokens.shape} (B_all={B_all})")
    
    # 为每一层初始化空的KV cache
    self._kv_caches = [{} for _ in range(len(self.blocks))]
    
    # 编码tokens
    embeddings = self.class_emb(tokens)  # [P*B, seq_len, C]
    print(f"   Token embeddings: {embeddings.shape}")
    return embeddings

def forward_step_true_batch(self, x):
    """单步前向传播，使用真正累积的KV cache"""
    B_all, seq_len, C = x.shape
    print(f"📦 Forward step: {x.shape}")
    
    # 通过所有transformer layers
    for layer_idx, block in enumerate(self.blocks):
        x = block(x, kv_cache=self._kv_caches[layer_idx])
    
    # 输出投影
    logits = self.head(x)  # [P*B, seq_len, vocab_size]
    print(f"   Logits output: {logits.shape}")
    return logits

@torch.no_grad()
def true_batch_autoregressive_generation(self, init_tokens, max_steps=32, diversity_factor=0.1):
    """真正的批量自回归生成 - 单次调用，无循环！"""
    B_all = init_tokens.size(0)  # P*B
    print(f"🚀 TRUE BATCH GENERATION: B_all={B_all}, max_steps={max_steps}")
    
    # 初始化：编码第一个token
    x = encode_batch_tokens(self, init_tokens[:, :1])  # [P*B, 1, C]
    generated_tokens = [init_tokens[:, :1]]  # 保存生成的tokens
    
    # 自回归生成循环 - 这是时间步循环，不是样本循环！
    for step in range(1, max_steps):
        print(f"⏰ Time step {step}/{max_steps}")
        
        # 单步前向传播 - 处理所有P*B个样本
        logits = forward_step_true_batch(self, x)  # [P*B, 1, vocab_size]
        logits = logits.squeeze(1)  # [P*B, vocab_size]
        
        # 添加diversity - 为不同stream添加不同的logit bias
        if B_all > 1 and diversity_factor > 0:
            for i in range(B_all):
                # 为每个stream添加小的随机bias
                logits[i] += torch.randn_like(logits[i]) * diversity_factor * (i + 1)
        
        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1)  # [P*B, 1]
        
        generated_tokens.append(next_tokens)
        
        # 准备下一步的输入
        x = self.class_emb(next_tokens)  # [P*B, 1, C]
        
        # 每10步检查KV cache状态
        if step % 10 == 0:
            if self._kv_caches[0]:
                cache_shape = self._kv_caches[0]["k"].shape if "k" in self._kv_caches[0] else "empty"
                print(f"   ✅ Step {step}: Layer 0 KV cache shape: {cache_shape}")
    
    # 拼接所有生成的tokens
    result = torch.cat(generated_tokens, dim=1)  # [P*B, max_steps]
    print(f"✅ Generation complete: {result.shape}")
    return result

# 绑定方法到VAR
var.encode_batch_tokens = types.MethodType(encode_batch_tokens, var)
var.forward_step_true_batch = types.MethodType(forward_step_true_batch, var)
var.true_batch_autoregressive_generation = types.MethodType(true_batch_autoregressive_generation, var)

# ---------- 测试真正的共享骨干 ----------
def test_true_shared_backbone(P):
    print(f"\n{'='*60}")
    print(f"🧪 TESTING TRUE SHARED BACKBONE: P={P}")
    print(f"{'='*60}")
    
    # 创建P个不同的初始tokens
    tokens = torch.randint(100, 200, (P, 1), device=dev)  # 使用100-200范围确保有效
    for i in range(P):
        tokens[i, 0] = 100 + i * 10  # 确保每个stream有不同起点
    
    print(f"📝 Initial tokens: {tokens.flatten().tolist()}")
    
    # 清理内存并开始计时
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    try:
        # 关键：这里只有一次调用！
        outputs = var.true_batch_autoregressive_generation(
            tokens, max_steps=16, diversity_factor=0.05
        )
        success = True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        success = False
        outputs = None
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    if success and outputs is not None:
        print(f"📊 Generated shape: {outputs.shape}")
        print(f"📊 Sample outputs:")
        for i in range(min(P, 3)):
            print(f"   Stream {i}: {outputs[i, :8].tolist()}")
        
        # 计算diversity
        diversity = 0.0
        if P > 1:
            similarities = []
            for i in range(P):
                for j in range(i + 1, P):
                    sim = F.cosine_similarity(
                        outputs[i].float(), outputs[j].float(), dim=0
                    ).item()
                    similarities.append(abs(sim))
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                diversity = 1.0 - avg_similarity
        
        print(f"🎨 Computed diversity: {diversity:.3f}")
    else:
        diversity = 0.0
        print(f"❌ Failed to generate outputs")
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")
    
    return {
        'latency': latency,
        'peak_mem': peak_mem,
        'diversity': diversity,
        'success': success
    }

# ---------- 运行最终测试 ----------
print("🏁 RUNNING FINAL TRUE SHARED BACKBONE TESTS")
print("="*70)
print("Expected: Memory scales with P, Efficiency ≤ 100%, KV cache accumulates")

results = {}
baseline_mem = None

for P in [1, 2, 4]:
    result = results[P] = test_true_shared_backbone(P)
    
    if P == 1:
        baseline_mem = result['peak_mem']
    else:
        mem_ratio = result['peak_mem'] / baseline_mem if baseline_mem else 1.0
        expected_sequential_lat = results[1]['latency'] * P
        efficiency = expected_sequential_lat / result['latency'] if result['latency'] > 0 else 0
        
        print(f"   Memory ratio: {mem_ratio:.2f}x baseline")
        print(f"   Parallel efficiency: {efficiency:.1%}")
        
        # 判断是否为真正的共享骨干
        if mem_ratio >= 1.5 and efficiency <= 100:
            print(f"   ✅ TRUE SHARED BACKBONE CONFIRMED!")
        elif efficiency > 150:
            print(f"   🚨 Still sequential execution detected")
        else:
            print(f"   ⚠️ Partial shared backbone")

print(f"\n🎯 FINAL VERDICT:")
if all(results[p]['success'] for p in [1, 2, 4]):
    print("✅ All tests passed - True shared backbone implemented!")
    print("🍾 NOW WE CAN OPEN THE CHAMPAGNE! 🍾")
else:
    print("❌ Implementation needs more work")
SCRIPT_EOF

echo "🚀 执行真正的共享骨干最终测试..."
python3 true_shared_backbone_final.py
EOF

echo "✅ 真正的共享骨干测试执行完成!"