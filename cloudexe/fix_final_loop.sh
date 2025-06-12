#!/bin/bash
# 修复最后的for-loop，实现真正的共享骨干
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 修复最后的for-loop，实现真正的共享骨干"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 创建真正的单次batch调用版本
cat > fix_true_batch.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

# ---------- 彻底修补attention和block ----------
import types

def _patch_all_attention_for_batch(model):
    """真正的批处理attention补丁"""
    for lid, blk in enumerate(model.blocks):
        attn = blk.attn
        if getattr(attn, "_patched", False):
            continue
            
        nh = attn.num_heads
        hdim = attn.head_dim
        
        def batched_attn_forward(self, x, cache):
            # x: [P*B, 1, C]
            B, L, C = x.shape
            print(f"🔥 Layer {lid} processing batch: {x.shape}")
            
            qkv = self.mat_qkv(x).view(B, L, 3, nh, hdim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, nh, 1, hdim]
            
            # KV cache累积 - 关键步骤！
            if "k" in cache:
                print(f"   Prev cache k shape: {cache['k'].shape}")
                k = torch.cat([cache["k"], k], dim=2)  # [B, nh, t+1, hdim]
                v = torch.cat([cache["v"], v], dim=2)  # [B, nh, t+1, hdim]
                print(f"   After concat k shape: {k.shape}")
            else:
                print(f"   Initial k shape: {k.shape}")
            
            cache["k"], cache["v"] = k, v
            
            # scaled dot-product attention
            scale = hdim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)  # [B, nh, 1, hdim]
            
            out = out.transpose(1, 2).reshape(B, L, C)
            return self.proj(out)
        
        attn.forward = types.MethodType(batched_attn_forward, attn)
        attn._patched = True
        
        # 修补block forward支持cache
        def block_forward_with_cache(self, x, cache):
            shortcut = x
            x = self.attn(x, cache)
            x = shortcut + self.drop_path(x)
            shortcut = x  
            x = self.ffn(x)
            x = shortcut + self.drop_path(x)
            return x
        
        blk.forward = types.MethodType(block_forward_with_cache, blk)

_patch_all_attention_for_batch(var)

# ---------- 真正的单次batch推理 ----------
def encode_tokens(self, tokens):
    """初始化KV cache"""
    print(f"🚀 Encoding initial tokens: {tokens.shape}")
    self._kv_cache = [{} for _ in self.blocks]
    return self.class_emb(tokens)

def forward_step_batch(self, x):
    """单步前向传播，使用共享KV cache"""
    print(f"📦 Forward step input: {x.shape}")
    for layer_idx, block in enumerate(self.blocks):
        x = block(x, cache=self._kv_cache[layer_idx])
    
    # 使用VAR的head进行输出投影
    logits = self.head(x)  # [P*B, 1, vocab_size]
    print(f"📤 Forward step output: {logits.shape}")
    return logits

@torch.no_grad()
def true_batch_autoregressive_infer(self, init_tokens, max_steps=64):
    """真正的批量自回归推理 - 删除所有内部循环！"""
    B_all = init_tokens.size(0)  # P*B
    print(f"🎯 Starting TRUE batch inference with B_all={B_all}")
    
    # 初始化
    x = encode_tokens(self, init_tokens[:, :1])  # [P*B, 1, C]
    generated = [init_tokens[:, :1]]  # list of [P*B, 1]
    
    # 自回归生成循环 - 这里是时间步循环，不是batch循环！
    for step in range(1, max_steps):
        if step % 10 == 0:
            print(f"⏰ Step {step}/{max_steps}")
            # 检查KV cache大小
            if self._kv_cache[0]:
                print(f"   KV Cache layer 0, key shape: {self._kv_cache[0]['k'].shape}")
        
        # 单步前向传播 - 处理整个batch
        logits = forward_step_batch(self, x).squeeze(1)  # [P*B, vocab_size]
        
        # 采样 - 对整个batch同时采样
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1)  # [P*B, 1]
        
        generated.append(next_tokens)
        x = self.class_emb(next_tokens)  # 下一步输入
    
    print(f"✅ Batch inference complete!")
    return torch.cat(generated, dim=1)  # [P*B, max_steps]

# 绑定方法到VAR实例
var.encode_tokens = types.MethodType(encode_tokens, var)
var.forward_step_batch = types.MethodType(forward_step_batch, var)
var.true_batch_autoregressive_infer = types.MethodType(true_batch_autoregressive_infer, var)

# ---------- 测试真正的批处理 ----------
def run_true_batch_test(P):
    print(f"\n{'='*50}")
    print(f"🧪 Testing TRUE batch processing with P={P}")
    print(f"{'='*50}")
    
    # 创建P个不同的初始token以确保diversity
    tokens = torch.randint(0, 1000, (P, 1), device=dev)
    for i in range(P):
        tokens[i, 0] = 42 + i  # 确保不同的起始token
    
    print(f"📝 Initial tokens: {tokens.flatten().tolist()}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    # 关键：这里只调用一次！
    outputs = var.true_batch_autoregressive_infer(tokens, max_steps=32)
    
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"📊 Generated outputs shape: {outputs.shape}")
    print(f"📊 First few tokens per stream:")
    for i in range(P):
        print(f"   Stream {i}: {outputs[i, :5].cpu().tolist()}")
    
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
        diversity = 1.0 - (sum(similarities) / len(similarities))
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")
    return latency, peak_mem, diversity

print("🚀 Running TRUE batch tests...")
print("Expected: Memory should scale with P, Latency should stay reasonable")

for P in (1, 2, 4):
    run_true_batch_test(P)
SCRIPT_EOF

echo "🚀 执行真正的批处理测试..."
python3 fix_true_batch.py
EOF

echo "✅ 真正的批处理测试执行完成!"