#!/bin/bash
# 执行真正的共享骨干VAR-ParScale（基于正确VAR结构）
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 执行真正的共享骨干VAR-ParScale（基于正确VAR结构）"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 基于正确的VAR结构创建真共享骨干实现
cat > test_true_shared_final.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var

# ---------- build models ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()

# ---------- patch attention for batch processing ----------
import types

def _patch_attention_for_batch(model):
    """修补attention以支持batch processing和KV caching"""
    for blk in model.blocks:
        attn = blk.attn
        if getattr(attn, "_patched", False):
            continue
            
        nh = attn.num_heads
        hdim = attn.head_dim
        orig_forward = attn.forward
        
        def batched_attn_forward(self, x, cache=None):
            # x: [B, 1, C]
            B, L, C = x.shape
            qkv = self.mat_qkv(x).view(B, L, 3, nh, hdim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, nh, 1, hdim]
            
            if cache is not None and "k" in cache:
                # 拼接历史KV cache
                k = torch.cat([cache["k"], k], dim=2)  # [B, nh, t+1, hdim]
                v = torch.cat([cache["v"], v], dim=2)  # [B, nh, t+1, hdim]
            
            if cache is not None:
                cache["k"], cache["v"] = k, v
            
            # scaled dot-product attention
            scale = hdim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)  # [B, nh, 1, hdim]
            
            # reshape and project
            out = out.transpose(1, 2).reshape(B, L, C)
            return self.proj(out)
        
        attn.forward = types.MethodType(batched_attn_forward, attn)
        attn._patched = True
        
        # 修补block的forward以支持cache参数
        orig_blk_forward = blk.forward
        def block_forward_with_cache(self, x, cache=None):
            if cache is None:
                return orig_blk_forward(x)
            else:
                # AdaLNSelfAttn forward with cache
                shortcut = x
                x = self.attn(x, cache)
                x = shortcut + self.drop_path(x)
                shortcut = x
                x = self.ffn(x)
                x = shortcut + self.drop_path(x)
                return x
        
        blk.forward = types.MethodType(block_forward_with_cache, blk)

_patch_attention_for_batch(var)

# ---------- add batched inference methods ----------
def encode_tokens(self, tokens):
    """初始化KV cache并编码tokens"""
    self._kv_cache = [{} for _ in self.blocks]
    return self.class_emb(tokens)

def forward_step(self, x):
    """前向传播一步，使用KV cache"""
    for layer_idx, block in enumerate(self.blocks):
        x = block(x, cache=self._kv_cache[layer_idx])
    return self.ln_f(x) @ self.class_emb.weight.T

@torch.no_grad()
def batched_autoregressive_infer(self, init_tokens, max_steps=128, top_p=0.95, top_k=900):
    """真正的批量自回归推理"""
    B = init_tokens.size(0)
    x = encode_tokens(self, init_tokens[:, :1])  # [B, 1, C]
    generated = [init_tokens[:, :1]]  # list of [B, 1]
    
    for step in range(1, max_steps):
        logits = forward_step(self, x).squeeze(1)  # [B, vocab_size]
        
        # 简化采样（可以改进为top-k/top-p）
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1)  # [B, 1]
        
        generated.append(next_tokens)
        x = self.class_emb(next_tokens)  # 准备下一步的输入
    
    return torch.cat(generated, dim=1)  # [B, max_steps]

# 将方法绑定到VAR实例
var.encode_tokens = types.MethodType(encode_tokens, var)
var.forward_step = types.MethodType(forward_step, var)
var.batched_autoregressive_infer = types.MethodType(batched_autoregressive_infer, var)

# ---------- run tests for P = 1/2/4 ----------
def run_test(P):
    tokens = torch.randint(0, 1000, (P, 1), device=dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    outputs = var.batched_autoregressive_infer(tokens, max_steps=128)
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
                    outputs[i].float(), outputs[j].float(), dim=0
                ).item()
                similarities.append(abs(sim))
        diversity = 1.0 - (sum(similarities) / len(similarities))
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")

print("🚀 Running true shared backbone tests...")
for P in (1, 2, 4):
    run_test(P)
SCRIPT_EOF

echo "🚀 执行真共享骨干测试..."
python3 test_true_shared_final.py
EOF

echo "✅ 真共享骨干测试执行完成!"