#!/bin/bash
# 最后一刀：删除for-loop，实现真正的KV cache累积
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🔪 最后一刀：删除for-loop，实现真正的KV cache累积"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 实现最后一刀：真正的单次batch调用
cat > final_last_cut.py << 'SCRIPT_EOF'
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

# ---------- 第一步：最小KV cache调试补丁 ----------
import types

def _patch_attention_for_kv_debug(model, debug_layers=[0, 1]):
    """先只patch前两层用于调试KV cache累积"""
    for layer_idx in debug_layers:
        if layer_idx >= len(model.blocks):
            continue
            
        block = model.blocks[layer_idx]
        attn = block.attn
        
        if getattr(attn, "_kv_patched", False):
            continue
            
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        def kv_accumulating_forward(self, x, kv_cache):
            # x: [B_all, 1, C]
            B_all, seq_len, C = x.shape
            
            # QKV projection
            qkv = self.mat_qkv(x)  # [B_all, 1, 3*C]
            qkv = qkv.view(B_all, seq_len, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_all, num_heads, 1, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # KV cache累积 - 这是关键！
            layer_key = f"layer_{layer_idx}"
            if layer_key in kv_cache and "k" in kv_cache[layer_key]:
                # 拼接历史KV
                prev_k = kv_cache[layer_key]["k"]  # [B_all, num_heads, t-1, head_dim]
                prev_v = kv_cache[layer_key]["v"]  # [B_all, num_heads, t-1, head_dim]
                
                k = torch.cat([prev_k, k], dim=2)  # [B_all, num_heads, t, head_dim]
                v = torch.cat([prev_v, v], dim=2)  # [B_all, num_heads, t, head_dim]
                
                if layer_idx == 0:  # 只在layer 0打印调试信息
                    print(f"   🔄 Layer {layer_idx}: KV {prev_k.shape} + {qkv[1].shape} → {k.shape}")
            else:
                if layer_idx == 0:
                    print(f"   🆕 Layer {layer_idx}: KV初始化 {k.shape}")
            
            # 更新cache
            if layer_key not in kv_cache:
                kv_cache[layer_key] = {}
            kv_cache[layer_key]["k"] = k
            kv_cache[layer_key]["v"] = v
            
            # Attention计算
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)
            
            # 重组输出
            out = out.transpose(1, 2).contiguous().view(B_all, seq_len, C)
            return self.proj(out)
        
        # 替换attention forward
        attn.forward = types.MethodType(kv_accumulating_forward, attn)
        attn._kv_patched = True
        
        # 修改block forward支持kv_cache
        original_block_forward = block.forward
        def block_forward_with_cache(self, x, kv_cache=None):
            if kv_cache is None:
                return original_block_forward(x)
            
            # 只在被patch的层使用新逻辑
            if layer_idx in debug_layers:
                shortcut = x
                attn_out = self.attn(x, kv_cache)
                x = shortcut + self.drop_path(attn_out)
                
                shortcut = x
                ffn_out = self.ffn(x)
                x = shortcut + self.drop_path(ffn_out)
                
                return x
            else:
                return original_block_forward(x)
        
        block.forward = types.MethodType(block_forward_with_cache, block)

print("🔧 Applying KV debug patches to layers 0,1...")
_patch_attention_for_kv_debug(var, debug_layers=[0, 1])

# ---------- 实现核心方法 ----------
def encode_prompt(self, tokens):
    """编码tokens并初始化KV cache"""
    print(f"🎯 Encoding prompt: {tokens.shape}")
    
    # 清空KV cache
    self._kv_cache = {}
    
    # 编码首个token
    return self.class_emb(tokens)

def forward_step_single(self, x, step):
    """单步前向传播，使用累积的KV cache"""
    B_all, seq_len, C = x.shape
    print(f"📦 Step {step}: input {x.shape}")
    
    # 只通过前两层（调试模式）
    for layer_idx in [0, 1]:
        if layer_idx < len(self.blocks):
            x = self.blocks[layer_idx](x, kv_cache=self._kv_cache)
    
    # 简化的输出投影（用于调试）
    logits = self.head(x)
    return logits

@torch.no_grad()
def debug_kv_accumulation(self, tokens, max_steps=5):
    """调试KV cache累积机制"""
    print(f"🔍 DEBUG: KV cache accumulation test")
    
    x = encode_prompt(self, tokens)
    
    for step in range(max_steps):
        print(f"\n⏰ Step {step}:")
        _ = forward_step_single(self, x, step)
        
        # 检查KV cache状态
        if "layer_0" in self._kv_cache and "k" in self._kv_cache["layer_0"]:
            k_shape = self._kv_cache["layer_0"]["k"].shape
            print(f"   ✅ Layer 0 KV cache shape: {k_shape}")
            
            # 验证时间维度增长
            expected_t = step + 1
            actual_t = k_shape[2]
            if actual_t == expected_t:
                print(f"   ✅ Time dimension correct: {actual_t}")
            else:
                print(f"   ❌ Time dimension mismatch: expected {expected_t}, got {actual_t}")
        else:
            print(f"   ❌ No KV cache found")
        
        # 为下一步准备输入（使用相同的token embedding）
        x = self.class_emb(tokens[:, :1])

# 绑定方法到VAR
var.encode_prompt = types.MethodType(encode_prompt, var)
var.forward_step_single = types.MethodType(forward_step_single, var)
var.debug_kv_accumulation = types.MethodType(debug_kv_accumulation, var)

print("\n🧪 第一步：KV Cache累积调试")
print("="*40)

# 测试KV cache累积
test_tokens = torch.randint(0, 1000, (2, 1), device=dev)
print(f"📝 Test tokens: {test_tokens.flatten().tolist()}")

var.debug_kv_accumulation(test_tokens, max_steps=5)

print("\n" + "="*60)
print("如果看到时间维度正确递增，则继续实现完整版本...")

# ---------- 第二步：完整实现（如果调试通过） ----------
def _patch_all_layers_for_batch(model):
    """为所有16层实现KV cache批处理"""
    print("🔧 Patching all 16 layers for batch processing...")
    
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        
        if getattr(attn, "_full_patched", False):
            continue
            
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        def full_kv_forward(self, x, kv_cache):
            B_all, seq_len, C = x.shape
            
            # QKV projection
            qkv = self.mat_qkv(x).view(B_all, seq_len, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # KV cache累积
            layer_key = f"layer_{layer_idx}"
            if layer_key in kv_cache and "k" in kv_cache[layer_key]:
                prev_k = kv_cache[layer_key]["k"]
                prev_v = kv_cache[layer_key]["v"]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            
            # 更新cache
            if layer_key not in kv_cache:
                kv_cache[layer_key] = {}
            kv_cache[layer_key]["k"] = k
            kv_cache[layer_key]["v"] = v
            
            # Attention
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)
            
            out = out.transpose(1, 2).contiguous().view(B_all, seq_len, C)
            return self.proj(out)
        
        attn.forward = types.MethodType(full_kv_forward, attn)
        attn._full_patched = True
        
        # Block forward
        def full_block_forward(self, x, kv_cache=None):
            if kv_cache is None:
                # 回退到原始forward
                shortcut = x
                x = shortcut + self.drop_path(self.attn(x))
                shortcut = x
                x = shortcut + self.drop_path(self.ffn(x))
                return x
            else:
                shortcut = x
                attn_out = self.attn(x, kv_cache)
                x = shortcut + self.drop_path(attn_out)
                
                shortcut = x
                ffn_out = self.ffn(x)
                x = shortcut + self.drop_path(ffn_out)
                
                return x
        
        block.forward = types.MethodType(full_block_forward, block)

def full_forward_step_single(self, x, step):
    """完整的单步前向传播（16层）"""
    for layer_idx, block in enumerate(self.blocks):
        x = block(x, kv_cache=self._kv_cache)
    
    logits = self.head(x)
    return logits

@torch.no_grad()
def autoregressive_infer_batch(self, tokens, max_steps=32):
    """真正的单次batch调用！删除所有for-loop！"""
    B_all = tokens.size(0)
    print(f"🚀 TRUE BATCH INFERENCE: B_all={B_all}, max_steps={max_steps}")
    print("🔪 NO MORE FOR-LOOPS!")
    
    # 初始化
    x = self.encode_prompt(tokens[:, :1])  # [B_all, 1, C]
    generated = [tokens[:, :1]]
    
    # 时间步循环 - 这是唯一的循环！
    for t in range(1, max_steps):
        if t % 10 == 0:
            print(f"⏰ Time step {t}/{max_steps}")
            # 检查KV cache状态
            if "layer_0" in self._kv_cache and "k" in self._kv_cache["layer_0"]:
                cache_shape = self._kv_cache["layer_0"]["k"].shape
                print(f"   ✅ Layer 0 KV cache: {cache_shape}")
        
        # 单步前向 - 处理整个batch
        logits = self.full_forward_step_single(x, t)  # [B_all, 1, vocab]
        logits = logits.squeeze(1)  # [B_all, vocab]
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1)  # [B_all, 1]
        
        generated.append(next_tokens)
        x = self.class_emb(next_tokens)
    
    result = torch.cat(generated, dim=1)  # [B_all, max_steps]
    print(f"✅ Batch inference complete: {result.shape}")
    return result

# 为完整测试准备
_patch_all_layers_for_batch(var)
var.full_forward_step_single = types.MethodType(full_forward_step_single, var)
var.autoregressive_infer_batch = types.MethodType(autoregressive_infer_batch, var)

# ---------- 第三步：最终测试并获得3行结果 ----------
def test_final_true_batch(P):
    print(f"\n{'='*60}")
    print(f"🏁 FINAL TRUE BATCH TEST: P={P}")
    print(f"{'='*60}")
    
    # 创建P个输入
    tokens = torch.randint(100, 200, (P, 1), device=dev)
    print(f"📝 Input tokens: {tokens.flatten().tolist()}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    try:
        # 关键：单次调用！
        outputs = var.autoregressive_infer_batch(tokens, max_steps=32)
        success = True
    except Exception as e:
        print(f"❌ Error: {e}")
        success = False
        outputs = None
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # 计算diversity
    diversity = 0.0
    if success and outputs is not None and P > 1:
        similarities = []
        for i in range(P):
            for j in range(i + 1, P):
                sim = F.cosine_similarity(
                    outputs[i].float(), outputs[j].float(), dim=0
                ).item()
                similarities.append(abs(sim))
        
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            diversity = 1.0 - avg_sim
    
    print(f"P={P:<2}  Lat {latency:>6.1f}ms  PeakMem {peak_mem:.2f}GB  Diversity {diversity:.3f}")
    
    return {
        'latency': latency,
        'peak_mem': peak_mem,
        'diversity': diversity,
        'success': success
    }

print("\n🏁 最后一刀测试：真正的单次batch调用")
print("="*50)

results = {}
baseline_mem = None

for P in [1, 2, 4]:
    result = results[P] = test_final_true_batch(P)
    
    if P == 1:
        baseline_mem = result['peak_mem']
    else:
        if baseline_mem:
            mem_ratio = result['peak_mem'] / baseline_mem
            print(f"   Memory scaling: {mem_ratio:.2f}x baseline")
            
            # 检查是否达标
            if P == 2 and mem_ratio >= 1.6:
                print(f"   ✅ P=2 memory scaling达标!")
            elif P == 4 and mem_ratio >= 2.5:
                print(f"   ✅ P=4 memory scaling达标!")
            elif mem_ratio < 1.2:
                print(f"   🚨 Memory scaling不足 - KV cache可能未累积")
            else:
                print(f"   ⚠️ Memory scaling部分达标")

print(f"\n🎯 最终3行结果:")
print("-" * 50)
for P in [1, 2, 4]:
    r = results[P]
    print(f"P={P}  Lat {r['latency']:>6.1f}ms  PeakMem {r['peak_mem']:.2f}GB  Diversity {r['diversity']:.3f}")

# 检查是否可以开香槟
mem_scaling_ok = (
    results[2]['peak_mem'] / results[1]['peak_mem'] >= 1.6 and
    results[4]['peak_mem'] / results[1]['peak_mem'] >= 2.5
)

if mem_scaling_ok and all(r['success'] for r in results.values()):
    print(f"\n🍾 CHAMPAGNE TIME! 🍾")
    print(f"✅ Memory scaling达标!")
    print(f"✅ All tests successful!")
    print(f"🎉 VAR-ParScale真正的共享骨干完成! 🎉")
else:
    print(f"\n🔧 Still working...")
    if not mem_scaling_ok:
        print(f"❌ Memory scaling needs work")
    for P, r in results.items():
        if not r['success']:
            print(f"❌ P={P} failed")
SCRIPT_EOF

echo "🔪 执行最后一刀..."
python3 final_last_cut.py
EOF

echo "✅ 最后一刀执行完成!"