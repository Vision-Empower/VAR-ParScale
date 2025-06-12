#!/bin/bash
# Execute final VAR-ParScale test on H100
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 Executing final VAR-ParScale test on H100"
echo "="*60

# Upload and execute test script
$SSH_CMD << 'EOF'
cd /root

# Create the final test script
cat > /root/test_shared_batch.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
"""
Final VAR-ParScale Implementation with True Shared Batch Processing
"""

import torch
import torch.nn.functional as F
import os
import sys
import time

def main():
    print("🚀 Final VAR-ParScale: True Shared Batch Processing")
    print("=" * 60)
    
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
    
    # Apply patches to VAR
    patch_var_for_batch(var)
    
    print("✅ Models loaded and patched")
    
    # Test P=1, P=2, P=4
    for P in [1, 2, 4]:
        print(f"\n🔧 Testing P={P}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        # Generate with true batch processing
        with torch.no_grad():
            outputs = var.autoregressive_infer_batch(
                torch.randint(0, var.class_embed.num_embeddings, (P, 1), device=device),
                max_steps=64, top_p=0.95, top_k=900
            )
        
        torch.cuda.synchronize()
        latency = (time.time() - start_time) * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Compute diversity
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

def _sample_topk_topp(logits, top_k, top_p):
    """Sample from logits with top-k and top-p filtering"""
    # Top-k filtering
    if top_k > 0:
        top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        indices_to_remove = logits < top_k_logits[..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)

def encode_prompt(self, toks):
    """Encode prompt tokens to embeddings"""
    return self.class_emb(toks)

def _step(self, x, t):
    """Single forward step"""
    return self.forward(x, t)

def autoregressive_infer_batch(self, toks, max_steps=256, top_p=.95, top_k=900):
    """Batched autoregressive inference"""
    B_all = toks.size(0)
    x = encode_prompt(self, toks)
    outs = [toks[:, :1]]
    
    for t in range(1, max_steps):
        logits = _step(self, x, t).squeeze(1)
        nxt = _sample_topk_topp(logits, top_k, top_p).unsqueeze(1)
        outs.append(nxt)
        x = self.class_emb(nxt)
    
    return torch.cat(outs, 1)

def _patch_all_attention_for_batch(self):
    """Patch attention layers for batch processing"""
    for block in self.blocks:
        if hasattr(block, 'attn'):
            original_forward = block.attn.forward
            
            def patched_forward(x, cache=None):
                B, L, C = x.shape
                
                # Use mat_qkv instead of qkv for VAR
                if hasattr(block.attn, 'mat_qkv'):
                    qkv = block.attn.mat_qkv(x)
                    q, k, v = qkv.chunk(3, dim=-1)
                else:
                    # Fallback to original method
                    return original_forward(x)
                
                # Reshape for multi-head attention
                num_heads = getattr(block.attn, 'num_heads', 16)
                head_dim = C // num_heads
                
                q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
                k = k.view(B, L, num_heads, head_dim).transpose(1, 2)
                v = v.view(B, L, num_heads, head_dim).transpose(1, 2)
                
                # Attention computation
                scale = head_dim ** -0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = F.softmax(attn, dim=-1)
                
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).contiguous().view(B, L, C)
                
                # Apply output projection if exists
                if hasattr(block.attn, 'proj'):
                    out = block.attn.proj(out)
                
                return out
            
            block.attn.forward = patched_forward

def patch_var_for_batch(var):
    """Apply all patches to VAR model"""
    # Add methods to VAR instance
    import types
    var.encode_prompt = types.MethodType(encode_prompt, var)
    var._step = types.MethodType(_step, var)
    var.autoregressive_infer_batch = types.MethodType(autoregressive_infer_batch, var)
    var._patch_all_attention_for_batch = types.MethodType(_patch_all_attention_for_batch, var)
    
    # Apply attention patches
    var._patch_all_attention_for_batch()

if __name__ == "__main__":
    main()
SCRIPT_EOF

# Download models if needed
if [ ! -f "vae_ch160v4096z32.pth" ]; then
    echo "📥 Downloading VAE model..."
    wget -q https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
fi
if [ ! -f "var_d16.pth" ]; then
    echo "📥 Downloading VAR model..."
    wget -q https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
fi

# Setup VAR if needed
if [ ! -d "VAR" ]; then
    echo "📂 Cloning VAR repository..."
    git clone https://github.com/FoundationVision/VAR.git
    echo "📦 Installing dependencies..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    pip3 install huggingface_hub scipy numpy pillow
fi

# Run the final test
echo "🚀 Running final VAR-ParScale test..."
python3 test_shared_batch.py
EOF

echo "✅ Final test execution complete!"