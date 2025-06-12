#!/bin/bash
# 执行最终修正版VAR-ParScale真共享骨干测试
set -e

SSH_CMD="ssh -p 11292 root@inst-gw.cloudexe.tech"
echo "🚀 执行最终修正版VAR-ParScale真共享骨干测试"
echo "="*60

$SSH_CMD << 'EOF'
cd /root/VAR

# 添加补丁到 models/basic_var.py
cat >> models/basic_var.py << 'PATCH_EOF'

# ------------- add once -------------
def _patch_all_attention_for_batch(model):
    """
    把 16 个 AdaLNSelfAttn.attn 打补丁，使其一次吃 [P·B,1,C]，
    并把 KV 累积到 cache["k"], cache["v"] （形状 [B_all,n_h,t,d]）
    """
    import types, torch, torch.nn.functional as F
    for blk in model.blocks:
        attn = blk.attn
        if getattr(attn, "_batched", False):
            continue

        nh   = attn.num_heads
        hdim = attn.head_dim

        def fwd(self, x, cache):
            # x:[B,1,C]
            B,_,C = x.shape
            qkv = self.mat_qkv(x).view(B,1,3,nh,hdim).permute(2,0,3,1,4)  # (3,B,nh,1,hdim)
            q,k,v = qkv[0],qkv[1],qkv[2]
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            y = F.scaled_dot_product_attention(q,k,v)                      # (B,nh,1,hdim)
            y = y.transpose(1,2).reshape(B,1,C)
            return self.proj(y)

        attn.forward = types.MethodType(fwd, attn)
        attn._batched = True
PATCH_EOF

# 创建测试脚本
cat > test_shared_batch.py << 'SCRIPT_EOF'
import torch, time, os, sys, torch.nn.functional as F
os.chdir("/root/VAR"); sys.path.append("/root/VAR")
from models import build_vae_var
from models.basic_var import _patch_all_attention_for_batch

# ---------- build & patch ----------
dev = "cuda"
vae, var = build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,
                         device=dev,patch_nums=(1,2,3,4,5,6,8,10,13,16),
                         num_classes=1000,depth=16,shared_aln=False)
vae.cuda().eval(); var.cuda().eval()
_patch_all_attention_for_batch(var)

# ---------- add batched AR loop ----------
import types
def encode(self,t): self._kv=[{}for _ in self.blocks]; return self.class_emb(t)
def step(self,x):                                   # one token
    for lid,b in enumerate(self.blocks): x=b(x,cache=self._kv[lid])
    return self.ln_f(x) @ self.class_emb.weight.T
@torch.no_grad()
def infer_batch(self, tok, T=256, p=.95,k=900):
    B=tok.size(0); x=encode(self,tok[:,:1])
    outs=[tok[:,:1]]
    for _ in range(1,T):
        log=step(self,x).squeeze(1)
        nxt=(log.softmax(-1)).multinomial(1)        # 简化采样
        outs.append(nxt); x=self.class_emb(nxt)
    return torch.cat(outs,1)
var.encode_prompt      = types.MethodType(encode,var)
var._step              = types.MethodType(step,var)
var.autoregressive_infer_batch = types.MethodType(infer_batch,var)

# ---------- run P = 1/2/4 ----------
def run(P):
    tok=torch.randint(0,1000,(P,1),device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0=time.time()
    out=var.autoregressive_infer_batch(tok,T=128)
    torch.cuda.synchronize()
    lat=(time.time()-t0)*1000
    mem=torch.cuda.max_memory_allocated()/1e9
    # diversity
    div=0.0
    if P>1:
        sims=[]
        for i in range(P):
            for j in range(i+1,P):
                sims.append(F.cosine_similarity(out[i].float(),out[j].float(),dim=0))
        div=1-float(torch.stack(sims).mean())
    print(f"P={P:<2}  Lat {lat:>6.1f}ms  PeakMem {mem:.2f}GB  Diversity {div:.3f}")

for P in (1,2,4):
    run(P)
SCRIPT_EOF

echo "🚀 执行最终真共享骨干测试..."
python3 test_shared_batch.py
EOF

echo "✅ 最终测试执行完成!"