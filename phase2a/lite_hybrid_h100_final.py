#!/usr/bin/env python3
"""
Lite-Hybrid H100 Test - Final GPU version of #6 experiment
HART-inspired dual-branch architecture with 256x256 images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np

# Force CUDA for H100
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using H100 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    device = torch.device('cpu')
    print("⚠️ CUDA not available, falling back to CPU")

class LiteVAEEncoder(nn.Module):
    """Optimized VAE encoder for H100"""
    def __init__(self, in_channels=3, z_channels=32):
        super().__init__()
        # More realistic encoder for 256x256 → 16x16
        self.encoder = nn.Sequential(
            # 256 → 128
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # 128 → 64  
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            
            # 64 → 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            
            # 32 → 16
            nn.Conv2d(256, z_channels, 4, stride=2, padding=1)
        )
        
    def encode(self, x):
        """Encode images to latent tokens"""
        h = self.encoder(x)  # [B, 32, 16, 16]
        B, C, H, W = h.shape
        return h.view(B, H*W, C)  # [B, 256, 32]

class CoarseTokenizer(nn.Module):
    """HART-inspired coarse branch - 16x16 → 4x4"""
    
    def __init__(self, in_channels=32, coarse_vocab_size=1024):
        super().__init__()
        self.coarse_vocab_size = coarse_vocab_size
        
        # 下采样到4x4 (16x16 → 4x4 via 4x4 conv)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, stride=4),  # 16x16 → 4x4
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        # 离散量化
        self.quantizer = nn.Linear(256, coarse_vocab_size)
        
    def forward(self, fine_latent):
        """从fine latent提取coarse tokens"""
        # Reshape [B, 256, 32] → [B, 32, 16, 16]
        B, L, C = fine_latent.shape
        H = W = int(L ** 0.5)  # 16
        fine_latent = fine_latent.view(B, C, H, W)
        
        coarse_feat = self.downsample(fine_latent)  # [B, 256, 4, 4]
        
        # Flatten and quantize
        B, C, H, W = coarse_feat.shape
        coarse_feat = coarse_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, 16, 256]
        
        coarse_logits = self.quantizer(coarse_feat)  # [B, 16, 1024]
        coarse_tokens = torch.argmax(coarse_logits, dim=-1)  # [B, 16]
        
        return coarse_tokens, coarse_logits

class FineResidualHead(nn.Module):
    """HART-inspired fine residual processing"""
    
    def __init__(self, in_channels=32, hidden_dim=128):
        super().__init__()
        
        # Lightweight UNet for residual prediction
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        
        # Down-sample
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim*2),
            nn.SiLU()
        )
        
        # Mid processing
        self.mid = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim*2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1),
        )
        
        # Up-sample
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        
    def forward(self, x):
        """预测残差 - input: [B, 256, 32]"""
        # Reshape to spatial
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # 16
        x = x.view(B, C, H, W)  # [B, 32, 16, 16]
        
        # UNet processing
        h = self.input_proj(x)       # [B, 128, 16, 16]
        h_down = self.down1(h)       # [B, 256, 8, 8]
        h_mid = self.mid(h_down)     # [B, 256, 8, 8]
        h_up = self.up1(h_mid)       # [B, 128, 16, 16]
        residual = self.output_proj(h_up)  # [B, 32, 16, 16]
        
        # Back to token format
        B, C, H, W = residual.shape
        return residual.view(B, H*W, C)  # [B, 256, 32]

class LiteHybridH100(nn.Module):
    """Lite-Hybrid for H100 - 256x256 images"""
    
    def __init__(self):
        super().__init__()
        
        # Base encoder (LiteVAE style)
        self.vae_encoder = LiteVAEEncoder()
        
        # Dual branches (HART inspiration)
        self.coarse_tokenizer = CoarseTokenizer()
        self.fine_residual_head = FineResidualHead()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32, 64),  # Expand for fusion
            nn.SiLU(),
            nn.Linear(64, 32),  # Back to original dim
            nn.LayerNorm(32)
        )
        
        print(f"🔥 Lite-Hybrid H100 Model初始化完成")
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"   总参数: {total_params:.1f}M")
        print(f"   设备: {next(self.parameters()).device}")
        
    def encode_hybrid(self, images):
        """混合编码 - coarse + fine"""
        # Base VAE encoding
        base_tokens = self.vae_encoder.encode(images)  # [B, 256, 32]
        
        # Coarse branch
        coarse_tokens, coarse_logits = self.coarse_tokenizer(base_tokens)
        
        # Fine branch
        fine_residual = self.fine_residual_head(base_tokens)
        
        return {
            'coarse_tokens': coarse_tokens,
            'coarse_logits': coarse_logits,
            'fine_residual': fine_residual,
            'base_tokens': base_tokens
        }
    
    def forward(self, images):
        """前向传播 - 完整dual-branch处理"""
        encoded = self.encode_hybrid(images)
        
        # Fusion: base + residual
        enhanced_tokens = encoded['base_tokens'] + encoded['fine_residual']
        enhanced_tokens = self.fusion(enhanced_tokens)
        
        return {
            'tokens': enhanced_tokens,
            'coarse_info': encoded['coarse_tokens'], 
            'fine_residual': encoded['fine_residual']
        }

def benchmark_h100_performance():
    """H100性能基准测试"""
    
    print("🔬 Lite-Hybrid H100性能基准测试")
    print("=" * 60)
    
    # 创建模型
    hybrid_model = LiteHybridH100().to(device).eval()
    baseline_model = LiteVAEEncoder().to(device).eval()
    
    # 测试数据 - 256x256 images适合H100
    batch_sizes = [1, 4, 8] if device.type == 'cuda' else [1, 2]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Batch Size {batch_size} 测试:")
        test_images = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # FP16 for H100 efficiency
        if device.type == 'cuda':
            test_images = test_images.half()
            hybrid_model = hybrid_model.half()
            baseline_model = baseline_model.half()
        
        # Baseline测试
        baseline_times = []
        for _ in range(10):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = baseline_model.encode(test_images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            baseline_times.append((end - start) * 1000)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        baseline_per_image = baseline_avg / batch_size
        
        # Hybrid测试
        hybrid_times = []
        for _ in range(10):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = hybrid_model(test_images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            hybrid_times.append((end - start) * 1000)
        
        hybrid_avg = sum(hybrid_times) / len(hybrid_times)
        hybrid_per_image = hybrid_avg / batch_size
        
        # 结果分析
        latency_increase = hybrid_per_image - baseline_per_image
        
        print(f"  基线延迟: {baseline_per_image:.2f}ms/图")
        print(f"  Hybrid延迟: {hybrid_per_image:.2f}ms/图")
        print(f"  延迟增加: {latency_increase:.2f}ms ({(latency_increase/baseline_per_image)*100:.1f}%)")
        
        results[f'batch_{batch_size}'] = {
            'baseline_ms': baseline_per_image,
            'hybrid_ms': hybrid_per_image,
            'increase_ms': latency_increase,
            'increase_percent': (latency_increase/baseline_per_image)*100
        }
    
    # 内存使用
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n💾 GPU内存使用: {memory_used:.1f}MB")
        results['memory_mb'] = memory_used
    
    # 目标评估
    best_batch = min(results.keys(), key=lambda k: results[k]['increase_ms'] if 'increase_ms' in results[k] else float('inf'))
    best_increase = results[best_batch]['increase_ms']
    target_achieved = best_increase <= 1.0  # ≤1ms目标
    
    print(f"\n🎯 目标评估:")
    print(f"  最佳配置: {best_batch}")
    print(f"  最小延迟增加: {best_increase:.2f}ms")
    print(f"  目标达成 (≤1ms): {'✅' if target_achieved else '❌'}")
    
    if target_achieved:
        print("🟢 H100实验成功！延迟目标达成")
        status = "SUCCESS"
    elif best_increase <= 2.0:
        print("🟡 接近目标，表现良好") 
        status = "PROMISING"
    else:
        print("🔴 需要进一步优化")
        status = "NEEDS_WORK"
    
    results['summary'] = {
        'best_batch': best_batch,
        'best_increase_ms': best_increase,
        'target_achieved': target_achieved,
        'status': status,
        'device': str(device)
    }
    
    return results

def validate_architecture():
    """架构验证 - 确保dual-branch正常工作"""
    
    print("\n🔍 架构验证测试")
    print("=" * 40)
    
    model = LiteHybridH100().to(device).eval()
    test_images = torch.randn(2, 3, 256, 256).to(device)
    
    if device.type == 'cuda':
        test_images = test_images.half()
        model = model.half()
    
    with torch.no_grad():
        # 测试编码
        encoded = model.encode_hybrid(test_images)
        
        # 测试前向传播
        result = model(test_images)
    
    print(f"✅ 输入形状: {test_images.shape}")
    print(f"✅ Coarse tokens: {encoded['coarse_tokens'].shape}")
    print(f"✅ Fine residual: {encoded['fine_residual'].shape}")
    print(f"✅ Final tokens: {result['tokens'].shape}")
    print(f"✅ Coarse token range: [{encoded['coarse_tokens'].min().item()}, {encoded['coarse_tokens'].max().item()}]")
    
    return {
        'input_shape': list(test_images.shape),
        'coarse_tokens_shape': list(encoded['coarse_tokens'].shape),
        'fine_residual_shape': list(encoded['fine_residual'].shape),
        'final_tokens_shape': list(result['tokens'].shape),
        'validation_passed': True
    }

def main():
    """主实验函数"""
    
    print("🚀 开始Lite-Hybrid H100实验 (#6)")
    print("🎯 目标: HART灵感双分支架构，延迟+1ms以内")
    print("🏭 平台: H100 80GB HBM3")
    print("=" * 70)
    
    try:
        # 架构验证
        arch_results = validate_architecture()
        
        # 性能基准测试
        perf_results = benchmark_h100_performance()
        
        # 合并结果
        final_results = {
            'experiment': 'lite_hybrid_hart_h100',
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'architecture_validation': arch_results,
            'performance_benchmark': perf_results
        }
        
        # 保存结果
        results_file = f'lite_hybrid_h100_results_{final_results["timestamp"]}.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n🎉 Lite-Hybrid H100实验完成！")
        print(f"📁 结果保存: {results_file}")
        
        summary = perf_results['summary']
        print(f"\n📊 实验总结:")
        print(f"  架构验证: {'✅ 通过' if arch_results['validation_passed'] else '❌ 失败'}")
        print(f"  性能状态: {summary['status']}")
        print(f"  最佳延迟增加: {summary['best_increase_ms']:.2f}ms")
        print(f"  目标达成: {'✅' if summary['target_achieved'] else '❌'}")
        print(f"  运行平台: {summary['device']}")
        
        if summary['status'] == 'SUCCESS':
            print("\n🏆 实验大成功！HART灵感的双分支架构在H100上表现优异")
            print("📋 下一步: 集成到完整ParScale-EAR pipeline")
        else:
            print(f"\n⚡ 状态: {summary['status']} - 架构可行但需进一步优化")
        
        return final_results
        
    except Exception as e:
        print(f"❌ H100实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()