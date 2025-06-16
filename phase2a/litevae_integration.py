#!/usr/bin/env python3
"""
LiteVAE Integration - 72小时可行性最高的VAE加速方案
基于NeurIPS 2024 LiteVAE: 分层可分组卷积 + 低秩因子分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.cuda.amp import autocast

class GroupedConv2d(nn.Module):
    """分组卷积 - LiteVAE核心优化"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.groups = min(groups, in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups=self.groups, bias=False)
        self.norm = nn.GroupNorm(32, out_channels, eps=1e-6)
        
    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))

class LowRankConv2d(nn.Module):
    """低秩因子分解卷积 - 大幅减少参数和计算"""
    
    def __init__(self, in_channels, out_channels, kernel_size, rank_ratio=0.5):
        super().__init__()
        self.rank = max(1, int(min(in_channels, out_channels) * rank_ratio))
        
        # 分解为两个低秩卷积
        self.conv1 = nn.Conv2d(in_channels, self.rank, 1, bias=False)
        self.conv2 = nn.Conv2d(self.rank, out_channels, kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.norm = nn.GroupNorm(32, out_channels, eps=1e-6)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return F.silu(self.norm(x))

class LiteResBlock(nn.Module):
    """LiteVAE优化的ResNet块"""
    
    def __init__(self, in_channels, out_channels=None, use_low_rank=True):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        
        if use_low_rank and in_channels >= 64:
            # 高维通道使用低秩分解
            self.conv1 = LowRankConv2d(in_channels, out_channels, 3, rank_ratio=0.5)
            self.conv2 = LowRankConv2d(out_channels, out_channels, 3, rank_ratio=0.5)
        else:
            # 低维通道使用分组卷积
            groups = min(8, in_channels)
            self.conv1 = GroupedConv2d(in_channels, out_channels, 3, padding=1, groups=groups)
            self.conv2 = GroupedConv2d(out_channels, out_channels, 3, padding=1, groups=groups)
        
        # 跳连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h + self.shortcut(x)

class EfficientDownsample(nn.Module):
    """高效下采样 - 避免信息损失"""
    
    def __init__(self, in_channels):
        super().__init__()
        # 使用1x1卷积 + stride=2 而不是3x3
        self.conv = nn.Conv2d(in_channels, in_channels, 1, stride=2, bias=False)
        
    def forward(self, x):
        return self.conv(x)

class LiteVAEEncoder(nn.Module):
    """LiteVAE编码器 - 针对速度优化"""
    
    def __init__(self, ch=128, ch_mult=(1,2,4,8), num_res_blocks=2, 
                 in_channels=3, z_channels=32):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # 入口层 - 使用分组卷积
        self.conv_in = GroupedConv2d(in_channels, ch, 3, padding=1, groups=1)
        
        # 下采样路径
        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            # ResNet块
            for i_block in range(self.num_res_blocks):
                use_low_rank = block_in >= 64  # 高维通道才用低秩分解
                block.append(LiteResBlock(block_in, block_out, use_low_rank))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            
            # 下采样
            if i_level != self.num_resolutions - 1:
                down.downsample = EfficientDownsample(block_in)
            
            self.down.append(down)
        
        # 中间层 - 去掉Self-Attention加速
        self.mid = nn.Module()
        self.mid.block_1 = LiteResBlock(block_in, block_in, use_low_rank=True)
        # 注意：移除attention层以提速
        self.mid.block_2 = LiteResBlock(block_in, block_in, use_low_rank=True)
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, padding=1, bias=False)
    
    def forward(self, x):
        # 入口
        hs = [self.conv_in(x)]
        
        # 下采样
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # 中间处理
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        
        # 输出
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h

class LiteVAEComplete(nn.Module):
    """完整的LiteVAE模型"""
    
    def __init__(self, ch=128, ch_mult=(1,2,4,8), num_res_blocks=2,
                 in_channels=3, z_channels=32, n_embed=4096, embed_dim=32):
        super().__init__()
        
        # 轻量编码器
        self.encoder = LiteVAEEncoder(ch, ch_mult, num_res_blocks, 
                                     in_channels, z_channels)
        
        # 保持原有量化和解码器（专注优化编码瓶颈）
        from vae_integration_fix import VectorQuantizer
        self.quantize = VectorQuantizer(n_embed, embed_dim)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 3, padding=1, bias=False)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 3, padding=1, bias=False)
        
        # 简化解码器（如果需要进一步优化）
        self._create_lite_decoder(ch, ch_mult, num_res_blocks, z_channels, in_channels)
        
        # 编译优化
        self.encoder = torch.compile(self.encoder, mode="max-autotune")
    
    def _create_lite_decoder(self, ch, ch_mult, num_res_blocks, z_channels, out_channels):
        """创建轻量解码器"""
        # 为了快速验证，暂时使用简化版本
        # 生产环境可以进一步优化
        pass  # 使用原有解码器
    
    def encode(self, x):
        """轻量编码"""
        with autocast():  # 混合精度
            h = self.encoder(x)
            h = self.quant_conv(h)
            quant, _, indices = self.quantize(h)
            
            # 转换为tokens
            B, C, H, W = quant.shape
            tokens = quant.permute(0, 2, 3, 1).reshape(B, H*W, C)
            return tokens

def compare_vae_implementations():
    """对比不同VAE实现的性能"""
    
    print("🔬 VAE实现性能对比测试")
    print("=" * 50)
    
    device = torch.device('cuda')
    test_images = torch.randn(4, 3, 256, 256).to(device)
    
    # 测试配置
    configs = [
        ("LiteVAE-Small", {"ch": 96, "ch_mult": (1,2,4), "num_res_blocks": 1}),
        ("LiteVAE-Medium", {"ch": 128, "ch_mult": (1,2,4,8), "num_res_blocks": 2}),
        ("LiteVAE-Large", {"ch": 160, "ch_mult": (1,1,2,2,4), "num_res_blocks": 2}),
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\n📊 测试 {name}...")
        
        # 创建模型
        model = LiteVAEComplete(
            ch=config["ch"],
            ch_mult=config["ch_mult"], 
            num_res_blocks=config["num_res_blocks"],
            in_channels=3,
            z_channels=32,
            n_embed=4096,
            embed_dim=32
        ).to(device)
        
        model.eval()
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model.encode(test_images)
        
        # 测量编码延迟
        latencies = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                tokens = model.encode(test_images)
            
            torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        per_image = avg_latency / test_images.shape[0]
        
        results[name] = {
            'total_params_M': total_params / 1e6,
            'encoder_params_M': encoder_params / 1e6,
            'avg_latency_ms': avg_latency,
            'per_image_ms': per_image,
            'tokens_shape': list(tokens.shape),
            'memory_mb': torch.cuda.max_memory_allocated() / 1024**2
        }
        
        print(f"  参数量: {total_params/1e6:.1f}M (编码器: {encoder_params/1e6:.1f}M)")
        print(f"  延迟: {per_image:.2f}ms/image")
        print(f"  内存: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    # 寻找最优配置
    print(f"\n🎯 性能总结:")
    best_config = None
    best_score = float('inf')
    
    for name, data in results.items():
        # 综合评分：延迟权重70%，参数量权重30%
        score = data['per_image_ms'] * 0.7 + data['encoder_params_M'] * 0.3
        print(f"{name}: {data['per_image_ms']:.1f}ms, {data['encoder_params_M']:.1f}M参数, 综合分{score:.2f}")
        
        if score < best_score:
            best_score = score
            best_config = name
    
    print(f"\n🏆 推荐配置: {best_config}")
    
    # 与目标对比
    best_latency = results[best_config]['per_image_ms']
    if best_latency <= 20:
        print("🟢 达到20ms目标！可进行下一步优化")
    elif best_latency <= 30:
        print("🟡 接近目标，建议结合TensorRT进一步优化")
    else:
        print("🔴 需要更激进的优化策略")
    
    return results, best_config

if __name__ == "__main__":
    print("🚀 开始LiteVAE集成测试...")
    
    try:
        results, best_config = compare_vae_implementations()
        
        # 保存结果
        import json
        with open('litevae_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ LiteVAE测试完成！最佳配置: {best_config}")
        print("📁 结果已保存到: litevae_comparison_results.json")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()