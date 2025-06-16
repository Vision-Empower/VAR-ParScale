#!/usr/bin/env python3
"""
实验#1: 一步扩散 Energy-Head 探索
动机: 能量-扩散同构，用单步Energy替代整个VAR+VAE pipeline
目标: Teacher(DiT) → Student(Energy) 蒸馏，loss < 0.05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

class TeacherDiT(nn.Module):
    """简化的DiT Teacher模型"""
    
    def __init__(self, dim=768, num_heads=12, num_layers=6):
        super().__init__()
        self.dim = dim
        
        # 简化的DiT架构
        self.patch_embed = nn.Conv2d(3, dim, 16, stride=16)  # 256→16
        self.pos_embed = nn.Parameter(torch.randn(1, 256, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=num_heads, 
                dim_feedforward=dim*4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 3 * 16 * 16),  # 预测噪声patch
        )
        
        print(f"🎓 Teacher DiT initialized: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
    def forward(self, x, timestep=None):
        """前向传播 - 预测噪声"""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, dim]
        x = x + self.pos_embed
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        # 预测噪声
        noise_pred = self.noise_head(x.mean(1))  # Global pooling
        noise_pred = noise_pred.view(B, 3, 16, 16)
        
        # 上采样到原尺寸
        noise_pred = F.interpolate(noise_pred, size=(H, W), mode='bilinear')
        
        return noise_pred

class OneStepEnergyHead(nn.Module):
    """一步Energy Head - 轻量级学生模型"""
    
    def __init__(self, dim=256, n_layers=4):
        super().__init__()
        self.dim = dim
        
        # 输入投影
        self.input_proj = nn.Conv2d(3, dim, 8, stride=8)  # 256→32
        
        # Energy processing layers
        self.energy_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            ) for _ in range(n_layers)
        ])
        
        # 输出投影 - 预测噪声
        self.noise_head = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 3, 3, padding=1)
        )
        
        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"⚡ OneStep Energy Head: {params:.1f}M params")
    
    def forward(self, x):
        """单步预测噪声"""
        B, C, H, W = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, dim, 32, 32]
        
        # Energy processing with residual connections
        for layer in self.energy_layers:
            residual = h
            h = layer(h) + residual
        
        # 预测噪声
        noise_pred = self.noise_head(h)  # [B, 3, 32, 32]
        
        # 上采样到原尺寸
        noise_pred = F.interpolate(noise_pred, size=(H, W), mode='bilinear')
        
        return noise_pred

class DiffusionDataSimulator:
    """扩散过程数据模拟器"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 线性noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, noise, timesteps):
        """添加噪声到原图"""
        alphas_cumprod_t = self.alphas_cumprod[timesteps]
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod_t).view(-1, 1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x
    
    def sample_timesteps(self, batch_size, device):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

def distillation_loss(pred_noise, target_noise, loss_type='mse'):
    """蒸馏损失函数"""
    if loss_type == 'mse':
        return F.mse_loss(pred_noise, target_noise)
    elif loss_type == 'l1':
        return F.l1_loss(pred_noise, target_noise)
    elif loss_type == 'huber':
        return F.huber_loss(pred_noise, target_noise)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def run_experiment_1(args):
    """实验#1主函数"""
    
    print("🚀 实验#1: 一步扩散 Energy-Head 探索")
    print("🎯 目标: 用单步Energy替代整个VAR+VAE pipeline")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 初始化模型
    print("\n📦 Initializing models...")
    teacher = TeacherDiT().to(device)
    student = OneStepEnergyHead().to(device)
    diffusion_simulator = DiffusionDataSimulator()
    
    if device.type == 'cuda':
        teacher = teacher.half()
        student = student.half()
    
    print("✅ Models initialized")
    
    # 2. 预训练Teacher (简化版本)
    print("\n🎓 Pre-training Teacher DiT...")
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4)
    
    # 简单的自训练循环
    teacher.train()
    for step in tqdm(range(args.teacher_steps), desc="Teacher pre-training"):
        # 生成随机"图片"
        fake_images = torch.randn(args.batch_size, 3, 256, 256, device=device)
        if device.type == 'cuda':
            fake_images = fake_images.half()
        
        # 添加噪声
        noise = torch.randn_like(fake_images)
        timesteps = diffusion_simulator.sample_timesteps(args.batch_size, device)
        noisy_images = diffusion_simulator.add_noise(fake_images, noise, timesteps)
        
        # Teacher预测噪声
        pred_noise = teacher(noisy_images)
        
        # 自一致性损失
        loss = F.mse_loss(pred_noise, noise)
        
        teacher_optimizer.zero_grad()
        loss.backward()
        teacher_optimizer.step()
        
        if step % 100 == 0:
            print(f"   Teacher step {step}: loss = {loss.item():.4f}")
    
    teacher.eval()
    print("✅ Teacher pre-training completed")
    
    # 3. 学生模型蒸馏
    print(f"\n⚡ Student distillation for {args.distill_steps} steps...")
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    
    # 记录训练过程
    train_losses = []
    best_loss = float('inf')
    convergence_threshold = 0.05
    
    student.train()
    for step in tqdm(range(args.distill_steps), desc="Student distillation"):
        # 生成训练数据
        fake_images = torch.randn(args.batch_size, 3, 256, 256, device=device)
        if device.type == 'cuda':
            fake_images = fake_images.half()
        
        noise = torch.randn_like(fake_images)
        timesteps = diffusion_simulator.sample_timesteps(args.batch_size, device)
        noisy_images = diffusion_simulator.add_noise(fake_images, noise, timesteps)
        
        # Teacher预测（frozen）
        with torch.no_grad():
            teacher_pred = teacher(noisy_images)
        
        # Student预测
        student_pred = student(noisy_images)
        
        # 蒸馏损失
        loss = distillation_loss(student_pred, teacher_pred, args.loss_type)
        
        # 反向传播
        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        
        # 记录
        train_losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            # 保存最佳模型
            torch.save(student.state_dict(), output_dir / 'best_student.pth')
        
        # 定期输出
        if step % 100 == 0:
            avg_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
            print(f"   Step {step}: loss = {loss.item():.4f}, avg_100 = {avg_loss:.4f}")
            
            # 检查收敛
            if avg_loss < convergence_threshold:
                print(f"🎉 Convergence achieved! avg_loss = {avg_loss:.4f} < {convergence_threshold}")
                break
    
    student.eval()
    
    # 4. 性能评估
    print("\n📊 Performance evaluation...")
    
    # 单步生成测试
    test_images = torch.randn(args.batch_size, 3, 256, 256, device=device)
    if device.type == 'cuda':
        test_images = test_images.half()
    
    # Teacher vs Student对比
    with torch.no_grad():
        # 时间测试
        teacher_times = []
        student_times = []
        
        for _ in range(10):
            # Teacher timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            teacher_output = teacher(test_images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            teacher_times.append((time.time() - start) * 1000)
            
            # Student timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            student_output = student(test_images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            student_times.append((time.time() - start) * 1000)
        
        teacher_latency = sum(teacher_times) / len(teacher_times) / args.batch_size
        student_latency = sum(student_times) / len(student_times) / args.batch_size
        speedup = teacher_latency / student_latency
        
        # 预测质量对比
        prediction_diff = F.mse_loss(student_output, teacher_output).item()
    
    # 5. 结果分析
    final_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
    convergence_achieved = final_loss < convergence_threshold
    
    results = {
        'experiment': 'one_step_energy_distillation',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'config': {
            'teacher_steps': args.teacher_steps,
            'distill_steps': args.distill_steps,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'loss_type': args.loss_type
        },
        'training': {
            'final_loss': final_loss,
            'best_loss': best_loss,
            'convergence_threshold': convergence_threshold,
            'convergence_achieved': convergence_achieved,
            'total_steps': len(train_losses)
        },
        'performance': {
            'teacher_latency_ms_per_image': teacher_latency,
            'student_latency_ms_per_image': student_latency,
            'speedup': speedup,
            'prediction_mse': prediction_diff
        },
        'model_sizes': {
            'teacher_params_m': sum(p.numel() for p in teacher.parameters()) / 1e6,
            'student_params_m': sum(p.numel() for p in student.parameters()) / 1e6
        }
    }
    
    # 实验评估
    if convergence_achieved:
        if speedup > 2.0 and prediction_diff < 0.1:
            status = "BREAKTHROUGH"
            print("🏆 实验#1 重大突破！一步Energy Head蒸馏成功")
        else:
            status = "SUCCESS"
            print("🟢 实验#1 成功！达到收敛条件")
    elif final_loss < 0.1:
        status = "PROMISING"
        print("🟡 实验#1 有希望，接近收敛")
    else:
        status = "NEEDS_WORK"
        print("🔴 实验#1 需要调试，未达到预期")
    
    results['status'] = status
    
    print(f"\n🎯 实验#1 结果总结:")
    print(f"  最终损失: {final_loss:.4f}")
    print(f"  收敛达成: {'✅' if convergence_achieved else '❌'}")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  预测差异: {prediction_diff:.4f}")
    print(f"  状态: {status}")
    
    # 6. 保存结果和模型
    results_file = output_dir / f"experiment_1_results_{results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.axhline(y=convergence_threshold, color='r', linestyle='--', label=f'Convergence threshold: {convergence_threshold}')
    plt.xlabel('Step')
    plt.ylabel('Distillation Loss')
    plt.title('实验#1: 一步Energy Head训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f'training_curve_{results["timestamp"]}.png')
    plt.close()
    
    print(f"\n📁 Results saved: {results_file}")
    print(f"🎉 实验#1 完成！状态: {status}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='实验#1: 一步扩散 Energy-Head 探索')
    parser.add_argument('--teacher_steps', type=int, default=500,
                       help='Teacher pre-training steps')
    parser.add_argument('--distill_steps', type=int, default=2000,
                       help='Student distillation steps')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate for student')
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'l1', 'huber'],
                       help='Distillation loss type')
    parser.add_argument('--output_dir', type=str, default='results/experiment_1_energy',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 运行实验#1
    results = run_experiment_1(args)
    
    return results

if __name__ == "__main__":
    main()