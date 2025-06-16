#!/usr/bin/env python3
"""
M3-1 Coarse Branch Ablation - Component validation
关闭coarse分支测试其贡献价值
"""

import torch
import json
import argparse
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ci.fid_eval import generate_model_samples, calculate_fid_pytorch

def disable_coarse_branch(model):
    """关闭coarse分支"""
    print("🔧 Disabling coarse branch...")
    
    # 遍历coarse tokenizer参数并置零
    coarse_module = model.hybrid_processor.coarse_tokenizer
    
    disabled_params = 0
    for name, param in coarse_module.named_parameters():
        param.data.zero_()
        param.requires_grad = False
        disabled_params += param.numel()
    
    print(f"   -> Disabled {disabled_params:,} parameters in coarse branch")
    print("   -> Coarse tokenizer output will be zero")
    
    return model

def generate_no_coarse_samples(num_samples, output_dir, seed=44):
    """生成关闭coarse分支的样本"""
    
    print(f"🎨 Generating samples with coarse branch disabled")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载并修改模型
    from e2e_lite_hybrid_pipeline_fixed import ParScaleEAR_E2E_System
    model = ParScaleEAR_E2E_System().to(device).eval()
    
    # 关闭coarse分支
    model = disable_coarse_branch(model)
    
    if device.type == 'cuda':
        model = model.half()
    
    # 生成样本
    batch_size = 32
    generated_count = 0
    
    from tqdm import tqdm
    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating no-coarse samples")
        
        while generated_count < num_samples:
            current_batch = min(batch_size, num_samples - generated_count)
            
            dummy_input = torch.randn(current_batch, 3, 256, 256, device=device)
            if device.type == 'cuda':
                dummy_input = dummy_input.half()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(dummy_input)
            
            # 确保输出范围正确
            output = torch.clamp(output, -1.0, 1.0)
            output = (output + 1.0) / 2.0
            
            # 保存图片
            from PIL import Image
            for i in range(output.size(0)):
                if generated_count < num_samples:
                    img_tensor = output[i].cpu()
                    img_np = img_tensor.numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    
                    img_path = output_dir / f"{generated_count:05d}.png"
                    img_pil.save(img_path, format="PNG")
                    generated_count += 1
            
            pbar.update(current_batch)
        
        pbar.close()
    
    print(f"✅ Generated {generated_count} no-coarse samples")
    return output_dir

def run_coarse_ablation(num_samples, ref_dir, output_file):
    """运行coarse分支消融实验"""
    
    print("🔬 M3-1 COARSE BRANCH ABLATION")
    print("🎯 Target: Prove coarse branch value (ΔFID ≥ +10)")
    print("=" * 50)
    
    # 生成无coarse分支的样本
    samples_dir = f"tmp/no_coarse_samples_{num_samples}"
    generate_no_coarse_samples(num_samples, samples_dir)
    
    # 计算FID
    print("\n📊 Calculating FID for no-coarse samples...")
    fid_score = calculate_fid_pytorch(samples_dir, ref_dir)
    
    if fid_score is None:
        print("🔴 COARSE ABLATION FAILED")
        return 1
    
    # 与完整hybrid比较
    hybrid_fid = 328.8  # 从M2-3结果
    delta_fid = fid_score - hybrid_fid
    
    print(f"\n📊 COARSE ABLATION RESULTS")
    print(f"   Hybrid FID (full): {hybrid_fid:.3f}")
    print(f"   No-coarse FID: {fid_score:.3f}")
    print(f"   Delta (degradation): +{delta_fid:.3f}")
    
    # 判断coarse分支价值
    target_degradation = 10.0  # 目标降级
    coarse_valuable = delta_fid >= target_degradation
    
    if coarse_valuable:
        print(f"🟢 COARSE BRANCH VALUABLE: +{delta_fid:.1f} ≥ +{target_degradation}")
        validation_status = "PASSED"
    else:
        print(f"🔴 COARSE BRANCH QUESTIONABLE: +{delta_fid:.1f} < +{target_degradation}")
        validation_status = "FAILED"
    
    # 保存结果
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': 'coarse_ablation',
        'num_samples': num_samples,
        'reference_dir': str(ref_dir),
        'samples_dir': samples_dir,
        'hybrid_fid_full': hybrid_fid,
        'no_coarse_fid': float(fid_score),
        'delta_fid_degradation': float(delta_fid),
        'target_degradation': target_degradation,
        'coarse_valuable': coarse_valuable,
        'validation_status': validation_status
    }
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved: {output_path}")
    
    return 0 if coarse_valuable else 1

def main():
    parser = argparse.ArgumentParser(description='M3-1 Coarse Branch Ablation')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--ref_dir', type=str, default='data/imagenet_ref_1k',
                       help='Reference dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    return run_coarse_ablation(
        num_samples=args.num_samples,
        ref_dir=args.ref_dir,
        output_file=args.output
    )

if __name__ == "__main__":
    exit(main())