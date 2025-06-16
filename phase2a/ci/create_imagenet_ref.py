#!/usr/bin/env python3
"""
Create ImageNet Reference Dataset
为FID计算生成参考数据集
"""

import numpy as np
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

def set_seeds(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    random.seed(seed)

def create_imagenet_reference(output_dir, num_samples=1000, seed=42):
    """创建ImageNet风格的参考数据集"""
    
    print(f"📦 Creating ImageNet reference: {num_samples} samples")
    
    set_seeds(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成多样化的自然图像模式
    for i in tqdm(range(num_samples), desc="Creating reference"):
        # 不同场景类型
        scene_type = i % 10
        
        if scene_type < 3:  # 自然场景 (30%)
            base_color = np.array([34, 139, 34]) + np.random.randint(-20, 20, 3)
        elif scene_type < 5:  # 天空场景 (20%) 
            base_color = np.array([135, 206, 235]) + np.random.randint(-30, 30, 3)
        elif scene_type < 7:  # 物体场景 (20%)
            base_color = np.array([160, 82, 45]) + np.random.randint(-40, 40, 3)
        else:  # 混合场景 (30%)
            base_color = np.random.randint(50, 200, 3)
        
        # 生成256x256图像
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :] = np.clip(base_color, 0, 255)
        
        # 添加多层纹理
        num_textures = np.random.randint(15, 25)
        for _ in range(num_textures):
            x = np.random.randint(0, 200)
            y = np.random.randint(0, 200)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            
            # 随机纹理强度
            intensity = np.random.randint(30, 70)
            texture = np.random.randint(-intensity, intensity, (h, w, 3))
            
            x_end = min(x + w, 256)
            y_end = min(y + h, 256)
            img[y:y_end, x:x_end] = np.clip(
                img[y:y_end, x:x_end] + texture[:y_end-y, :x_end-x], 0, 255
            )
        
        # 添加高斯噪声提升真实感
        noise_level = np.random.uniform(3, 8)
        noise = np.random.normal(0, noise_level, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 随机几何变换 (轻微)
        if np.random.rand() < 0.3:
            # 轻微旋转或翻转
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img)
            
            if np.random.rand() < 0.5:
                # 水平翻转
                img_pil = img_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
            else:
                # 轻微旋转 (-5 to +5 degrees)
                angle = np.random.uniform(-5, 5)
                img_pil = img_pil.rotate(angle, fillcolor=(128, 128, 128))
            
            img = np.array(img_pil)
        
        # 保存为PNG
        img_pil = Image.fromarray(img)
        img_path = output_dir / f"{i:05d}.png"
        img_pil.save(img_path, format="PNG", optimize=False)
    
    print(f"✅ Reference dataset created: {output_dir}")
    
    # 生成统计信息
    sample_files = list(output_dir.glob("*.png"))[:5]
    if sample_files:
        print("📊 Sample statistics:")
        for img_path in sample_files:
            img = np.array(Image.open(img_path))
            mean = img.mean()
            std = img.std()
            print(f"   {img_path.name}: mean={mean:.1f}, std={std:.1f}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Create ImageNet Reference Dataset')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    create_imagenet_reference(
        output_dir=args.output,
        num_samples=args.num_samples,
        seed=args.seed
    )

if __name__ == "__main__":
    main()