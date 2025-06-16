#!/usr/bin/env python3
"""
M2-1 Sanity Check - Architecture validation
确保端到端pipeline正常运行
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_deterministic(seed=42):
    """设置确定性环境"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Deterministic mode enabled (seed={seed})")

def test_e2e_pipeline(batch_size=4):
    """测试端到端pipeline"""
    
    print(f"🔍 Testing E2E pipeline with batch_size={batch_size}")
    
    # Import models
    from e2e_lite_hybrid_pipeline_fixed import ParScaleEAR_E2E_System
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Initialize model
    model = ParScaleEAR_E2E_System().to(device).eval()
    
    if device.type == 'cuda':
        model = model.half()
        print("   Precision: FP16")
    else:
        print("   Precision: FP32")
    
    # Create test input
    dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)
    if device.type == 'cuda':
        dummy_input = dummy_input.half()
    
    print(f"   Input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Output shape: {output.shape}")
    
    # Validation checks
    assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} vs {dummy_input.shape}"
    print("   ✅ Shape consistency check passed")
    
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    print("   ✅ NaN check passed")
    
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    print("   ✅ Inf check passed")
    
    # Output range check
    output_min, output_max = output.min().item(), output.max().item()
    print(f"   Output range: [{output_min:.3f}, {output_max:.3f}]")
    
    # Diversity check
    output_std = output.std().item()
    assert output_std > 0.01, f"Output std too low: {output_std}, possible mode collapse"
    print(f"   ✅ Output diversity check passed (std: {output_std:.3f})")
    
    # Memory usage (if CUDA)
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   GPU memory used: {memory_mb:.1f}MB")
        torch.cuda.reset_peak_memory_stats()
    
    return True

def test_model_components():
    """测试模型组件独立性"""
    
    print("🔧 Testing model components...")
    
    from e2e_lite_hybrid_pipeline_fixed import LiteHybridLatentProcessor, LiteVAEComplete
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test LiteVAE
    vae = LiteVAEComplete().to(device).eval()
    test_input = torch.randn(2, 3, 256, 256, device=device)
    
    if device.type == 'cuda':
        vae = vae.half()
        test_input = test_input.half()
    
    with torch.no_grad():
        encoded = vae.encode(test_input)
        print(f"   VAE encode output: {encoded.shape}")
        
        decoded = vae.decode(encoded)
        print(f"   VAE decode output: {decoded.shape}")
    
    # Test Hybrid Processor
    hybrid = LiteHybridLatentProcessor().to(device).eval()
    latent_input = torch.randn(2, 32, 16, 16, device=device)
    
    if device.type == 'cuda':
        hybrid = hybrid.half()
        latent_input = latent_input.half()
    
    with torch.no_grad():
        enhanced = hybrid(latent_input)
        print(f"   Hybrid processor output: {enhanced.shape}")
    
    print("   ✅ All components working properly")
    return True

def main():
    parser = argparse.ArgumentParser(description='M2-1 Sanity Check')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("🚨 M2-1 SANITY CHECK")
    print("=" * 40)
    
    # Setup environment
    setup_deterministic(args.seed)
    
    try:
        # Test individual components
        test_model_components()
        
        # Test end-to-end pipeline
        test_e2e_pipeline(args.batch)
        
        print("\n🟢 M2-1 SANITY CHECK PASSED")
        print("✅ All architecture validations successful")
        
        return 0
        
    except Exception as e:
        print(f"\n🔴 M2-1 SANITY CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())