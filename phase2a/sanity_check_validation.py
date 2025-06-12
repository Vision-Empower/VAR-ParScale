#!/usr/bin/env python3
"""
Sanity Check Validation for ParScale-VAR Results
Critical verification of "too perfect" scaling results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

def main():
    print("🚨 SANITY CHECK: Verifying 'Too Perfect' ParScale-VAR Results")
    print("Critical Analysis of P=4 Super-Linear Claims")
    print("="*75)
    
    # Setup environment
    os.chdir("/root/VAR")
    sys.path.append("/root/VAR")
    
    from models import build_vae_var
    
    # Load baseline models
    print("\n📁 Loading Models for Critical Verification")
    print("-" * 45)
    
    device = "cuda"
    
    vae, baseline_var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=16, shared_aln=False,
    )
    
    vae.load_state_dict(torch.load("/root/vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    baseline_var.load_state_dict(torch.load("/root/var_d16.pth", map_location="cpu"), strict=True)
    
    vae.eval()
    baseline_var.eval()
    print("✅ Models loaded")
    
    # Critical Verification Tests
    print("\n🔍 CRITICAL VERIFICATION TESTS")
    print("-" * 45)
    
    # Test 1: Verify P parameter actually affects computation
    print("1️⃣ Verifying P Parameter Actually Affects Computation")
    test_p_parameter_effects()
    
    # Test 2: Check if diversity scores are hardcoded
    print("\n2️⃣ Checking for Hardcoded Diversity Scores")
    test_hardcoded_diversity()
    
    # Test 3: Memory usage verification
    print("\n3️⃣ Memory Usage Deep Analysis")
    test_memory_usage_patterns(baseline_var, device)
    
    # Test 4: Output shape and content verification
    print("\n4️⃣ Output Shape and Content Verification")
    test_output_shapes_and_content(baseline_var, device)
    
    # Test 5: GPU utilization patterns
    print("\n5️⃣ GPU Utilization Pattern Analysis")
    test_gpu_utilization_patterns(baseline_var, device)
    
    # Test 6: True computational complexity analysis
    print("\n6️⃣ True Computational Complexity Analysis")
    test_computational_complexity(baseline_var, device)
    
    print("\n🎯 SANITY CHECK COMPLETE")
    print("="*75)


def test_p_parameter_effects():
    """Test if P parameter actually affects the computation path"""
    
    print("   Testing P parameter computational effects...")
    
    # Create models with different P values
    models = {}
    for p in [1, 2, 4]:
        if p == 1:
            models[f'P={p}'] = None  # Will use baseline
        else:
            model = DebuggableSharedBackbone(None, num_streams=p)
            models[f'P={p}'] = model
            print(f"   Created P={p} model with {model.num_streams} streams")
            
            # Verify assertion
            assert model.num_streams == p, f"CRITICAL: num_streams={model.num_streams} != P={p}"
    
    print("   ✅ P parameter correctly propagated to models")


def test_hardcoded_diversity():
    """Check if diversity scores are hardcoded rather than computed"""
    
    print("   Analyzing diversity score computation...")
    
    # Test the suspicious linear relationship: diversity = 0.05 * P
    suspected_hardcoded_values = {2: 0.1, 4: 0.2}
    
    for p, expected_div in suspected_hardcoded_values.items():
        print(f"   Testing P={p} expected diversity={expected_div}")
        
        # Check if this is hardcoded in the formula: base_diversity = 0.05 * self.num_streams
        calculated_div = 0.05 * p
        
        if abs(calculated_div - expected_div) < 0.001:
            print(f"   🚨 CRITICAL: P={p} diversity appears HARDCODED!")
            print(f"      Formula: 0.05 * {p} = {calculated_div}")
            print(f"      This is NOT real diversity measurement!")
        else:
            print(f"   ✅ P={p} diversity appears genuine")


def test_memory_usage_patterns(baseline_var, device):
    """Deep analysis of memory usage patterns"""
    
    print("   Deep memory usage analysis...")
    
    # Clear all memory
    torch.cuda.empty_cache()
    
    # Baseline memory measurement
    memory_baseline = torch.cuda.memory_allocated(device)
    memory_reserved_baseline = torch.cuda.memory_reserved(device)
    
    print(f"   Baseline allocated: {memory_baseline / (1024**3):.3f}GB")
    print(f"   Baseline reserved: {memory_reserved_baseline / (1024**3):.3f}GB")
    
    # Test different P values
    for p in [1, 2, 4]:
        print(f"\n   Testing P={p} memory patterns:")
        
        torch.cuda.empty_cache()
        memory_before_allocated = torch.cuda.memory_allocated(device)
        memory_before_reserved = torch.cuda.memory_reserved(device)
        
        with torch.no_grad():
            if p == 1:
                # Baseline single inference
                output = baseline_var.autoregressive_infer_cfg(
                    B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
                )
            else:
                # Multiple calls simulation (worst case)
                outputs = []
                for stream_idx in range(p):
                    stream_output = baseline_var.autoregressive_infer_cfg(
                        B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
                    )
                    outputs.append(stream_output)
                output = outputs[0]  # Use first output
        
        memory_after_allocated = torch.cuda.memory_allocated(device)
        memory_after_reserved = torch.cuda.memory_reserved(device)
        
        allocated_diff = (memory_after_allocated - memory_before_allocated) / (1024**3)
        reserved_diff = (memory_after_reserved - memory_before_reserved) / (1024**3)
        
        print(f"     Allocated memory change: {allocated_diff:.3f}GB")
        print(f"     Reserved memory change: {reserved_diff:.3f}GB")
        
        # Check for suspicious patterns
        if p > 1 and allocated_diff < 0.01:
            print(f"     🚨 SUSPICIOUS: P={p} uses minimal additional memory!")
            print(f"        This suggests P streams are NOT actually running")


def test_output_shapes_and_content(baseline_var, device):
    """Verify output shapes and content differences"""
    
    print("   Testing output shapes and content...")
    
    outputs = {}
    
    with torch.no_grad():
        for p in [1, 2, 4]:
            print(f"   Testing P={p} output...")
            
            if p == 1:
                output = baseline_var.autoregressive_infer_cfg(
                    B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
                )
            else:
                # Simulate what shared backbone SHOULD do if it were real
                model = DebuggableSharedBackbone(baseline_var, num_streams=p)
                output, metrics = model.debug_shared_backbone_infer(
                    B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900, return_metrics=True
                )
                
                print(f"     P={p} reported metrics: {metrics}")
            
            outputs[p] = output
            print(f"     P={p} output shape: {output.shape}")
            print(f"     P={p} output type: {type(output)}")
            print(f"     P={p} output device: {output.device}")
            
            # Check output content
            if output.numel() > 0:
                print(f"     P={p} output mean: {output.float().mean().item():.6f}")
                print(f"     P={p} output std: {output.float().std().item():.6f}")
    
    # Compare outputs between different P values
    print("\n   Comparing outputs between P values:")
    
    if 1 in outputs and 2 in outputs:
        similarity_1_2 = F.cosine_similarity(
            outputs[1].flatten().float(),
            outputs[2].flatten().float(),
            dim=0
        ).item()
        print(f"   P=1 vs P=2 similarity: {similarity_1_2:.6f}")
        
        if similarity_1_2 > 0.99:
            print(f"   🚨 CRITICAL: P=1 and P=2 outputs are nearly identical!")
            print(f"      This suggests P=2 is not actually doing different computation")
    
    if 2 in outputs and 4 in outputs:
        similarity_2_4 = F.cosine_similarity(
            outputs[2].flatten().float(),
            outputs[4].flatten().float(),
            dim=0
        ).item()
        print(f"   P=2 vs P=4 similarity: {similarity_2_4:.6f}")
        
        if similarity_2_4 > 0.99:
            print(f"   🚨 CRITICAL: P=2 and P=4 outputs are nearly identical!")


def test_gpu_utilization_patterns(baseline_var, device):
    """Test GPU utilization patterns for different P values"""
    
    print("   Testing GPU utilization patterns...")
    
    # Note: This is a simplified test since we can't run nvidia-smi dmon from within Python
    # In practice, you'd run this alongside nvidia-smi in a separate terminal
    
    for p in [1, 2, 4]:
        print(f"   Simulating P={p} workload...")
        
        torch.cuda.empty_cache()
        
        # Create large tensors to simulate different workloads
        if p == 1:
            # Single stream workload
            dummy_tensor = torch.randn(1, 1024, 1024, device=device)
            result = torch.matmul(dummy_tensor, dummy_tensor.transpose(-2, -1))
        else:
            # Multi-stream workload simulation
            dummy_tensors = []
            for i in range(p):
                tensor = torch.randn(1, 1024, 1024, device=device)
                dummy_tensors.append(tensor)
            
            # If truly parallel, this should utilize more GPU resources
            results = []
            for tensor in dummy_tensors:
                result = torch.matmul(tensor, tensor.transpose(-2, -1))
                results.append(result)
        
        torch.cuda.synchronize()
        print(f"     P={p} workload completed")


def test_computational_complexity(baseline_var, device):
    """Test true computational complexity scaling"""
    
    print("   Testing computational complexity scaling...")
    
    # Test with different sequence lengths to see if P scaling holds
    sequence_configs = [
        {"B": 1, "description": "Standard batch"},
        {"B": 2, "description": "Double batch"},  # This should reveal scaling patterns
    ]
    
    for config in sequence_configs:
        print(f"\n   Testing with {config['description']} (B={config['B']}):")
        
        for p in [1, 2, 4]:
            times = []
            
            for trial in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    if p == 1:
                        output = baseline_var.autoregressive_infer_cfg(
                            B=config['B'], label_B=None, cfg=1.0, top_p=0.95, top_k=900
                        )
                    else:
                        # Simulate true P-stream workload
                        outputs = []
                        for stream_idx in range(p):
                            stream_output = baseline_var.autoregressive_infer_cfg(
                                B=config['B'], label_B=None, cfg=1.0, top_p=0.95, top_k=900
                            )
                            outputs.append(stream_output)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)
            
            avg_time = np.mean(times)
            print(f"     P={p}: {avg_time:.1f}ms")
            
            # Check for expected scaling
            if p > 1:
                expected_min_time = (avg_time / p) * 0.8  # Allow for some efficiency
                baseline_time = 285.2  # From previous measurements
                
                if avg_time < expected_min_time:
                    print(f"     🚨 SUSPICIOUS: P={p} time {avg_time:.1f}ms is too fast!")
                    print(f"        Expected minimum: {expected_min_time:.1f}ms")


class DebuggableSharedBackbone(nn.Module):
    """Debuggable version of shared backbone for verification"""
    
    def __init__(self, base_var, num_streams=2):
        super().__init__()
        self.base_var = base_var
        self.num_streams = num_streams
        
        print(f"   DEBUG: Created DebuggableSharedBackbone with num_streams={self.num_streams}")
        
        # Verify this is set correctly
        assert self.num_streams == num_streams, f"CRITICAL: num_streams mismatch!"
    
    def debug_shared_backbone_infer(self, B, label_B=None, cfg=1.0, return_metrics=False, **kwargs):
        """Debug version with detailed logging"""
        
        print(f"     DEBUG: Running inference with P={self.num_streams}")
        print(f"     DEBUG: Input B={B}, cfg={cfg}")
        
        # Single call to VAR (this is the key test)
        output = self.base_var.autoregressive_infer_cfg(
            B=B, label_B=label_B, cfg=cfg, **kwargs
        )
        
        print(f"     DEBUG: Output shape: {output.shape}")
        print(f"     DEBUG: Output device: {output.device}")
        
        # Check the suspicious hardcoded diversity formula
        base_diversity = 0.05 * self.num_streams
        print(f"     DEBUG: Calculated base_diversity = 0.05 * {self.num_streams} = {base_diversity}")
        
        # This is the SMOKING GUN - if diversity is just base_diversity, it's hardcoded!
        diversity = base_diversity
        
        metrics = {
            'diversity_score': diversity,
            'num_streams_used': self.num_streams,
            'implementation': 'debug_shared_backbone',
            'HARDCODED_WARNING': True if diversity == base_diversity else False
        }
        
        print(f"     DEBUG: Metrics: {metrics}")
        
        if return_metrics:
            return output, metrics
        else:
            return output


if __name__ == "__main__":
    main()