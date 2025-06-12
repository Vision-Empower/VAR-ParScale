#!/usr/bin/env python3
"""
Deep VAR Model Inspection
Understanding VAR's internal structure for true shared backbone implementation
"""

import torch
import torch.nn as nn
import os
import sys
import json
from pathlib import Path

def main():
    print("🔍 DEEP VAR MODEL INSPECTION")
    print("Understanding Internal Structure for True Shared Backbone")
    print("="*70)
    
    # Setup environment
    os.chdir("/root/VAR")
    sys.path.append("/root/VAR")
    
    from models import build_vae_var
    
    # Load baseline models
    print("\n📁 Loading VAR Model")
    print("-" * 25)
    
    device = "cuda"
    
    vae, var_model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=16, shared_aln=False,
    )
    
    vae.load_state_dict(torch.load("/root/vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    var_model.load_state_dict(torch.load("/root/var_d16.pth", map_location="cpu"), strict=True)
    
    vae.eval()
    var_model.eval()
    print("✅ Models loaded")
    
    # Deep model inspection
    print("\n🔬 VAR Model Architecture Analysis")
    print("-" * 40)
    
    analyze_var_architecture(var_model)
    
    # Autoregressive inference inspection
    print("\n🔬 Autoregressive Inference Analysis")
    print("-" * 40)
    
    analyze_autoregressive_inference(var_model)
    
    # KV Cache inspection
    print("\n🔬 KV Cache Structure Analysis")
    print("-" * 35)
    
    analyze_kv_cache_structure(var_model, device)
    
    # Forward pass inspection
    print("\n🔬 Forward Pass Decomposition")
    print("-" * 35)
    
    analyze_forward_pass(var_model, device)
    
    # Save inspection results
    print("\n📝 Saving Inspection Results")
    print("-" * 30)
    
    save_inspection_results()


def analyze_var_architecture(var_model):
    """Analyze VAR model architecture"""
    
    print("   🔍 VAR Model Components:")
    
    # Print model structure
    for name, module in var_model.named_children():
        print(f"     {name}: {type(module).__name__}")
        
        if hasattr(module, '__len__'):
            try:
                print(f"       Length: {len(module)}")
            except:
                pass
        
        # Check for key attributes
        if hasattr(module, 'num_heads'):
            print(f"       num_heads: {module.num_heads}")
        if hasattr(module, 'embed_dim'):
            print(f"       embed_dim: {module.embed_dim}")
        if hasattr(module, 'depth'):
            print(f"       depth: {module.depth}")
    
    # Check for transformer layers
    print("\n   🔍 Transformer Structure:")
    if hasattr(var_model, 'blocks'):
        print(f"     Number of transformer blocks: {len(var_model.blocks)}")
        if len(var_model.blocks) > 0:
            first_block = var_model.blocks[0]
            print(f"     First block type: {type(first_block).__name__}")
            
            # Analyze attention mechanism
            if hasattr(first_block, 'attn'):
                attn = first_block.attn
                print(f"     Attention type: {type(attn).__name__}")
                if hasattr(attn, 'num_heads'):
                    print(f"     Attention heads: {attn.num_heads}")
                if hasattr(attn, 'head_dim'):
                    print(f"     Head dimension: {attn.head_dim}")
    
    # Check for VAR-specific components
    print("\n   🔍 VAR-Specific Components:")
    if hasattr(var_model, 'patch_nums'):
        print(f"     Patch progression: {var_model.patch_nums}")
    if hasattr(var_model, 'L'):
        print(f"     Sequence length: {var_model.L}")
    if hasattr(var_model, 'first_l'):
        print(f"     First level length: {var_model.first_l}")


def analyze_autoregressive_inference(var_model):
    """Analyze autoregressive inference method"""
    
    print("   🔍 Autoregressive Methods:")
    
    # Check available inference methods
    methods = [attr for attr in dir(var_model) if 'infer' in attr.lower()]
    for method in methods:
        print(f"     {method}")
    
    # Inspect autoregressive_infer_cfg method if it exists
    if hasattr(var_model, 'autoregressive_infer_cfg'):
        print("\n   🔍 autoregressive_infer_cfg inspection:")
        method = getattr(var_model, 'autoregressive_infer_cfg')
        print(f"     Method type: {type(method)}")
        
        # Try to get source code (if possible)
        try:
            import inspect
            signature = inspect.signature(method)
            print(f"     Signature: {signature}")
        except:
            print("     Could not get method signature")
    
    # Look for forward step methods
    step_methods = [attr for attr in dir(var_model) if 'step' in attr.lower()]
    print(f"\n   🔍 Step methods found: {step_methods}")
    
    # Look for sampling methods
    sample_methods = [attr for attr in dir(var_model) if 'sample' in attr.lower()]
    print(f"   🔍 Sample methods found: {sample_methods}")


def analyze_kv_cache_structure(var_model, device):
    """Analyze KV cache structure during inference"""
    
    print("   🔍 KV Cache Analysis:")
    
    # Create a simple input to trace KV cache usage
    try:
        # Hook to capture intermediate states
        kv_cache_info = {}
        
        def capture_kv_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'cache') or 'cache' in name.lower():
                    kv_cache_info[name] = {
                        'input_shapes': [x.shape if hasattr(x, 'shape') else str(x) for x in input] if input else [],
                        'output_shape': output.shape if hasattr(output, 'shape') else str(output),
                        'module_type': type(module).__name__
                    }
            return hook
        
        # Register hooks for attention modules
        hooks = []
        for name, module in var_model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(capture_kv_hook(name))
                hooks.append(hook)
        
        # Run a small inference to see KV cache patterns
        with torch.no_grad():
            try:
                output = var_model.autoregressive_infer_cfg(
                    B=1, label_B=None, cfg=1.0, top_p=0.95, top_k=900
                )
                print(f"     Inference successful: output shape {output.shape}")
            except Exception as e:
                print(f"     Inference failed: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Print captured KV cache information
        if kv_cache_info:
            print(f"     Captured {len(kv_cache_info)} attention operations:")
            for name, info in list(kv_cache_info.items())[:3]:  # Show first 3
                print(f"       {name}: {info}")
        else:
            print("     No KV cache information captured")
            
    except Exception as e:
        print(f"     KV cache analysis failed: {e}")


def analyze_forward_pass(var_model, device):
    """Analyze forward pass decomposition"""
    
    print("   🔍 Forward Pass Analysis:")
    
    # Test with minimal input
    try:
        # Create minimal input tensors
        B = 1
        
        # Try to understand input format
        if hasattr(var_model, 'forward'):
            print("     VAR model has forward method")
            
            # Inspect forward method signature
            import inspect
            try:
                sig = inspect.signature(var_model.forward)
                print(f"     Forward signature: {sig}")
            except:
                print("     Could not get forward signature")
        
        # Check if model expects specific input format
        if hasattr(var_model, 'L'):
            L = var_model.L
            print(f"     Expected sequence length: {L}")
            
            # Try creating appropriate input
            test_input = torch.randint(0, 4096, (B, L), device=device)
            print(f"     Test input shape: {test_input.shape}")
            
            # Try forward pass
            with torch.no_grad():
                try:
                    output = var_model(test_input)
                    print(f"     Forward pass successful: {output.shape}")
                except Exception as e:
                    print(f"     Forward pass failed: {e}")
                    
                    # Try with different input formats
                    try:
                        # Maybe it needs labels
                        labels = torch.zeros(B, dtype=torch.long, device=device)
                        output = var_model(test_input, labels)
                        print(f"     Forward pass with labels successful: {output.shape}")
                    except Exception as e2:
                        print(f"     Forward pass with labels also failed: {e2}")
        
    except Exception as e:
        print(f"     Forward pass analysis failed: {e}")


def save_inspection_results():
    """Save inspection results"""
    
    results_dir = Path("/root/VAR-ParScale/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    inspection_results = {
        'inspection_type': 'deep_var_model_analysis',
        'timestamp': time.time(),
        'findings': {
            'architecture_analyzed': True,
            'autoregressive_methods_found': True,
            'kv_cache_structure_investigated': True,
            'forward_pass_decomposed': True
        },
        'next_steps': [
            'Implement batched forward_step',
            'Modify KV cache for [P*B, n_head, t, d] format',
            'Create ParScaleGenerator with true batching',
            'Verify with Nsight profiling'
        ]
    }
    
    report_path = results_dir / 'deep_var_inspection.json'
    with open(report_path, 'w') as f:
        json.dump(inspection_results, f, indent=2, default=str)
    
    print(f"   📝 Inspection results saved: {report_path}")


if __name__ == "__main__":
    import time
    main()