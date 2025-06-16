#!/usr/bin/env python3
"""
M2-2 Latency Profile - Performance validation
复现0.566ms P99延迟性能
"""

import torch
import time
import json
import argparse
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_cuda_environment():
    """设置CUDA环境以获得稳定性能测量"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ CUDA optimized for performance measurement")
    else:
        print("⚠️ CPU mode - measurements will be less meaningful")

def benchmark_model_precise(model, device, batch_size=8, num_warmup=30, num_runs=100):
    """精确的模型延迟测量 - 遵循NVIDIA性能测量最佳实践"""
    
    model.eval()
    
    # 创建测试数据
    test_input = torch.randn(batch_size, 3, 256, 256, device=device)
    if device.type == 'cuda':
        test_input = test_input.half()
        model = model.half()
    
    print(f"🔧 Benchmarking with batch_size={batch_size}")
    
    # 预热阶段 - 关键！
    print(f"   -> Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(test_input)
    
    # 同步等待
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 精确测量阶段
    print(f"   -> Measuring ({num_runs} runs)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms
    
    # 统计分析
    latencies = np.array(latencies)
    per_image_times = latencies / batch_size
    
    p50 = np.percentile(per_image_times, 50)
    p95 = np.percentile(per_image_times, 95)
    p99 = np.percentile(per_image_times, 99)
    avg = np.mean(per_image_times)
    
    print(f"   -> Avg: {avg:.3f}ms/img")
    print(f"   -> P50: {p50:.3f}ms/img")
    print(f"   -> P95: {p95:.3f}ms/img") 
    print(f"   -> P99: {p99:.3f}ms/img")
    
    return {
        'avg_ms_per_image': float(avg),
        'p50_ms_per_image': float(p50),
        'p95_ms_per_image': float(p95),
        'p99_ms_per_image': float(p99),
        'total_avg_ms': float(np.mean(latencies)),
        'batch_size': batch_size,
        'num_runs': num_runs,
        'raw_latencies_ms': latencies.tolist()
    }

def run_latency_validation(batch_list, num_runs=100, output_file=None):
    """运行完整的延迟验证"""
    
    print("⚡ M2-2 LATENCY VALIDATION")
    print("🎯 Target: P99 ≤ 0.6ms (validation: 0.566ms)")
    print("=" * 50)
    
    # 1. 环境设置
    setup_cuda_environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️ Warning: CPU measurements may not match H100 performance")
    
    # 2. 加载模型
    print("\n📦 Loading model...")
    from e2e_lite_hybrid_pipeline_fixed import ParScaleEAR_E2E_System
    
    model = ParScaleEAR_E2E_System().to(device).eval()
    print("✅ Model loaded successfully")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Model size: {total_params:.1f}M parameters")
    
    # 3. 批处理测试
    results = {}
    best_p99 = float('inf')
    best_batch = None
    
    for batch_size in batch_list:
        print(f"\n📊 Testing batch_size = {batch_size}")
        
        # 性能测量
        perf_data = benchmark_model_precise(
            model, device, batch_size, num_runs=num_runs
        )
        results[f'batch_{batch_size}'] = perf_data
        
        # 跟踪最佳性能
        if perf_data['p99_ms_per_image'] < best_p99:
            best_p99 = perf_data['p99_ms_per_image']
            best_batch = batch_size
        
        # 延迟检查
        if perf_data['p99_ms_per_image'] > 1.0:
            print(f"🟡 WARNING: P99 {perf_data['p99_ms_per_image']:.3f}ms > 1.0ms")
        else:
            print(f"🟢 GOOD: P99 {perf_data['p99_ms_per_image']:.3f}ms ≤ 1.0ms")
    
    # 4. 总结结果
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'best_batch_size': best_batch,
        'best_p99_ms': float(best_p99),
        'target_p99_ms': 0.566,
        'validation_passed': best_p99 <= 0.6,  # Allow 5% tolerance
        'results': results
    }
    
    print(f"\n📋 LATENCY VALIDATION SUMMARY")
    print(f"   Best configuration: batch_{best_batch}")
    print(f"   Best P99 latency: {best_p99:.3f}ms/image")
    print(f"   Target (0.566ms): {'✅ PASSED' if best_p99 <= 0.6 else '❌ FAILED'}")
    
    # 5. 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Results saved: {output_path}")
    
    # 6. 返回验证状态
    if summary['validation_passed']:
        print("🟢 M2-2 LATENCY VALIDATION PASSED")
        return 0
    else:
        print("🔴 M2-2 LATENCY VALIDATION FAILED")
        return 1

def main():
    parser = argparse.ArgumentParser(description='M2-2 Latency Profile')
    parser.add_argument('--batch_list', nargs='+', type=int, default=[4, 8, 16],
                       help='List of batch sizes to test')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of measurement runs per batch')
    parser.add_argument('--output', type=str, default='results/ci_latency.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    return run_latency_validation(
        batch_list=args.batch_list,
        num_runs=args.num_runs,
        output_file=args.output
    )

if __name__ == "__main__":
    exit(main())