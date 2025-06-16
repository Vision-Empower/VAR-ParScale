#!/usr/bin/env python3
"""
CI Latency Tests - 验证性能指标
确保0.566ms P99延迟可复现
"""

import json
import math
import pytest
from pathlib import Path

def test_latency_results_exist():
    """测试延迟结果文件是否存在"""
    results_file = Path("results/ci_latency.json")
    assert results_file.exists(), f"Latency results file not found: {results_file}"

def test_latency_performance():
    """测试延迟性能是否达标"""
    results_file = Path("results/ci_latency.json")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 检查关键字段存在
    assert 'best_p99_ms' in data, "Missing best_p99_ms in results"
    assert 'validation_passed' in data, "Missing validation_passed in results"
    assert 'results' in data, "Missing detailed results"
    
    best_p99 = data['best_p99_ms']
    target_p99 = 0.566
    tolerance = 0.01  # 1% tolerance
    
    # 验证P99延迟在允许范围内
    assert math.isclose(best_p99, target_p99, rel_tol=tolerance), \
        f"P99 latency drift: {best_p99:.3f}ms vs target {target_p99:.3f}ms (tolerance: {tolerance*100}%)"
    
    # 验证验证状态
    assert data['validation_passed'], f"Latency validation failed: P99={best_p99:.3f}ms"

def test_batch_scaling():
    """测试批处理缩放性能"""
    results_file = Path("results/ci_latency.json")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 检查至少有3个批处理大小的结果
    assert len(results) >= 3, f"Insufficient batch size tests: {len(results)}"
    
    # 验证每个批处理结果的完整性
    for batch_key, batch_data in results.items():
        assert 'p99_ms_per_image' in batch_data, f"Missing p99 in {batch_key}"
        assert 'batch_size' in batch_data, f"Missing batch_size in {batch_key}"
        assert 'num_runs' in batch_data, f"Missing num_runs in {batch_key}"
        
        # 检查P99在合理范围内
        p99 = batch_data['p99_ms_per_image']
        assert 0.1 <= p99 <= 2.0, f"P99 out of range in {batch_key}: {p99:.3f}ms"

def test_performance_consistency():
    """测试性能一致性"""
    results_file = Path("results/ci_latency.json")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 获取所有P99值
    p99_values = [r['p99_ms_per_image'] for r in results.values()]
    
    # 检查P99值的变异性（不应相差太大）
    min_p99 = min(p99_values)
    max_p99 = max(p99_values)
    variation = (max_p99 - min_p99) / min_p99
    
    # 批处理间的变异性不应超过50%
    assert variation <= 0.5, f"High P99 variation between batches: {variation:.2%}"

def test_device_info():
    """测试设备信息记录"""
    results_file = Path("results/ci_latency.json")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 检查设备信息
    assert 'device' in data, "Missing device info"
    assert 'device_name' in data, "Missing device name"
    
    # 如果是CUDA，应该有设备名称
    if 'cuda' in data['device']:
        assert data['device_name'] != 'CPU', "CUDA device should not report as CPU"

if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v"])