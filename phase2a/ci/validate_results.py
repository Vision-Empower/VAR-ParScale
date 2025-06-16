#!/usr/bin/env python3
"""
Results Validation - 汇总验证所有CI结果
M2-M3 milestone综合检查
"""

import json
import sys
from pathlib import Path

def load_json_result(filepath):
    """安全加载JSON结果文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Result file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        return None

def validate_latency_results(results_dir):
    """验证延迟结果"""
    print("⚡ Validating latency results...")
    
    latency_file = results_dir / "ci_latency.json"
    data = load_json_result(latency_file)
    
    if not data:
        return False
    
    # 检查关键指标
    best_p99 = data.get('best_p99_ms', float('inf'))
    validation_passed = data.get('validation_passed', False)
    
    print(f"   Best P99: {best_p99:.3f}ms")
    print(f"   Target: 0.566ms (±5% tolerance)")
    print(f"   Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    return validation_passed

def validate_fid_results(results_dir):
    """验证FID结果"""
    print("\n🎯 Validating FID results...")
    
    # 检查baseline和hybrid FID
    baseline_file = results_dir / "fid_baseline.json"
    hybrid_file = results_dir / "fid_hybrid.json"
    
    baseline_data = load_json_result(baseline_file)
    hybrid_data = load_json_result(hybrid_file)
    
    if not baseline_data or not hybrid_data:
        print("   ❌ Missing FID result files")
        return False
    
    baseline_fid = baseline_data.get('fid_score')
    hybrid_fid = hybrid_data.get('fid_score')
    
    if baseline_fid is None or hybrid_fid is None:
        print("   ❌ Missing FID scores in result files")
        return False
    
    delta_fid = hybrid_fid - baseline_fid
    
    print(f"   Baseline FID: {baseline_fid:.3f}")
    print(f"   Hybrid FID: {hybrid_fid:.3f}")
    print(f"   Delta FID: {delta_fid:.3f}")
    print(f"   Target: ≤ +3.0")
    
    fid_passed = delta_fid <= 3.0
    print(f"   Status: {'✅ PASSED' if fid_passed else '❌ FAILED'}")
    
    return fid_passed

def validate_ablation_results(results_dir):
    """验证消融实验结果"""
    print("\n🔬 Validating ablation results...")
    
    # 检查coarse和fine消融
    coarse_file = results_dir / "fid_no_coarse.json"
    fine_file = results_dir / "fid_no_fine.json"
    
    coarse_data = load_json_result(coarse_file)
    fine_data = load_json_result(fine_file)
    
    coarse_passed = False
    fine_passed = False
    
    # Coarse ablation
    if coarse_data:
        coarse_valuable = coarse_data.get('coarse_valuable', False)
        delta_degradation = coarse_data.get('delta_fid_degradation', 0)
        print(f"   Coarse ablation: +{delta_degradation:.1f} FID degradation")
        print(f"   Coarse valuable: {'✅ YES' if coarse_valuable else '❌ NO'}")
        coarse_passed = coarse_valuable
    else:
        print("   ❌ Coarse ablation results missing")
    
    # Fine ablation  
    if fine_data:
        fine_valuable = fine_data.get('fine_valuable', False)
        delta_degradation = fine_data.get('delta_fid_degradation', 0)
        print(f"   Fine ablation: +{delta_degradation:.1f} FID degradation")
        print(f"   Fine valuable: {'✅ YES' if fine_valuable else '❌ NO'}")
        fine_passed = fine_valuable
    else:
        print("   ❌ Fine ablation results missing")
    
    return coarse_passed and fine_passed

def generate_summary_report(results_dir):
    """生成总结报告"""
    print("\n📊 Generating summary report...")
    
    # 收集所有结果
    summary = {
        'timestamp': None,
        'milestones': {},
        'overall_status': 'UNKNOWN'
    }
    
    # M2-2 延迟
    latency_file = results_dir / "ci_latency.json"
    latency_data = load_json_result(latency_file)
    if latency_data:
        summary['milestones']['M2-2_latency'] = {
            'status': 'PASSED' if latency_data.get('validation_passed') else 'FAILED',
            'best_p99_ms': latency_data.get('best_p99_ms'),
            'target_p99_ms': 0.566
        }
        summary['timestamp'] = latency_data.get('timestamp')
    
    # M2-3 FID
    baseline_file = results_dir / "fid_baseline.json"
    hybrid_file = results_dir / "fid_hybrid.json"
    baseline_data = load_json_result(baseline_file)
    hybrid_data = load_json_result(hybrid_file)
    
    if baseline_data and hybrid_data:
        delta_fid = hybrid_data['fid_score'] - baseline_data['fid_score']
        fid_passed = delta_fid <= 3.0
        
        summary['milestones']['M2-3_fid'] = {
            'status': 'PASSED' if fid_passed else 'FAILED',
            'baseline_fid': baseline_data['fid_score'],
            'hybrid_fid': hybrid_data['fid_score'],
            'delta_fid': delta_fid,
            'target_delta': 3.0
        }
    
    # M3-1 消融
    coarse_file = results_dir / "fid_no_coarse.json"
    fine_file = results_dir / "fid_no_fine.json"
    coarse_data = load_json_result(coarse_file)
    fine_data = load_json_result(fine_file)
    
    if coarse_data and fine_data:
        ablation_passed = (coarse_data.get('coarse_valuable', False) and 
                          fine_data.get('fine_valuable', False))
        
        summary['milestones']['M3-1_ablation'] = {
            'status': 'PASSED' if ablation_passed else 'FAILED',
            'coarse_valuable': coarse_data.get('coarse_valuable'),
            'fine_valuable': fine_data.get('fine_valuable'),
            'coarse_degradation': coarse_data.get('delta_fid_degradation'),
            'fine_degradation': fine_data.get('delta_fid_degradation')
        }
    
    # 总体状态
    all_passed = all(
        milestone.get('status') == 'PASSED' 
        for milestone in summary['milestones'].values()
    )
    summary['overall_status'] = 'PASSED' if all_passed else 'FAILED'
    
    # 保存总结
    summary_file = results_dir / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved: {summary_file}")
    return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CI Results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    print("🔍 CI RESULTS VALIDATION")
    print("=" * 50)
    
    # 验证各个组件
    latency_ok = validate_latency_results(results_dir)
    fid_ok = validate_fid_results(results_dir)
    ablation_ok = validate_ablation_results(results_dir)
    
    # 生成总结报告
    summary = generate_summary_report(results_dir)
    
    # 最终判断
    print(f"\n🏆 FINAL VALIDATION STATUS")
    print(f"   M2-2 Latency: {'✅ PASSED' if latency_ok else '❌ FAILED'}")
    print(f"   M2-3 FID: {'✅ PASSED' if fid_ok else '❌ FAILED'}")
    print(f"   M3-1 Ablation: {'✅ PASSED' if ablation_ok else '❌ FAILED'}")
    
    overall_passed = latency_ok and fid_ok and ablation_ok
    print(f"\n   Overall: {'🟢 ALL TESTS PASSED' if overall_passed else '🔴 TESTS FAILED'}")
    
    if overall_passed:
        print("\n🎉 v0.3.0-lite-hybrid READY FOR RELEASE!")
    else:
        print("\n❌ Fix issues before release")
    
    return 0 if overall_passed else 1

if __name__ == "__main__":
    exit(main())