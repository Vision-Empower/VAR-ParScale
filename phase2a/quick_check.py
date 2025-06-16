#!/usr/bin/env python3
"""
Quick validation script - 快速自检关键指标
"""

import json
import math
from pathlib import Path

def check_latency():
    """检查延迟结果"""
    try:
        with open("results/ci_latency.json") as f:
            lat = json.load(f)
        
        best = min(v["p99_ms_per_image"] for v in lat["results"].values())
        target = 0.566
        tolerance = 0.01
        
        ok = math.isclose(best, target, rel_tol=tolerance)
        
        print(f"⚡ Latency Check:")
        print(f"   P99: {best:.3f}ms")
        print(f"   Target: {target:.3f}ms")
        print(f"   Status: {'✅ OK' if ok else '❌ FAIL'}")
        
        return ok
    except Exception as e:
        print(f"❌ Latency check failed: {e}")
        return False

def check_fid():
    """检查FID结果"""
    try:
        baseline_file = Path("results/fid_baseline.json")
        hybrid_file = Path("results/fid_hybrid.json")
        
        if not baseline_file.exists() or not hybrid_file.exists():
            print("🎯 FID Check: Files not found (run ci/run.sh first)")
            return False
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(hybrid_file) as f:
            hybrid = json.load(f)
        
        baseline_fid = baseline["fid_score"]
        hybrid_fid = hybrid["fid_score"]
        delta = hybrid_fid - baseline_fid
        
        ok = delta <= 3.0
        
        print(f"🎯 FID Check:")
        print(f"   Baseline: {baseline_fid:.1f}")
        print(f"   Hybrid: {hybrid_fid:.1f}")
        print(f"   Delta: {delta:.1f}")
        print(f"   Target: ≤ +3.0")
        print(f"   Status: {'✅ OK' if ok else '❌ FAIL'}")
        
        return ok
    except Exception as e:
        print(f"❌ FID check failed: {e}")
        return False

def check_ablation():
    """检查消融结果"""
    try:
        coarse_file = Path("results/fid_no_coarse.json")
        fine_file = Path("results/fid_no_fine.json")
        
        if not coarse_file.exists() or not fine_file.exists():
            print("🔬 Ablation Check: Files not found (run ablation first)")
            return False
        
        with open(coarse_file) as f:
            coarse = json.load(f)
        with open(fine_file) as f:
            fine = json.load(f)
        
        coarse_valuable = coarse.get("coarse_valuable", False)
        fine_valuable = fine.get("fine_valuable", False)
        
        coarse_delta = coarse.get("delta_fid_degradation", 0)
        fine_delta = fine.get("delta_fid_degradation", 0)
        
        print(f"🔬 Ablation Check:")
        print(f"   Coarse off: +{coarse_delta:.1f} FID {'✅' if coarse_valuable else '❌'}")
        print(f"   Fine off: +{fine_delta:.1f} FID {'✅' if fine_valuable else '❌'}")
        
        return coarse_valuable and fine_valuable
    except Exception as e:
        print(f"❌ Ablation check failed: {e}")
        return False

def main():
    print("🚀 ParScale-EAR Quick Validation")
    print("=" * 40)
    
    # Check what exists
    latency_ok = check_latency()
    fid_ok = check_fid()
    ablation_ok = check_ablation()
    
    print(f"\n📊 Summary:")
    print(f"   Latency: {'✅' if latency_ok else '❌'}")
    print(f"   FID: {'✅' if fid_ok else '❌'}")
    print(f"   Ablation: {'✅' if ablation_ok else '❌'}")
    
    if latency_ok and fid_ok and ablation_ok:
        print(f"\n🎉 ALL CHECKS PASSED - Ready for release!")
        return 0
    else:
        print(f"\n⚠️ Some checks failed - Run ci/run.sh to generate missing results")
        return 1

if __name__ == "__main__":
    exit(main())