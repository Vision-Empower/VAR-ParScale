#!/usr/bin/env python3
"""
Experiment 3: Direct Comparison (ParScale-VAR P=2 vs. Baseline VAR)
目标: 直接对比ParScale-VAR P=2与基线VAR性能

验证假设:
1. 质量假设: ParScale-VAR FID更低, IS更高 (相似参数数量)
2. 效率假设: 延迟 ≤1.2x基线, 显存 <<2x基线
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats

@dataclass
class ComparisonConfig:
    """对比实验配置"""
    num_evaluation_samples: int = 10000
    batch_size: int = 20
    num_repeated_runs: int = 3  # 统计显著性验证
    significance_threshold: float = 0.05
    efficiency_latency_target: float = 1.2  # 1.2x基线延迟目标
    
class StatisticalComparison:
    """统计显著性测试"""
    
    @staticmethod
    def paired_t_test(baseline_metrics: List[float], 
                     parscale_metrics: List[float], 
                     alpha: float = 0.05) -> Dict:
        """配对t检验"""
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(parscale_metrics, baseline_metrics)
        
        # 计算效应大小 (Cohen's d)
        diff = np.array(parscale_metrics) - np.array(baseline_metrics)
        cohens_d = np.mean(diff) / np.std(diff)
        
        result = {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
            'mean_baseline': np.mean(baseline_metrics),
            'mean_parscale': np.mean(parscale_metrics),
            'improvement_pct': (np.mean(parscale_metrics) - np.mean(baseline_metrics)) / np.mean(baseline_metrics) * 100
        }
        
        return result

class DirectComparisonExperiment:
    """直接对比实验实施"""
    
    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()
        self.setup_logging()
        
        # 结果存储
        self.baseline_results = []
        self.parscale_results = []
        self.comparison_summary = {}
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - EXP3-Comparison - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('exp3_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def compare_model_parameters(self, baseline_model, parscale_model) -> Dict:
        """对比模型参数数量"""
        
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        parscale_params = sum(p.numel() for p in parscale_model.parameters())
        
        # ParScale额外参数 (变换头 + 聚合头)
        shared_backbone_params = sum(p.numel() for p in parscale_model.shared_var_backbone.parameters())
        additional_params = parscale_params - shared_backbone_params
        
        param_comparison = {
            'baseline_parameters': baseline_params,
            'parscale_total_parameters': parscale_params,
            'shared_backbone_parameters': shared_backbone_params,
            'additional_parameters': additional_params,
            'parameter_ratio': parscale_params / baseline_params,
            'additional_param_percentage': (additional_params / baseline_params) * 100
        }
        
        self.logger.info(f"📊 参数对比: {param_comparison}")
        
        # 验证参数效率
        if param_comparison['parameter_ratio'] > 1.3:  # 超过30%增加
            self.logger.warning("⚠️  ParScale参数增加超过30%, 可能影响公平对比")
            
        return param_comparison
        
    def measure_model_performance(self, model, model_name: str, 
                                device='cuda', is_parscale=False) -> Dict:
        """测量单个模型性能"""
        
        self.logger.info(f"📊 测量 {model_name} 性能...")
        
        model.eval()
        
        # 性能指标收集
        generated_images = []
        inference_times = []
        memory_usages = []
        
        num_samples = self.config.num_evaluation_samples
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for batch_idx in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - batch_idx)
                
                # 显存基线
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(device)
                
                # 推理时间测量
                start_time = time.time()
                torch.cuda.synchronize()
                
                try:
                    if is_parscale:
                        # ParScale推理 (包含多流处理)
                        generated_tokens = model.autoregressive_infer_cfg(
                            B=current_batch_size,
                            label_B=None,
                            cfg=1.0,
                            top_k=900,
                            top_p=0.95
                        )
                    else:
                        # 基线VAR推理
                        generated_tokens = model.autoregressive_infer_cfg(
                            B=current_batch_size,
                            label_B=None,
                            cfg=1.0,
                            top_k=900,
                            top_p=0.95
                        )
                        
                    # TODO: 解码为图像 (需要VQVAE)
                    # generated_imgs = vae_model.decode(generated_tokens)
                    
                except Exception as e:
                    self.logger.error(f"{model_name} 生成失败: {e}")
                    continue
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                memory_after = torch.cuda.memory_allocated(device)
                
                # 记录性能
                batch_inference_time = (end_time - start_time) / current_batch_size
                inference_times.append(batch_inference_time)
                memory_usages.append(memory_after - memory_before)
                
                # generated_images.extend(generated_imgs.cpu())
                
                if batch_idx % 1000 == 0:
                    avg_time = np.mean(inference_times) * 1000
                    self.logger.info(f"{model_name} 进度: {batch_idx + current_batch_size}/{num_samples}, "
                                   f"平均时间: {avg_time:.2f}ms")
        
        # 计算最终指标
        # TODO: 实现真实FID/IS计算
        fid_score = self.calculate_fid_placeholder()
        is_score = self.calculate_is_placeholder()
        
        results = {
            'model_name': model_name,
            'fid_score': fid_score,
            'inception_score': is_score,
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'inference_time_std': np.std(inference_times) * 1000,
            'peak_memory_gb': max(memory_usages) / (1024**3),
            'avg_memory_gb': np.mean(memory_usages) / (1024**3),
            'samples_evaluated': len(inference_times) * batch_size
        }
        
        self.logger.info(f"✅ {model_name} 结果: {results}")
        return results
        
    def calculate_fid_placeholder(self) -> float:
        """FID计算占位符"""
        # 基线: 2.1, ParScale目标: 1.8-1.9 (改善)
        return np.random.uniform(1.8, 2.2)
        
    def calculate_is_placeholder(self) -> float:
        """IS计算占位符"""
        # 基线: 342, ParScale目标: 350-360 (改善)
        return np.random.uniform(340, 365)
        
    def run_repeated_comparison(self, baseline_model, parscale_model) -> Dict:
        """执行多次重复对比 (统计显著性)"""
        
        self.logger.info(f"🔄 执行 {self.config.num_repeated_runs} 次重复对比...")
        
        baseline_fids = []
        baseline_iss = []
        baseline_latencies = []
        
        parscale_fids = []
        parscale_iss = []
        parscale_latencies = []
        
        # 多次运行
        for run_idx in range(self.config.num_repeated_runs):
            self.logger.info(f"📊 执行第 {run_idx + 1}/{self.config.num_repeated_runs} 次运行...")
            
            # 基线VAR
            baseline_result = self.measure_model_performance(
                baseline_model, f"Baseline_Run_{run_idx}", is_parscale=False
            )
            baseline_fids.append(baseline_result['fid_score'])
            baseline_iss.append(baseline_result['inception_score'])
            baseline_latencies.append(baseline_result['avg_inference_time_ms'])
            
            # ParScale-VAR
            parscale_result = self.measure_model_performance(
                parscale_model, f"ParScale_P2_Run_{run_idx}", is_parscale=True
            )
            parscale_fids.append(parscale_result['fid_score'])
            parscale_iss.append(parscale_result['inception_score'])
            parscale_latencies.append(parscale_result['avg_inference_time_ms'])
            
        # 统计分析
        statistical_results = {
            'fid_comparison': StatisticalComparison.paired_t_test(baseline_fids, parscale_fids),
            'is_comparison': StatisticalComparison.paired_t_test(baseline_iss, parscale_iss),
            'latency_comparison': StatisticalComparison.paired_t_test(baseline_latencies, parscale_latencies)
        }
        
        return statistical_results
        
    def evaluate_hypotheses(self, statistical_results: Dict, param_comparison: Dict) -> Dict:
        """评估验证假设"""
        
        self.logger.info("🎯 评估核心假设...")
        
        # 假设1: 质量改善 (FID降低, IS提升)
        fid_improved = (statistical_results['fid_comparison']['improvement_pct'] < 0 and 
                       statistical_results['fid_comparison']['is_significant'])
        
        is_improved = (statistical_results['is_comparison']['improvement_pct'] > 0 and
                      statistical_results['is_comparison']['is_significant'])
        
        quality_hypothesis_met = fid_improved or is_improved
        
        # 假设2: 效率可接受 (延迟 ≤1.2x)
        latency_ratio = (statistical_results['latency_comparison']['mean_parscale'] / 
                        statistical_results['latency_comparison']['mean_baseline'])
        
        efficiency_hypothesis_met = latency_ratio <= self.config.efficiency_latency_target
        
        # 参数效率
        parameter_efficiency_good = param_comparison['parameter_ratio'] <= 1.5  # 参数增加<50%
        
        hypothesis_evaluation = {
            'quality_hypothesis': {
                'met': quality_hypothesis_met,
                'fid_improved': fid_improved,
                'is_improved': is_improved,
                'fid_improvement_pct': statistical_results['fid_comparison']['improvement_pct'],
                'is_improvement_pct': statistical_results['is_comparison']['improvement_pct']
            },
            'efficiency_hypothesis': {
                'met': efficiency_hypothesis_met,
                'latency_ratio': latency_ratio,
                'target_ratio': self.config.efficiency_latency_target,
                'latency_improvement_pct': statistical_results['latency_comparison']['improvement_pct']
            },
            'parameter_efficiency': {
                'acceptable': parameter_efficiency_good,
                'parameter_ratio': param_comparison['parameter_ratio'],
                'additional_params_pct': param_comparison['additional_param_percentage']
            },
            'overall_success': quality_hypothesis_met and efficiency_hypothesis_met and parameter_efficiency_good
        }
        
        self.logger.info(f"🎯 假设评估结果: {hypothesis_evaluation}")
        
        return hypothesis_evaluation
        
    def generate_decision_recommendation(self, hypothesis_results: Dict) -> str:
        """生成决策建议"""
        
        if hypothesis_results['overall_success']:
            decision = "✅ PROCEED: ParScale-VAR概念验证成功!"
            reasoning = [
                "质量假设得到验证",
                "效率假设满足目标", 
                "参数效率可接受",
                "建议进行Phase 2详细消融研究"
            ]
        else:
            failed_aspects = []
            if not hypothesis_results['quality_hypothesis']['met']:
                failed_aspects.append("质量改善不显著")
            if not hypothesis_results['efficiency_hypothesis']['met']:
                failed_aspects.append("效率成本过高")
            if not hypothesis_results['parameter_efficiency']['acceptable']:
                failed_aspects.append("参数开销过大")
                
            decision = "⚠️  INVESTIGATE: 需要诊断和调优"
            reasoning = [
                f"失败方面: {', '.join(failed_aspects)}",
                "建议检查实现细节",
                "调优超参数 (λ, 温度调度)",
                "考虑改进变换T_i设计",
                "消融研究变为诊断工具"
            ]
            
        recommendation = {
            'decision': decision,
            'reasoning': reasoning,
            'next_steps': reasoning[-2:] if not hypothesis_results['overall_success'] else ["Phase 2消融研究", "扩展P值实验"]
        }
        
        self.logger.info(f"🎯 决策建议: {recommendation}")
        
        return recommendation
        
    def save_comparison_report(self, all_results: Dict, output_dir: Path):
        """保存完整对比报告"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comprehensive_report = {
            'experiment': 'ParScale-VAR P=2 vs Baseline VAR Direct Comparison',
            'configuration': {
                'num_evaluation_samples': self.config.num_evaluation_samples,
                'num_repeated_runs': self.config.num_repeated_runs,
                'significance_threshold': self.config.significance_threshold
            },
            'results': all_results,
            'timestamp': time.time(),
            'device': 'H100 80GB'
        }
        
        report_path = output_dir / 'exp3_comparison_report.json'
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
            
        self.logger.info(f"📝 对比报告已保存: {report_path}")

def main():
    """执行直接对比实验"""
    print("🚀 开始 Experiment 3: ParScale-VAR P=2 vs 基线VAR直接对比")
    print("🎯 验证假设: 质量改善 + 效率可接受")
    
    experiment = DirectComparisonExperiment()
    
    # TODO: 加载训练好的模型
    print("⚠️  TODO: 加载基线VAR模型")
    print("⚠️  TODO: 加载训练好的ParScale-VAR P=2模型")
    print("⚠️  TODO: 执行多次重复对比")
    print("⚠️  TODO: 统计显著性分析")
    
    # baseline_model = load_baseline_var()
    # parscale_model = load_parscale_var_p2()
    
    # param_comparison = experiment.compare_model_parameters(baseline_model, parscale_model)
    # statistical_results = experiment.run_repeated_comparison(baseline_model, parscale_model)
    # hypothesis_results = experiment.evaluate_hypotheses(statistical_results, param_comparison)
    # decision = experiment.generate_decision_recommendation(hypothesis_results)
    
    print("⚖️  直接对比实验框架已就绪")

if __name__ == "__main__":
    main()