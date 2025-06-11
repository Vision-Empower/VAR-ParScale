#!/usr/bin/env python3
"""
Decision Framework: ParScale-VAR概念验证决策点
基于实验3结果确定下一步行动

决策逻辑:
✅ 成功 → Phase 2详细消融研究
⚠️  问题 → 诊断调优 → 重新评估
❌ 失败 → 重新设计概念
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class DecisionOutcome(Enum):
    """决策结果枚举"""
    PROCEED_TO_ABLATION = "proceed_to_phase2_ablation"
    INVESTIGATE_AND_TUNE = "investigate_and_tune"
    REDESIGN_CONCEPT = "redesign_concept"

@dataclass
class DecisionCriteria:
    """决策标准"""
    # 质量改善标准
    min_fid_improvement_pct: float = -5.0    # FID至少改善5%
    min_is_improvement_pct: float = 2.0      # IS至少改善2%
    
    # 效率标准
    max_latency_ratio: float = 1.2           # 延迟最多1.2x
    max_memory_ratio: float = 1.8            # 显存最多1.8x (远小于2x)
    
    # 参数效率标准
    max_parameter_ratio: float = 1.5         # 参数最多增加50%
    
    # 统计显著性
    significance_threshold: float = 0.05

class ParScaleVARDecisionFramework:
    """ParScale-VAR决策框架"""
    
    def __init__(self, criteria: DecisionCriteria = None):
        self.criteria = criteria or DecisionCriteria()
        
    def evaluate_quality_hypothesis(self, comparison_results: Dict) -> Dict:
        """评估质量假设"""
        
        fid_results = comparison_results.get('fid_comparison', {})
        is_results = comparison_results.get('is_comparison', {})
        
        # FID改善 (降低)
        fid_improved = (
            fid_results.get('improvement_pct', 0) <= self.criteria.min_fid_improvement_pct and
            fid_results.get('is_significant', False)
        )
        
        # IS改善 (提升)
        is_improved = (
            is_results.get('improvement_pct', 0) >= self.criteria.min_is_improvement_pct and
            is_results.get('is_significant', False)
        )
        
        quality_score = 0
        if fid_improved: quality_score += 1
        if is_improved: quality_score += 1
        
        quality_evaluation = {
            'fid_improved': fid_improved,
            'is_improved': is_improved,
            'quality_score': quality_score,  # 0-2分
            'quality_hypothesis_met': quality_score >= 1,  # 至少一个指标改善
            'details': {
                'fid_improvement_pct': fid_results.get('improvement_pct', 0),
                'fid_significant': fid_results.get('is_significant', False),
                'is_improvement_pct': is_results.get('improvement_pct', 0),
                'is_significant': is_results.get('is_significant', False)
            }
        }
        
        return quality_evaluation
        
    def evaluate_efficiency_hypothesis(self, comparison_results: Dict) -> Dict:
        """评估效率假设"""
        
        latency_results = comparison_results.get('latency_comparison', {})
        
        # 延迟比率
        latency_ratio = (latency_results.get('mean_parscale', 0) / 
                        max(latency_results.get('mean_baseline', 1), 1))
        
        latency_acceptable = latency_ratio <= self.criteria.max_latency_ratio
        
        # TODO: 添加显存比率评估 (需要从实验结果获取)
        memory_ratio = 1.3  # 占位符
        memory_acceptable = memory_ratio <= self.criteria.max_memory_ratio
        
        efficiency_score = 0
        if latency_acceptable: efficiency_score += 1
        if memory_acceptable: efficiency_score += 1
        
        efficiency_evaluation = {
            'latency_acceptable': latency_acceptable,
            'memory_acceptable': memory_acceptable,
            'efficiency_score': efficiency_score,  # 0-2分
            'efficiency_hypothesis_met': efficiency_score >= 1,
            'details': {
                'latency_ratio': latency_ratio,
                'latency_target': self.criteria.max_latency_ratio,
                'memory_ratio': memory_ratio,
                'memory_target': self.criteria.max_memory_ratio
            }
        }
        
        return efficiency_evaluation
        
    def evaluate_parameter_efficiency(self, param_comparison: Dict) -> Dict:
        """评估参数效率"""
        
        param_ratio = param_comparison.get('parameter_ratio', 1.0)
        param_acceptable = param_ratio <= self.criteria.max_parameter_ratio
        
        additional_param_pct = param_comparison.get('additional_param_percentage', 0)
        
        param_evaluation = {
            'parameter_ratio_acceptable': param_acceptable,
            'parameter_efficiency_met': param_acceptable,
            'details': {
                'parameter_ratio': param_ratio,
                'parameter_target': self.criteria.max_parameter_ratio,
                'additional_param_pct': additional_param_pct
            }
        }
        
        return param_evaluation
        
    def make_decision(self, experiment_results: Dict) -> Dict:
        """做出最终决策"""
        
        # 解析实验结果
        comparison_results = experiment_results.get('statistical_results', {})
        param_comparison = experiment_results.get('parameter_comparison', {})
        
        # 评估各项假设
        quality_eval = self.evaluate_quality_hypothesis(comparison_results)
        efficiency_eval = self.evaluate_efficiency_hypothesis(comparison_results)
        param_eval = self.evaluate_parameter_efficiency(param_comparison)
        
        # 计算总分
        total_score = (quality_eval['quality_score'] + 
                      efficiency_eval['efficiency_score'] + 
                      (2 if param_eval['parameter_efficiency_met'] else 0))
        
        # 决策逻辑
        if total_score >= 5:  # 满分6分，至少5分
            decision = DecisionOutcome.PROCEED_TO_ABLATION
            confidence = "high"
            rationale = "ParScale-VAR概念验证成功，所有核心假设得到验证"
            
        elif total_score >= 3:  # 中等分数
            decision = DecisionOutcome.INVESTIGATE_AND_TUNE
            confidence = "medium"
            rationale = "ParScale-VAR显示潜力但需要调优"
            
        else:  # 低分
            decision = DecisionOutcome.REDESIGN_CONCEPT
            confidence = "low"
            rationale = "当前设计未达到预期，需要重新思考概念"
            
        # 生成具体行动建议
        action_plan = self.generate_action_plan(decision, quality_eval, efficiency_eval, param_eval)
        
        decision_result = {
            'decision': decision.value,
            'confidence': confidence,
            'total_score': total_score,
            'max_score': 6,
            'rationale': rationale,
            'evaluations': {
                'quality': quality_eval,
                'efficiency': efficiency_eval,
                'parameter_efficiency': param_eval
            },
            'action_plan': action_plan
        }
        
        return decision_result
        
    def generate_action_plan(self, decision: DecisionOutcome, 
                           quality_eval: Dict, efficiency_eval: Dict, param_eval: Dict) -> List[str]:
        """生成具体行动计划"""
        
        if decision == DecisionOutcome.PROCEED_TO_ABLATION:
            return [
                "🚀 立即启动Phase 2详细消融研究",
                "📊 优先P值扩展实验 (P=4, P=8)",
                "🔍 分析Transform层T_i有效性",
                "📈 研究多样性正则化最优λ值",
                "⚙️ 探索Token-wise vs Global聚合策略",
                "📝 准备高质量研究论文"
            ]
            
        elif decision == DecisionOutcome.INVESTIGATE_AND_TUNE:
            issues = []
            solutions = []
            
            if not quality_eval['quality_hypothesis_met']:
                issues.append("质量改善不足")
                solutions.extend([
                    "🔧 调整多样性损失权重λ (当前可能过小/过大)",
                    "🔄 实验不同T_i变换 (rotation, scaling, color)",
                    "🌡️ 优化温度调度策略",
                    "📊 增加训练数据多样性"
                ])
                
            if not efficiency_eval['efficiency_hypothesis_met']:
                issues.append("效率成本过高")
                solutions.extend([
                    "⚡ 优化共享KV缓存实现",
                    "🔀 简化Token-wise聚合计算",
                    "💾 实现更激进的内存优化",
                    "⏱️ 分析推理瓶颈并优化"
                ])
                
            if not param_eval['parameter_efficiency_met']:
                issues.append("参数开销过大")
                solutions.extend([
                    "📉 减少聚合头参数",
                    "🔗 探索参数共享策略",
                    "✂️ 精简变换层设计"
                ])
                
            action_plan = [f"⚠️  识别问题: {', '.join(issues)}"] + solutions + [
                "🔄 重新执行实验3验证改进效果",
                "📋 如果仍未达标，考虑重新设计"
            ]
            
            return action_plan
            
        else:  # REDESIGN_CONCEPT
            return [
                "🔄 重新评估ParScale-VAR核心假设",
                "📚 深入研究VAR架构限制",
                "💡 探索替代并行化策略",
                "🎯 考虑不同的多样性机制",
                "🔬 或转向其他模型架构融合",
                "⏸️ 暂停当前实现，重新设计概念"
            ]
            
    def save_decision_report(self, decision_result: Dict, output_dir: Path):
        """保存决策报告"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        decision_report = {
            'decision_framework': 'ParScale-VAR Concept Verification',
            'criteria_used': {
                'min_fid_improvement_pct': self.criteria.min_fid_improvement_pct,
                'min_is_improvement_pct': self.criteria.min_is_improvement_pct,
                'max_latency_ratio': self.criteria.max_latency_ratio,
                'max_memory_ratio': self.criteria.max_memory_ratio,
                'max_parameter_ratio': self.criteria.max_parameter_ratio
            },
            'decision_result': decision_result,
            'timestamp': time.time()
        }
        
        report_path = output_dir / 'decision_framework_report.json'
        with open(report_path, 'w') as f:
            json.dump(decision_report, f, indent=2)
            
        # 生成可读性总结
        summary_path = output_dir / 'decision_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"ParScale-VAR概念验证决策报告\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"🎯 决策结果: {decision_result['decision']}\n")
            f.write(f"🎚️ 置信度: {decision_result['confidence']}\n")
            f.write(f"📊 总分: {decision_result['total_score']}/{decision_result['max_score']}\n\n")
            f.write(f"📝 决策理由:\n{decision_result['rationale']}\n\n")
            f.write(f"📋 行动计划:\n")
            for i, action in enumerate(decision_result['action_plan'], 1):
                f.write(f"{i}. {action}\n")
                
        print(f"📝 决策报告已保存: {report_path}")
        print(f"📄 决策总结已保存: {summary_path}")

def main():
    """决策框架主程序"""
    print("🎯 ParScale-VAR概念验证决策框架")
    print("📊 基于实验3结果进行智能决策")
    
    # TODO: 加载实验3结果
    print("⚠️  TODO: 加载实验3对比结果")
    print("⚠️  TODO: 应用决策框架")
    print("⚠️  TODO: 生成行动计划")
    
    # 示例用法:
    # decision_framework = ParScaleVARDecisionFramework()
    # experiment3_results = load_experiment3_results()
    # decision = decision_framework.make_decision(experiment3_results)
    # decision_framework.save_decision_report(decision, Path('/Users/peter/VAR-ParScale/results'))
    
    print("🎯 决策框架已就绪")

if __name__ == "__main__":
    main()