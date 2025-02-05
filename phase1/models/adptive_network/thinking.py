from typing import List, Optional
import numpy as np

class ThinkingController:
    def __init__(self,
                 max_depth: int = 50,
                 convergence_threshold: float = 0.95,
                 min_confidence: float = 0.7,
                 patience: int = 5):
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.patience = patience
        
        self.current_depth = 0
        self.confidence_history: List[float] = []
        self.best_confidence: float = 0.0
        self.steps_without_improvement = 0
        
    def reset(self) -> None:
        """重置控制器状态"""
        self.current_depth = 0
        self.confidence_history.clear()
        self.best_confidence = 0.0
        self.steps_without_improvement = 0
        
    def update(self, confidence: float) -> None:
        """更新控制器状态"""
        self.current_depth += 1
        self.confidence_history.append(confidence)
        
        # 更新最佳置信度
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
            
    def should_stop(self) -> bool:
        """判断是否应该停止思考"""
        # 检查是否达到最大深度
        if self.current_depth >= self.max_depth:
            return True
            
        # 检查是否已经达到足够高的置信度
        if self.best_confidence >= self.convergence_threshold:
            return True
            
        # 检查是否已经收敛
        if self._check_convergence():
            return True
            
        # 检查是否长时间没有改进
        if self.steps_without_improvement >= self.patience:
            return True
            
        return False
        
    def _check_convergence(self) -> bool:
        """检查思考过程是否收敛"""
        if len(self.confidence_history) < 3:
            return False
            
        # 计算最近几次结果的方差
        recent_confidences = self.confidence_history[-3:]
        variance = np.var(recent_confidences)
        
        # 如果方差很小，认为已经收敛
        return variance < 0.01
        
    def get_thinking_stats(self) -> dict:
        """获取思考过程的统计信息"""
        if not self.confidence_history:
            return {
                'current_depth': 0,
                'best_confidence': 0.0,
                'avg_confidence': 0.0,
                'convergence_rate': 0.0
            }
            
        return {
            'current_depth': self.current_depth,
            'best_confidence': self.best_confidence,
            'avg_confidence': np.mean(self.confidence_history),
            'convergence_rate': self._calculate_convergence_rate()
        }
        
    def _calculate_convergence_rate(self) -> float:
        """计算收敛率"""
        if len(self.confidence_history) < 2:
            return 0.0
            
        # 计算置信度的变化率
        changes = np.diff(self.confidence_history)
        return np.mean(np.abs(changes))
        
    def adjust_parameters(self, performance: float) -> None:
        """根据性能调整控制参数"""
        # 动态调整收敛阈值
        if performance > 0.8:
            self.convergence_threshold = min(0.98, self.convergence_threshold * 1.05)
        else:
            self.convergence_threshold = max(0.9, self.convergence_threshold * 0.95)
            
        # 动态调整耐心值
        if self.steps_without_improvement > self.patience:
            self.patience = min(10, self.patience + 1)
        else:
            self.patience = max(3, self.patience - 1) 