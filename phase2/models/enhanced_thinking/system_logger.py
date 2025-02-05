"""日志系统模块

负责系统运行时的日志记录，包括性能指标、状态信息和错误信息。
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch

class SystemLogger:
    """系统日志器"""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._setup_log_dir()
        self._setup_loggers()
        
        # 性能指标历史
        self.performance_history: List[Dict[str, Any]] = []
        # 状态历史
        self.state_history: List[Dict[str, Any]] = []
        # 错误历史
        self.error_history: List[Dict[str, Any]] = []
        
    def _setup_log_dir(self) -> None:
        """设置日志目录"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'performance'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'state'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'error'), exist_ok=True)
        
    def _setup_loggers(self) -> None:
        """设置日志记录器"""
        # 性能日志
        self.perf_logger = logging.getLogger('performance')
        perf_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'performance.log')
        )
        perf_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
        # 状态日志
        self.state_logger = logging.getLogger('state')
        state_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'state.log')
        )
        state_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.state_logger.addHandler(state_handler)
        self.state_logger.setLevel(logging.INFO)
        
        # 错误日志
        self.error_logger = logging.getLogger('error')
        error_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'error.log')
        )
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)
        
    def log_performance(self, 
                       level: str,
                       metrics: Dict[str, float],
                       step: Optional[int] = None) -> None:
        """记录性能指标
        
        Args:
            level: 思维层次名称
            metrics: 性能指标字典
            step: 当前步骤（可选）
        """
        timestamp = datetime.now().isoformat()
        record = {
            'timestamp': timestamp,
            'level': level,
            'metrics': metrics,
            'step': step
        }
        
        # 添加到历史记录
        self.performance_history.append(record)
        
        # 记录到日志文件
        self.perf_logger.info(
            f"Level: {level}, Metrics: {json.dumps(metrics)}"
        )
        
        # 保存详细记录
        if step is not None:
            filename = f"step_{step:06d}.json"
        else:
            filename = f"{timestamp}.json"
            
        save_path = os.path.join(
            self.log_dir, 'performance', filename
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
    def log_state(self,
                  level: str,
                  state: Dict[str, Any],
                  step: Optional[int] = None) -> None:
        """记录状态信息
        
        Args:
            level: 思维层次名称
            state: 状态信息字典
            step: 当前步骤（可选）
        """
        timestamp = datetime.now().isoformat()
        
        # 处理不可序列化的对象
        processed_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                processed_state[k] = v.detach().cpu().numpy().tolist()
            else:
                processed_state[k] = v
                
        record = {
            'timestamp': timestamp,
            'level': level,
            'state': processed_state,
            'step': step
        }
        
        # 添加到历史记录
        self.state_history.append(record)
        
        # 记录到日志文件
        self.state_logger.info(
            f"Level: {level}, State: {json.dumps(processed_state)}"
        )
        
        # 保存详细记录
        if step is not None:
            filename = f"step_{step:06d}.json"
        else:
            filename = f"{timestamp}.json"
            
        save_path = os.path.join(
            self.log_dir, 'state', filename
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
    def log_error(self,
                  level: str,
                  error: Exception,
                  context: Dict[str, Any]) -> None:
        """记录错误信息
        
        Args:
            level: 思维层次名称
            error: 异常对象
            context: 错误发生时的上下文信息
        """
        timestamp = datetime.now().isoformat()
        
        # 处理不可序列化的对象
        processed_context = {}
        for k, v in context.items():
            if isinstance(v, torch.Tensor):
                processed_context[k] = v.detach().cpu().numpy().tolist()
            else:
                processed_context[k] = v
                
        record = {
            'timestamp': timestamp,
            'level': level,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': processed_context
        }
        
        # 添加到历史记录
        self.error_history.append(record)
        
        # 记录到日志文件
        self.error_logger.error(
            f"Level: {level}, Error: {type(error).__name__}, "
            f"Message: {str(error)}, Context: {json.dumps(processed_context)}"
        )
        
        # 保存详细记录
        filename = f"error_{timestamp}.json"
        save_path = os.path.join(
            self.log_dir, 'error', filename
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
    def get_performance_summary(self,
                              level: Optional[str] = None,
                              last_n: Optional[int] = None) -> Dict[str, Any]:
        """获取性能总结
        
        Args:
            level: 指定层次（可选）
            last_n: 最近n条记录（可选）
            
        Returns:
            性能统计信息
        """
        records = self.performance_history
        
        if level is not None:
            records = [r for r in records if r['level'] == level]
            
        if last_n is not None:
            records = records[-last_n:]
            
        if not records:
            return {}
            
        # 计算统计信息
        metrics_summary = {}
        for record in records:
            for metric, value in record['metrics'].items():
                if metric not in metrics_summary:
                    metrics_summary[metric] = []
                metrics_summary[metric].append(value)
                
        # 计算统计值
        summary = {}
        for metric, values in metrics_summary.items():
            summary[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }
            
        return summary
        
    def get_error_summary(self,
                         level: Optional[str] = None,
                         error_type: Optional[str] = None) -> Dict[str, Any]:
        """获取错误总结
        
        Args:
            level: 指定层次（可选）
            error_type: 指定错误类型（可选）
            
        Returns:
            错误统计信息
        """
        records = self.error_history
        
        if level is not None:
            records = [r for r in records if r['level'] == level]
            
        if error_type is not None:
            records = [r for r in records if r['error_type'] == error_type]
            
        if not records:
            return {}
            
        # 统计错误类型
        error_counts = {}
        for record in records:
            error_type = record['error_type']
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
            
        return {
            'total_errors': len(records),
            'error_types': error_counts,
            'latest_error': records[-1]
        }
        
    def clear_history(self,
                     clear_performance: bool = True,
                     clear_state: bool = True,
                     clear_error: bool = True) -> None:
        """清除历史记录
        
        Args:
            clear_performance: 是否清除性能历史
            clear_state: 是否清除状态历史
            clear_error: 是否清除错误历史
        """
        if clear_performance:
            self.performance_history.clear()
            
        if clear_state:
            self.state_history.clear()
            
        if clear_error:
            self.error_history.clear() 