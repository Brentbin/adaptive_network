import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import time

from node import AdaptiveNode, ActivationPattern
from subgraph import SubgraphFormation
from thinking import ThinkingController

@dataclass
class NetworkState:
    active_nodes: Set[int]
    thinking_depth: int
    confidence: float
    task_type: str

class AdaptiveNetwork(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 max_thinking_depth: int = 50):
        super().__init__()
        
        # 初始化节点
        self.nodes = nn.ModuleDict({
            str(i): AdaptiveNode(input_size, hidden_size, input_size)  # 中间节点保持输入维度
            for i in range(num_nodes - 1)  # 除了最后一个节点
        })
        
        # 最后一个节点负责输出维度转换
        self.nodes[str(num_nodes - 1)] = AdaptiveNode(input_size, hidden_size, output_size)
        
        # 控制器和组件
        self.subgraph_formation = SubgraphFormation()
        self.thinking_controller = ThinkingController(max_thinking_depth)
        
        # 状态跟踪
        self.current_state: Optional[NetworkState] = None
        self.thought_history: List[Tuple[torch.Tensor, float]] = []
        
    def forward(self, x: torch.Tensor, task_type: str) -> torch.Tensor:
        self.thinking_controller.reset()
        self.thought_history.clear()
        
        while not self.thinking_controller.should_stop():
            # 获取活跃子图
            active_nodes = self.subgraph_formation.form_subgraph(
                x, task_type, self.nodes
            )
            
            # 在子图上进行计算
            result = self._process_subgraph(x, active_nodes)
            
            # 评估结果
            confidence = self._evaluate_result(result)
            
            # 更新状态
            self.current_state = NetworkState(
                active_nodes=active_nodes,
                thinking_depth=self.thinking_controller.current_depth,
                confidence=confidence,
                task_type=task_type
            )
            
            self.thought_history.append((result, confidence))
            
            # 更新思考控制器
            self.thinking_controller.update(confidence)
            
            if self.thinking_controller.should_stop():
                break
                
        # 返回最佳结果
        return self._get_best_result()
        
    def _process_subgraph(self, 
                         x: torch.Tensor, 
                         active_nodes: Set[int]) -> torch.Tensor:
        """在活跃子图上进行计算
        Args:
            x: 输入张量，形状为 [batch_size, input_size]
            active_nodes: 活跃节点集合
        Returns:
            result: 输出张量，形状为 [batch_size, output_size]
        """
        # 确保输入是二维的 [batch_size, input_size]
        if x.dim() == 1:
            x = x.view(1, -1)  # [input_size] -> [1, input_size]
            
        result = x
        processed_nodes = set()
        
        # 按节点ID排序处理，确保最后一个节点最后处理
        sorted_nodes = sorted(active_nodes)
        for node_id in sorted_nodes:
            node = self.nodes[str(node_id)]
            
            try:
                result = node(result)
                processed_nodes.add(node_id)
            except Exception as e:
                print(f"Error processing node {node_id}:")
                print(f"Input shape: {result.shape}")
                print(f"Node input size: {node.input_size}")
                print(f"Node output size: {node.output_size}")
                raise e
                
        return result
        
    def _get_dependencies(self, node_id: int) -> Set[int]:
        """获取节点的依赖节点"""
        node = self.nodes[str(node_id)]
        return {
            dep_id for dep_id, strength in node.connection_strength.items()
            if strength > 0.5  # 连接强度阈值
        }
        
    def _evaluate_result(self, result: torch.Tensor) -> float:
        """评估结果的质量"""
        if not self.thought_history:
            return 0.5
            
        # 计算与历史结果的一致性
        prev_result, _ = self.thought_history[-1]
        consistency = torch.nn.functional.cosine_similarity(
            result.view(-1), 
            prev_result.view(-1), 
            dim=0
        ).item()
        
        return consistency
        
    def _get_best_result(self) -> torch.Tensor:
        """获取最佳结果"""
        if not self.thought_history:
            raise RuntimeError("No results available")
            
        # 选择置信度最高的结果
        best_idx = max(range(len(self.thought_history)),
                      key=lambda i: self.thought_history[i][1])
        
        return self.thought_history[best_idx][0]
        
    def update_network(self, performance: float) -> None:
        """根据性能更新网络"""
        if self.current_state is None:
            return
            
        # 更新节点专门化
        timestamp = time.time()
        for node_id in self.current_state.active_nodes:
            node = self.nodes[str(node_id)]
            pattern = ActivationPattern(
                task_type=self.current_state.task_type,
                input_pattern=self.thought_history[0][0],  # 初始输入
                output_pattern=self._get_best_result(),    # 最终输出
                performance=performance,
                timestamp=timestamp
            )
            node.update_specialization(pattern)
            
        # 更新连接强度
        self._update_connections(performance)
        
    def _update_connections(self, performance: float) -> None:
        """更新节点间的连接强度"""
        if self.current_state is None:
            return
            
        active_nodes = list(self.current_state.active_nodes)
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                node_i = self.nodes[str(active_nodes[i])]
                node_j = self.nodes[str(active_nodes[j])]
                
                # 计算连接强度变化
                strength_delta = performance * 0.1  # 学习率
                
                # 双向更新连接
                node_i.update_connection(active_nodes[j], strength_delta)
                node_j.update_connection(active_nodes[i], strength_delta)
                
    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        return {
            'num_nodes': len(self.nodes),
            'active_nodes': len(self.current_state.active_nodes) if self.current_state else 0,
            'thinking_depth': self.thinking_controller.current_depth,
            'avg_confidence': sum(conf for _, conf in self.thought_history) / len(self.thought_history) if self.thought_history else 0.0
        } 