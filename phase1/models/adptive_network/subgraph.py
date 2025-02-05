import torch
import torch.nn as nn
from typing import Dict, List, Set, Any
import numpy as np

class SubgraphFormation:
    def __init__(self,
                 min_relevance_score: float = 0.3,
                 max_active_nodes: int = 5):
        self.min_relevance_score = min_relevance_score
        self.max_active_nodes = max_active_nodes
        
    def form_subgraph(self,
                     x: torch.Tensor,
                     task_type: str,
                     nodes: nn.ModuleDict) -> Set[int]:
        """形成处理子图"""
        # 计算相关性分数
        relevance_scores = self._compute_relevance_scores(x, task_type, nodes)
        
        # 选择初始节点
        active_nodes = self._select_initial_nodes(relevance_scores)
        
        # 动态扩展子图
        while len(active_nodes) < self.max_active_nodes:
            expansion = self._expand_subgraph(
                active_nodes,
                relevance_scores,
                nodes
            )
            
            if not expansion:
                break
                
            active_nodes.update(expansion)
            
        return active_nodes
        
    def _compute_relevance_scores(self,
                                x: torch.Tensor,
                                task_type: str,
                                nodes: nn.ModuleDict) -> Dict[int, float]:
        """计算节点与当前任务的相关性分数"""
        scores = {}
        for node_id, node in nodes.items():
            # 专门化匹配度
            spec_score = self._compute_specialization_score(
                node.specialization,
                task_type
            )
            
            # 历史性能
            perf_score = self._compute_performance_score(node)
            
            # 输入特征匹配度
            feature_score = self._compute_feature_match_score(node, x)
            
            # 综合评分
            scores[int(node_id)] = (
                spec_score * 0.4 +
                perf_score * 0.3 +
                feature_score * 0.3
            )
            
        return scores
        
    def _compute_specialization_score(self,
                                    specialization: Dict[str, float],
                                    task_type: str) -> float:
        """计算专门化匹配分数"""
        if not specialization:
            return 0.5
            
        return specialization.get(task_type, 0.0)
        
    def _compute_performance_score(self, node: Any) -> float:
        """计算历史性能分数"""
        stats = node.get_performance_stats()
        return stats['mean']
        
    def _compute_feature_match_score(self,
                                   node: Any,
                                   x: torch.Tensor) -> float:
        """计算输入特征匹配分数"""
        if not node.activation_patterns:
            return 0.5
            
        # 确保输入是二维的
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 计算与历史输入模式的相似度
        similarities = []
        for pattern in node.activation_patterns[-10:]:  # 只使用最近的10个模式
            # 确保历史模式也是二维的
            pattern_input = pattern.input_pattern
            if pattern_input.dim() == 1:
                pattern_input = pattern_input.unsqueeze(0)
                
            # 计算相似度
            try:
                sim = torch.nn.functional.cosine_similarity(
                    x.view(1, -1),
                    pattern_input.view(1, -1),
                    dim=1
                ).mean().item()
                similarities.append(sim)
            except:
                # 如果维度不匹配，返回中性分数
                similarities.append(0.5)
            
        return np.mean(similarities) if similarities else 0.5
        
    def _select_initial_nodes(self,
                            relevance_scores: Dict[int, float]) -> Set[int]:
        """选择初始节点"""
        # 按相关性分数排序
        sorted_nodes = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择分数高于阈值的节点
        initial_nodes = {
            node_id for node_id, score in sorted_nodes
            if score >= self.min_relevance_score
        }
        
        # 确保至少有一个节点
        if not initial_nodes and sorted_nodes:
            initial_nodes.add(sorted_nodes[0][0])
            
        return initial_nodes
        
    def _expand_subgraph(self,
                        active_nodes: Set[int],
                        relevance_scores: Dict[int, float],
                        nodes: nn.ModuleDict) -> Set[int]:
        """扩展子图"""
        expansion = set()
        
        # 考虑所有活跃节点的强连接节点
        for node_id in active_nodes:
            node = nodes[str(node_id)]
            
            # 获取强连接节点
            connected_nodes = {
                dep_id for dep_id, strength in node.connection_strength.items()
                if strength > 0.5  # 连接强度阈值
            }
            
            # 在强连接节点中选择相关性高的
            for connected_id in connected_nodes:
                if (connected_id not in active_nodes and
                    relevance_scores[connected_id] >= self.min_relevance_score):
                    expansion.add(connected_id)
                    
        return expansion 