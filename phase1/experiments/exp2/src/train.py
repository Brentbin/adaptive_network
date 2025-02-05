"""训练脚本"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
from typing import Dict, List, Union

root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.append(str(root_path))
from experiments.exp2.config import CONFIG
from experiments.exp2.src.network import MultiTaskSequenceNetwork
from experiments.exp2.src.utils import (
    load_data, calculate_metrics, analyze_node_specialization,
    plot_specialization_heatmap, save_training_log
)
from models.baselines.mlp import MLPNetwork

class ExperimentResults:
    def __init__(self, model_name: str):
        self.results = {phase: {} for phase in ['pretrain', 'joint', 'adapt']}
        self.model_name = model_name
        self.specialization_history = []
    
    def add_result(self, phase: str, task: str, epoch: int,
                  train_loss: float, eval_loss: float,
                  specialization_score: float = None):
        if task not in self.results[phase]:
            self.results[phase][task] = {
                'train_losses': [],
                'eval_losses': [],
                'specialization_scores': []
            }
        
        self.results[phase][task]['train_losses'].append(train_loss)
        self.results[phase][task]['eval_losses'].append(eval_loss)
        if specialization_score is not None:
            self.results[phase][task]['specialization_scores'].append(specialization_score)
    
    def add_specialization_matrix(self, matrix: np.ndarray, epoch: int, phase: str):
        """仅用于自适应网络"""
        if self.model_name == 'adaptive':
            self.specialization_history.append({
                'matrix': matrix.tolist(),
                'epoch': epoch,
                'phase': phase
            })
    
    def save_results(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练结果
        results_file = save_dir / f"{self.model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存专门化历史（仅限自适应网络）
        if self.model_name == 'adaptive' and self.specialization_history:
            spec_file = save_dir / f"{self.model_name}_specialization_history.json"
            with open(spec_file, 'w') as f:
                json.dump(self.specialization_history, f, indent=2)
        
        # 打印结果摘要
        self.print_summary()
    
    def print_summary(self):
        print(f"\n=== {self.model_name.upper()} Model Results Summary ===")
        for phase in self.results:
            print(f"\nPhase: {phase.upper()}")
            for task, metrics in self.results[phase].items():
                train_losses = metrics['train_losses']
                eval_losses = metrics['eval_losses']
                
                print(f"\n{task.upper()} Task:")
                print(f"Training Loss:")
                print(f"  Initial: {train_losses[0]:.4f}")
                print(f"  Final:   {train_losses[-1]:.4f}")
                print(f"  Best:    {min(train_losses):.4f}")
                print(f"  Mean:    {np.mean(train_losses):.4f}")
                print(f"Evaluation Loss:")
                print(f"  Initial: {eval_losses[0]:.4f}")
                print(f"  Final:   {eval_losses[-1]:.4f}")
                print(f"  Best:    {min(eval_losses):.4f}")
                print(f"  Mean:    {np.mean(eval_losses):.4f}")
                
                if 'specialization_scores' in metrics and metrics['specialization_scores']:
                    spec_scores = metrics['specialization_scores']
                    print(f"Specialization Score:")
                    print(f"  Initial: {spec_scores[0]:.4f}")
                    print(f"  Final:   {spec_scores[-1]:.4f}")
                    print(f"  Best:    {max(spec_scores):.4f}")
                    print(f"  Mean:    {np.mean(spec_scores):.4f}")

def train_epoch(model: Union[MultiTaskSequenceNetwork, MLPNetwork],
                dataloader: DataLoader,
                task: str,
                criterion: nn.Module,
                optimizer: optim.Optimizer) -> Dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    metrics_list = []
    
    # 遍历每个batch
    for batch_X, batch_y in dataloader:
        # 对batch中的每个样本单独处理
        for i in range(len(batch_X)):
            sequence = batch_X[i]
            target = batch_y[i]
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(sequence.unsqueeze(0), task)
            loss = criterion(output, target.unsqueeze(0))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算性能并更新适应性（仅对自适应网络）
            if isinstance(model, MultiTaskSequenceNetwork):
                performance = 1.0 - loss.item()  # 将损失转换为性能分数
                model.update_network(performance)  # 更新网络适应性
            
            # 记录指标
            metrics = calculate_metrics(output.detach(), target.unsqueeze(0))
            metrics['loss'] = loss.item()
            metrics_list.append(metrics)
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(dataloader.dataset))
    return {
        'avg_loss': avg_loss,
        'metrics': metrics_list
    }

def evaluate(model: Union[MultiTaskSequenceNetwork, MLPNetwork],
            dataloader: DataLoader,
            task: str,
            criterion: nn.Module) -> Dict:
    """评估模型"""
    model.eval()
    total_loss = 0
    metrics_list = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            # 对batch中的每个样本单独评估
            for i in range(len(batch_X)):
                sequence = batch_X[i]
                target = batch_y[i]
                
                # 预测
                output = model(sequence.unsqueeze(0), task)
                loss = criterion(output, target.unsqueeze(0))
                
                # 记录指标
                metrics = calculate_metrics(output, target.unsqueeze(0))
                metrics['loss'] = loss.item()
                metrics_list.append(metrics)
                total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    return {
        'avg_loss': avg_loss,
        'metrics': metrics_list
    }

def train_model(model: Union[MultiTaskSequenceNetwork, MLPNetwork],
                model_name: str,
                dataloaders: Dict,
                criterion: nn.Module,
                optimizer: optim.Optimizer) -> ExperimentResults:
    """训练模型（包括所有阶段）"""
    results = ExperimentResults(model_name)
    
    # 1. 预训练阶段
    print(f"\nPhase 1: Pretraining ({model_name})")
    for task in CONFIG['tasks']:
        print(f"\nPretraining on {task} task...")
        for epoch in tqdm(range(CONFIG['num_epochs']['pretrain'])):
            # 训练
            train_stats = train_epoch(
                model, dataloaders[task]['train'],
                task, criterion, optimizer
            )
            
            # 评估
            eval_stats = evaluate(
                model, dataloaders[task]['test'],
                task, criterion
            )
            
            # 分析节点专门化（仅限自适应网络）
            spec_score = None
            if isinstance(model, MultiTaskSequenceNetwork):
                spec_matrix = model.get_specialization_matrix()
                spec_matrix, spec_score = analyze_node_specialization(
                    spec_matrix, CONFIG['tasks']
                )
                results.add_specialization_matrix(spec_matrix, epoch, 'pretrain')
            
            # 记录结果
            results.add_result(
                'pretrain', task, epoch,
                train_stats['avg_loss'],
                eval_stats['avg_loss'],
                spec_score
            )
            
            # 保存日志
            save_training_log(train_stats, eval_stats, task, epoch, 'pretrain')
    
    # 2. 联合训练阶段
    print(f"\nPhase 2: Joint Training ({model_name})")
    for epoch in tqdm(range(CONFIG['num_epochs']['joint'])):
        for task in CONFIG['tasks']:
            # 训练
            train_stats = train_epoch(
                model, dataloaders[task]['train'],
                task, criterion, optimizer
            )
            
            # 评估
            eval_stats = evaluate(
                model, dataloaders[task]['test'],
                task, criterion
            )
            
            # 分析节点专门化（仅限自适应网络）
            spec_score = None
            if isinstance(model, MultiTaskSequenceNetwork):
                spec_matrix = model.get_specialization_matrix()
                spec_matrix, spec_score = analyze_node_specialization(
                    spec_matrix, CONFIG['tasks']
                )
                results.add_specialization_matrix(spec_matrix, epoch, 'joint')
                
                # 绘制专门化热力图
                if epoch % 5 == 0:
                    plot_specialization_heatmap(
                        spec_matrix,
                        CONFIG['tasks'],
                        CONFIG['results_dir'] / 'figures' / f'specialization_epoch_{epoch}.png'
                    )
            
            # 记录结果
            results.add_result(
                'joint', task, epoch,
                train_stats['avg_loss'],
                eval_stats['avg_loss'],
                spec_score
            )
            
            # 保存日志
            save_training_log(train_stats, eval_stats, task, epoch, 'joint')
    
    # 3. 适应性测试阶段
    print(f"\nPhase 3: Adaptation Testing ({model_name})")
    for task in CONFIG['tasks']:
        print(f"\nTesting adaptation on {task} task...")
        # 生成稍微不同分布的测试数据
        test_data = TensorDataset(
            dataloaders[task]['test'].dataset.tensors[0] * 1.2,
            dataloaders[task]['test'].dataset.tensors[1]
        )
        adapt_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'])
        
        for epoch in tqdm(range(CONFIG['num_epochs']['adapt'])):
            # 训练
            train_stats = train_epoch(
                model, adapt_loader,
                task, criterion, optimizer
            )
            
            # 评估
            eval_stats = evaluate(
                model, dataloaders[task]['test'],
                task, criterion
            )
            
            # 分析节点专门化（仅限自适应网络）
            spec_score = None
            if isinstance(model, MultiTaskSequenceNetwork):
                spec_matrix = model.get_specialization_matrix()
                spec_matrix, spec_score = analyze_node_specialization(
                    spec_matrix, CONFIG['tasks']
                )
                results.add_specialization_matrix(spec_matrix, epoch, 'adapt')
            
            # 记录结果
            results.add_result(
                'adapt', task, epoch,
                train_stats['avg_loss'],
                eval_stats['avg_loss'],
                spec_score
            )
            
            # 保存日志
            save_training_log(train_stats, eval_stats, task, epoch, 'adapt')
    
    return results

def main():
    # 加载数据
    data = load_data()
    
    # 创建数据加载器
    dataloaders = {}
    for task in CONFIG['tasks']:
        train_data = TensorDataset(
            data[f'{task}_X_train'],
            data[f'{task}_y_train']
        )
        test_data = TensorDataset(
            data[f'{task}_X_test'],
            data[f'{task}_y_test']
        )
        dataloaders[task] = {
            'train': DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True),
            'test': DataLoader(test_data, batch_size=CONFIG['batch_size'])
        }
    
    # 训练自适应网络
    print("\nTraining Adaptive Network...")
    adaptive_model = MultiTaskSequenceNetwork()
    adaptive_optimizer = optim.Adam(adaptive_model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    adaptive_results = train_model(
        adaptive_model, 'adaptive',
        dataloaders, criterion, adaptive_optimizer
    )
    
    # 训练基准MLP
    print("\nTraining Baseline MLP...")
    mlp_model = MLPNetwork(
        input_size=CONFIG['input_window'],
        hidden_size=CONFIG['hidden_size']
    )
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=CONFIG['learning_rate'])
    
    mlp_results = train_model(
        mlp_model, 'mlp',
        dataloaders, criterion, mlp_optimizer
    )
    
    # 保存实验结果
    adaptive_results.save_results(CONFIG['results_dir'])
    mlp_results.save_results(CONFIG['results_dir'])
    
    # 打印模型参数数量比较
    print("\nModel Parameters Comparison:")
    print(f"Adaptive Network: {adaptive_model.count_parameters():,} parameters")
    print(f"Baseline MLP: {mlp_model.count_parameters():,} parameters")

if __name__ == "__main__":
    main() 