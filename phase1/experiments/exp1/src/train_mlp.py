import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.insert(0, str(root_path))
from experiments.exp1.src.mlp import MLPNetwork
from experiments.exp1.src.utils import load_data, save_training_log

class ExperimentResults:
    def __init__(self, model_name="mlp"):
        self.results = {}
        self.model_name = model_name
    
    def add_result(self, pattern_type, epoch, train_loss, test_loss):
        if pattern_type not in self.results:
            self.results[pattern_type] = {
                'train_losses': [],
                'test_losses': []
            }
        self.results[pattern_type]['train_losses'].append(train_loss)
        self.results[pattern_type]['test_losses'].append(test_loss)
    
    def get_summary_dict(self):
        summary = {}
        for pattern_type, metrics in self.results.items():
            train_losses = metrics['train_losses']
            test_losses = metrics['test_losses']
            
            summary[pattern_type] = {
                'train_loss': {
                    'initial': float(train_losses[0]),
                    'final': float(train_losses[-1]),
                    'best': float(min(train_losses)),
                    'mean': float(np.mean(train_losses))
                },
                'test_loss': {
                    'initial': float(test_losses[0]),
                    'final': float(test_losses[-1]),
                    'best': float(min(test_losses)),
                    'mean': float(np.mean(test_losses))
                },
                'history': {
                    'train_losses': [float(x) for x in train_losses],
                    'test_losses': [float(x) for x in test_losses]
                }
            }
        return summary
    
    def save_results(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        results_file = save_dir / f"{self.model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.get_summary_dict(), f, indent=2)
        
        # 打印结果到控制台
        self.print_summary()
    
    def print_summary(self):
        print("\n=== Experiment Results Summary ===")
        for pattern_type, metrics in self.results.items():
            train_losses = metrics['train_losses']
            test_losses = metrics['test_losses']
            
            print(f"\n{pattern_type.upper()} Pattern:")
            print(f"Training Loss:")
            print(f"  Initial: {train_losses[0]:.4f}")
            print(f"  Final:   {train_losses[-1]:.4f}")
            print(f"  Best:    {min(train_losses):.4f}")
            print(f"  Mean:    {np.mean(train_losses):.4f}")
            print(f"Test Loss:")
            print(f"  Initial: {test_losses[0]:.4f}")
            print(f"  Final:   {test_losses[-1]:.4f}")
            print(f"  Best:    {min(test_losses):.4f}")
            print(f"  Mean:    {np.mean(test_losses):.4f}")

def train_epoch(model, X_train, y_train, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for sequence, target in tqdm(zip(X_train, y_train), total=len(X_train), desc="Training", leave=False):
        target = target.view(-1)
        optimizer.zero_grad()
        prediction = model(sequence).view(-1)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(X_train)

def evaluate(model, X_test, y_test, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    
    with torch.no_grad():
        for sequence, target in zip(X_test, y_test):
            target = target.view(-1)
            prediction = model(sequence).view(-1)
            loss = criterion(prediction, target)
            total_loss += loss.item()
            predictions.append(prediction.cpu().numpy())
    
    return {
        'avg_loss': total_loss / len(X_test),
        'predictions': predictions
    }

def main():
    # 加载数据
    data = load_data()
    results = ExperimentResults(model_name="mlp")
    
    # 打印数据维度
    print("\nData dimensions:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 模型参数
    input_size = data['input_window']
    hidden_size = 64
    output_size = 1
    num_epochs = 10
    learning_rate = 0.001
    
    # 训练循环
    for pattern_type in ['arithmetic', 'geometric', 'multiplier']:
        print(f"\nTraining on {pattern_type} pattern...")
        
        X_train = data[f'{pattern_type}_X_train']
        y_train = data[f'{pattern_type}_y_train']
        X_test = data[f'{pattern_type}_X_test']
        y_test = data[f'{pattern_type}_y_test']
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}\n")
        
        # 创建新的模型实例
        model = MLPNetwork(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print("Training details:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"First sequence shape: {X_train[0].shape}")
        print(f"First target shape: {y_train[0].shape}")
        
        for epoch in tqdm(range(num_epochs), desc=f"Epochs"):
            # 训练
            train_loss = train_epoch(model, X_train, y_train, criterion, optimizer)
            
            # 评估
            eval_stats = evaluate(model, X_test, y_test, criterion)
            test_loss = eval_stats['avg_loss']
            
            # 记录结果
            results.add_result(pattern_type, epoch, train_loss, test_loss)
            
            # 保存日志
            train_stats = {
                'avg_loss': train_loss,
                'metrics': [{'loss': train_loss}]
            }
            save_training_log(train_stats, eval_stats, pattern_type, epoch)
    
    # 保存实验结果
    results_dir = Path(root_path) / "experiments/exp1/results"
    results.save_results(results_dir)

if __name__ == "__main__":
    main() 