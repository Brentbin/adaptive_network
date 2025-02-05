import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # 确保输入是二维的 [batch_size, input_size]
        if len(x.shape) == 1:
            x = x.view(1, -1)
        return self.network(x)

    def predict(self, sequence):
        """对输入序列进行预测"""
        with torch.no_grad():
            return self.forward(sequence) 