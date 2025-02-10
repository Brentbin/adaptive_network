"""通信管理模块

实现层级间的消息传递和协调机制。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import torch
import numpy as np

@dataclass
class Message:
    """通信消息"""
    source: str              # 消息来源层级
    target: str              # 目标层级
    message_type: str        # 消息类型
    content: Dict[str, Any]  # 消息内容
    timestamp: float         # 时间戳
    priority: float = 0.5    # 消息优先级

class CommunicationManager:
    """通信管理器
    
    负责:
    1. 消息的传递和路由
    2. 消息优先级管理
    3. 通信效率监控
    4. 消息历史记录
    """
    
    def __init__(self, buffer_size: int = 50):
        # 消息缓冲区
        self.message_buffers = {
            'bottom_up': [],    # 自下而上的信息流
            'top_down': [],     # 自上而下的控制流
            'lateral': []       # 同层交互
        }
        
        # 消息历史
        self.message_history = []
        
        # 通信统计
        self.communication_stats = {
            'message_counts': {},  # 消息计数
            'latencies': [],       # 传递延迟
            'success_rates': []    # 成功率
        }
        
        # 配置
        self.buffer_size = buffer_size
        self.priority_threshold = 0.8  # 高优先级阈值
        
    def send_message(self, message: Message) -> bool:
        """发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            是否发送成功
        """
        try:
            # 验证消息
            if not self._validate_message(message):
                return False
                
            # 确定消息类型
            if message.source < message.target:  # 数字小的层级向数字大的层级发送
                buffer_type = 'bottom_up'
            elif message.source > message.target:
                buffer_type = 'top_down'
            else:
                buffer_type = 'lateral'
                
            # 处理高优先级消息
            if message.priority >= self.priority_threshold:
                self._handle_priority_message(message, buffer_type)
            else:
                # 添加到相应的缓冲区
                self.message_buffers[buffer_type].append(message)
                
                # 维护缓冲区大小
                if len(self.message_buffers[buffer_type]) > self.buffer_size:
                    self.message_buffers[buffer_type].pop(0)
            
            # 更新统计信息
            self._update_stats(message, True)
            
            # 记录历史
            self.message_history.append(message)
            if len(self.message_history) > self.buffer_size:
                self.message_history.pop(0)
                
            return True
            
        except Exception as e:
            print(f"Error sending message: {str(e)}")
            self._update_stats(message, False)
            return False
            
    def receive_messages(self, 
                        target: str, 
                        message_type: Optional[str] = None) -> List[Message]:
        """接收消息
        
        Args:
            target: 目标层级
            message_type: 可选的消息类型过滤
            
        Returns:
            消息列表
        """
        messages = []
        
        # 检查所有相关缓冲区
        for buffer_type, buffer in self.message_buffers.items():
            # 根据消息类型过滤
            if message_type and buffer_type != message_type:
                continue
                
            # 获取目标消息
            target_messages = [
                msg for msg in buffer 
                if msg.target == target
            ]
            
            messages.extend(target_messages)
            
            # 从缓冲区移除已处理的消息
            for msg in target_messages:
                buffer.remove(msg)
                
        return sorted(messages, key=lambda x: x.priority, reverse=True)
    
    def broadcast_message(self, message: Message) -> bool:
        """广播消息到所有层级
        
        Args:
            message: 要广播的消息
            
        Returns:
            是否广播成功
        """
        try:
            # 复制消息到所有缓冲区
            for buffer_type in self.message_buffers:
                broadcast_msg = Message(
                    source=message.source,
                    target='all',
                    message_type=message.message_type,
                    content=message.content,
                    timestamp=time.time(),
                    priority=message.priority
                )
                self.message_buffers[buffer_type].append(broadcast_msg)
                
                # 维护缓冲区大小
                if len(self.message_buffers[buffer_type]) > self.buffer_size:
                    self.message_buffers[buffer_type].pop(0)
            
            # 更新统计信息
            self._update_stats(message, True)
            
            return True
            
        except Exception as e:
            print(f"Error broadcasting message: {str(e)}")
            self._update_stats(message, False)
            return False
            
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'message_counts': dict(self.communication_stats['message_counts']),
            'average_latency': np.mean(self.communication_stats['latencies']) if self.communication_stats['latencies'] else 0,
            'success_rate': np.mean(self.communication_stats['success_rates']) if self.communication_stats['success_rates'] else 1.0,
            'buffer_usage': {
                buffer_type: len(buffer) / self.buffer_size
                for buffer_type, buffer in self.message_buffers.items()
            }
        }
        
        return stats
    
    def clear_buffers(self) -> None:
        """清空所有消息缓冲区"""
        for buffer_type in self.message_buffers:
            self.message_buffers[buffer_type].clear()
            
    def _validate_message(self, message: Message) -> bool:
        """验证消息的有效性
        
        Args:
            message: 要验证的消息
            
        Returns:
            消息是否有效
        """
        try:
            assert message.source is not None, "Message source cannot be None"
            assert message.target is not None, "Message target cannot be None"
            assert message.message_type in [
                'bottom_up',           # 自下而上的信息流
                'top_down',           # 自上而下的控制流
                'lateral',            # 同层交互
                'preparation',        # 准备消息
                'processing_complete', # 处理完成消息
                'control',            # 控制消息
                'feedback',           # 反馈消息
                'alert'              # 警报消息
            ], "Invalid message type"
            assert isinstance(message.content, dict), "Message content must be a dictionary"
            assert 0 <= message.priority <= 1, "Message priority must be between 0 and 1"
            
            return True
            
        except AssertionError as e:
            print(f"Message validation failed: {str(e)}")
            return False
            
    def _handle_priority_message(self, 
                               message: Message,
                               buffer_type: str) -> None:
        """处理高优先级消息
        
        Args:
            message: 高优先级消息
            buffer_type: 缓冲区类型
        """
        # 如果缓冲区已满,移除最低优先级的消息
        if len(self.message_buffers[buffer_type]) >= self.buffer_size:
            # 找到最低优先级的消息
            min_priority_idx = min(
                range(len(self.message_buffers[buffer_type])),
                key=lambda i: self.message_buffers[buffer_type][i].priority
            )
            # 移除它
            self.message_buffers[buffer_type].pop(min_priority_idx)
            
        # 添加新消息
        self.message_buffers[buffer_type].append(message)
        
    def _update_stats(self, message: Message, success: bool) -> None:
        """更新通信统计信息
        
        Args:
            message: 处理的消息
            success: 是否成功
        """
        # 更新消息计数
        msg_type = f"{message.source}->{message.target}"
        self.communication_stats['message_counts'][msg_type] = \
            self.communication_stats['message_counts'].get(msg_type, 0) + 1
            
        # 更新延迟
        latency = time.time() - message.timestamp
        self.communication_stats['latencies'].append(latency)
        
        # 更新成功率
        self.communication_stats['success_rates'].append(float(success))
        
        # 维护统计历史长度
        if len(self.communication_stats['latencies']) > self.buffer_size:
            self.communication_stats['latencies'] = self.communication_stats['latencies'][-self.buffer_size:]
        if len(self.communication_stats['success_rates']) > self.buffer_size:
            self.communication_stats['success_rates'] = self.communication_stats['success_rates'][-self.buffer_size:] 