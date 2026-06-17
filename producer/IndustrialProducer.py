#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级数据生产者 - 支持背压控制和异常注入
"""

import json
import time
import random
import uuid
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


class IndustrialProducer:
    """工业级电商数据生产者"""

    def __init__(self, qps: float = 100.0, chaos_rate: float = 0.01, enable_backpressure: bool = True):
        """
        初始化生产者

        Args:
            qps: 目标生产速率（每秒事件数）
            chaos_rate: 异常数据注入率（0.01 = 1%）
            enable_backpressure: 是否启用背压机制
        """
        self.qps = qps
        self.chaos_rate = chaos_rate
        self.enable_backpressure = enable_backpressure

        # 背压状态
        self.backpressure_active = False
        self.current_delay = 1.0 / qps if qps > 0 else 0.01

        # 漏斗概率配置
        self.behavior_probs = {
            'view': 0.80,
            'cart': 0.15,
            'purchase': 0.05
        }

        # 数据池配置
        self.num_users = 10000  # 用户池大小
        self.num_items = 5000   # 商品池大小
        self.num_categories = 1000  # 类别池大小

        # 会话状态管理
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.session_history: Dict[str, List[Dict]] = defaultdict(list)  # session_id -> 行为历史

        # 商品热度分布 (Zipf分布参数)
        self.zipf_a = 1.5  # Zipf分布参数，控制长尾效应强度

        # 运行状态
        self.running = True
        self.total_events = 0

    def set_backpressure_state(self, active: bool):
        """设置背压状态并调整生产延迟"""
        self.backpressure_active = active

        if active:
            # 背压激活：增加延迟，降低生产速率
            self.current_delay = min(self.current_delay * 2, 2.0)  # 最大延迟2秒
        else:
            # 背压解除：减少延迟，恢复正常生产速率
            self.current_delay = max(self.current_delay / 2, 1.0 / self.qps)

    def _generate_user_id(self) -> str:
        """生成用户ID"""
        return f"user_{random.randint(1, self.num_users):06d}"

    def _generate_item_id(self) -> str:
        """使用Zipf分布生成商品ID，模拟长尾效应"""
        rank = int(np.random.zipf(self.zipf_a)) % self.num_items
        return f"item_{rank + 1:06d}"

    def _generate_category_id(self) -> str:
        """生成类别ID"""
        return f"category_{random.randint(1, self.num_categories):06d}"

    def _get_or_create_session(self, user_id: str) -> str:
        """获取或创建用户会话"""
        current_time = time.time()

        if user_id not in self.user_sessions:
            # 新用户，创建新会话
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            self.user_sessions[user_id] = session_id
        else:
            # 检查会话是否超时（30分钟）
            session_id = self.user_sessions[user_id]
            if session_id in self.session_history:
                last_event_time = self.session_history[session_id][-1]['timestamp']
                if current_time - last_event_time > 1800:  # 30分钟超时
                    session_id = f"session_{uuid.uuid4().hex[:8]}"
                    self.user_sessions[user_id] = session_id

        return self.user_sessions[user_id]

    def _can_perform_behavior(self, session_id: str, behavior_type: str) -> bool:
        """检查行为是否逻辑自洽"""
        if behavior_type == 'view':
            return True

        session_events = self.session_history[session_id]
        if not session_events:
            return False

        # 检查是否有view行为
        has_view = any(event['behavior_type'] == 'view' for event in session_events)

        if behavior_type == 'cart':
            return has_view
        elif behavior_type == 'purchase':
            # purchase需要view，最好有cart
            if not has_view:
                return False
            # 50%概率要求有cart行为
            has_cart = any(event['behavior_type'] == 'cart' for event in session_events)
            return has_cart or random.random() > 0.5

        return False

    def _generate_behavior_type(self, session_id: str) -> str:
        """生成符合业务逻辑的行为类型"""
        # 根据漏斗概率生成行为
        rand_val = random.random()

        if rand_val < self.behavior_probs['view']:
            target_behavior = 'view'
        elif rand_val < self.behavior_probs['view'] + self.behavior_probs['cart']:
            target_behavior = 'cart'
        else:
            target_behavior = 'purchase'

        # 检查行为是否逻辑自洽
        if self._can_perform_behavior(session_id, target_behavior):
            return target_behavior
        else:
            # 如果不合理，降级为view行为
            return 'view'

    def _inject_chaos(self, event: Dict) -> Optional[Dict]:
        """注入异常数据（混沌测试）"""
        if random.random() >= self.chaos_rate:
            return event

        # 随机选择一种异常类型
        chaos_type = random.choice(['missing_fields', 'invalid_types', 'corrupted_data', 'extreme_values'])

        corrupted_event = event.copy()

        if chaos_type == 'missing_fields':
            # 随机删除必需字段
            required_fields = ['user_id', 'item_id', 'category_id', 'timestamp']
            field_to_remove = random.choice(required_fields)
            if field_to_remove in corrupted_event:
                del corrupted_event[field_to_remove]

        elif chaos_type == 'invalid_types':
            # 生成无效的数据类型
            field_to_corrupt = random.choice(['user_id', 'item_id', 'category_id', 'timestamp'])
            corrupted_event[field_to_corrupt] = 'corrupted_string'

        elif chaos_type == 'corrupted_data':
            # 生成完全损坏的数据
            corrupted_event = {'corrupted': True, 'data': 'invalid_structure'}

        elif chaos_type == 'extreme_values':
            # 生成极端值
            corrupted_event['timestamp'] = '999999999999999999'

        return corrupted_event

    def generate_data(self) -> Optional[Dict]:
        """生成单个事件数据"""
        try:
            user_id = self._generate_user_id()
            item_id = self._generate_item_id()
            category_id = self._generate_category_id()
            session_id = self._get_or_create_session(user_id)
            behavior_type = self._generate_behavior_type(session_id)

            # 生成ISO8601格式时间戳
            event_time = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            timestamp = int(datetime.now(timezone.utc).timestamp())

            event = {
                'event_time': event_time,
                'user_id': user_id,
                'item_id': item_id,
                'category_id': category_id,
                'behavior_type': behavior_type,
                'session_id': session_id,
                'timestamp': timestamp
            }

            # 记录会话历史（用于逻辑验证）
            self.session_history[session_id].append({
                'behavior_type': behavior_type,
                'timestamp': time.time()
            })

            # 注入异常数据（混沌测试）
            if self.chaos_rate > 0:
                event = self._inject_chaos(event)
                if event is None:
                    return None

            self.total_events += 1
            return event

        except Exception as e:
            print(f"[Producer] 数据生成错误: {e}")
            return None

    def get_current_delay(self) -> float:
        """获取当前生产延迟"""
        return self.current_delay

    def get_stats(self) -> Dict:
        """获取生产者统计信息"""
        return {
            'total_events': self.total_events,
            'backpressure_active': self.backpressure_active,
            'current_delay': self.current_delay,
            'target_qps': self.qps,
            'chaos_rate': self.chaos_rate
        }