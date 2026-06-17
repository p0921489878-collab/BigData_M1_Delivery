#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级数据流Pipeline统一入口 - 基于前序实验资产整合
支持配置化启动、完整监控、异常处理和ML推理
"""

import argparse
import threading
import time
import json
import os
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import queue
import joblib
import pandas as pd
import numpy as np


class PipelineOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Pipeline编排器

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.running = False
        self.stats = {
            'start_time': time.time(),
            'processed_count': 0,
            'produced_count': 0,
            'dead_letter_count': 0,
            'ml_inferences': 0,
            'backpressure_events': 0
        }

        # 数据队列
        self.data_queue = queue.Queue(maxsize=config['queue_limit'])

        # 死信队列
        self.dead_letter_queue = queue.Queue()

        # 线程控制
        self.producer_thread = None
        self.consumer_thread = None
        self.monitor_thread = None

        # ML模型
        self.ml_model = None
        if not config.get('no_ml', False):
            self._load_ml_model()

        # 死信监控已移至独立模块

    def _load_ml_model(self):
        """加载ML模型"""
        try:
            model_path = 'model.pkl'
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                print(f"[PIPELINE] ML模型加载成功: {model_path}")
            else:
                print(f"[PIPELINE] 警告: 未找到ML模型文件 {model_path}")
                self.ml_model = None
        except Exception as e:
            print(f"[PIPELINE] ML模型加载失败: {e}")
            self.ml_model = None

    def _producer_worker(self):
        """生产者工作线程"""
        print(f"[PRODUCER] 启动 - QPS: {self.config['qps']}, 异常率: {self.config['chaos_rate']*100:.1f}%")

        # 导入Producer类
        from IndustrialProducer import IndustrialProducer

        producer = IndustrialProducer(
            qps=self.config['qps'],
            chaos_rate=self.config['chaos_rate'],
            enable_backpressure=not self.config.get('no_backpressure', False)
        )

        # 背压控制方法
        def custom_backpressure_handler(active: bool):
            if active:
                self.stats['backpressure_events'] += 1
            producer.set_backpressure_state(active)

        # 背压监控（如果启用）
        def backpressure_monitor():
            if self.config.get('no_backpressure', False):
                return

            while self.running:
                load_pct = self.data_queue.qsize() / self.config['queue_limit']
                if load_pct >= 0.85:  # 高水位线
                    custom_backpressure_handler(True)
                elif load_pct <= 0.30:  # 低水位线
                    custom_backpressure_handler(False)
                time.sleep(0.1)

        # 启动背压监控
        if not self.config.get('no_backpressure', False):
            monitor_t = threading.Thread(target=backpressure_monitor, daemon=True)
            monitor_t.start()

        # 生产数据
        try:
            while self.running:
                try:
                    data = producer.generate_data()
                    if data:
                        # 写入文件
                        with open('streaming_logs.jsonl', 'a', encoding='utf-8') as f:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            f.flush()

                        # 放入队列
                        try:
                            self.data_queue.put(data, timeout=1.0)
                            self.stats['produced_count'] += 1
                        except queue.Full:
                            # 队列满，记录到死信
                            self._add_dead_letter(data, "QUEUE_FULL", "数据队列已满")
                            self.stats['produced_count'] += 1

                    # 控制速率
                    time.sleep(producer.get_current_delay())

                except Exception as e:
                    print(f"[PRODUCER] 数据生成错误: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"[PRODUCER] 生产者错误: {e}")

    def _consumer_worker(self):
        """消费者工作线程"""
        print(f"[CONSUMER] 启动 - 批处理大小: {self.config.get('batch_size', 1)}")

        batch_size = self.config.get('batch_size', 1)
        batch_timeout = 0.5  # 批处理超时
        batch_buffer = []
        last_batch_time = time.time()

        def process_batch(data_batch):
            """处理一批数据"""
            if not data_batch:
                return

            # 批处理特征提取
            features_batch = []
            valid_data = []

            for data in data_batch:
                try:
                    features = self._extract_features(data)
                    if features is not None:
                        features_batch.append(features)
                        valid_data.append(data)
                    else:
                        self._add_dead_letter(data, "FEATURE_EXTRACTION_ERROR", "特征提取失败")
                except Exception as e:
                    self._add_dead_letter(data, "FEATURE_EXTRACTION_ERROR", str(e))

            # ML推理
            if self.ml_model and features_batch:
                try:
                    features_df = pd.DataFrame(features_batch)
                    predictions = self.ml_model.predict(features_df)
                    probabilities = self.ml_model.predict_proba(features_df)[:, 1]

                    # 保存结果
                    for i, (data, pred, prob) in enumerate(zip(valid_data, predictions, probabilities)):
                        result = {
                            **data,
                            'predicted_label': int(pred),
                            'buy_probability': float(prob),
                            'processed_time': datetime.now().isoformat()
                        }

                        with open('scored_events.jsonl', 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                        self.stats['ml_inferences'] += 1

                except Exception as e:
                    for data in valid_data:
                        self._add_dead_letter(data, "ML_INFERENCE_ERROR", str(e))

            self.stats['processed_count'] += len(data_batch)

            # 添加处理延迟以触发背压机制
            time.sleep(0.01)  # 10ms延迟，减慢消费速度

        try:
            while self.running:
                try:
                    # 获取数据
                    data = self.data_queue.get(timeout=0.1)
                    batch_buffer.append(data)

                    # 检查是否触发批处理
                    current_time = time.time()
                    should_process = (
                        len(batch_buffer) >= batch_size or
                        (batch_buffer and (current_time - last_batch_time) >= batch_timeout)
                    )

                    if should_process:
                        process_batch(batch_buffer)
                        batch_buffer = []
                        last_batch_time = current_time

                except queue.Empty:
                    # 超时处理剩余批次
                    if batch_buffer and (time.time() - last_batch_time) >= batch_timeout:
                        process_batch(batch_buffer)
                        batch_buffer = []
                        last_batch_time = time.time()
                    continue

                except Exception as e:
                    print(f"[CONSUMER] 数据处理错误: {e}")

            # 处理剩余批次
            if batch_buffer:
                process_batch(batch_buffer)

        except Exception as e:
            print(f"[CONSUMER] 消费者错误: {e}")

    def _extract_features(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """特征提取（容错版本）"""
        try:
            # 必需字段检查
            required_fields = ['user_id', 'item_id', 'category_id', 'timestamp']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"缺失字段: {field}")

            # 安全类型转换
            try:
                user_id = int(str(data['user_id']).replace('user_', ''))
            except:
                user_id = 0

            try:
                item_id = int(str(data['item_id']).replace('item_', ''))
            except:
                item_id = 0

            try:
                category_id = int(str(data['category_id']).replace('category_', ''))
            except:
                category_id = 0

            try:
                timestamp = int(data['timestamp'])
            except:
                timestamp = int(time.time())

            # 衍生特征
            hour_of_day = datetime.fromtimestamp(timestamp).hour

            return {
                'user_id': user_id,
                'item_id': item_id,
                'category_id': category_id,
                'timestamp': timestamp,
                'hour_of_day': hour_of_day
            }

        except Exception as e:
            raise ValueError(f"特征提取失败: {e}")

    def _add_dead_letter(self, data: Dict[str, Any], error_type: str, error_msg: str):
        """添加死信记录"""
        dead_letter = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_msg,
            'original_data': data
        }

        try:
            self.dead_letter_queue.put(dead_letter, block=False)
        except queue.Full:
            pass  # 死信队列也满了，跳过

        # 写入死信文件
        with open('dead_letter.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(dead_letter, ensure_ascii=False) + '\n')

        self.stats['dead_letter_count'] += 1

    def _monitor_worker(self):
        """监控工作线程"""
        last_stats = self.stats.copy()
        last_time = time.time()

        while self.running:
            time.sleep(2.0)  # 2秒刷新一次

            current_time = time.time()
            elapsed = current_time - last_time

            # 计算速率
            processed_rate = (self.stats['processed_count'] - last_stats['processed_count']) / elapsed
            dead_letter_rate = (self.stats['dead_letter_count'] - last_stats['dead_letter_count']) / elapsed

            # 队列状态
            queue_size = self.data_queue.qsize()
            queue_load = queue_size / self.config['queue_limit']

            # 背压状态
            backpressure_status = "[BACKPRESSURE]" if queue_load > 0.85 else "[NORMAL]"
            load_status = "[HIGH_LOAD]" if queue_load > 0.7 else "[MED_LOAD]" if queue_load > 0.3 else "[LOW_LOAD]"

            # 计算实际生产速率
            elapsed_total = current_time - self.stats['start_time']
            actual_qps = self.stats['produced_count'] / elapsed_total if elapsed_total > 0 else 0

            # 显示状态
            print(f"\r[{elapsed_total:6.1f}s] {backpressure_status} {load_status} "
                  f"队列:{queue_size}/{self.config['queue_limit']} | "
                  f"生产:{actual_qps:.0f} | "
                  f"处理:{self.stats['processed_count']} | "
                  f"死信:{self.stats['dead_letter_count']}", end="")

            # 只在死信数量增加时显示提示信息
            if self.stats['dead_letter_count'] > last_stats['dead_letter_count']:
                new_dead_letters = self.stats['dead_letter_count'] - last_stats['dead_letter_count']
                print(f"\n  [INFO] 新增 {new_dead_letters} 条死信，详细监控请运行: python view_dead_letters.py", end="")

            # 更新统计
            last_stats = self.stats.copy()
            last_time = current_time

    def _get_recent_dead_letters(self) -> list:
        """获取最近的死信记录"""
        recent = []
        temp_list = []

        # 从队列中取出所有死信
        while not self.dead_letter_queue.empty():
            try:
                dl = self.dead_letter_queue.get_nowait()
                temp_list.append(dl)
            except queue.Empty:
                break

        # 保留最近的5条
        recent = temp_list[-5:]

        # 将死信放回队列
        for dl in temp_list:
            try:
                self.dead_letter_queue.put_nowait(dl)
            except queue.Full:
                break

        return recent

    # 死信监控已移至独立模块 dead_letter_monitor.py

    def signal_handler(self, signum, frame):
        """信号处理"""
        print(f"\n\n[PIPELINE] 收到停止信号，正在优雅停止...")
        self.running = False

    def run(self):
        """运行Pipeline"""
        print("[PIPELINE] 工业级数据流Pipeline启动")
        print(f"配置参数:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print()

        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # 初始化文件
        open('streaming_logs.jsonl', 'w').close()
        open('scored_events.jsonl', 'w').close()
        open('dead_letter.jsonl', 'w').close()

        self.running = True

        # 启动工作线程
        self.producer_thread = threading.Thread(target=self._producer_worker, daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer_worker, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)

        self.producer_thread.start()
        self.consumer_thread.start()
        self.monitor_thread.start()

        print("[PIPELINE] 所有组件已启动，开始处理数据流...\n")

        # 主循环
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        """优雅关闭"""
        print("\n[PIPELINE] 正在关闭...")
        self.running = False

        # 等待线程结束
        if self.producer_thread:
            self.producer_thread.join(timeout=2.0)
        if self.consumer_thread:
            self.consumer_thread.join(timeout=2.0)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        # 打印最终统计
        elapsed = time.time() - self.stats['start_time']
        print(f"\n[STATS] 最终统计:")
        print(f"  运行时间: {elapsed:.1f}秒")
        print(f"  处理数据: {self.stats['processed_count']}条")
        print(f"  死信数量: {self.stats['dead_letter_count']}条")
        print(f"  ML推理: {self.stats['ml_inferences']}次")
        print(f"  背压事件: {self.stats['backpressure_events']}次")

        if self.stats['processed_count'] > 0:
            dead_letter_rate = (self.stats['dead_letter_count'] / self.stats['processed_count']) * 100
            print(f"  死信率: {dead_letter_rate:.1f}%")

        print(f"\n输出文件:")
        print(f"  - streaming_logs.jsonl: 原始数据流")
        print(f"  - scored_events.jsonl: ML推理结果")
        print(f"  - dead_letter.jsonl: 死信记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='工业级数据流Pipeline统一入口')
    parser.add_argument('--qps', type=int, default=1000, help='生产者QPS')
    parser.add_argument('--queue_limit', type=int, default=100, help='队列容量限制')
    parser.add_argument('--chaos_rate', type=float, default=0.01, help='异常数据注入率 (0.01表示1%)')
    parser.add_argument('--batch_size', type=int, default=10, help='批处理大小')
    parser.add_argument('--no_ml', action='store_true', help='禁用ML推理')
    parser.add_argument('--no_backpressure', action='store_true', help='禁用背压机制')
    parser.add_argument('--duration', type=int, default=None, help='运行时长（秒）')

    args = parser.parse_args()

    config = {
        'qps': args.qps,
        'queue_limit': args.queue_limit,
        'chaos_rate': args.chaos_rate,
        'batch_size': args.batch_size,
        'no_ml': args.no_ml,
        'no_backpressure': args.no_backpressure
    }

    # 检查ML模型文件
    if not args.no_ml and not os.path.exists('model.pkl'):
        print("警告: 未找到model.pkl文件，ML推理将被禁用")
        config['no_ml'] = True

    orchestrator = PipelineOrchestrator(config)

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        orchestrator._shutdown()


if __name__ == '__main__':
    main()