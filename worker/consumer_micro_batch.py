# Micro-Batch优化的ML推理Consumer
import joblib
import pandas as pd
import queue
import time
import csv
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

class MicroBatchMLConsumer:
    def __init__(self, model_path: str, output_file: str, input_queue: queue.Queue,
                 batch_size: int = 50, batch_timeout: float = 0.5):
        """
        初始化Micro-Batch ML推理Consumer
        :param model_path: 模型文件路径
        :param output_file: 输出文件路径
        :param input_queue: 输入队列
        :param batch_size: 批量大小
        :param batch_timeout: 批量超时时间（秒）
        """
        self.model_path = model_path
        self.output_file = output_file
        self.queue = input_queue
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.model = None

        # Micro-Batch相关
        self.buffer = []
        self.last_flush = time.time()

        # 初始化输出文件
        self.init_output_file()

    def load_model(self):
        """加载ML模型"""
        try:
            print(f"正在加载模型：{self.model_path}")
            self.model = joblib.load(self.model_path)
            print("模型加载成功！")
            return True
        except Exception as e:
            print(f"模型加载失败：{e}")
            return False

    def init_output_file(self):
        """初始化输出CSV文件"""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp',
                    'predicted_label', 'buy_probability', 'error'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"输出文件已初始化：{self.output_file}")
        except Exception as e:
            print(f"输出文件初始化失败：{e}")

    def extract_features_batch(self, events: List[Dict[str, str]]) -> Optional[pd.DataFrame]:
        """
        批量提取特征（优化版本）
        :param events: 事件列表
        :return: 批量特征DataFrame或None（失败时）
        """
        try:
            features_list = []

            for event in events:
                try:
                    # 类型转换：将字符串转为数值
                    user_id = int(event['user_id'])
                    item_id = int(event['item_id'])
                    category_id = int(event['category_id'])
                    timestamp = int(event['timestamp'])

                    # 特征衍生：从timestamp提取hour和dayofweek
                    ts = pd.Timestamp(timestamp, unit='s')
                    hour = ts.hour
                    dayofweek = ts.dayofweek

                    features_list.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'category_id': category_id,
                        'timestamp': timestamp,
                        'hour': hour,
                        'dayofweek': dayofweek
                    })

                except Exception as e:
                    print(f"事件特征提取失败：{e}, 事件：{event}")
                    # 添加错误标记
                    features_list.append({
                        'user_id': -1, 'item_id': -1, 'category_id': -1,
                        'timestamp': -1, 'hour': -1, 'dayofweek': -1
                    })

            # 构造批量特征DataFrame
            features_df = pd.DataFrame(features_list)
            return features_df

        except Exception as e:
            print(f"批量特征提取失败：{e}")
            return None

    def run_batch_inference(self, features: pd.DataFrame) -> tuple:
        """
        运行批量模型推理
        :param features: 批量特征DataFrame
        :return: (预测标签数组, 购买概率数组)
        """
        try:
            # 批量模型推理
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)[:, 1]  # 正类概率

            return predictions, probabilities

        except Exception as e:
            print(f"批量模型推理失败：{e}")
            # 返回错误标记
            error_pred = [-1] * len(features)
            error_prob = [-1.0] * len(features)
            return error_pred, error_prob

    def save_batch_results(self, events: List[Dict[str, Any]]):
        """保存批量打标结果到CSV文件"""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp',
                    'predicted_label', 'buy_probability', 'error'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                for event in events:
                    writer.writerow(event)

        except Exception as e:
            print(f"批量结果保存失败：{e}")

    def process_batch(self):
        """处理当前buffer中的批量数据"""
        if not self.buffer:
            return

        try:
            # 批量特征提取
            batch_features = self.extract_features_batch(self.buffer)

            if batch_features is None:
                # 特征提取失败，标记所有事件为错误
                for event in self.buffer:
                    event['predicted_label'] = -1
                    event['buy_probability'] = -1.0
                    event['error'] = '批量特征提取失败'
                    self.error_count += len(self.buffer)
            else:
                # 批量模型推理
                predictions, probabilities = self.run_batch_inference(batch_features)

                # 结果回流：将预测结果追加到原始事件中
                for i, event in enumerate(self.buffer):
                    if i < len(predictions):
                        event['predicted_label'] = int(predictions[i])
                        event['buy_probability'] = float(probabilities[i])
                        event['error'] = ''

                        if predictions[i] == -1:
                            self.error_count += 1
                            event['error'] = '模型推理失败'
                    else:
                        event['predicted_label'] = -1
                        event['buy_probability'] = -1.0
                        event['error'] = '结果索引错误'
                        self.error_count += 1

            # 终端展示：打印批量打标统计
            batch_size = len(self.buffer)
            self.processed_count += batch_size

            if self.processed_count % 250 == 0:  # 每250条显示一次
                print(f"批量打标完成：批次大小={batch_size}, 累计处理={self.processed_count}条, "
                      f"错误={self.error_count}条")

            # 结果持久化
            self.save_batch_results(self.buffer)

            # 清空buffer
            self.buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            print(f"批量处理异常：{e}")
            # 异常处理：标记所有事件为错误
            for event in self.buffer:
                event['predicted_label'] = -1
                event['buy_probability'] = -1.0
                event['error'] = str(e)
                self.error_count += 1
                self.save_batch_results([event])

            self.processed_count += len(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()

    def start_consuming(self):
        """开始消费数据（Micro-Batch版本）"""
        print(f"Micro-Batch Consumer启动：批量大小={self.batch_size}, 超时={self.batch_timeout}秒")

        # 加载模型
        if not self.load_model():
            print("模型加载失败，Consumer退出")
            return

        self.running = True

        # 主消费循环
        while self.running:
            try:
                # 从队列获取事件（带超时）
                event = self.queue.get(timeout=0.1)
                self.buffer.append(event)

            except queue.Empty:
                # 队列为空，继续检查触发条件
                pass
            except Exception as e:
                print(f"数据获取异常：{e}")
                break

            # 双触发条件：攒满B条 或 超时
            current_time = time.time()
            if (len(self.buffer) >= self.batch_size or
                (self.buffer and current_time - self.last_flush > self.batch_timeout)):

                trigger_reason = "数量触发" if len(self.buffer) >= self.batch_size else "超时触发"
                print(f"触发推理：{trigger_reason}, 批次大小={len(self.buffer)}")
                self.process_batch()

        # 退出前处理剩余数据
        if self.buffer:
            print(f"退出前处理剩余数据：{len(self.buffer)}条")
            self.process_batch()

        print(f"Micro-Batch Consumer完成：共处理 {self.processed_count} 条数据")
        print(f"处理失败：{self.error_count} 条")

    def start(self):
        """启动Consumer线程"""
        self.thread = threading.Thread(target=self.start_consuming, daemon=True)
        self.thread.start()

    def stop(self):
        """停止Consumer"""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """获取Consumer统计信息"""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count - self.error_count) / max(self.processed_count, 1) * 100,
            'running': self.running,
            'buffer_size': len(self.buffer),
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout
        }

# 测试代码
if __name__ == "__main__":
    # 创建队列
    data_queue = queue.Queue(maxsize=1000)

    # 创建Micro-Batch Consumer实例
    consumer = MicroBatchMLConsumer(
        model_path="model.pkl",
        output_file="micro_batch_scored_events.csv",
        input_queue=data_queue,
        batch_size=50,
        batch_timeout=0.5
    )

    # 启动Consumer
    consumer.start()

    # 运行30秒后停止
    time.sleep(30)
    consumer.stop()

    # 显示统计信息
    stats = consumer.get_stats()
    print(f"\nMicro-Batch Consumer统计：")
    print(f"  处理数量：{stats['processed_count']} 条")
    print(f"  错误数量：{stats['error_count']} 条")
    print(f"  成功率：{stats['success_rate']:.2f}%")
    print(f"  批量大小：{stats['batch_size']}")
    print(f"  超时设置：{stats['batch_timeout']}秒")
    print(f"  运行状态：{'运行中' if stats['running'] else '已停止'}")