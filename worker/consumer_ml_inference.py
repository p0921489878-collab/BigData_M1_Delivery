# 集成ML推理的Consumer - 端到端打标流水线
import joblib
import pandas as pd
import queue
import time
import csv
from typing import Dict, Any, Optional
from datetime import datetime
import threading

class MLConsumer:
    def __init__(self, model_path: str, output_file: str, input_queue: queue.Queue):
        """
        初始化ML推理Consumer
        :param model_path: 模型文件路径
        :param output_file: 输出文件路径
        :param input_queue: 输入队列
        """
        self.model_path = model_path
        self.output_file = output_file
        self.queue = input_queue
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.model = None

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

    def extract_features(self, event: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        从事件字典中提取特征
        :param event: 事件字典
        :return: 特征DataFrame或None（失败时）
        """
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

            # 构造特征DataFrame
            features = pd.DataFrame([{
                'user_id': user_id,
                'item_id': item_id,
                'category_id': category_id,
                'timestamp': timestamp,
                'hour': hour,
                'dayofweek': dayofweek
            }])

            return features

        except Exception as e:
            print(f"特征提取失败：{e}, 事件：{event}")
            return None

    def run_inference(self, features: pd.DataFrame) -> tuple:
        """
        运行模型推理
        :param features: 特征DataFrame
        :return: (预测标签, 购买概率)
        """
        try:
            # 模型推理
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0][1]

            return int(prediction), float(probability)

        except Exception as e:
            print(f"模型推理失败：{e}")
            return -1, -1.0

    def save_result(self, event: Dict[str, Any]):
        """保存打标结果到CSV文件"""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp',
                    'predicted_label', 'buy_probability', 'error'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(event)
        except Exception as e:
            print(f"结果保存失败：{e}")

    def process_event(self, event: Dict[str, str]):
        """处理单个事件"""
        try:
            # 特征提取
            features = self.extract_features(event)

            if features is None:
                # 特征提取失败
                event['predicted_label'] = -1
                event['buy_probability'] = -1.0
                event['error'] = '特征提取失败'
                self.error_count += 1
            else:
                # 模型推理
                predicted_label, buy_probability = self.run_inference(features)

                # 结果回流：将预测结果追加到原始事件中
                event['predicted_label'] = predicted_label
                event['buy_probability'] = buy_probability
                event['error'] = ''

                if predicted_label == -1:
                    self.error_count += 1
                    event['error'] = '模型推理失败'

            # 终端展示：打印打标后的事件
            if self.processed_count % 50 == 0:  # 每50条显示一次
                print(f"打标结果：用户{event['user_id']}, 预测标签:{event['predicted_label']}, "
                      f"购买概率:{event['buy_probability']:.4f}")

            # 结果持久化
            self.save_result(event)
            self.processed_count += 1

        except Exception as e:
            # 异常容错：单条坏数据不会击垮消费者进程
            print(f"事件处理异常：{e}")
            event['predicted_label'] = -1
            event['buy_probability'] = -1.0
            event['error'] = str(e)
            self.error_count += 1
            self.save_result(event)
            self.processed_count += 1

    def start_consuming(self):
        """开始消费数据"""
        print("Consumer启动：开始处理数据流...")

        # 加载模型
        if not self.load_model():
            print("模型加载失败，Consumer退出")
            return

        self.running = True

        # 主消费循环
        while self.running:
            try:
                # 从队列获取事件
                event = self.queue.get(timeout=1.0)

                # 处理事件
                self.process_event(event)

                # 模拟处理延迟
                time.sleep(0.01)

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"Consumer异常：{e}")
                break

        print(f"Consumer完成：共处理 {self.processed_count} 条数据")
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
            'running': self.running
        }

# 测试代码
if __name__ == "__main__":
    # 创建队列
    data_queue = queue.Queue(maxsize=1000)

    # 创建Consumer实例
    consumer = MLConsumer(
        model_path="model.pkl",
        output_file="scored_events.csv",
        input_queue=data_queue
    )

    # 启动Consumer
    consumer.start()

    # 运行30秒后停止
    time.sleep(30)
    consumer.stop()

    # 显示统计信息
    stats = consumer.get_stats()
    print(f"\nConsumer统计：")
    print(f"  处理数量：{stats['processed_count']} 条")
    print(f"  错误数量：{stats['error_count']} 条")
    print(f"  成功率：{stats['success_rate']:.2f}%")
    print(f"  运行状态：{'运行中' if stats['running'] else '已停止'}")