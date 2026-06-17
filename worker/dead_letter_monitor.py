#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
死信队列监控模块 - 独立封装版本
用于监控Pipeline运行时的异常数据处理情况
"""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any


class DeadLetterMonitor:
    """死信队列监控器"""

    def __init__(self, dead_letter_file: str = 'dead_letter.jsonl'):
        """
        初始化死信监控器

        Args:
            dead_letter_file: 死信记录文件路径
        """
        self.dead_letter_file = dead_letter_file
        self.file_position = 0
        self.running = True
        self.error_stats = {}
        self.total_count = 0

    def _get_file_size(self) -> int:
        """获取文件大小"""
        try:
            return os.path.getsize(self.dead_letter_file)
        except OSError:
            return 0

    def _read_new_lines(self) -> List[str]:
        """读取文件新增的行"""
        new_lines = []

        try:
            with open(self.dead_letter_file, 'r', encoding='utf-8') as f:
                # 移动到上次读取的位置
                f.seek(self.file_position)

                # 读取新增内容
                new_content = f.read()
                if new_content:
                    lines = new_content.strip().split('\n')
                    new_lines = [line for line in lines if line.strip()]

                # 更新文件位置
                self.file_position = f.tell()

        except FileNotFoundError:
            print(f"等待死信文件 {self.dead_letter_file} 创建...")
            time.sleep(2)
        except Exception as e:
            print(f"读取死信文件错误: {e}")

        return new_lines

    def _process_dead_letter(self, line: str) -> Dict[str, Any]:
        """处理单个死信记录"""
        try:
            record = json.loads(line)
            error_type = record.get('error_type', 'UNKNOWN')

            # 更新错误统计
            if error_type not in self.error_stats:
                self.error_stats[error_type] = 0
            self.error_stats[error_type] += 1

            return record

        except json.JSONDecodeError:
            print(f"死信JSON解析错误: {line[:50]}...")
            return {}
        except Exception as e:
            print(f"处理死信记录错误: {e}")
            return {}

    def _print_error_summary(self):
        """打印错误统计摘要"""
        if not self.error_stats:
            return

        total_errors = sum(self.error_stats.values())
        print(f"\n=== 死信统计摘要 (总计: {total_errors} 条) ===")

        # 按错误类型排序
        sorted_errors = sorted(self.error_stats.items(), key=lambda x: x[1], reverse=True)

        for error_type, count in sorted_errors:
            percentage = (count / total_errors) * 100
            print(f"  {error_type}: {count} 条 ({percentage:.1f}%)")

        print("=" * 50)

    def start_monitoring(self):
        """开始监控死信队列"""
        print(f"开始监控死信队列: {self.dead_letter_file}")
        print("按 Ctrl+C 停止监控\n")

        last_summary_time = time.time()

        try:
            while self.running:
                # 检查文件是否有新内容
                current_size = self._get_file_size()

                if current_size > self.file_position:
                    # 读取新增行
                    new_lines = self._read_new_lines()

                    # 处理每个新死信记录
                    for line in new_lines:
                        record = self._process_dead_letter(line)
                        if record:
                            self.total_count += 1

                            # 打印最新死信
                            timestamp = record.get('timestamp', '未知时间')
                            error_type = record.get('error_type', 'UNKNOWN')
                            error_msg = record.get('error_message', '未知错误')

                            print(f"\n[死信 #{self.total_count}] {timestamp}")
                            print(f"  类型: {error_type}")
                            print(f"  错误: {error_msg}")

                            # 显示原始数据的部分内容
                            original_data = record.get('original_data', {})
                            if original_data:
                                print(f"  数据: {str(original_data)[:100]}...")

                # 每30秒打印一次统计摘要
                current_time = time.time()
                if current_time - last_summary_time >= 30:
                    self._print_error_summary()
                    last_summary_time = current_time

                # 短暂休眠，避免CPU占用过高
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n接收到停止信号，正在停止...")
        except Exception as e:
            print(f"监控循环错误: {e}")
        finally:
            # 最终统计
            print(f"\n=== 最终死信统计 ===")
            print(f"总计死信数量: {self.total_count} 条")
            self._print_error_summary()
            print("死信监控已停止")


def main():
    """主函数"""
    monitor = DeadLetterMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()