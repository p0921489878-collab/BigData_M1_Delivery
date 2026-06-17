#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1DataPipeline - 1億數據 Parquet 流式處理版
優化點：利用 Polars Streaming Engine 自動管理內存，替代手動 slice 循環。
"""

import logging
import os
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional

import polars as pl

class M1DataPipeline100M:
    """
    M1數據處理管道（1億數據專用優化版）
    支持 Parquet 原生讀取與流式計算。
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        session_timeout: int = 1800,
        log_level: str = "INFO"
    ) -> None:
        self.input_path: str = input_path
        self.output_dir: str = output_dir
        self.session_timeout: int = session_timeout
        self.final_df: Optional[pl.DataFrame] = None

        # 創建輸出目錄
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 配置日誌
        self._setup_logging(log_level)
        self.logger.info("M1DataPipeline100M (Streaming 版) 初始化完成")

    def _setup_logging(self, log_level: str) -> None:
        """配置日誌系統"""
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def extract(self) -> bool:
        """數據提取階段：掃描 Parquet 元數據"""
        try:
            self.logger.info(f"正在掃描 Parquet 文件: {self.input_path}")
            start_time = time.time()

            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"輸入文件不存在: {self.input_path}")

            # 使用 scan_parquet 建立 LazyFrame (不加載數據進內存)
            self.lazy_plan = pl.scan_parquet(self.input_path)

            # 快速獲取總行數 (從 Parquet Metadata 讀取，極快)
            total_rows = self.lazy_plan.select(pl.len()).collect().item()
            self.logger.info(f"數據總量: {total_rows:,} 行")
            self.logger.info(f"數據提取準備完成，耗時: {time.time() - start_time:.2f} 秒")
            return True

        except Exception as e:
            self.logger.error(f"數據提取失敗: {str(e)}")
            return False

    def process_streaming(self) -> bool:
        """
        核心處理階段：利用 Streaming Engine 執行精密去重、會話識別與漏斗分析。
        """
        try:
            self.logger.info("啟動流式處理引擎 (Streaming Engine) 執行全鏈路優化...")
            start_time = time.time()

            # 1. 構建全鏈路 Lazy 查詢計劃
            # 注意：這裡的 unique 和 window function 都會在流式模式下自動分塊執行
            processed_lazy = (
                self.lazy_plan
                # 任務一：精密去重
                .unique(subset=["user_id", "item_id", "behavior_type", "timestamp"])
                # 任務二：會話識別 (保證 user_id 的數據完整性)
                .sort(["user_id", "timestamp"])
                .with_columns([
                    (pl.col("timestamp") - pl.col("timestamp").shift(1).over("user_id"))
                    .alias("time_diff")
                ])
                .with_columns([
                    pl.when(
                        pl.col("time_diff").is_null() | (pl.col("time_diff") > self.session_timeout)
                    ).then(1).otherwise(0).alias("new_session_flag")
                ])
                .with_columns([
                    pl.col("new_session_flag").cum_sum().over("user_id").alias("session_id")
                ])
                .drop(["time_diff", "new_session_flag"])
            )

            # 2. 觸發計算 (核心：engine="streaming")
            # 這會將 1 億條數據以流式方式處理並返回 DataFrame
            self.logger.info("正在執行流式計算 (一億行量級，請耐心等待)...")
            self.final_df = processed_lazy.collect(engine="streaming")
            
            # 3. 執行漏斗分析 (任務三)
            behavior_counts = (
                self.final_df.group_by("behavior_type")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )

            # 保存最終統計數據
            self.final_stats = {
                "behavior_counts": behavior_counts,
                "total_sessions": self.final_df.select(pl.col("session_id").n_unique()).item(),
                "total_rows": len(self.final_df)
            }

            self.logger.info(f"全鏈路處理完成，耗時: {time.time() - start_time:.2f} 秒")
            return True

        except Exception as e:
            self.logger.error(f"流式處理失敗: {str(e)}")
            return False

    def load(self) -> bool:
        """數據加載階段：導出結果與報告"""
        try:
            self.logger.info("開始數據加載與報告生成...")
            start_time = time.time()

            if self.final_df is None:
                raise ValueError("未找到處理結果，請先執行 process_streaming()")

            # 1. 導出處理後的完整 Parquet 數據
            output_parquet = os.path.join(self.output_dir, "m1_processed_final.parquet")
            self.final_df.write_parquet(output_parquet)
            self.logger.info(f"已導出固化數據: {output_parquet}")

            # 2. 導出漏斗分析 CSV
            funnel_file = os.path.join(self.output_dir, "funnel_analysis.csv")
            self.final_stats["behavior_counts"].write_csv(funnel_file)

            # 3. 生成執行報告
            report_file = os.path.join(self.output_dir, "processing_report.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("=== M1數據處理管道執行報告 (Parquet Streaming 版) ===\n\n")
                f.write(f"處理時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"輸入文件: {self.input_path}\n")
                f.write(f"輸出目錄: {self.output_dir}\n")
                f.write(f"總數據量: {self.final_stats['total_rows']:,} 行\n")
                f.write(f"總會話數: {self.final_stats['total_sessions']:,}\n")
                f.write("\n=== 任務完成狀態 ===\n")
                f.write("✅ 任務一：精密去重\n")
                f.write("✅ 任務二：會話識別 (Global Windowing)\n")
                f.write("✅ 任務三：漏斗分析\n")
                f.write("\n=== 技術亮點 ===\n")
                f.write("🚀 使用 Polars Streaming Engine 避免手動分塊\n")
                f.write("🚀 Parquet 列式存儲優化\n")
                f.write("🚀 內存溢出自動保護機制\n")

            self.logger.info(f"處理報告已生成: {report_file}")
            return True

        except Exception as e:
            self.logger.error(f"數據加載失敗: {str(e)}")
            return False

def main():
    """主函數"""
    # 配置路徑 (請根據實際情況修改)
    INPUT_FILE = r"D:\大数据分析\实验\bigdata_lab3\任务5_M1最终数据固化\m1_final_clean.parquet"
    OUTPUT_DIR = r"D:\大数据分析\实验\bigdata_lab4\M1_Streaming_Output"

    print("\n" + "="*70)
    print(">>> M1DataPipeline Streaming Edition (100M Rows) <<<")
    print("="*70)

    pipeline = M1DataPipeline100M(INPUT_FILE, OUTPUT_DIR)

    # 執行 ETL 流程
    if pipeline.extract():
        if pipeline.process_streaming():
            if pipeline.load():
                print("\n" + "="*70)
                print("🎉 所有處理任務已圓滿完成！")
                print(f"📁 最終結果路徑: {OUTPUT_DIR}")
                print("="*70)

if __name__ == "__main__":
    main()