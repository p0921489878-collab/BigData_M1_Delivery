#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1DataPipeline 100M 數據主入口文件 (Parquet Streaming 優化版)
一鍵運行 1 億數據的精密去重、會話識別、漏斗分析全流程
"""

import sys
import os
import time
# 確保檔名與你保存 Pipeline 類別的文件名一致
from M1DataPipeline_100M import M1DataPipeline100M 

def main():
    """主函數：一鍵運行 M1DataPipeline 100M 數據處理全流程"""

    # 配置參數 (請確認路徑與你磁碟實際位置一致)
    INPUT_FILE = r"D:\大数据分析\实验\bigdata_lab3\任务5_M1最终数据固化\m1_final_clean.parquet"
    OUTPUT_DIR = r"D:\大数据分析\实验\bigdata_lab4\M1_100M_Streaming_Output"

    print("\n" + "="*70)
    print(">>> 啟動 M1 大數據工程管道 (1億數據 Parquet Streaming 版) <<<")
    print("="*70)
    print("目標: 成功處理 100,150,807 行數據的完整 ETL 流程")
    print("技術: Polars Streaming Engine | 全鏈路 Lazy 優化")
    print("="*70 + "\n")

    # 創建 100M 數據管道實例
    pipeline = M1DataPipeline100M(
        input_path=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        session_timeout=1800,  # 30 分鐘會話超時
        log_level="INFO"
    )

    # 執行完整的 ETL 流程
    try:
        # [1/3] 數據提取階段
        print("[1/3] 數據提取階段...")
        if not pipeline.extract():
            print("❌ 數據提取失敗，請檢查輸入文件路徑")
            sys.exit(1)
        print("✅ 數據提取完成")

        # [2/3] 流式處理階段
        print("\n[2/3] 核心處理階段...")
        print("執行: 精密去重 + 會話識別 + 漏斗分析")
        print("技術: Polars 流式計算引擎 (自動管理內存)")
        
        # 這裡調用我們新改好的流式處理函數
        if not pipeline.process_streaming():
            print("❌ 處理失敗，請檢查日誌或磁碟空間")
            sys.exit(1)
        print("✅ 數據計算完成")

        # [3/3] 數據加載階段
        print("\n[3/3] 數據加載階段...")
        print("導出: 最終 Parquet 數據 + 漏斗分析 CSV + 執行報告")
        if not pipeline.load():
            print("❌ 數據加載失敗，請檢查輸出目錄權限")
            sys.exit(1)
        print("✅ 數據加載與導出完成")

        print("\n" + "="*70)
        print("🎉 [成功] M1DataPipeline 100M 數據處理全流程執行完畢！")
        print(f"📁 結果已保存至: {OUTPUT_DIR}")
        print("\n[實現的實驗任務]:")
        print("  ✅ 任務一：精密去重 (user_id, item_id, behavior, ts)")
        print("  ✅ 任務二：會話識別 (Global Session Partitioning)")
        print("  ✅ 任務三：漏斗分析 (Behavior Conversion)")
        print("\n[應用的黑科技]:")
        print("  ✅ Parquet 列式存儲優化")
        print("  ✅ Streaming Engine (流式引擎)")
        print("  ✅ 內存自動溢出保護 (Auto Spill-to-disk)")
        print("="*70)

    except KeyboardInterrupt:
        print("\n⚠️  程序被用戶手動中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [嚴重錯誤] 程序運行中斷: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()