import polars as pl
import time
import os
import gc

# ================= 配置區域 =================
# 請確保此路徑正確指向你的 1 億行數據檔案
RAW_DATA_PATH = r"D:\大数据分析\实验\bigdata_lab3\任务5_M1最终数据固化\m1_final_clean.parquet"
# ===========================================

def test_original_approach(file_path):
    """
    測試原始方法：模擬 Eager 模式（多次觸發 Collect / 冗餘 I/O）
    """
    print("\n" + "-"*20)
    print("--- [Step 1/2] 正在測試：原始方法 (多個獨立 Collect) ---")
    start_time = time.perf_counter()

    # 模擬冗餘操作 1：先讀取所有數據並計算行數
    # 注意：這裡模擬的是不帶投影的掃描
    lf = pl.scan_parquet(file_path)
    total_count = lf.select(pl.len()).collect().item()
    print(f"  [Log] 步驟 1-1 (讀取元數據) 完成，原始數據: {total_count:,} 行")

    # 模擬冗餘操作 2：重新掃描並執行物理去重 (最耗內存)
    # 這裡不加 select，模擬原始未優化狀態
    dedup_df = pl.scan_parquet(file_path).unique().collect()
    dedup_count = len(dedup_df)
    print(f"  [Log] 步驟 1-2 (物理去重) 完成，去重後: {dedup_count:,} 行")

    elapsed_time = time.perf_counter() - start_time
    print(f"  ✅ 原始方法執行完畢，耗時: {elapsed_time:.2f} 秒")
    
    # 必須徹底手動釋放內存，防止 Step 2 崩潰
    del dedup_df
    del lf
    gc.collect()
    
    return elapsed_time, total_count

def test_optimized_approach(file_path):
    """
    測試優化方法：全鏈路 Lazy 優化 + 投影下推 + 流式引擎
    """
    print("\n" + "-"*20)
    print("--- [Step 2/2] 正在測試：優化方法 (Lazy API + 單次 Collect) ---")
    
    # 為了公平，不計入獲取總數的時間
    start_time = time.perf_counter()

    # 1. 構建「全鏈路」Lazy 查詢圖
    # 關鍵黑科技：Projection Pushdown (投影下推) - 只選 4 個欄位讀取
    optimized_query = (
        pl.scan_parquet(file_path)
        .select(["user_id", "item_id", "behavior_type", "timestamp"]) 
        .unique()
        .select(pl.len())
    )

    print("  [Log] 啟動 Polars Streaming Engine 執行全鏈路優化計算...")
    print("  [提示] 1 億行數據計算中，請觀察任務管理員 CPU 變化，預計耗時 1-2 分鐘...")
    
    # 2. 執行單次物理計算，並開啟流式引擎
    result = optimized_query.collect(engine="streaming")
    final_count = result.item()

    elapsed_time = time.perf_counter() - start_time
    print(f"  [Log] 最終處理量: {final_count:,}")
    print(f"  ✅ 優化方法執行完畢，耗時: {elapsed_time:.2f} 秒")

    return elapsed_time

def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 錯誤：找不到文件 {RAW_DATA_PATH}")
        return

    print("=" * 65)
    print("🚀 M1 Milestone: 1億條真實數據性能對比 (工程化穩定版)")
    print("技術栈: Parquet | Projection Pushdown | Streaming Engine")
    print("=" * 65)

    # 執行 Step 1
    t_old, count = test_original_approach(RAW_DATA_PATH)

    # --- 關鍵防崩潰保護：中間停頓與深度清理 ---
    print("\n[系統] 偵測到大數據負載，正在強制釋放 RAM 並冷卻 10 秒...")
    gc.collect()
    time.sleep(10) 
    # ---------------------------------------

    # 執行 Step 2
    try:
        t_new = test_optimized_approach(RAW_DATA_PATH)

        # 輸出最終對比表格
        print("\n" + "📊 性能對比報告 (1億行數據量級)".center(60, " "))
        print("-" * 65)
        print(f"| 測試指標           | 原始方法 (Eager) | 優化方法 (Lazy) |")
        print("-" * 65)
        print(f"| 執行耗時 (秒)      | {t_old:>14.2f} | {t_new:>14.2f} |")
        print(f"| 處理速度 (行/秒)   | {count/t_old:>14,.0f} | {count/t_new:>14,.0f} |")
        print("-" * 65)
        
        speedup = t_old / t_new
        improvement = ((t_old - t_new) / t_old) * 100
        
        print(f"\n💡 結論: 優化後性能提升了 {improvement:.1f}%")
        print(f"🚀 總體加速倍率: {speedup:.2f} 倍")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ Step 2 運行失敗: {e}")
        print("建議原因：記憶體仍不足以支撐 1 億行去重 Hash Table，請嘗試關閉 Chrome 後重試。")

if __name__ == "__main__":
    main()