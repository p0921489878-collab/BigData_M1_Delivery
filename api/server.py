import os
import re
import logging
import traceback
from collections import Counter
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="大数据分析看板 API - 实验十四 E2E 系统联调")

# ================= 1. 解决跨域问题 (任务3前置配置) =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. 鲁棒性：API Key 检查与优雅降级 =================
API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
LLM_ACTIVE = bool(API_KEY)

if not LLM_ACTIVE:
    logger.warning("API_KEY 未设置（SILICONFLOW_API_KEY / DASHSCOPE_API_KEY），LLM 功能已优雅降级")
    logger.warning("请复制 .env.example 为 .env 并填入 API Key")
else:
    logger.info("API Key 已检测，LLM 功能可用")


# ================= 3. 鲁棒性：DuckDB 只读连接 + 零崩溃数据加载 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "batch_1000_features.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "online_shopping_10_cats.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "analytics.db")

# DuckDB 只读连接 —— Worker 负责写入，FastAPI 只读
duck_conn = None
if os.path.exists(DB_PATH):
    try:
        duck_conn = duckdb.connect(database=DB_PATH, read_only=True)
        logger.info(f"DuckDB 只读连接已建立: {DB_PATH}")
    except Exception as e:
        logger.warning(f"DuckDB 连接失败（降级到 CSV 模式）: {e}")

# 零崩溃回退：CSV 加载
df = None
load_error = None

try:
    if os.path.exists(FEATURES_PATH):
        df = pd.read_csv(FEATURES_PATH)
        logger.info(f"成功加载 LLM 增强数据，共 {len(df)} 条")
    elif os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        logger.info(f"增强数据不存在，回退加载原始数据，共 {len(df)} 条")
    elif duck_conn is not None:
        df = duck_conn.execute("SELECT * FROM events LIMIT 10000").fetchdf()
        logger.info(f"从 DuckDB 加载数据，共 {len(df)} 条")
    else:
        load_error = "所有数据源均不可用"
        logger.warning(load_error)
        df = pd.DataFrame({"cat": [], "review": [], "label": [], "sentiment": []})
except Exception as e:
    load_error = str(e)
    logger.warning(f"数据加载异常，启用空数据集: {e}")
    df = pd.DataFrame({"cat": [], "review": [], "label": [], "sentiment": []})

# 核心防御：智能生成与检查标准 sentiment 字段（空DataFrame也安全）
if len(df) > 0 and "sentiment" not in df.columns:
    possible_label_cols = ["label", "label_llm", "pred", "prediction"]
    found_col = next((c for c in possible_label_cols if c in df.columns), None)
    
    if found_col:
        df["sentiment"] = df[found_col].map({1: "正面", 0: "负面"})
        print(f"【系统提示】已成功将字段 '{found_col}' 映射为标准的 'sentiment' 字段。")
    else:
        df["sentiment"] = "正面"
        print("【⚠️警告】数据中未找到任何情感标签字段，已启用默认填充。")

# 确保确定品类与评论的标准字段名称，方便后续过滤
CAT_COL = "cat" if "cat" in df.columns else "category"
REVIEW_COL = "review" if "review" in df.columns else "text"


# ================= 3. API 路由接口 =================

# 任务1: 健康检查
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "服务运行正常"}


@app.get("/api/system-status")
def system_status():
    """系统状态检查：LLM 可用性、数据源状态"""
    status = {
        "llm_active": LLM_ACTIVE,
        "llm_reason": "API_KEY_OK" if LLM_ACTIVE else "API_KEY_MISSING",
        "data_source": "duckdb" if duck_conn else ("csv" if df is not None and len(df) > 0 else "empty"),
        "data_count": len(df) if df is not None else 0,
        "database": os.path.basename(DB_PATH) if os.path.exists(DB_PATH) else None,
        "load_error": load_error,
    }
    return status


# 任务2 - 接口A: 品类分布统计 (静态总览)
@app.get("/api/category-distribution")
def get_category_distribution():
    """返回各品类的样本数量，供前端柱状图作静态总览使用"""
    stats = df[CAT_COL].value_counts()
    return {
        "categories": stats.index.tolist(),
        "counts": stats.values.tolist()
    }


# 🌟 任务3 - 选项C新增强调接口: 异步自适应高频分词统计
@app.get("/api/keywords")
def get_keywords(cat: str = None, sentiment: str = None, query: str = None, limit: int = 15):
    """
    提取当前全局筛选条件下的高频词。
    同样受品类、情感、搜索框三重影响，保持与评论明细一致的切片状态。
    """
    filtered = df

    # 1. 品类维度过滤
    if cat and cat != "null" and cat != "undefined" and cat != "":
        filtered = filtered[filtered[CAT_COL] == cat]
        
    # 2. 情感维度过滤
    if sentiment and sentiment != "null" and sentiment != "undefined" and sentiment != "":
        filtered = filtered[filtered["sentiment"] == sentiment]
        
    # 3. 正则检索与防御性编程
    if query and query.strip() != "":
        try:
            filtered = filtered[filtered[REVIEW_COL].fillna("").str.contains(query, case=False, regex=True)]
        except Exception:
            filtered = filtered[filtered[REVIEW_COL].fillna("").str.contains(query, case=False, regex=False)]
            
    # 将过滤后的文本聚合进行分词计数
    all_text = " ".join(filtered[REVIEW_COL].fillna("").astype(str).tolist())
    
    # 提取 2-4 字的中文短语
    words = re.findall(r'[\u4e00-\u9fa5]{2,4}', all_text)
    
    # 电商高频噪音频蔽词过滤
    stop_words = {
        "可以", "这个", "感觉", "觉得", "非常", "以后", "东西", "买的", "一个", "已经", 
        "结果", "还是", "时候", "没有", "比较", "商品", "不错", "质量", "很好", "一样",
        "买来", "一次", "哈哈", "收到", "日常", "交易", "真的", "看到"
    }
    if cat: 
        stop_words.add(cat) # 避免当前选择的品类名自身在词云里刷屏
        
    filtered_words = [w for w in words if w not in stop_words and len(w) >= 2]
    word_counts = Counter(filtered_words).most_common(limit)
    
    return [{"word": k, "count": v} for k, v in word_counts]


# 任务2 - 接口B: 情感分析概览 (支持品类动态联动过滤)
@app.get("/api/sentiment-overview")
def get_sentiment_overview(cat: str = None):
    """
    对接前端状态机：返回各品类的情感分布。
    如果传入了 cat 参数，则只对该品类进行计算，实现品类到情感图表的单向动态联动。
    """
    try:
        if cat == "null" or cat == "undefined" or cat == "":
            cat = None
            
        filtered_df = df if cat is None else df[df[CAT_COL] == cat]
        
        if filtered_df.empty:
            return {"status": "ok", "data": []}
            
        pivot = filtered_df.groupby([CAT_COL, "sentiment"]).size().unstack(fill_value=0)
        
        result = []
        for cat_name in pivot.index:
            pos_count = int(pivot.loc[cat_name, "正面"]) if "正面" in pivot.columns else 0
            neg_count = int(pivot.loc[cat_name, "负面"]) if "负面" in pivot.columns else (int(pivot.loc[cat_name, "負面"]) if "負面" in pivot.columns else 0)
            
            result.append({
                "category": str(cat_name),
                "正面": pos_count,
                "负面": neg_count
            })
        return {"status": "ok", "data": result}
        
    except Exception as e:
        print("【错误】/api/sentiment-overview 聚合计算失败：")
        print(traceback.format_exc())
        return {"status": "error", "message": f"后端透视计算失败: {str(e)}", "data": []}


# 任务2 - 接口C: 多维交叉筛选评论 (对接 SDD 联动核心 + 正则安全防御)
@app.get("/api/reviews")
def get_reviews(cat: str = None, sentiment: str = None, query: str = None, limit: int = 20):
    """按品类、情感、正则表达式综合筛选评论明细。"""
    filtered = df
    
    if cat and cat != "null" and cat != "undefined" and cat != "":
        filtered = filtered[filtered[CAT_COL] == cat]
        
    if sentiment and sentiment != "null" and sentiment != "undefined" and sentiment != "":
        filtered = filtered[filtered["sentiment"] == sentiment]
        
    if query and query.strip() != "":
        try:
            filtered = filtered[filtered[REVIEW_COL].fillna("").str.contains(query, case=False, regex=True)]
        except Exception as e:
            print(f"【⚠️检测到非法正则语法】: '{query}'。系统已自动降级为普通字符串匹配。原因: {e}")
            filtered = filtered[filtered[REVIEW_COL].fillna("").str.contains(query, case=False, regex=False)]
            
    records = filtered.head(limit).to_dict(orient="records")
    return {"total": len(filtered), "data": records}


# ================= 4. 挂载前端静态文件 (任务3前置配置) =================
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"【系统提示】前端静态目录挂载成功，可直接访问 http://127.0.0.1:8000/")
else:
    print(f"【⚠️警告】未检测到 '{frontend_dir}' 文件夹，请确保其存在。")