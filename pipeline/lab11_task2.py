import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import hstack, csr_matrix

print("📂 正在读取平衡数据集...")
df = pd.read_csv("batch_1000_features.csv")

print("⚡ [前置步骤] 自动运行任务 1 的文本特征提取...")
tfidf = TfidfVectorizer(analyzer='char', max_features=500)
X_text_sparse = tfidf.fit_transform(df['review'].fillna(''))

print("⚡ [核心步骤] 正在执行任务 2：大模型稠密特征编码与水平拼接...")
# 1. 定义需要融合的大模型稠密特征列
llm_cols = ["cat", "extracted_feature"]

# 2. 安全处理：将可能存在的缺失值填充为 "Unknown"
df[llm_cols] = df[llm_cols].fillna("Unknown")

# 3. 数字化编码：使用 OrdinalEncoder 将文本分类标签转换为整数字形（2维稠密矩阵）
encoder = OrdinalEncoder()
X_dense = encoder.fit_transform(df[llm_cols])

# 4. 异构矩阵水平拼接：必须先用 csr_matrix() 把稠密的 X_dense 转换为稀疏格式
# 然后用 hstack (Horizontal Stack) 在列的方向上拼接 (500维 + 2维 = 502维)
X_fused = hstack([X_text_sparse, csr_matrix(X_dense)])

print("\n================ 任务 2 验证成功 ================")
print(f"📊 稠密编码矩阵维度 (X_dense)     : {X_dense.shape}  (包含 cat 和 extracted_feature)")
print(f"✅ 融合后特征矩阵维度 (X_fused)     : {X_fused.shape}  (500维文本 + 2维大模型特征)")
print("=================================================")