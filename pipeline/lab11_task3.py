import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

print("📂 1. 正在加载拥有健康平衡标签的数据集...")
df = pd.read_csv("batch_1000_features.csv")
y = df["label"]

print("⚡ 2. 正在复现前两个任务的特征流水线...")
# [任务 1 复现] 提取文本 TF-IDF (500维)
tfidf = TfidfVectorizer(analyzer='char', max_features=500)
X_text_sparse = tfidf.fit_transform(df['review'].fillna(''))

# [任务 2 复现] 大模型特征编码与水平拼接 (502维)
llm_cols = ["cat", "extracted_feature"]
df[llm_cols] = df[llm_cols].fillna("Unknown")
encoder = OrdinalEncoder()
X_dense = encoder.fit_transform(df[llm_cols])
X_fused = hstack([X_text_sparse, csr_matrix(X_dense)])

print("🏋️ 3. 正在启动标准消融实验对照训练...")
# 定义三种对照实验的特征源
experiments = {
    "Baseline A (纯 TF-IDF文本)": X_text_sparse,
    "Baseline B (纯大模型特征)": X_dense,
    "Fused C (异构特征融合)": X_fused
}

results = {}

# 开始循环训练与评估
for name, X_data in experiments.items():
    # 划分数据集：80% 训练，20% 测试
    # 固定 random_state=42 确保所有模型面对的样本完全一致，保证实验的公平性
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y, test_size=0.2, random_state=42
    )
    
    # 初始化 LightGBM 分类器
    # verbose=-1 用于关闭底层的冗余调试日志，保持终端整洁
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测并计算准确率
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = acc
    print(f" └─ 模型 [{name}] 评估完毕，准确率: {acc:.4f}")

# ==========================================
# 4. 打印最终的消融实验对照表
# ==========================================
print("\n==========================================")
print("       📊 最终消融实验对照表 (Summary)       ")
print("==========================================")
for name, score in results.items():
    print(f" {name:<22} | 测试集准确率: {score:.4f}")
print("==========================================")