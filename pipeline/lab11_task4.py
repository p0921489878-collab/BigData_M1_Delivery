import os
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 规范化配置 1：严格修复 Matplotlib 中文与负号显示
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 严格将负号设置为标准的 ASCII 连字符

print("📂 1. 正在加载平衡数据集并规范化构建特征流水线...")
if not os.path.exists("batch_1000_features.csv"):
    raise FileNotFoundError("未找到平衡数据集 batch_1000_features.csv，请先确保数据落盘成功。")

df = pd.read_csv("batch_1000_features.csv")
y = df["label"]

# [任务 1 复现] 提取文本 TF-IDF
tfidf = TfidfVectorizer(analyzer='char', max_features=500)
X_text_sparse = tfidf.fit_transform(df['review'].fillna(''))
df_text = pd.DataFrame(X_text_sparse.toarray())  # 保持纯数字列名 (0, 1, 2...)

# [任务 2 复现] 稠密特征编码
llm_cols = ["cat", "extracted_feature"]
df[llm_cols] = df[llm_cols].fillna("Unknown")
encoder = OrdinalEncoder()
X_dense = encoder.fit_transform(df[llm_cols])
df_dense = pd.DataFrame(X_dense)  # 保持纯数字列名

# 水平拼接得到纯净无污染的特征矩阵 (1000, 502)
X_fused_df = pd.concat([df_text, df_dense], axis=1)
# 🚨 核心规范：强行将列名重命名为 LightGBM 绝对支持的纯文本格式 (f0, f1, f2...)
X_fused_df.columns = [f"f{i}" for i in range(X_fused_df.shape[1])]

# 🚀 统一提取供 SHAP 绘图显示的【漂亮特征名称列表】
# 包含：具体的文本字词和大模型特征名
pretty_feature_names = tfidf.get_feature_names_out().tolist() + [f"llm_{col}" for col in llm_cols]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_fused_df, y, test_size=0.2, random_state=42
)

print("🏋️ 2. 正在严格训练标准融合模型 C (LightGBM)...")
# 此时特征名全部是 f0, f1... LightGBM 绝对不会报错
clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
clf.fit(X_train, y_train)

print("🔮 3. 正在基于统一特征命名空间计算 SHAP 解释价值...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

# 🚨 在计算完后，将漂亮的中文和标点符号名称强行绑定给 SHAP 对象，用于图片渲染
shap_values.feature_names = pretty_feature_names

print("📊 4. 正在生成规范化 SHAP 瀑布图...")
plt.figure(figsize=(11, 7))

# 绘制测试集第一条样本（index 0）的瀑布图
shap.plots.waterfall(shap_values[0], max_display=12, show=False)

# 布局美化
plt.title("实验十一 任务四：融合模型单个样本预测的 SHAP 瀑布图解释", fontsize=14, pad=25, weight='bold')
plt.xlabel("SHAP Value (对预测概率的正负向拉扯贡献)", fontsize=11, labelpad=10)
plt.tight_layout()

# 高清保存图片
output_img = "shap_waterfall.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"🎉 规范化 SHAP 瀑布图已生成！高清图片成功保存至：{output_img}")
plt.show()