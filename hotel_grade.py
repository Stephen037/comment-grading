from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# 1. 设置文件路径
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "hotel.csv"
OUTPUT_PATH = PROJECT_DIR / "hotel_grade_results.csv"
hotel_df = pd.read_csv(DATA_PATH)

# 2. 数据清洗
hotel_df = hotel_df.dropna(
    subset=["review", "label"]
).copy()

hotel_df["review"] = (
    hotel_df["review"]
    .astype(str)
    .str.strip()
)

hotel_df = hotel_df[
    hotel_df["review"] != ""
].copy()

hotel_df["label"] = hotel_df["label"].astype(int)

hotel_df = hotel_df.drop_duplicates(
    subset=["review"]
).reset_index(drop=True)

print("\nCleaned Data Shape")
print(hotel_df.shape)

print("\nCleaned Label Distribution")
print(hotel_df["label"].value_counts())


# 3. 设置特征和目标变量
X = hotel_df["review"]
y = hotel_df["label"]


# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=1,
)

print("\nX_train Shape")
print(X_train.shape)

print("\nX_test Shape")
print(X_test.shape)

print("\ny_train Distribution")
print(y_train.value_counts())

print("\ny_test Distribution")
print(y_test.value_counts())


# 5. 建立字符级 TF-IDF 特征提取器
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)


# 6. 使用训练集建立 TF-IDF 特征
X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nX_train TF-IDF Shape")
print(X_train_tfidf.shape)

print("\nX_test TF-IDF Shape")
print(X_test_tfidf.shape)


# 7. 建立并训练 Logistic Regression 模型
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=1,
)

model.fit(
    X_train_tfidf,
    y_train,
)


# 8. 预测测试集的 0/1 标签
predicted_labels = model.predict(
    X_test_tfidf
)


# 9. 预测测试集属于正面评论的概率
recommendation_scores = model.predict_proba(
    X_test_tfidf
)[:, 1]


# 10. 计算基础模型指标
accuracy = accuracy_score(
    y_test,
    predicted_labels,
)

precision = precision_score(
    y_test,
    predicted_labels,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predicted_labels,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predicted_labels,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    recommendation_scores,
)

print("\nModel Evaluation")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        predicted_labels,
    )
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predicted_labels,
        digits=4,
        zero_division=0,
    )
)


# 11. 整理测试集的连续评分结果
result_df = pd.DataFrame(
    {
        "review": X_test,
        "actual_label": y_test,
        "predicted_label": predicted_labels,
        "recommendation_score": recommendation_scores,
    }
).reset_index(drop=True)

# 根据 recommendation_score 从高到低排列，方便检查结果
result_df = result_df.sort_values(
    by="recommendation_score",
    ascending=False,
).reset_index(drop=True)

print("\nHighest Recommendation Scores")
print(
    result_df[
        [
            "review",
            "actual_label",
            "predicted_label",
            "recommendation_score",
        ]
    ].head(10)
)

print("\nLowest Recommendation Scores")
print(
    result_df[
        [
            "review",
            "actual_label",
            "predicted_label",
            "recommendation_score",
        ]
    ].tail(10)
)


# 12. 保存测试集评分结果
result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\nResult File")
print(OUTPUT_PATH)