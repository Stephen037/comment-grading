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


# 1.设置文件路径
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "ecommerce.csv"
OUTPUT_PATH = PROJECT_DIR / "ecommerce_grade_results.csv"


# 2.读取电商评论数据
ecommerce_df = pd.read_csv(DATA_PATH)

print("\nOriginal Data Shape")
print(ecommerce_df.shape)

print("\nColumns")
print(ecommerce_df.columns.tolist())

print("\nOriginal Label Distribution")
print(ecommerce_df["label"].value_counts())


# 3.数据清洗

ecommerce_df = ecommerce_df.dropna(
    subset=["review", "label"]
).copy()


ecommerce_df["review"] = (
    ecommerce_df["review"]
    .astype(str)
    .str.strip()
)


ecommerce_df = ecommerce_df[
    ecommerce_df["review"] != ""
].copy()


ecommerce_df["label"] = (
    ecommerce_df["label"]
    .astype(int)
)


ecommerce_df = ecommerce_df[
    ecommerce_df["label"].isin([0, 1])
].copy()

ecommerce_df = ecommerce_df.drop_duplicates(
    subset=["review"]
).reset_index(drop=True)


print("\nCleaned Data Shape")
print(ecommerce_df.shape)

print("\nCleaned Label Distribution")
print(ecommerce_df["label"].value_counts())


# 4.设置特征和目标变量

X = ecommerce_df["review"]
y = ecommerce_df["label"]


# 5.划分训练集和测试集

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


# 6.建立字符级 TF-IDF 特征提取器

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)


# 7.使用训练集建立 TF-IDF 特征

X_train_tfidf = vectorizer.fit_transform(
    X_train
)



X_test_tfidf = vectorizer.transform(
    X_test
)


print("\nX_train TF-IDF Shape")
print(X_train_tfidf.shape)

print("\nX_test TF-IDF Shape")
print(X_test_tfidf.shape)


# 8.建立并训练 Logistic Regression 模型

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=1,
)


model.fit(
    X_train_tfidf,
    y_train,
)


# 9.预测测试集的 0/1 标签

predicted_labels = model.predict(
    X_test_tfidf
)


# 10.预测测试集属于好评的概率

recommendation_scores = model.predict_proba(
    X_test_tfidf
)[:, 1]


# 11.计算基础模型指标

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

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print(
    "ROC AUC:",
    round(roc_auc, 4)
)


# 12.输出混淆矩阵

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predicted_labels,
    )
)


# 13.输出详细分类报告

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predicted_labels,
        digits=4,
        zero_division=0,
    )
)


# 14.整理测试集的连续评分结果

result_df = pd.DataFrame(
    {
        "review": X_test,
        "actual_label": y_test,
        "predicted_label": predicted_labels,
        "recommendation_score": recommendation_scores,
    }
).reset_index(drop=True)


result_df = result_df.sort_values(
    by="recommendation_score",
    ascending=False,
).reset_index(drop=True)


# 15.查看推荐倾向评分最高的 10 条评论

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


# 16.查看推荐倾向评分最低的 10 条评论

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


# 17.保存测试集的连续评分结果

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("\nResult File")
print(OUTPUT_PATH)