from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# 下载MNIST数据集
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

# 分割数据集
# X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)
# 分割测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=1/7, random_state=42)
# 分割验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1/6, random_state=42)

# 初始化分类器
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
svm_clf = SVC(kernel='linear', probability=True, random_state=42)
lda_clf = LinearDiscriminantAnalysis()

# 训练分类器
random_forest_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)
lda_clf.fit(X_train, y_train)

# 验证集上评估分类器
rf_val_accuracy = accuracy_score(y_val, random_forest_clf.predict(X_val))
svm_val_accuracy = accuracy_score(y_val, svm_clf.predict(X_val))
lda_val_accuracy = accuracy_score(y_val, lda_clf.predict(X_val))
# 打印准确率
print(f"Random Forest validation accuracy: {rf_val_accuracy}")
print(f"SVM validation accuracy: {svm_val_accuracy}")
print(f"LDA validation accuracy: {lda_val_accuracy}")

# 创建投票分类器
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("svm_clf", svm_clf),
    ("lda_clf", lda_clf)
]
voting_clf = VotingClassifier(named_estimators)
voting_clf.voting = "soft"  # 软投票集成
# 训练投票分类器
voting_clf.fit(X_train, y_train)

# 验证集上评估投票分类器
voting_val_accuracy = accuracy_score(y_val, voting_clf.predict(X_val))
# 测试集上评估投票分类器
voting_test_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
# 打印准确率
print(f"Voting classifier (soft) validation accuracy: {voting_val_accuracy}")
print(f"Voting classifier (soft) test accuracy: {voting_test_accuracy}")


voting_clf.voting = "hard"  # 硬投票集成
# 训练投票分类器
voting_clf.fit(X_train, y_train)

# 验证集上评估投票分类器
voting_val_accuracy = accuracy_score(y_val, voting_clf.predict(X_val))
# 测试集上评估投票分类器
voting_test_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
# 打印准确率
print(f"Voting classifier (hard) validation accuracy: {voting_val_accuracy}")
print(f"Voting classifier (hard) test accuracy: {voting_test_accuracy}")

