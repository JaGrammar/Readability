
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. 加载数据集
Training_dataset = pd.read_csv("Training_dataset.csv",encoding="UTF-8")

X_train = Training_dataset.iloc[:,2:]
y_train = Training_dataset.iloc[:,1]

Test_dataset = pd.read_csv("Test_dataset.csv",encoding="UTF-8")
X_test = Test_dataset.iloc[:,2:]
y_test = Test_dataset.iloc[:,1]

# 3. 训练决策树模型
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 4. 进行预测
y_pred = clf.predict(X_test)

# 5. 评估模型
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
