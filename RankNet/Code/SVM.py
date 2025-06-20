
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics

Training_dataset = pd.read_csv("train1.csv",encoding="UTF-8")

X_train = Training_dataset.iloc[:,2:]
y_train = Training_dataset.iloc[:,0]

Test_dataset = pd.read_csv("test1.csv",encoding="UTF-8")
X_test = Test_dataset.iloc[:,2:]
y_test = Test_dataset.iloc[:,0]


svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

