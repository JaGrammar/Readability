
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

Training_dataset = pd.read_csv("newtrain1.csv",encoding="UTF-8")

X_train = Training_dataset.iloc[:,2:]
y_train = Training_dataset.iloc[:,0]

Test_dataset = pd.read_csv("newtest1.csv",encoding="UTF-8")
X_test = Test_dataset.iloc[:,2:]
y_test = Test_dataset.iloc[:,0]


Rf_classifier = RandomForestClassifier(n_estimators=100)
Rf_classifier.fit(X_train, y_train)


y_pred = Rf_classifier.predict(X_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

