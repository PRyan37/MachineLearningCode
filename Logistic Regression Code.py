import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
training_file="wildfires_training.csv"
test_file="wildfires_test.csv"
independent_cols=['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day','month','wind_speed']
dependent_col='fire'

df_training = pd.read_csv(training_file)
print(df_training.head())
print(df_training.shape)

X_training = df_training.loc[:,independent_cols]
print(X_training.head())
print(X_training.shape)

y_training = df_training.loc[:,dependent_col]
print(y_training.head())
print(y_training.shape)

df_test = pd.read_csv(test_file)
print(df_test.head())
print(df_test.shape)

X_test = df_test.loc[:,independent_cols]
print(X_test.head())
print(X_test.shape)

y_test = df_test.loc[:,dependent_col]
print(y_test.head())
print(y_test.shape)


model = LogisticRegression(max_iter=1000)
model.fit(X_training, y_training)

predictions_training = model.predict(X_training)
predictions_test = model.predict(X_test)

accuracy_training = metrics.accuracy_score(y_training, predictions_training)
accuracy_test = metrics.accuracy_score(y_test, predictions_test)
print("Accuracy on training data:",accuracy_training)
print("Accuracy on test data:",accuracy_test)

c_values=[0.01, 0.1, 1, 10, 100]
for c in c_values:
    model = LogisticRegression(C=c, max_iter=20000)
    model.fit(X_training, y_training)
    predictions_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("C:", c, " - Accuracy on test data:", accuracy_test)

penalty_list=['l1', 'l2', 'elasticnet', None]
for penalty in penalty_list:
    if penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=20000)
    elif penalty == 'l2' or penalty is None:
        model = LogisticRegression(penalty=penalty, solver='lbfgs', max_iter=20000)
    else:
        model = LogisticRegression(penalty=penalty,solver="liblinear", max_iter=20000 )
    model.fit(X_training, y_training)
    predictions_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("Penalty:", penalty, " - Accuracy on test data:", accuracy_test)


for c in c_values:
    for penalty in penalty_list:
        if penalty == 'elasticnet':
            model = LogisticRegression(C=c,penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=20000)
        elif penalty == 'l2' or penalty is None:
            model = LogisticRegression(C=c,penalty=penalty, solver='lbfgs', max_iter=20000)
        else:
            model = LogisticRegression(C=c,penalty=penalty,solver="liblinear", max_iter=20000 )
        model.fit(X_training, y_training)
        predictions_test = model.predict(X_test)
        accuracy_test = metrics.accuracy_score(y_test, predictions_test)
        print("Penalty:", penalty, "C: ",c," - Accuracy on test data:", accuracy_test)