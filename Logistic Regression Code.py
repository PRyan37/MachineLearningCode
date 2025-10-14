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

print("\n ------------TESTING C------------- \n")
c_values=[0.01, 0.1, 1, 10, 100]
for c in c_values:
    model = LogisticRegression(C=c, max_iter=20000)
    model.fit(X_training, y_training)
    predictions_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("C:", c, " - Accuracy on test data:", accuracy_test)

print("\n ------------TESTING PENALTY------------- \n")
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

penalty_list_no_none=['l1', 'l2', 'elasticnet']
l1_accuracy=[]
l2_accuracy=[]
elasticnet_accuracy=[]

print("\n ------------TESTING BOTH C AND PENALTY------------- \n")
for c in c_values:
    for penalty in penalty_list_no_none:
        if penalty == 'elasticnet':
            model = LogisticRegression(C=c,penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=20000)
        elif penalty == 'l2':
            model = LogisticRegression(C=c,penalty=penalty, solver='lbfgs', max_iter=20000)
        elif penalty =='l1':
            model = LogisticRegression(C=c,penalty=penalty,solver="liblinear", max_iter=20000 )


        model.fit(X_training, y_training)
        predictions_test = model.predict(X_test)
        accuracy_test = metrics.accuracy_score(y_test, predictions_test)
        if penalty == 'l1':
            l1_accuracy.append(accuracy_test)
        elif penalty == 'l2':
            l2_accuracy.append(accuracy_test)
        elif penalty == 'elasticnet':
            elasticnet_accuracy.append(accuracy_test)
        print("Penalty:", penalty, "| C: ",c," - Accuracy on test data:", accuracy_test)

model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=20000)
model.fit(X_training, y_training)
predictions_test = model.predict(X_test)
none_accuracy = metrics.accuracy_score(y_test, predictions_test)
print("Penalty: None - Accuracy on test data:", none_accuracy)



plt.plot(c_values, l1_accuracy, marker='o', label='L1')
plt.plot(c_values, l2_accuracy, marker='o', label='L2')
plt.plot(c_values, elasticnet_accuracy, marker='o', label='ElasticNet')
plt.axhline(y=none_accuracy, color='gray', linestyle='--', label='None')


plt.xscale('log')
plt.xlabel('C (inverse regularization strength)')
plt.ylabel('Accuracy on Test Data')
plt.title('Effect of C and Penalty on Logistic Regression Accuracy')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()