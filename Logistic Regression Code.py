import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
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


model = LogisticRegression(max_iter=20000)
model.fit(X_training, y_training)

predictions_training = model.predict(X_training)
predictions_test = model.predict(X_test)

default_accuracy_training = metrics.accuracy_score(y_training, predictions_training)
default_accuracy_test = metrics.accuracy_score(y_test, predictions_test)
print("Accuracy on training data:",default_accuracy_training)
print("Accuracy on test data:",default_accuracy_test)


penalty_list=['l1', 'l2', 'elasticnet', None]
c_values=[0.01, 0.1, 1, 10, 100]
print("C values:", c_values)
print("Penalty values:", penalty_list)


print("\n ------------TESTING C------------- \n")

for c in c_values:
    model = LogisticRegression(C=c, max_iter=20000)
    model.fit(X_training, y_training)
    predictions_training = model.predict(X_training)
    predictions_test = model.predict(X_test)
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("C: ",c,"Accuracy on training data:",accuracy_training)
    print("C:", c, " - Accuracy on test data:", accuracy_test)

print("\n ------------TESTING PENALTY------------- \n")

for penalty in penalty_list:
    if penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=20000)
    elif penalty == 'l2' or penalty is None:
        model = LogisticRegression(penalty=penalty, solver='lbfgs', max_iter=20000)
    else:
        model = LogisticRegression(penalty=penalty,solver="liblinear", max_iter=20000 )
    model.fit(X_training, y_training)
    predictions_test = model.predict(X_test)
    predictions_training = model.predict(X_training)
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("Penalty: ",penalty,"Accuracy on training data:",accuracy_training)
    print("Penalty:", penalty, " - Accuracy on test data:", accuracy_test)

penalty_list_no_none=['l1', 'l2', 'elasticnet']

best_penalty_training=""
best_c_training=0
best_accuracy_training=0
worst_penalty_training=""
worst_c_training=0
worst_accuracy_training=1



l1_accuracy_training=[]
l2_accuracy_training=[]
elasticnet_accuracy_training=[]


best_penalty_test=""
best_c_test=0
best_accuracy_test=0
worst_penalty_test=""
worst_c_test=0
worst_accuracy_test=1

l1_accuracy_test=[]
l2_accuracy_test=[]
elasticnet_accuracy_test=[]

count=0
total_accuracy_test=0
total_accuracy_training=0

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
        predictions_training = model.predict(X_training)
        accuracy_training = metrics.accuracy_score(y_training, predictions_training)
        accuracy_test = metrics.accuracy_score(y_test, predictions_test)

        if penalty == 'l1':
            l1_accuracy_training.append(accuracy_training)
        elif penalty == 'l2':
            l2_accuracy_training.append(accuracy_training)
        elif penalty == 'elasticnet':
            elasticnet_accuracy_training.append(accuracy_training)
        if accuracy_training > best_accuracy_training:
            best_accuracy_training=accuracy_training
            best_penalty_training=penalty
            best_c_training=c
        if accuracy_training < worst_accuracy_training:
            worst_accuracy_training=accuracy_training
            worst_penalty_training=penalty
            worst_c_training=c


        if penalty == 'l1':
            l1_accuracy_test.append(accuracy_test)
        elif penalty == 'l2':
            l2_accuracy_test.append(accuracy_test)
        elif penalty == 'elasticnet':
            elasticnet_accuracy_test.append(accuracy_test)
        if accuracy_test > best_accuracy_test:
            best_accuracy_test=accuracy_test
            best_penalty_test=penalty
            best_c_test=c
        if accuracy_test < worst_accuracy_test:
            worst_accuracy_test=accuracy_test
            worst_penalty_test=penalty
            worst_c_test=c




        count+=1
        total_accuracy_test=total_accuracy_test+accuracy_test
        total_accuracy_training=total_accuracy_training+accuracy_training
        print("Penalty:", penalty, "| C: ",c," - Accuracy on training data:", accuracy_training)
        print("Penalty:", penalty, "| C: ",c," - Accuracy on test data:", accuracy_test)

model = LogisticRegression(penalty=None, max_iter=20000)
model.fit(X_training, y_training)
predictions_training = model.predict(X_training)
predictions_test = model.predict(X_test)
none_accuracy_training = metrics.accuracy_score(y_training, predictions_training)
none_accuracy_test = metrics.accuracy_score(y_test, predictions_test)
print("Penalty: None - Accuracy on training data:", none_accuracy_training)
print("Penalty: None - Accuracy on test data:", none_accuracy_test)


print("------------TRAINING RESULTS-----------")
print("Default training accuracy is ",default_accuracy_training)
print("Best Penalty:", best_penalty_training, "| Best C:", best_c_training, " | Best Accuracy:", best_accuracy_training)
print(" Worst Penalty:", worst_penalty_training, "| Worst C:", worst_c_training, " | Worst Accuracy:", worst_accuracy_training)
print("Average Accuracy:", total_accuracy_training/count)

print("------------TEST RESULTS----------- ")
print("Default test accuracy is ",default_accuracy_test)
print("Best Penalty:", best_penalty_test, "| Best C:", best_c_test, " | Best Accuracy:", best_accuracy_test)
print(" Worst Penalty:", worst_penalty_test, "| Worst C:", worst_c_test, " | Worst Accuracy:", worst_accuracy_test)
print("Average Accuracy:", total_accuracy_test/count)

plt.plot(c_values, l1_accuracy_training, marker='o', label='L1')
plt.plot(c_values, l2_accuracy_training, marker='o', label='L2')
plt.plot(c_values, elasticnet_accuracy_training, marker='o', label='ElasticNet')
plt.axhline(y=none_accuracy_training, color='red', label='None')


plt.xscale('log')
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Accuracy on Training Data')
plt.title('Effect of C and Penalty on Logistic Regression Accuracy (TRAINING)')
plt.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


plt.plot(c_values, l1_accuracy_test, marker='o', label='L1')
plt.plot(c_values, l2_accuracy_test, marker='o', label='L2')
plt.plot(c_values, elasticnet_accuracy_test, marker='o', label='ElasticNet')
plt.axhline(y=none_accuracy_test, color='red', label='None')


plt.xscale('log')
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Accuracy on Test Data')
plt.title('Effect of C and Penalty on Logistic Regression Accuracy (TEST)')
plt.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()