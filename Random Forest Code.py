import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
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


model = RandomForestClassifier(random_state=43)
model.fit(X_training, y_training)

predictions_training = model.predict(X_training)
predictions_test = model.predict(X_test)

default_accuracy_training = metrics.accuracy_score(y_training, predictions_training)
default_accuracy_test = metrics.accuracy_score(y_test, predictions_test)
print("---------DEFAULT MODEL RESULTS---------")
print("Accuracy on training data:",default_accuracy_training)
print("Accuracy on test data:",default_accuracy_test)

print("\n ------------TESTING N_ESTIMATORS------------ \n")
n_estimator_values=[100, 200, 300, 400, 500]
for n_estimator in n_estimator_values:
    model = RandomForestClassifier(n_estimators=n_estimator, random_state=43)
    model.fit(X_training, y_training)
    predictions_training = model.predict(X_training)
    predictions_test = model.predict(X_test)
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("n_estimators:", n_estimator, " - Accuracy on training data:", accuracy_training)
    print("n_estimators:", n_estimator, " - Accuracy on test data:", accuracy_test)

print("\n ----------TESTING MAX_DEPTH----------- \n")
criterion_values=['gini', 'entropy', 'log_loss']
max_depth_values=[3, 6, 9, 12, 15, 18,None]
for max_depth in max_depth_values:
    model = RandomForestClassifier(max_depth=max_depth,random_state=43)
    model.fit(X_training, y_training)
    predictions_training = model.predict(X_training)
    predictions_test = model.predict(X_test)
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("max_depth:", max_depth, " - Accuracy on training data:", accuracy_training)
    print("max_depth:", max_depth, " - Accuracy on test data:", accuracy_test)

best_accuracy_training=0
best_max_depth_training=0
best_n_estimator_training=0
worst_accuracy_training=1
worst_max_depth_training=0
worst_n_estimator_training=0
total_accuracy_training=0

max_depth_list_training = []
n_estimators_list_training = []
accuracy_list_training = []



best_accuracy_test=0
best_max_depth_test=0
best_n_estimator_test=0
worst_accuracy_test=1
worst_max_depth_test=0
worst_n_estimator_test=0
total_accuracy_test=0

max_depth_list_test = []
n_estimators_list_test = []
accuracy_list_test = []


count=0

print("\n ------------TESTING BOTH MAX_DEPTH AND N_ESTIMATORS------------- \n")
for max_depth in max_depth_values:
    for n_estimator in n_estimator_values:
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator, random_state=43)
        model.fit(X_training, y_training)
        predictions_training = model.predict(X_training)
        predictions_test = model.predict(X_test)
        accuracy_training = metrics.accuracy_score(y_training, predictions_training)
        accuracy_test = metrics.accuracy_score(y_test, predictions_test)

        if max_depth is None:
            max_depth_list_training.append('None')
        else:
            max_depth_list_training.append(max_depth)
        n_estimators_list_training.append(n_estimator)
        accuracy_list_training.append(accuracy_training)

        if accuracy_training>=best_accuracy_training:
            best_accuracy_training=accuracy_training
            best_max_depth_training=max_depth
            best_n_estimator_training=n_estimator
        if accuracy_training<=worst_accuracy_training:
            worst_accuracy_training=accuracy_training
            worst_max_depth_training=max_depth
            worst_n_estimator_training=n_estimator


        if max_depth is None:
            max_depth_list_test.append('None')
        else:
            max_depth_list_test.append(max_depth)
        n_estimators_list_test.append(n_estimator)
        accuracy_list_test.append(accuracy_test)

        if accuracy_test>=best_accuracy_test:
            best_accuracy_test=accuracy_test
            best_max_depth_test=max_depth
            best_n_estimator_test=n_estimator
        if accuracy_test<=worst_accuracy_test:
            worst_accuracy_test=accuracy_test
            worst_max_depth_test=max_depth
            worst_n_estimator_test=n_estimator
        count=count+1
        total_accuracy_training = total_accuracy_training + accuracy_training
        total_accuracy_test=total_accuracy_test+accuracy_test
        print("max_depth:", max_depth, " n_estimators:", n_estimator, " - Accuracy on test data:", accuracy_test)
        print("max_depth:", max_depth, " n_estimators:", n_estimator, " - Accuracy on training data:", accuracy_training)

print("------------TRAINING RESULTS----------- ")
print("Default training accuracy is ",default_accuracy_training)
print("The best accuracy is ",best_accuracy_training," with max_depth=",best_max_depth_training," and n_estimators=",best_n_estimator_training)
print("The worst accuracy is ",worst_accuracy_training," with max_depth=",worst_max_depth_training," and n_estimators=",worst_n_estimator_training)
print("The average accuracy is ",total_accuracy_training/count)


print("------------TEST RESULTS----------- ")
print("Default test accuracy is ",default_accuracy_test)
print("The best accuracy is ",best_accuracy_test," with max_depth=",best_max_depth_test," and n_estimators=",best_n_estimator_test)
print("The worst accuracy is ",worst_accuracy_test," with max_depth=",worst_max_depth_test," and n_estimators=",worst_n_estimator_test)
print("The average accuracy is ",total_accuracy_test/count)


df = pd.DataFrame({
    'max_depth': max_depth_list_training,
    'n_estimators': n_estimators_list_training,
    'accuracy': accuracy_list_training
})
# plot


pivot = df.pivot(index='max_depth', columns='n_estimators', values='accuracy')
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True,cmap='mako', fmt=".2f")
plt.title("The Effect of max_depth and n_estimators on Random Forest Accuracy (TRAINING)")
plt.ylabel("max_depth")
plt.xlabel("n_estimators")
plt.show()

df = pd.DataFrame({
    'max_depth': max_depth_list_test,
    'n_estimators': n_estimators_list_test,
    'accuracy': accuracy_list_test
})
# plot


pivot = df.pivot(index='max_depth', columns='n_estimators', values='accuracy')
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True,cmap='mako', fmt=".2f")
plt.title("The Effect of max_depth and n_estimators on Random Forest Accuracy (TEST)")
plt.ylabel("max_depth")
plt.xlabel("n_estimators")
plt.show()
# print(model.feature_importances_)
#
# criteria = ['gini', 'entropy', 'log_loss']
# n_estimators = [100, 200, 300, 400, 500]
#
# # Accuracy results
# accuracies = {
#     'gini': [0.84, 0.84, 0.82, 0.84, 0.84],
#     'entropy': [0.86, 0.84, 0.84, 0.84, 0.86],
#     'log_loss': [0.86, 0.84, 0.84, 0.84, 0.86]
# }
#
# # Plot each criterion as a line
# plt.figure(figsize=(8, 5))
# for criterion in criteria:
#     plt.plot(n_estimators, accuracies[criterion], marker='o', label=criterion)
#
# # Labels and title
# plt.title("Random Forest Accuracy vs. Number of Estimators")
# plt.xlabel("Number of Estimators (n_estimators)")
# plt.ylabel("Accuracy on Test Data")
# plt.xticks(n_estimators)
# plt.ylim(0.8, 0.9)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Criterion")
#
#
# plt.show()


# ✅ after loop: create a dataframe for plotting
