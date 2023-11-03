import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

def plot_dataset(X, Y):
  # Plot positive values
  plt.plot(X[Y == 1][:, 0], 
           X[Y == 1][:, 1],
           marker="+",
           linestyle="None",
           color="green",
           label="Target is +1")
  # Plot negative values
  plt.plot(X[Y == -1][:, 0], 
           X[Y == -1][:, 1],
           linestyle="None",
           marker="+",
           color="red",
           label="Target is -1")
  plt.xlabel("x_1")
  plt.ylabel("x_2")
  plt.legend()
  plt.show()

def print_dataset_stats(X, Y):
  total_vals = X.shape[0]

  total_pos  = np.count_nonzero(Y == 1)
  total_neg  = np.count_nonzero(Y == -1)

  percent_pos = (total_pos/total_vals)*100
  percent_neg = (total_neg/total_vals)*100

  X1_min, X1_max = np.min(X[0]), np.max(X[0])
  X2_min, X2_max = np.min(X[1]), np.max(X[1])

  print(f'{total_vals} total data points.')
  print(f'{total_pos} ({percent_pos}%) positive targets.')
  print(f'{total_neg} ({percent_neg}%) negative targets.')
  print(f'X1 range: ({X1_min}, {X1_max})')
  print(f'X2 range: ({X2_min}, {X2_max})')

def test_logistic_parameter_config(X, Y, C, max_poly_order, folds=5):
  folder =  KFold(n_splits=folds)
  poly = PolynomialFeatures(max_poly_order, include_bias=False)
  extended_X = poly.fit_transform(X)

  classifiers = []
  scores = []

  for train_idxs, test_idxs in folder.split(extended_X):
    trainX, trainY = extended_X[train_idxs], Y[train_idxs]
    testX, testY   = extended_X[test_idxs],  Y[test_idxs]

    classifier = LogisticRegression(random_state=0, max_iter=1000, penalty="l2", C=C).fit(trainX, trainY)  
    classifiers.append(classifier)
    test_preds = classifier.predict(testX)
    f1 = f1_score(testY, test_preds, average='binary')
    scores.append(f1)

  scores = np.array(scores)
  return scores.mean(), scores.std(), classifiers

# Load first dataset
dataset_file = "dataset2.csv"
df = pd.read_csv(dataset_file, header=None)
data = np.array(df)
X = data[:, :2]
Y = data[:, 2]

# Plot dataset
plot_dataset(X, Y)
print_dataset_stats(X, Y)

def plot_hyperparameter_vals(C_VALS, POLY_ORDERS, COLORS):
  poly_orders_to_f1_means = {}
  poly_orders_to_f1_std   = {}

  for POLY_ORDER in POLY_ORDERS:
    poly_orders_to_f1_means[POLY_ORDER] = []
    poly_orders_to_f1_std[POLY_ORDER] = []
    for C_VAL in C_VALS:
      f1_mean, f1_std, classifiers = test_logistic_parameter_config(X, Y, C_VAL, POLY_ORDER)
      poly_orders_to_f1_means[POLY_ORDER].append(f1_mean)
      poly_orders_to_f1_std[POLY_ORDER].append(f1_std)

  fig, ax = plt.subplots()
  i = 0
  lines = []
  for POLY_ORDER in POLY_ORDERS:
    f1_means = poly_orders_to_f1_means[POLY_ORDER]
    f1_stds  = poly_orders_to_f1_std[POLY_ORDER]
    lines.append(ax.errorbar(C_VALS, f1_means, yerr=f1_stds, color=COLORS[i]))
    i += 1
  ax.set_xlabel("C")
  ax.set_ylabel("F1 Score")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.legend(lines, [f"Max Poly Order = {POLY_ORDER}" for POLY_ORDER in POLY_ORDERS])

C_VALS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
POLY_ORDERS = [1, 2, 3, 4, 5]
COLORS = [(0.5, 0, 0, 1), (0.5, 0.5, 0, 1), (0, 0.5, 0, 1), (0, 0.5, 0.5, 1), (0.5, 0, 0.5, 1)]
plot_hyperparameter_vals(C_VALS, POLY_ORDERS, COLORS)
plt.show()  

def plot_classifier(pred, plot, decision_boundary_xs=None, decision_boundary_ys=None, legend=True):
  # Plot positive train values
  plot.plot(X[Y == 1][:, 0], 
           X[Y == 1][:, 1],
           marker="+",
           linestyle="None",
           color="green",
           label="+1 Train")
  # Plot negative train values
  plot.plot(X[Y == -1][:, 0], 
           X[Y == -1][:, 1],
           linestyle="None",
           marker="+",
           color="red",
           label="-1 Train")
  # Plot positive predicted values
  plot.plot(X[pred == 1][:, 0], 
           X[pred == 1][:, 1],
           marker="o",
           linestyle="None",
           markeredgecolor="green",
           markerfacecolor="none",
           label="+1 Predicted")
  # Plot negative predicted values
  plot.plot(X[pred == -1][:, 0], 
           X[pred == -1][:, 1],
           linestyle="None",
           marker="o",
           markeredgecolor="red",
           markerfacecolor="none",
           label="-1 Predicted")

  if decision_boundary_xs != None:
    plot.plot(decision_boundary_xs, decision_boundary_ys)
  plot.set_xlabel("x_1")
  plot.set_ylabel("x_2")
  if legend:
    plot.legend(loc="upper left")

poly = PolynomialFeatures(2, include_bias=False)
extended_X = poly.fit_transform(X)
logistic_classifier = LogisticRegression(random_state=0, max_iter=1000, penalty="l2", C=0.0001).fit(extended_X, Y)  
preds = logistic_classifier.predict(extended_X)

fig, ax = plt.subplots()
plot_classifier(preds, ax)
plt.show()

def test_knn_parameter_config(X, Y, k, folds=5):
  folder =  KFold(n_splits=folds)

  classifiers = []
  scores = []

  for train_idxs, test_idxs in folder.split(X):
    trainX, trainY = X[train_idxs], Y[train_idxs]
    testX, testY   = X[test_idxs],  Y[test_idxs]

    classifier = KNeighborsClassifier(n_neighbors=k).fit(trainX, trainY)  
    classifiers.append(classifier)
    test_preds = classifier.predict(testX)
    f1 = f1_score(testY, test_preds, average='binary')
    scores.append(f1)

  scores = np.array(scores)
  return scores.mean(), scores.std(), classifiers

K_VALS = [1, 2, 3, 4, 5, 6, 7, 8]
f1_means = []
f1_stds  = []

for K in K_VALS:
  f1_mean, f1_std, _ = test_knn_parameter_config(X, Y, K, folds=5)
  f1_means.append(f1_mean)
  f1_stds.append(f1_std)

fig, ax = plt.subplots()
ax.errorbar(K_VALS, f1_means, yerr=f1_stds)
ax.set_xlabel("K")
ax.set_ylabel("F1 Score")
ax.set_yscale("log")
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=7).fit(X, Y)
knn_preds = knn_classifier.predict(X)

fig, ax = plt.subplots()
plot_classifier(knn_preds, ax)
plt.show()

most_frequent_classifier = DummyClassifier(strategy="most_frequent").fit(X, Y)
most_frequent_preds = most_frequent_classifier.predict(X)

# Plot confusion matrices
print("Logistic Regression")
cm = confusion_matrix(Y, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print(cm)

print("KNN")
cm = confusion_matrix(Y, knn_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print(cm)

print("Most Frequent")
cm = confusion_matrix(Y, most_frequent_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print(cm)

# Plot ROC Curves
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(Y, logistic_classifier.predict_proba(extended_X)[:, 1])
knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(Y, knn_classifier.predict_proba(X)[:, 1])
mf_fpr, mf_tpr, mf_thresholds = metrics.roc_curve(Y, most_frequent_classifier.predict_proba(X)[:, 1])

plt.plot(lr_fpr, lr_tpr, label="Logistic Regression ROC")
plt.plot(knn_fpr, knn_tpr, label="KNN ROC")
plt.plot(mf_fpr, mf_tpr, label="Baseline ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
