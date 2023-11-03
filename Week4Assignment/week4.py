import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

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
df = pd.read_csv("dataset1.csv", header=None)
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
