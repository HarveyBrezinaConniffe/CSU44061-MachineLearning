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

  scores = []

  for train_idxs, test_idxs in folder.split(extended_X):
    trainX, trainY = extended_X[train_idxs], Y[train_idxs]
    testX, testY   = extended_X[test_idxs],  Y[test_idxs]

    classifier = LogisticRegression(random_state=0, penalty="l2", C=C).fit(trainX, trainY)  
    test_preds = classifier.predict(testX)
    f1 = f1_score(testY, test_preds, average='binary')
    scores.append(f1)

  scores = np.array(scores)
  return scores.mean(), scores.std()

# Load first dataset
df = pd.read_csv("dataset1.csv", header=None)
data = np.array(df)
X = data[:, :2]
Y = data[:, 2]

# Plot dataset
plot_dataset(X, Y)
print_dataset_stats(X, Y)
print(test_logistic_parameter_config(X, Y, 0.5, 5))
