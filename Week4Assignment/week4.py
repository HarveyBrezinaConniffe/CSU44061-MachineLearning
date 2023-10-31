import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

def test_logistic_parameter_config(X, Y, C, max_poly_order):
  poly = PolynomialFeatures(max_poly_order, include_bias=False)
  extended_X = poly.fit_transform(X)
  classifier = LogisticRegression(random_state=0, penalty="l2", C=C).fit(X, Y)  
  scores = cross_val_score(classifier, extended_X, Y, scoring="f1", cv=5)
  return scores.mean(), scores.std()

# Load first dataset
df = pd.read_csv("dataset1.csv", header=None)
data = np.array(df)
X = data[:, :2]
Y = data[:, 2]

# Plot dataset
plot_dataset(X, Y)
print(test_logistic_parameter_config(X, Y, 0.5, 5))
