import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Load data
df = pd.read_csv("week2.csv", header=None)
data = np.array(df)
print(df.head())

# (a)(i)
# Plot positive values
plt.plot(data[data[:, 2] == 1][:, 0], 
         data[data[:, 2] == 1][:, 1],
         marker="+",
         linestyle="None",
         color="green",
         label="Target is +1")
# Plot negative values
plt.plot(data[data[:, 2] == -1][:, 0], 
         data[data[:, 2] == -1][:, 1],
         linestyle="None",
         marker="+",
         color="red",
         label="Target is -1")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend()
plt.show()

# (a)(ii)
logistic_classifier = LogisticRegression(penalty="none", random_state=0).fit(
  data[:, :2], 
  data[:, 2])
print("Logistic regression parameters:")
print("Parameters: {}".format(logistic_classifier.coef_[0]))
print("Intercept: {}".format(logistic_classifier.intercept_[0]))

# (a)(iii)
# Generate predictions

def plot_classifier(pred, plot, decision_boundary_xs, decision_boundary_ys):
  # Plot positive train values
  plot.plot(data[data[:, 2] == 1][:, 0], 
           data[data[:, 2] == 1][:, 1],
           marker="+",
           linestyle="None",
           color="green",
           label="+1 Train")
  # Plot negative train values
  plot.plot(data[data[:, 2] == -1][:, 0], 
           data[data[:, 2] == -1][:, 1],
           linestyle="None",
           marker="+",
           color="red",
           label="-1 Train")
  # Plot positive predicted values
  plot.plot(data[pred == 1][:, 0], 
           data[pred == 1][:, 1],
           marker="x",
           linestyle="None",
           color="green",
           label="+1 Predicted")
  # Plot negative predicted values
  plot.plot(data[pred == -1][:, 0], 
           data[pred == -1][:, 1],
           linestyle="None",
           marker="x",
           color="red",
           label="-1 Predicted")

  plot.plot(decision_boundary_xs, decision_boundary_ys)
  plot.xlabel("x_1")
  plot.ylabel("x_2")
  plot.legend()
  plot.show()

def calculate_linear_decision_boundary(classifier):
  # Plot decision boundary
  coefs = classifier.coef_
  bias = classifier.intercept_[0]
  X_POINTS = (-1, 1)
  Y_POINTS = tuple(((coefs[0][0]*x)+bias)/-coefs[0][1] for x in X_POINTS)
  return X_POINTS, Y_POINTS

xs, ys = calculate_linear_decision_boundary(logistic_classifier)
preds = logistic_classifier.predict(data[:, :2])
plot_classifier(preds, plt, xs, ys)

# Part 2
svm_classifiers = {}
C_VALS = [0.001, 0.01, 0.1, 1, 10, 100]

for c_val in C_VALS:
  classifier = LinearSVC(C=c_val).fit(
    data[:, :2], 
    data[:, 2])
  svm_classifiers[c_val] = classifier

fig, axs = plt.subplots(3, 2)

for c_val in svm_classifiers:
  classifier = svm_classifiers[c_val]
  print("C = {}".format(c_val))
  print("Parameters: {}".format(classifier.coef_[0]))
  print("Intercept: {}".format(classifier.intercept_[0]))
  print()
