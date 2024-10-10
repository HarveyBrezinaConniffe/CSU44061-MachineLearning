import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv("week3.csv", header=None)
data = np.array(df)
X = data[:, :2]
Y = data[:, 2]

# Part (i)

# (a)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], Y)
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.set_zlabel("Y")
plt.show()

# (b)
poly = PolynomialFeatures(5, include_bias=False)
extended_X = poly.fit_transform(X)

def analyse_model(model_func, C_VALS=[]):
  regressors = {}

  for C_VAL in C_VALS:
    model = model_func(alpha=1/(2*C_VAL))  
    model.fit(extended_X, Y)
    regressors[C_VAL] = model

  for C_VAL in C_VALS:
    model = regressors[C_VAL]
    print(model.intercept_, model.coef_)

  # Generate table
  columns = ["Feature Name"]+["C = {}".format(C_VAL) for C_VAL in C_VALS]
  rows = ["$\Theta$_{}".format(t) for t in range(0, 21)]

  cell_vals = []
  for i in range(21):
    cell_vals.append([0]*(len(C_VALS)+1))

  feature_names = ["Bias"]+list(poly.get_feature_names_out())
  for i in range(21):
    cell_vals[i][0] = feature_names[i].replace("x1", "x2").replace("x0", "x1")

  print(np.array(cell_vals))

  i = 1
  for C_VAL in C_VALS:
    model = regressors[C_VAL]
    cell_vals[0][i] = "{:.3f}".format(model.intercept_)
    j = 1 
    for param in model.coef_:
      cell_vals[j][i] = "{:.3f}".format(param)
      j += 1
    i += 1

  fig = plt.figure()
  ax = plt.subplot(111)
  fig.patch.set_visible(False)
  ax.axis('off')
  ax.axis('tight')

  t = plt.table(
    cellText=cell_vals,
    rowLabels=rows,
    colLabels=columns,
    loc='center')
  t.auto_set_font_size(False)  
  t.set(fontsize=20)
  plt.show()

  X_test = []
  grid = np.linspace(-2, 2)
  for i in grid:
    for j in grid:
      X_test.append([i, j])

  X_test = np.array(X_test)
  X_test = poly.fit_transform(X_test)

  ALPHA = 0.4

  COLORS = [(0.5, 0, 0, ALPHA), (0.5, 0.5, 0, ALPHA), (0, 0.5, 0, ALPHA), (0, 0.5, 0.5, ALPHA)]

  fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
  fig.tight_layout()
  axs = axs.flat
  i = 0

  for C_VAL, COLOUR in zip(C_VALS, COLORS):
    model = regressors[C_VAL]
    Y_test = model.predict(X_test)
    axs[i].scatter(X[:, 0], X[:, 1], Y)
    axs[i].plot_trisurf(X_test[:, 0], X_test[:, 1], Y_test, color=COLOUR)
    axs[i].set_xlabel("X_1")
    axs[i].set_ylabel("X_2")
    axs[i].set_zlabel("Y")
    axs[i].set_xlim([-2, 2])
    axs[i].set_ylim([-2, 2])
    axs[i].set_zlim([-1, 3])
    axs[i].set_title("C = {}".format(C_VAL))
    i += 1

  plt.show()

analyse_model(Lasso, C_VALS=[0.5, 10, 100, 10000])
analyse_model(Ridge, C_VALS=[0.0001, 0.001, 1, 10])

def plot_model_C(model_func, model_name, C_VALS, ax):
  MSE_VALS = []
  STD_VALS = []

  for C_VAL in C_VALS:
    model = model_func(alpha=1/(2*C_VAL))
    scores = cross_val_score(model, extended_X, Y, scoring="neg_mean_squared_error", cv=5)
    MSE_VALS.append(scores.mean())
    STD_VALS.append(scores.std())

  MSE_VALS = -np.array(MSE_VALS)

  ax.errorbar(C_VALS, MSE_VALS, yerr=STD_VALS)
  ax.set_xlabel("C")
  ax.set_ylabel("MSE")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_title(model_name)
  print(C_VALS)
  print(MSE_VALS)
  print(STD_VALS)

C_VALS = [0.0000001, 0.00001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]

fig, ax = plt.subplots()
plot_model_C(Lasso, "Lasso", C_VALS, ax)
plt.show()

fig, ax = plt.subplots()
plot_model_C(Ridge, "Ridge", C_VALS, ax)
plt.show()
