import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
logistic_classifier = LogisticRegression(random_state=0).fit(
  data[:, :2], 
  data[:, 2])
print("Logistic regression parameters:")
print(logistic_classifier.coef_)

# (a)(iii)
# Generate predictions
pred = logistic_classifier.predict(data[:, :2])
# Plot positive train values
plt.plot(data[data[:, 2] == 1][:, 0], 
         data[data[:, 2] == 1][:, 1],
         marker="+",
         linestyle="None",
         color="green",
         label="+1 Train")
# Plot negative train values
plt.plot(data[data[:, 2] == -1][:, 0], 
         data[data[:, 2] == -1][:, 1],
         linestyle="None",
         marker="+",
         color="red",
         label="-1 Train")
# Plot positive predicted values
plt.plot(data[pred == 1][:, 0], 
         data[pred == 1][:, 1],
         marker="x",
         linestyle="None",
         color="green",
         label="+1 Predicted")
# Plot negative predicted values
plt.plot(data[pred == -1][:, 0], 
         data[pred == -1][:, 1],
         linestyle="None",
         marker="x",
         color="red",
         label="-1 Predicted")

# Plot decision boundary
coefs = logistic_classifier.coef_
bias = logistic_classifier.intercept_[0]
X_POINTS = (-1, 1)
Y_POINTS = tuple(((coefs[0][0]*x)+bias)/-coefs[0][1] for x in X_POINTS)
plt.plot(X_POINTS, Y_POINTS)

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend()
plt.show()
