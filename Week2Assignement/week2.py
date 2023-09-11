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
