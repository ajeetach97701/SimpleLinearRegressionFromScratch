# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
# from LinearRegression import LinearRegression

# df = pd.read_csv('Simple_Linear_Regression/Advertising.csv')
# X = df[['TV']].to_numpy() # 200,1
# y = df['sales']

# fig = plt.figure(figsize=(10,6))
# plt.scatter(X, y, color ="red")
# plt.show()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1234)



# reg = LinearRegression(lr = 0.01)
# reg.fit(X_train, y_train)

# def mse(y_pred, y):
#     mse = np.sqrt((y_pred - y)**2)
#     return mse

# prediction = reg.predict(X_test)

# error = mse(prediction, y_test)




# # y_pred_line = reg.predict(X)
# # cmap = plt.get_cmap('viridis')
# # fig = plt.figure(figsize=(8,6))
# # plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# # m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# # m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

# # plt.ylim(0, 40)
# # plt.show()



# y_pred_line = reg.predict(X)
# cmap = plt.get_cmap('viridis')
# # fig = plt.figure(figsize=(8,6))
# # plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# m1 = plt.scatter(X_train, y_train, color='red', s=10)
# m2 = plt.scatter(X_test, y_test, color='black', s=10)
# # plt.xlim(0,350)
# # plt.ylim(0, 40)
# plt.plot(X, y_pred_line, color='pink', linewidth=10, label='Prediction')
# plt.show()










import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.001)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
