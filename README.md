# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HARSHITHA D
RegisterNumber:  212224040110
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)
np.mean(predict(res.x,X)==y)

*/
```

## Output:
## Array value of X:

<img width="351" height="134" alt="271177728-d2ae898a-254e-4bb6-8287-85471943f310" src="https://github.com/user-attachments/assets/7e969595-caf0-48cd-8b4b-cb9b146c92c5" />
## Array value of Y:

<img width="287" height="52" alt="271177927-081abc04-623f-4cca-9b81-a0544d8b5c69" src="https://github.com/user-attachments/assets/4ee9db90-4c4b-43b3-b24a-42cf76cbe054" />
## Exam 1-Score graph:
<img width="753" height="547" alt="271178005-e5324611-68e0-4a45-b85a-ef44a2bef856" src="https://github.com/user-attachments/assets/127070c6-41e4-4ae9-a9e0-ceceac92c112" />

## Sigmoid function graph:
<img width="723" height="530" alt="271178062-71d11017-566c-4850-8f73-43428f9f2369" src="https://github.com/user-attachments/assets/d93a1de6-80ae-49b0-bf82-937d97f61880" />
## X_Train_grad value:
<img width="428" height="75" alt="271178121-095605bc-1834-4a65-ac74-d0b2c5a75231" src="https://github.com/user-attachments/assets/28b59f6c-eadb-4aeb-9594-0b4d400eeda6" />

## Y_Train_grad value:
<img width="366" height="74" alt="271178309-778a2f58-57b2-493b-8096-269f0618dc33" src="https://github.com/user-attachments/assets/cc433018-59cb-4b81-8954-667db6b7f376" />
## Print res.X:

<img width="407" height="73" alt="271178353-760cccb9-84ff-4e14-8328-b505c6bfed6f" src="https://github.com/user-attachments/assets/bc46ba26-2dc7-476a-bf7c-099b15191f47" />


## Decision boundary-gragh for exam score:
<img width="737" height="527" alt="271178399-23be234f-1d03-4273-b869-a3b271c1883b" src="https://github.com/user-attachments/assets/0cb04625-f437-46e3-a0a0-d426cab75d0c" />

## Probability value:

<img width="210" height="46" alt="271178457-d40871f1-acef-4abe-b197-2443d5810318" src="https://github.com/user-attachments/assets/357fd519-3041-429d-83f5-89fdbb4cf76e" />

## Prediction value of mean:
<img width="129" height="60" alt="271178510-9a117131-6df3-460a-aa65-8185815f55fa" src="https://github.com/user-attachments/assets/04c830f3-de18-4b06-9731-8fc06f0feaa2" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

