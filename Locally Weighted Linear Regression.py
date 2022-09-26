import numpy as np
import matplotlib.pyplot as plt


trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_train.csv", delimiter=',',
                encoding='utf8')

test_short = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_test_short.csv", delimiter=',',
                encoding='utf8')

X_train = trng[:,0:6]
X_train_max = np.max(X_train, axis=0, keepdims=True)
X_train_min = np.min(X_train, axis=0, keepdims=True)
X_train = (X_train - X_train_min)/(X_train_max - X_train_min)
ones_array = np.ones([len(X_train),1])
X_train = np.c_[ones_array,X_train]


X_test = test_short[:,0:6]
X_test_max = np.max(X_test, axis=0, keepdims=True)
X_test_min = np.min(X_test, axis=0, keepdims=True)
X_test = (X_test - X_train_min)/(X_train_max - X_train_min)
ones_array = np.ones([len(X_test),1])
X_test = np.c_[ones_array,X_test]


Y_train = trng[:,6]
Y_test = test_short[:,6]

w_train = [0]*10
y_pred = np.zeros(100)
error = np.zeros(10)
tau_scale =2**np.linspace(-2,1,10)
for index , tau in enumerate(tau_scale):  
    for i in range(len(X_test)):
        X0 = X_test[i]
        R = np.zeros((len(X_train),len(X_train)))
        for j in range(len(X_train)):
            R[j][j] = np.exp(-((X_train[j] - X0).T @ (X_train[j] - X0))/ (2*(tau**2)))
        w_train = np.linalg.solve(X_train.T @ R @ X_train, X_train.T @ R @ Y_train)
        y_pred[i] = np.dot(X0,w_train)
    error[index] = np.sqrt(np.mean((y_pred-Y_test)**2))
    
    
plt.plot(tau_scale,error,marker='.',markersize=7.5)
plt.xlabel('$ln\tau$')
plt.ylabel('Test Error')
plt.legend(['Test Error'])
plt.title('Test error against hyperparamter $\tau$')
plt.show()