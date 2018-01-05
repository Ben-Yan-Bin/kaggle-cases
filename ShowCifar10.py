from keras.datasets import cifar10
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

plt.figure(facecolor='white')

for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(X_train[i])
    plt.axis("off")

plt.show()
