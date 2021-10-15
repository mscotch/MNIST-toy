import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 28x28 pixels
digits = pd.read_csv(r'C:\Users\skoca\PycharmProjects\MNIST-toy\Dataset\Kaggle\train.csv')
digits_x = digits.iloc[:, 1:]
digits_y = digits.iloc[:, 0]

# Show random number as 28x28 pixel image
plt.imshow(np.array(digits_x.iloc[np.random.randint(0, 1000), :]).reshape(28, 28))
plt.show()