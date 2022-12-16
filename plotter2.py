
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import numba as nb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv

df = pd.read_csv('output2.csv')

MAPs = [df['p'].to_numpy(), df['ic'].to_numpy(), df['sum_error'].to_numpy()]

fig = plt.figure(figsize=(10,15))
X, Y = np.meshgrid(MAPs[0],MAPs[1])
Z = np.array([MAPs[2]])
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("p : % train / test")
ax.set_ylabel("ic : % confiance")
ax.set_zlabel("sum_error : sum of errors")
surf = ax.scatter(MAPs[0], MAPs[1], MAPs[2])
plt.show()

