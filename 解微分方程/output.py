import matplotlib.pyplot as plt
import numpy as np
# Load data from files
x = np.loadtxt('x.txt')
y1 = np.loadtxt('y1.txt')
y2 = np.loadtxt('y2.txt')
y3 = np.loadtxt('y3.txt')
# Create 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 16))
# Plot data on each subplot
axs[0].plot(x, y1)
axs[0].set_title('y1 vs x')
axs[1].plot(x, y2)
axs[1].set_title('y2 vs x')
axs[2].plot(x, y3)
axs[2].set_title('y3 vs x')
# Set shared labels
fig.text(0.5, 0.04, 'x', ha='center')
fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')
plt.show()
