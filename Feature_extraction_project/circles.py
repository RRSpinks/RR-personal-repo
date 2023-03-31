# Python program to Plot Circle

# importing libraries
import numpy as np
from matplotlib import pyplot as plt

# Creating equally spaced 100 data in range 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Setting radius
radius = 50

# Generating x and y data
x = 100 + radius * np.cos(theta)
y = radius * np.sin(theta)

# Plotting
plt.plot(x, y)
plt.axis('equal')
plt.title('Circle')
plt.show()