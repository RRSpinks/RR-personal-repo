import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import data
from skimage.util import img_as_ubyte


# Load the image
#image = img_as_float(io.imread('brain_ct_slice.tif'))
img = img_as_ubyte(data.brain())
image = img[0]

# Smooth the image with a Gaussian filter
sigma = 1.5
image_smooth = gaussian(image, sigma=sigma)

# Define the initial contour
x = np.linspace(100, 200, 100)
y = np.linspace(100, 200, 100)
x, y = np.meshgrid(x, y)
init_contour = np.array([x.ravel(), y.ravel()]).T

# Define the parameters for active contour model
# snake = [0, 0]          # Maximum iterations to optimize snake shape.
alpha = 0.1             # Snake length shape parameter. Higher values makes snake contract faster.
beta = 1.0              # Snake smoothness shape parameter. Higher values makes snake smoother.
gamma = 0.01            # Explicit time stepping parameter.
w_line = 0.5            # Controls attraction to brightness. Use negative values to attract toward dark regions.
w_edge = 0.5            # Controls attraction to edges. Use negative values to repel snake from edges.
max_iterations = 5000   # Maximum iterations to optimize snake shape.
tolerance = 0.001

# Apply active contour model
snake = active_contour(image_smooth, init_contour, alpha=alpha, beta=beta, gamma=gamma, w_line=w_line, w_edge=w_edge,
                       max_iterations=max_iterations, convergence_check='none', tol=tolerance)

# Create a binary segmentation map
seg_map = np.zeros_like(image)
seg_map[np.round(snake[:, 1]).astype(int), np.round(snake[:, 0]).astype(int)] = 1

# Plot the original image and the segmentation map
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(seg_map, cmap='gray')
ax[1].set_title('Segmentation Map')
plt.show()
