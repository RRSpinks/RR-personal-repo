import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Function to load a DICOM file and return the CT slice as a 2D numpy array
def load_dicom_slice(dicom_file): 
    # Read the DICOM file
    dataset = pydicom.dcmread(dicom_file)
    # Get the pixel data and convert it to a numpy float32 array
    ct_slice = dataset.pixel_array.astype(np.float32)
    # Apply rescale slope and intercept to convert pixel values to Hounsfield Units (HU)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice

# Function to create an initial snake (contour) for active_contour function
def create_initial_snake(ct_slice, brain_r, brain_x, brain_y):
    x_dim, y_dim = ct_slice.shape
    # Create a circular snake with a radius slightly smaller than half of the image dimensions
    s = np.linspace(0, 2 * np.pi, 400)
    x = brain_x + brain_r * np.cos(s)
    y = brain_y + brain_r * np.sin(s)
    return np.array([x, y]).T

# Function to segment the brain slice using the active_contour method
def segment_brain_slice(ct_slice):
    # Smooth the input image using a Gaussian filter
    smoothed_ct_slice = gaussian(ct_slice, sigma=3)
    # Create the initial snake
    initial_snake = create_initial_snake(ct_slice, brain_r, brain_x, brain_y)
    # Apply the active_contour function to refine the snake
    snake = active_contour(smoothed_ct_slice, initial_snake, 
                           alpha=0.06,  # Snake length shape parameter. Higher values makes snake contract faster.
                           beta=1.0,    # Snake smoothness shape parameter. Higher values makes snake smoother.
                           gamma=0.1,   # Explicit time stepping parameter.
                           w_line=1,    # Controls attraction to brightness. Use negative values to attract toward dark regions.   
                           w_edge=0.1)    # Controls attraction to edges. Use negative values to repel snake from edges.
    return snake

# Function to plot the original CT slice and the segmentation result
def plot_results(ct_slice, initial_snake, final_snake):
    plt.imshow(ct_slice, cmap='gray')
    plt.plot(initial_snake[:, 0], initial_snake[:, 1], '-b', lw=2, label='Initial Snake')
    plt.plot(final_snake[:, 0], final_snake[:, 1], '-r', lw=2, label='Final Snake')
    plt.legend(loc='upper right')
    #plt.axis('off')
    plt.show()

# Execute all functions
if __name__ == '__main__':
    # Replace this path with the path to your DICOM file
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1640.dcm' #### INSERT FILE NAME HERE
    # Load the DICOM slice
    ct_slice = load_dicom_slice(dicom_file)
    # Create the initial snake
    brain_r = 125
    brain_x = 250
    brain_y = 250
    initial_snake = create_initial_snake(ct_slice, brain_r, brain_x, brain_y)
    # Segment the brain slice using active_contour to create the final snake
    final_snake = segment_brain_slice(ct_slice)
    # Plot the results with the initial and final snake
    plot_results(ct_slice, initial_snake, final_snake)
