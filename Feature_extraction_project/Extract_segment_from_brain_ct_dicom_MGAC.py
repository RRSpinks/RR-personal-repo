import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, exposure, morphology
from matplotlib.lines import Line2D
from skimage.filters import sobel
import os

# =================================================
# Note: This script is intended to be called by other, not run itself
# This script performs Morphological Geodesic Active Contour (MGAC) segmentation on a brain CT slice, 
# utilizing various image processing techniques to improve contrast and visualize the segmentation results.
# =================================================


# Load a DICOM CT slice and convert it to Hounsfield units
def load_dicom_slice(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    ct_slice = dataset.pixel_array.astype(np.float32)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice

# Store intermediate results of the MGAC iterations
def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))
    return _store

# Define custome function to create a small circle level set
def circle_level_set(shape, center, radius):
    grid = np.mgrid[list(map(slice, shape))].T - center
    squared_distance = np.sum(grid ** 2, -1)
    return np.where(squared_distance < radius**2, 1, 0).astype(np.float64)

# # Create custom function to close small gaps in the skull
# def close_small_gaps(image, iterations=3, selem_radius=1):
#     # Define a disk footprint
#     selem = morphology.disk(selem_radius)
#     selem2 = morphology.disk(selem_radius/2)
#     # Dilate the image to close gaps
#     dilated = morphology.dilation(image, selem)
#     # Erode image to reduce size to normal but with closed gaps
#     closed = morphology.erosion(dilated, selem2)
#     return closed

# Segment a brain CT slice using MGAC
def segment_brain_slice_mgac(ct_slice, seed, iterations=50, smoothing=4, balloon=1, threshold=0.95):
    # Normalize the image to the range of 0 to 1
    ct_slice_normalized = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    # Apply adaptive histogram equalization (CLAHE) to improve the contrast
    ct_slice_equalized = exposure.equalize_adapthist(ct_slice_normalized, clip_limit=0.03, nbins=256)
    # Calculate the gradient magnitude of the equalized CT slice
    ct_slice_sobel = sobel(ct_slice_equalized)
    ct_slice_inverted = ct_slice_sobel.max() - ct_slice_sobel
    # Close small gaps in the skull
    #ct_slice_closed = close_small_gaps(ct_slice_inverted, iterations=3, selem_radius=3)
    # Normalize image again
    gimage = (ct_slice_inverted - ct_slice_inverted .min()) / (ct_slice_inverted .max() - ct_slice_inverted .min())
    # Set the initial level set as a small circle centered at the seed point
    init_ls = circle_level_set(ct_slice.shape, seed, radius=25)
    # Store intermediate results during iterations
    evolution = []
    callback = store_evolution_in(evolution)
    # Perform MGAC segmentation
    ls = segmentation.morphological_geodesic_active_contour(
        gimage, iterations, init_ls, smoothing=smoothing, balloon=balloon, threshold=threshold, iter_callback=callback)
    return ls, evolution, gimage

# Custom function to access voxel dimensions in metadata
def get_voxel_dimensions(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    voxel_spacing = dataset.PixelSpacing
    slice_thickness = dataset.SliceThickness
    voxel_dims = (voxel_spacing[0], voxel_spacing[1], slice_thickness)
    return voxel_dims

# Plot the results of the MGAC segmentation
def plot_results_mgac(ct_slice, segmented_brain, evolution, gimage, save_path, dicom_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # Plot the final processed image
    ax1.imshow(gimage, cmap="gray")
    ax1.set_title("Final processed image", fontsize=12)
    ax2.imshow(ct_slice, cmap="gray")
    # In a seperate graph, plot the initial shape
    contour_initial = ax2.contour(evolution[0], [0.5], colors='r')
    # Plot the 5th iteration if available
    if len(evolution) > 4:
        contour_5th = ax2.contour(evolution[4], [0.5], colors='g')
    # Plot the final iteration
    contour_final = ax2.contour(segmented_brain, [0.5], colors='b')
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color='r', lw=2, label="Starting shape"),
        Line2D([0], [0], color='g', lw=2, label="5th Iteration"),
        Line2D([0], [0], color='b', lw=2, label="Final iteration")
    ]
    ax2.legend(handles=legend_elements, loc="upper right")
    ax2.set_title("Morphological GAC evolution", fontsize=12)
    # Save the figure to the specified folder as a PNG file
    plt.savefig(os.path.join(save_path, f"{os.path.basename(dicom_file)}_segmentation.png"), dpi=300)
    plt.close(fig)  # Close the figure to free up memory

# Main script
if __name__ == '__main__':
    # Load the DICOM CT slice
    dicom_file = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Input/000_000_001/SCANS/1/DICOM/vhf.1521.dcm'  # INSERT FILE NAME HERE
    ct_slice = load_dicom_slice(dicom_file)
    # Define the seed point for the MGAC segmentation
    seed = (250, 250)  # Replace this with the desired seed point
    # Perform MGAC segmentation
    segmented_brain, evolution, gimage = segment_brain_slice_mgac(ct_slice, seed)
    # Plot the results
    save_path = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Output/output' 
    plot_results_mgac(ct_slice, segmented_brain, evolution, gimage, save_path, dicom_file) # INSERT SAVE PATH HERE

