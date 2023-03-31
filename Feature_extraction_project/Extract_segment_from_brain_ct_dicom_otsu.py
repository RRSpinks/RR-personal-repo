import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk
from scipy.ndimage import binary_closing

# Function to load a DICOM file and return the CT slice as a 2D numpy array
def load_dicom_slice(dicom_file): 
    # Read the DICOM file
    dataset = pydicom.dcmread(dicom_file)
    # Get the pixel data and convert it to a numpy float32 array
    ct_slice = dataset.pixel_array.astype(np.float32)
    # Apply rescale slope and intercept to convert pixel values to Hounsfield Units (HU)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice
 
# Function to close any small holes in the mask    
def apply_binary_closing(mask, selem_radius=5):
    selem = disk(selem_radius)
    closed_mask = binary_closing(mask, structure=selem)
    return closed_mask

# Function to segment the brain slice using the Otsu threshold method
def segment_brain_slice_otsu(ct_slice):
    # Smooth the input image using a Gaussian filter
    smoothed_ct_slice = gaussian(ct_slice, sigma=3)
    # Calculate Otsu's threshold
    threshold = threshold_otsu(smoothed_ct_slice)
    # Create a binary mask using the threshold value
    mask = smoothed_ct_slice > threshold
    # Close small holes in the mask
    closed_mask = apply_binary_closing(mask)
    # Label connected regions in the closed mask
    labeled_mask = label(closed_mask)
    # Find the largest region, assuming it's the brain region
    largest_region = max(regionprops(labeled_mask), key=lambda r: r.area)
    # Create a new mask with only the largest region
    segmented_brain = np.zeros_like(labeled_mask, dtype=bool)
    segmented_brain[labeled_mask == largest_region.label] = True
    return segmented_brain

# Function to plot the original CT slice and the segmentation result
def plot_results_otsu(ct_slice, segmented_brain):
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(segmented_brain, alpha=0.3, cmap='jet')
    # plt.axis('off')
    plt.show()

# Execute all functions
if __name__ == '__main__':
    # Replace this path with the path to your DICOM file
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1640.dcm' #### INSERT FILE NAME HERE
    # Load the DICOM slice
    ct_slice = load_dicom_slice(dicom_file)
    # Segment the brain slice using Otsu's thresholding
    segmented_brain = segment_brain_slice_otsu(ct_slice)
    # Plot the results
    plot_results_otsu(ct_slice, segmented_brain)
    
