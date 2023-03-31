import pydicom
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.morphology import binary_closing, disk

# Function to load a DICOM file and return the CT slice as a SimpleITK image
def load_dicom_slice_sitk(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    ct_slice = dataset.pixel_array.astype(np.float32)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return sitk.GetImageFromArray(ct_slice)

# Function to segment the brain region using the connected threshold method
def segment_brain_slice_connected_threshold(ct_slice, seed, lower_threshold, upper_threshold):
    connected_threshold_filter = sitk.ConnectedThresholdImageFilter()
    connected_threshold_filter.SetLower(lower_threshold)
    connected_threshold_filter.SetUpper(upper_threshold)
    connected_threshold_filter.AddSeed(seed)
    connected_threshold_filter.SetReplaceValue(1)
    
    segmented_brain = connected_threshold_filter.Execute(ct_slice)
    
    return segmented_brain

# Function to apply morphological closing to the segmented brain mask
def apply_morphological_closing(segmented_brain_array, selem_radius=5):
    selem = disk(selem_radius)
    closed_mask = binary_closing(segmented_brain_array, selem)
    return closed_mask

# Function to plot the original CT slice and the segmentation result
def plot_results_connected_threshold(ct_slice, segmented_brain):
    ct_slice_array = sitk.GetArrayFromImage(ct_slice)
    segmented_brain_array = sitk.GetArrayFromImage(segmented_brain)
    
    plt.imshow(ct_slice_array, cmap='gray')
    plt.imshow(segmented_brain_array, alpha=0.35, cmap='gist_ncar')
    #plt.axis('off')
    plt.show()

# Main script
if __name__ == '__main__':
    # Load the CT slice
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1640.dcm' #### INSERT FILE NAME HERE
    ct_slice = load_dicom_slice_sitk(dicom_file)
    
    # Define seed point and intensity thresholds
    x_dim, y_dim = ct_slice.GetSize()
    seed = (y_dim // 2, x_dim // 2)
    lower_threshold = -50  #### ADJUSTED AS NEEDED
    upper_threshold = 100  #### ADJUSTED AS NEEDED
    
    # Segment the brain region using the connected threshold method
    segmented_brain = segment_brain_slice_connected_threshold(ct_slice, seed, lower_threshold, upper_threshold)
    segmented_brain_array = sitk.GetArrayFromImage(segmented_brain)

    # Apply morphological closing to the segmented brain mask
    closed_brain_mask = apply_morphological_closing(segmented_brain_array)
    closed_brain = sitk.GetImageFromArray(closed_brain_mask.astype(np.uint8))

    # Plot the results
    plot_results_connected_threshold(ct_slice, closed_brain)