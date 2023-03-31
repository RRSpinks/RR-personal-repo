import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, segmentation

# Function to load a DICOM file and convert it into a CT slice as a NumPy array
def load_dicom_slice(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    ct_slice = dataset.pixel_array.astype(np.float32)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice

# Function to perform segmentation using dilation and erosion (rolling-ball effect)
def segment_brain_slice_rolling_ball(ct_slice, threshold, selem_radius=5):
    # Create a binary image by thresholding the CT slice
    binary_image = ct_slice > threshold
    # Create a disk-shaped structuring element with the specified radius
    selem = morphology.disk(selem_radius)
    # Perform dilation using the structuring element (expanding the binary image)
    dilated_image = morphology.dilation(binary_image, selem)
    # Perform erosion using the structuring element (refining the dilated image)
    eroded_image = morphology.erosion(dilated_image, selem)
    return eroded_image

# Function to display the original CT slice and the segmentation result
def plot_results(ct_slice, segmented_brain):
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(segmented_brain, alpha=0.5, cmap='jet')
    #plt.axis('off')
    plt.show()
 
if __name__ == '__main__':
    # Specify the path to your DICOM file
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1630.dcm' #### INSERT FILE NAME HERE
    # Load the DICOM file as a CT slice (NumPy array)
    ct_slice = load_dicom_slice(dicom_file)
    # Set a threshold value for segmentation (e.g., 70th percentile of intensity values)
    threshold = np.percentile(ct_slice, 70)
    # Perform segmentation using the rolling-ball effect
    segmented_brain = segment_brain_slice_rolling_ball(ct_slice, threshold)
    # Display the results
    plot_results(ct_slice, segmented_brain)
