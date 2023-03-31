import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, filters, exposure, morphology

def load_dicom_slice(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    ct_slice = dataset.pixel_array.astype(np.float32)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice

def segment_brain_slice_watershed(ct_slice, compactness=0.01):
    # Equalize the histogram for better contrast
    ct_slice_equalized = exposure.equalize_hist(ct_slice)
      # Compute the gradient magnitude of the CT slice
    gradient = filters.rank.gradient(ct_slice_equalized, morphology.disk(5))
    # Apply the Watershed algorithm to segment the internal brain cavity
    labels = segmentation.watershed(gradient, compactness=compactness, markers=1)
    return labels == 1

def plot_results_watershed(ct_slice, segmented_brain):
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(segmented_brain, alpha=0.5, cmap='jet')
    #plt.axis('off')
    plt.show()

if __name__ == '__main__':
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1630.dcm' #### INSERT FILE NAME HERE
    ct_slice = load_dicom_slice(dicom_file)
    segmented_brain = segment_brain_slice_watershed(ct_slice)
    plot_results_watershed(ct_slice, segmented_brain)
