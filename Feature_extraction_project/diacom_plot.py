import pydicom
import numpy as np
import matplotlib.pyplot as plt

def load_dicom_slice(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    ct_slice = dataset.pixel_array.astype(np.float32)
    ct_slice = ct_slice * dataset.RescaleSlope + dataset.RescaleIntercept
    return ct_slice

def show_ct_slice(ct_slice):
    plt.imshow(ct_slice, cmap='gray')
    #plt.axis('off')
    plt.show()

if __name__ == '__main__':
    dicom_file = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/vhf.1630.dcm' #### INSERT FILE NAME HERE
    ct_slice = load_dicom_slice(dicom_file)
    show_ct_slice(ct_slice)
