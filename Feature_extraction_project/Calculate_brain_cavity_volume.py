import os
import numpy as np
from skimage import io
from Extract_segment_from_brain_ct_dicom_MGAC import (load_dicom_slice, get_voxel_dimensions, segment_brain_slice_mgac, 
                                plot_results_mgac, circle_level_set)

# =================================================
# This script calculates the total brain cavity volume from segmented mask images. It takes an input folder containing mask 
# images and the voxel dimensions from the original DICOM metadata to compute the volume in cubic millimeters.
# =================================================

# Create masks for all DICOM files in a folder and save them as PNG images
def create_masks_for_dicom_folder(dicom_folder, seed, mask_folder):
    dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')])
    
    # Check if the mask folder exists, if not, create it
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    voxel_dims = None
    for dicom_file in dicom_files:
        mask_filename = os.path.basename(dicom_file).replace('.dcm', '_mask.png')
        mask_filepath = os.path.join(mask_folder, mask_filename)
        
        # If the mask file already exists, skip processing for this DICOM file
        if os.path.isfile(mask_filepath):
            print(f"Mask file {mask_filename} already exists, skipping...")
            if voxel_dims is None:
                voxel_dims = get_voxel_dimensions(dicom_file)
            continue
        
        ct_slice = load_dicom_slice(dicom_file)
        voxel_dims = get_voxel_dimensions(dicom_file)
        segmented_brain, evolution, gimage = segment_brain_slice_mgac(ct_slice, seed)
        mask = np.where(segmented_brain > 0.5, 255, 0).astype(np.uint8)
        io.imsave(mask_filepath, mask)
        plot_results_mgac(ct_slice, segmented_brain, evolution, gimage, mask_folder, dicom_file)

    return voxel_dims


# Define a function to calculate brain cavity volume from mask images
def calculate_brain_cavity_volume(mask_folder, voxel_dims):
    # Get a sorted list of all mask files in the input folder
    mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')])
    
    # Initialize the total volume variable
    total_volume = 0

    # Iterate through all mask files and calculate their volume
    for mask_file in mask_files:
        # Read the mask image
        mask = io.imread(mask_file)
        # Calculate the cavity area in the mask
        cavity_area = np.sum(mask > 0)
        # Calculate the volume of the slice
        slice_volume = cavity_area * voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        # Add the slice volume to the total volume
        total_volume += slice_volume
        
    # Return the total volume
    return total_volume

# Main script
if __name__ == '__main__':
    dicom_folder = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Input/000_000_001/SCANS/1/DICOM'  # INSERT DICOM FOLDER PATH HERE
    mask_folder = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Input/000_000_001/SCANS/1/masks'  # INSERT MASK FOLDER PATH HERE
    seed = (250, 250)  # Replace this with the desired seed point
    voxel_dims = create_masks_for_dicom_folder(dicom_folder, seed, mask_folder)
    volume = calculate_brain_cavity_volume(mask_folder, voxel_dims)
    print(f"Total brain cavity volume: {volume} cubic millimeters")
    
