import os
import numpy as np
from skimage import io

def calculate_brain_cavity_volume(mask_folder, voxel_dims):
    mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')])
    
    total_volume = 0
    for mask_file in mask_files:
        mask = io.imread(mask_file)
        cavity_area = np.sum(mask > 0)
        slice_volume = cavity_area * voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        total_volume += slice_volume
        
    return total_volume

if __name__ == '__main__':
    mask_folder = 'masks'  # INSERT MASK FOLDER PATH HERE
    voxel_dims = (1, 1, 1)  # REPLACE WITH ACTUAL VOXEL DIMENSIONS (in mm) FROM DICOM METADATA
    volume = calculate_brain_cavity_volume(mask_folder, voxel_dims)
    print(f"Total brain cavity volume: {volume} cubic millimeters")
