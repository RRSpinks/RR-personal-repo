import os
from Extract_segment_from_brain_ct_dicom_MGAC import load_dicom_slice, segment_brain_slice_mgac, plot_results_mgac

# =================================================
# This script processes a whole brain CT scan by segmenting each slice using Morphological Geodesic Active Contour (MGAC) 
# segmentation. It takes an input folder containing DICOM files, a seed point for MGAC segmentation, and an output folder 
# to save the segmentation results as images.
# =================================================


# Define the main processing function to process all DICOM files in a folder
def process_whole_brain_ct(input_folder, seed, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a sorted list of all DICOM files in the input folder
    dicom_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.dcm')])

    # Iterate through all DICOM files and process them
    for i, dicom_file in enumerate(dicom_files, 1):
        print(f"Processing slice {i}/{len(dicom_files)}: {dicom_file}")
        # Load the CT slice from the DICOM file
        ct_slice = load_dicom_slice(dicom_file)
        # Perform MGAC segmentation
        segmented_brain, evolution, gimage = segment_brain_slice_mgac(ct_slice, seed)
        # Plot and save the segmentation results
        plot_results_mgac(ct_slice, segmented_brain, evolution, gimage, output_folder, dicom_file)

# Main script
if __name__ == '__main__':
    input_folder = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Input/000_000_001/SCANS/1/DICOM'  # INSERT FOLDER PATH HERE
    seed = (250, 250)  # REPLACE WITH DESIRED SEED POINT
    output_folder = '/home/richard/Richard/RR-personal-repo/Data/Extract_01/Output/output'  # INSERT OUTPUT FOLDER PATH HERE
    process_whole_brain_ct(input_folder, seed, output_folder)
