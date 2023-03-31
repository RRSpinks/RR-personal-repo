import os
from Extract_segment_from_brain_ct_dicom_MGAC import load_dicom_slice, segment_brain_slice_mgac, plot_results_mgac

def process_whole_brain_ct(folder_path, seed, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dicom_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')])

    for i, dicom_file in enumerate(dicom_files, 1):
        print(f"Processing slice {i}/{len(dicom_files)}: {dicom_file}")
        ct_slice = load_dicom_slice(dicom_file)
        segmented_brain, evolution, gimage = segment_brain_slice_mgac(ct_slice, seed)
        plot_results_mgac(ct_slice, segmented_brain, evolution, gimage, output_folder, dicom_file)

if __name__ == '__main__':
    folder_path = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM'  # INSERT FOLDER PATH HERE
    seed = (250, 250)  # REPLACE WITH DESIRED SEED POINT
    output_folder = '/home/richard/Richard/RS_git/RS-git-test/000_000_001/SCANS/1/DICOM/OUTPUT'  # INSERT OUTPUT FOLDER PATH HERE
    process_whole_brain_ct(folder_path, seed, output_folder)
