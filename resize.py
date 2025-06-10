import os
import cv2

# Set input and output directories
input_dir = 'new_data'
output_dir = 'resizeData'

# Loop through all subfolders and files in input directory
for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        # Check if file is an image
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Set input and output paths
            input_path = os.path.join(subdir, file)
            output_subdir = os.path.join(output_dir, os.path.relpath(subdir, input_dir))
            output_path = os.path.join(output_subdir, file)
            
            # Create output subdirectory if it does not exist
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
                
            # Load image and resize
            img = cv2.imread(input_path)
            resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Save resized image to output directory
            cv2.imwrite(output_path, resized)