"""
double_check_headmodel

the code below reads the head point from coreg in every 
step up until final fit and saves a csv file to 
compare them together.

process_digpoint_list: A helper function that takes in a 
        ist of DigPoint objects, processes each element, and extracts 
        the label and coordinates.
Regex: The code uses regular expressions to extract the 
        label (before the :) and the coordinates (between parentheses).
Convert to DataFrame: Each processed list is converted 
        into a Pandas DataFrame with two columns (label and coordinates).
Concatenate DataFrames: The DataFrames are concatenated side by side, 
        resulting in a final DataFrame with 130 rows and 8 columns.
Save as CSV: The resulting DataFrame is saved as a CSV file called 
        digpoint_data.csv.
This script creates the required table and saves it as a CSV file.
"""

import pandas as pd
import re

check_dig_points_csv_fname = op.join(deriv_folder, 'info_dig_head_points_fit2.csv')
check_dig_dict_points_csv_fname = op.join(deriv_folder, 'dig_dict_head_points_fit2.csv')
check_trans_points_csv_fname = op.join(deriv_folder, 'trans_head_points_fit2.csv')


# Helper function to process the DigPoint strings
def process_digpoint_list(dig_list):
    processed_data = []
    
    for digpoint in dig_list:
        # Convert DigPoint object to string
        dig_str = str(digpoint)
        
        # Extract the label (everything before the ":")
        label_match = re.search(r'<DigPoint \|(.+?):', dig_str)
        label = label_match.group(1).strip() if label_match else None
        
        # Extract the coordinates (everything between parentheses and "mm")
        coord_match = re.search(r'\((.+?)\) mm', dig_str)
        coords = coord_match.group(1).strip() if coord_match else None
        
        # Add the processed data
        processed_data.append([label, coords])
    
    return processed_data

# Process each list
before_processed = process_digpoint_list(dig_info_before)
after_fit_fiducials_processed = process_digpoint_list(dig_info_after_fit_fiducials)
after_fit_icp_processed = process_digpoint_list(dig_info_after_fit_icp)
after_omit_head_point_processed = process_digpoint_list(dig_info_after_omit_head_points)
after_final_icp_processed = process_digpoint_list(dig_info_after_final_fit_icp)

# Convert the lists into DataFrames
before_df = pd.DataFrame(before_processed, columns=["Label_Before", "Coords_Before"])
after_fit_fiducials_df = pd.DataFrame(after_fit_fiducials_processed, columns=["Label_After_Fit_Fiducials", "Coords_After_Fit_Fiducials"])
after_fit_icp_df = pd.DataFrame(after_fit_icp_processed, columns=["Label_After_Fit_ICP", "Coords_After_Fit_ICP"])
after_omit_head_point_df = pd.DataFrame(after_omit_head_point_processed, columns=["Label_After_Omit_Head_Point", "Coords_After_Omit_Head_Point"])
after_final_icp_df = pd.DataFrame(after_final_icp_processed, columns=["Label_After_Final_ICP", "Coords_After_Final_ICP"])

# Concatenate the DataFrames horizontally (side by side)
final_df = pd.concat([before_df, 
                      after_fit_fiducials_df, 
                      after_fit_icp_df, 
                      after_omit_head_point_df, 
                      after_final_icp_df], axis=1)

# Save to CSV
final_df.to_csv(check_dig_points_csv_fname, index=False)

print(f"Data saved to {check_dig_points_csv_fname}")



"""
double_check_headmodel

the code below reads the head point from coreg in every 
step up until final fit and saves a csv file to 
compare them together.

Dictionaries: The dictionaries are assumed to be before, 
        after_fit_fiducials, after_fit_icp, after_omit_head_point, 
        and after_final_icp. These contain the coordinates under 
        keys like 'nasion', 'lpa', 'rpa', 'hsp', 'hpi', and 'elp'.

CSV Writing: The code uses Python's built-in csv module to write 
        the extracted data into a CSV file called check_dig_points.csv. 
        Each row in the CSV will contain:

Label: The label of the point (nasion, lpa, rpa, etc.).
Coordinates: The x, y, z coordinates.
Set: The dictionary/set it belongs to (e.g., before, 
        after_fit_fiducials, etc.).
Coordinate Handling: The coordinates are converted to a 
        2D array even if there's only one set of coordinates. 
        This ensures that the code can handle both single points 
        and multiple points (like for hsp and hpi).

Output: The CSV file will contain the concatenated tables, 
        with rows corresponding to points and columns for label, 
        coordinates, and set.

Once you run the code, it will create the check_dig_points.csv file with all the requested data.

"""

import csv
import numpy as np

# Sample dictionaries
dictionaries = {
    'dig_info_before': dig_dict_before,
    'dig_info_after_fit_fiducials': dig_dict_after_fit_fiducials,
    'dig_info_after_fit_icp': dig_dict_after_fit_icp,
    'dig_info_after_omit_head_point': dig_dict_after_omit_head_points,
    'dig_info_after_final_icp': dig_dict_after_final_fit_icp
}

# List of point labels to extract
point_labels = ['nasion', 'lpa', 'rpa', 'hsp', 'hpi', 'elp']

# Prepare CSV file to write
with open(check_dig_dict_points_csv_fname, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Label', 'X', 'Y', 'Z', 'Set'])
    
    # Iterate over the dictionaries
    for set_name, data in dictionaries.items():
        # Iterate over the labels
        for label in point_labels:
            # Check if the label exists in the dictionary
            if label in data:
                # Get the coordinates array for the label
                coordinates = np.array(data[label])
                
                # If only one set of coordinates is present, make it a 2D array
                if coordinates.ndim == 1:
                    coordinates = coordinates[np.newaxis, :]
                
                # Write each coordinate set to the CSV file
                for coord in coordinates:
                    writer.writerow([label, coord[0], coord[1], coord[2], set_name])

print(f"Data has been saved to {check_dig_dict_points_csv_fname}")


"""
double_check_trans

the code below reads the trans matrix in every 
step up until final fit and saves a csv file to 
compare them together.

extract_transform_matrix function: This function 
        takes a transformation object, converts it 
        into a string, and splits it into lines. It 
        then removes the first line (the header) and 
        processes the remaining lines as the transformation 
        matrix.

Loop through transformations: It extracts the matrices 
        from each transformation object and stores them 
        in a list.

Concatenate the matrices: All extracted matrices are 
        concatenated into a single array using np.vstack().

Save to CSV: The combined matrix is saved as a CSV file using pandas.
"""

import numpy as np
import pandas as pd

# Function to extract the matrix and ignore the header
def extract_transform_matrix(transform):
    # Convert the transformation object to string and split by lines
    lines = str(transform).split('\n')
    
    # Ignore the first line (the descriptive header)
    matrix_lines = lines[1:]
  
    # Clean up any extra characters (like '[[' or ']]') and extract valid numeric content
    cleaned_lines = []
    for line in matrix_lines:
        # Remove brackets and extra characters, keep only the numeric content
        cleaned_line = line.replace('[', '').replace(']', '').strip()
        if cleaned_line:  # Skip any empty or invalid lines
            cleaned_lines.append(cleaned_line)
    
    # Convert the cleaned lines into a NumPy array
    matrix = np.array([list(map(float, line.split())) for line in cleaned_lines])
    
    
    return matrix

transforms = {
    'trans_before': trans_before,
    'trans_after_fit_fiducials': trans_after_fit_fiducials,
    'trans_after_fit_icp': trans_after_fit_icp,
    'trans_after_omit_head_points': trans_after_omit_head_points,
    'trans_after_final_fit_icp': trans_after_final_fit_icp
}

# Initialize an empty list to collect all matrices
all_matrices = []

# Loop through the transforms, extract the matrix, and append to the list
for name, transform in transforms.items():
    matrix = extract_transform_matrix(transform)
    all_matrices.append(matrix)

# Concatenate all matrices vertically into a single array
combined_matrix = np.hstack(all_matrices)

# Save the combined matrix into a CSV file
df = pd.DataFrame(combined_matrix)
df.to_csv(check_trans_points_csv_fname, header=False, index=False)

print("Transformations saved to check_trans_points_csv_fname")
