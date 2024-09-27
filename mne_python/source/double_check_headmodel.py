
"""
===============================================
double_check_headmodel
this scripts reads the head point from coreg in every 
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

# Assuming the lists are called: before, after_fit_fiducials, after_fit_icp, after_omit_head_point, after_final_icp
# These lists have the DigPoint format.

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
before_processed = process_digpoint_list(before)
after_fit_fiducials_processed = process_digpoint_list(after_fit_fiducials)
after_fit_icp_processed = process_digpoint_list(after_fit_icp)
after_omit_head_point_processed = process_digpoint_list(after_omit_head_point)
after_final_icp_processed = process_digpoint_list(after_final_icp)

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
final_df.to_csv('digpoint_data.csv', index=False)

print("Data saved to 'digpoint_data.csv'")
