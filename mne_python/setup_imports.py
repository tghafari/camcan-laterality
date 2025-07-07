import sys
import os

def add_source_to_path():
    source_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/programming/camcan-laterality/mne_python'
    if source_path not in sys.path:
        sys.path.append(source_path)
