import numpy as np
import sys

file_path = r"biomechanics_retarget\processed_data\S02_long_165\keypoints\S02_15ms_Long.npy"

try:
    data = np.load(file_path, allow_pickle=True)
    print(f"Type: {type(data)}")
    print(f"Shape: {data.shape}")
    
    if data.shape == ():
        print("Data is a 0-d array (scalar). Content:")
        print(data.item())
        if isinstance(data.item(), dict):
            print("Keys:", data.item().keys())
            for k, v in data.item().items():
                if isinstance(v, np.ndarray):
                    print(f"Key '{k}': shape {v.shape}")
    
except Exception as e:
    print(f"Error: {e}")
