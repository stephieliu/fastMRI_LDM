import os
import h5py
import pickle

# Specify the directory where the H5 files are located
directory = "fastMRIData/multicoil_train"  # Please replace with your directory path
output_pkl = "fastMRIData/multicoil_train_slices/data_slices.pkl"  # Path to the output PKL file

# Get all H5 files in the directory
h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]

# Read the old PKL file if it exists
try:
    with open(output_pkl, "rb") as pkl_file:
        old_data_dict = pickle.load(pkl_file)
except FileNotFoundError:
    old_data_dict = {}


# create dictionary to store size of first dimension
data_dict = {}

# go through each h5 file
for h5_file in h5_files:
    full_path = os.path.join(directory, h5_file)

    # open h5 file
    with h5py.File(full_path, "r") as h5f:
        # kspace
        if "kspace" in h5f.keys():
            data = h5f["kspace"][:]
            # record size of first dict
            data_dict[h5_file] = data.shape[0]
            print(h5_file)

old_data_dict.update(data_dict)
# save dict as pkl file
with open(output_pkl, "wb") as pkl_file:
    pickle.dump(old_data_dict, pkl_file)

print(f"Data has been saved to {output_pkl}")
print(len(old_data_dict.values()))