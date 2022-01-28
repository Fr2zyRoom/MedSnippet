import os
import numpy as np

FILE_EXTENSION = ['.csv', '.CSV', '.dcm', '.DCM', '.tiff', '.TIFF', \
                  '.svs', '.SVS', '.xml', '.XML', '.png', '.PNG', \
                  '.jpg', '.JPG', '.jpeg', '.JPEG', '.img', '.IMG', '.txt', '.TXT']
CSV_EXTENSION = ['.csv', '.CSV']
DCM_EXTENSION = ['.dcm', '.DCM']
TIFF_EXTENSION = ['.tiff', '.TIFF']
SVS_EXTENSION = ['.svs', '.SVS']
XML_EXTENSION = ['.xml', '.XML']
IMG_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.img', '.IMG']
TXT_EXTENSION = ['.txt', '.TXT']


def check_extension(filename, extension_ls=FILE_EXTENSION):
    return any(filename.endswith(extension) for extension in extension_ls)


def load_file_path(folder_path, extension_ls=FILE_EXTENSION, all_sub_folders=False):
    """find 'extension_ls' file paths in folder.
    
    Parameters:
        folder_path (str) -- folder directory
        extension_ls (list) -- list of extensions
    
    Return:
        file_paths (list) -- list of 'extension_ls' file paths
    """
    
    file_paths = []
    assert os.path.isdir(folder_path), f'{folder_path} is not a valid directory'

    for root, _, fnames in sorted(os.walk(folder_path)):
        for fname in fnames:
            if check_extension(fname, extension_ls):
                path = os.path.join(root, fname)
                file_paths.append(path)
        if not all_sub_folders:
            break

    return sorted(file_paths[:])


def gen_new_dir(new_dir):
    """make new directory
    
    Parameters:
        new_dir (str) -- new directory
    
    Return:
        None
    """
    try: 
        if not os.path.exists(new_dir): 
            os.makedirs(new_dir) 
            print(f"New directory!: {new_dir}")
    except OSError: 
        print("Error: Failed to create the directory.")   


def normalize(arr, bit8=False):
    """min-max normalization
    
    Parameters:
        arr (np.array) -- image array
        bit8 (bool) -- 0to1 or 0to255(8bit)
    
    Return:
        arr_norm (np.array) -- normalized image array
    """
    arr_norm = arr - np.min(arr)
    if np.max(arr_norm) != 0:
        arr_norm /= np.max(arr_norm)
    if bit8 == True:
        arr_norm = np.array(arr_norm*255).astype(np.uint8)
    return arr_norm
