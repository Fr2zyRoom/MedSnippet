import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut
from util.util import *


def load_mr_info(patient_folder_path):
    """
    Loading MR-info "per patient" from the case folder
    
    Parameters:
        patient_folder_path (str) -- case folder directory
        
    Return:
        slices (list) -- list of DICOM Header info(PydicomDataset)
        
    """
    # call all image paths in directory
    dcm_paths = load_file_path(patient_folder_path, DCM_EXTENSION)
    
    slices = [pydicom.read_file(dcm_path, force=True) for dcm_path in dcm_paths]
    
    # **Sort the slices in ascending order
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    
    return slices


def mr_loader(patient_folder_path):
    """
    Loading MR-scans "per patient" from the case folder
    patient path to MR slices
    
    Parameters:
        patient_folder_path (str) -- case folder directory
        
    Return:
        slices[0] (PydicomDataset) -- DICOM Info
        mr_images (list) -- list of MR images
        
    """
    slices = load_mr_info(patient_folder_path)
    
    mr_images = np.stack([s.pixel_array for s in slices])
    
    # return header of first dicom and stacked CT slices
    return slices[0], mr_images