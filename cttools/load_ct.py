import numpy as np
import pydicom
from util.util import *


def set_outside_scanner_to_air(hu_pixelarrays):
    """
    Pixel Padding Value Attribute(0028,0120) -> air
    """
    hu_pixelarrays[hu_pixelarrays < -1024] = -1024
    
    return hu_pixelarrays


def load_ct_info(patient_folder_path):
    """
    Loading CT-info "per patient" from the case folder
    
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


def dcm2ct(slices):
    """
    Loading CT scans "per patient" from the list of DICOM Header info
    
    Parameters:
        slices (list of PydicomDataset) -- list of DICOM Header info
        
    Return:
        hu_images (np.array) -- 3D CT scans with Houndsfield Unit 
        
    """
    # stack slices
    # Modality LUT: convert to HU
    hu_images = np.stack([apply_modality_lut(s.pixel_array, s) for s in slices])
    # Pixel Padding Value
    hu_images = set_outside_scanner_to_air(hu_images)
    return hu_images.astype(np.int16)


def ct_loader(patient_folder_path):
    """
    Loading CT-scans "per patient" from the case folder
    patient path to CT slices(HU)
    
    Parameters:
        patient_folder_path (str) -- case folder directory
        
    Return:
        slices[0] (PydicomDataset) -- DICOM Info
        hu_images (list) -- list of CT images(HU)
        
    """
    slices = load_ct_info(patient_folder_path)
    
    hu_images = dcm2ct(slices)
    
    # return header of first dicom and stacked CT slices
    return slices[0], hu_images
