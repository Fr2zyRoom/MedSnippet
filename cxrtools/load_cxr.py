import numpy as np
import pydicom
from util.util import *


def cxr_loader(path):
    """load cxr image from dicom file
    
    Parameters:
        path (str) -- path to cxr image(.dcm)
    
    Return:
        norm_img (np.array) -- cxr image
    """
    dcm_info = pydicom.read_file(path, force=True)
    dcm_img = dcm_info.pixel_array
    # Rescale
    if dcm_info.get("RescaleIntercept"):
        intercept = dcm_info.RescaleIntercept
        slope = dcm_info.RescaleSlope
        
        dcm_img = dcm_img.astype(np.float64) * slope + intercept
    # VOI lookup table or windowing operation to arr
    dcm_img = pydicom.pixel_data_handlers.apply_voi_lut(dcm_img, dcm_info)
    dcm_img = dcm_img.astype(np.float64)
    #complement and normalize
    if dcm_info[0x0028, 0x0004].value != "MONOCHROME2":
        norm_img = 1 - normalize(dcm_img)
    else:
        norm_img = normalize(dcm_img)
    
    return np.uint8(norm_img*255)
