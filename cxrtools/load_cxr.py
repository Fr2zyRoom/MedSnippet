import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
from util.util import *


def cxr_loader(path):
    """load cxr image from dicom file
    
    Parameters:
        path (str) -- path to cxr image(.dcm)
    
    Return:
        norm_img (np.array) -- cxr image
    """
    dcm_info = pydicom.read_file(path, force=True)
    if dcm_info.file_meta.get('TransferSyntaxUID') is None:
        dcm_info.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dcm_img = dcm_info.pixel_array.astype(np.float64)
    #complement and normalize
    if dcm_info[0x0028, 0x0004].value != "MONOCHROME2":
        norm_img = 1 - normalize(dcm_img)
    else:
        norm_img = normalize(dcm_img)
    
    return np.uint8(norm_img*255)


def standard_cxr_loader(path):
    """load cxr image from dicom file
    
    Parameters:
        path (str) -- path to cxr image(.dcm)
    
    Return:
        norm_img (np.array) -- cxr image
    """
    dcm_info = pydicom.read_file(path, force=True)
    if dcm_info.file_meta.get('TransferSyntaxUID') is None:
        dcm_info.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dcm_img = dcm_info.pixel_array
    # Modality LUT
    dcm_img = apply_modality_lut(dcm_img, dcm_info)
    # VOI LUT
    dcm_img = apply_voi_lut(dcm_img, dcm_info)
    # Presentation LUT
    # complement and normalize
    if dcm_info[0x0028, 0x0004].value != "MONOCHROME2":
        norm_img = 1 - normalize(dcm_img)
    else:
        norm_img = normalize(dcm_img)
    
    return np.uint8(norm_img*255)