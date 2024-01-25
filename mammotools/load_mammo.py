import numpy as np
import pydicom
import pylibjpeg
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut

from skimage.filters import threshold_otsu
from util.util import *
from tools.segmentation_tools import *

def mammography_loader(path):
    """load cxr image from dicom file
    
    Parameters:
        path (str) -- path to mammography image(.dcm)
    
    Return:
        norm_img (np.array) -- mammography image
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


def standard_mammography_loader(path, voi_lut=False, get_roi=False):
    """load mammography image from dicom file
    
    Parameters:
        path (str) -- path to mammography image(.dcm)
        voi_lut (bool) -- apply value of interest lookup table
        get_roi (bool) -- get region of interst
    
    Return:
        norm_img (np.array) -- mammography image
        [optional] coor (list) -- coordinate of roi
    """
    dcm_info = pydicom.read_file(path, force=True)
    if dcm_info.file_meta.get('TransferSyntaxUID') is None:
        dcm_info.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dcm_img = dcm_info.pixel_array
    if voi_lut:
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
        
    norm_img = np.uint8(norm_img*255)
    
    if get_roi:
        # get region of interest with PixelPaddingValue or threshold(median)
        if dcm_info.get('PixelPaddingValue'):
            roi_img = dcm_img!=dcm_info.PixelPaddingValue
        else:
            roi_img = dcm_img>np.median(dcm_img)+dcm_img.mean()*0.2

        label_img = split_island(roi_img)

        largest_island = np.where(np.where(label_img>1, 0, label_img)>0, 1, 0)
        horizontal_indicies = np.where(np.any(largest_island, axis=0))[0]
        vertical_indicies = np.where(np.any(largest_island, axis=1))[0]

        x1 = horizontal_indicies[0]
        x2 = horizontal_indicies[-1]
        y1 = vertical_indicies[0]
        y2 = vertical_indicies[-1]

        coor = [x1,x2,y1,y2]
        
        return norm_img, coor 
    
    return norm_img
