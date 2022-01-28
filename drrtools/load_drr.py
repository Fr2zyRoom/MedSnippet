import numpy as np

from cttools.load_ct import *


def calculate_ps(slices):
    """calculate pixel space"""
    #(x, y) pixel spacing
    ps = slices[0].PixelSpacing 
    # z pixel spacing
    ss = abs((slices[-1][0x020, 0x032][-1] - slices[0][0x020, 0x032][-1])/len(slices)) #(0020, 0032) Image Position (patient)
    # aspect ratio
    ax_aspect = ps[1]/ps[0] #horizontal
    sag_aspect = ss/ps[1] #sagittal
    cor_aspect = ss/ps[0] #frontal: z/x
    
    aspects = (ax_aspect, sag_aspect, cor_aspect)
    return aspects


def ct2drr_raysum(ct_img, cor_aspect):
    """convert ct slices(3d) to drr(ray-sum)"""
    drr = np.mean(ct_img, axis=1)[1:]
    drr_norm = normalize(drr, bit8=True)
    drr_shape = drr.shape
    return np.array(Image.fromarray(drr_norm).resize((drr_shape[1], round(drr_shape[0]*cor_aspect))).convert("L"))
    #return cv2.resize(drr, dsize=(drr_shape[1], round(drr_shape[0]*cor_aspect)),interpolation=cv2.INTER_LINEAR)


def load_drr_raysum(patient_folder_path):
    slices = load_ct_info(patient_folder_path)
    hu_images = dcm2ct(slices)
    ax_aspect, sag_aspect, cor_aspect = calculate_ps(slices)
    drr = ct2drr_raysum(hu_images, cor_aspect)
    return drr
