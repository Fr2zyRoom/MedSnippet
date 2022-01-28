import numpy as np
import nibabel as nib


def load_nifti_file(nifti_file):
    ni_img = nib.load(nifti_file).get_fdata()
    ni_img = ni_img[:,:,::-1].transpose(2,1,0)
    return ni_img
