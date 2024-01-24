import numpy as np
import cv2
from skimage.measure import label, regionprops

def split_island(mask):
    """
    split mask into connected regions (starting with the largest area, in descending order)

    Parameters:
        mask (np.array)

    Return:
        labels (np.array) -- Labeled array, where all connected regions are assigned the same integer value
    """
    # Label the mask, each island gets a unique label
    lbl_mask = label(mask)

    # Get properties of labeled regions
    regions = regionprops(lbl_mask)

    # Sort regions by area in descending order and get their labels
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
    sorted_labels = [region.label for region in sorted_regions]

    # Create a new mask with sorted labels
    labels = np.zeros_like(lbl_mask)
    for new_val, label_val in enumerate(sorted_labels, 1):
        labels[lbl_mask == label_val] = new_val

    return labels


def check_left_or_right(lung_mask):
    if lung_mask.shape[1]%2 == 1:
        lung_mask = lung_mask[:, :lung_mask.shape[1]-1]
    left_mask, right_mask = np.hsplit(lung_mask,2) # cut half horizontally
    if np.sum(left_mask) < np.sum(right_mask): # left mask!
        return "left"
    else:
        return "right"


def split_lung(lung_mask):
    """
    split lung mask(left+right) to left mask and right mask
    
    Parameters:
        lung_mask (np.array) -- lung mask(left+right)
    
    Return:
        left_mask, right_mask (np.array, np.array)
    
    """
    tmp = label(lung_mask)
    if tmp.max() > 1: #left and right lungs
        mask1 = np.where(tmp==1, 255, 0).astype(np.uint8)
        mask2 = np.where(tmp==2, 255, 0).astype(np.uint8)
        if check_left_or_right(mask1)=="left":
            return mask1, mask2
        else:
            return mask2, mask1
    elif tmp.max() == 1:
        mask1 = np.where(tmp, 255, 0).astype(np.uint8)
        other_mask = np.zeros_like(mask1).astype(np.uint8)
        if check_left_or_right(mask1)=="left":
            return mask1, other_mask
        else:
            return other_mask, mask1
    elif tmp.max() == 0:
        mask = np.zeros_like(tmp).astype(np.uint8)
        return mask, mask
    else:
        mask = np.zeros_like(tmp).astype(np.uint8)
        print("Someting Wrong..")
        return mask, mask


def fill_hole(mask_img):
    """
    fill hole of the mask
    
    Parameters:
        mask_img (np.array) -- gray binary image
    
    Return:
        filled_mask (np.array) -- hole-filled mask image
    
    """
    img_flood_fill = mask_img.copy()
    h, w = mask_img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    img_flood_fill = img_flood_fill.astype(np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
    filled_mask = mask_img | img_flood_fill_inv
    return filled_mask
