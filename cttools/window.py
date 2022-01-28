import numpy as np
import pydicom
from util.util import *


class CT_Windowing:
    """
    CT image windowing : WL_Window Level, WW_Window Width
    """
   
    def __init__(self, mode='custom', custom_window=None, norm=False):
        """
        Parameters:
            mode (str)(WL|WW) -- 
                'abdomen'(60|400) , 'bone'(300|1500), 'brain'(40|80), 
                'chest'(40|400), 'lung'(-700|1500), 'custom'(WL|WW)
            custom_window (list or tuple) -- 
                if mode == 'custom', set custom_window(WL, WW)
            norm (bool) -- normalize to uint8 (0~255)

        """
        option = ['abdomen' , 'bone', 'brain', 'chest', 'lung', 'custom']
        assert mode in option, "Wrong mode: Enter \'abdomen\' , \'bone\', \
                                \'brain\', \'chest\', \'lung\', \'custom\'"
        
        self.mode = "window_" + mode
        if custom_window is not None:
            self.w_level = custom_window[0]
            self.w_width = custom_window[1]
            
        self.norm = norm
        
    def windowing(self):
        self.w_min = self.w_level - (self.w_width / 2)
        self.w_max = self.w_level + (self.w_width / 2)
        window_image = self.img.copy()
        window_image = np.clip(window_image, self.w_min, self.w_max)
        
        if self.norm:
            window_image = np.uint8(((window_image - self.w_min) / \
                                     (self.w_max - self.w_min)) * 255.0)
        return window_image
        
    def window_abdomen(self):
        self.w_level = 60
        self.w_width = 400
        
        return self.windowing()
        
    def window_bone(self):
        self.w_level = 300
        self.w_width = 1500
        
        return self.windowing()
    
    def window_brain(self):
        self.w_level = 40
        self.w_width = 80
        
        return self.windowing()
    
    def window_chest(self):
        self.w_level = 40
        self.w_width = 400
        
        return self.windowing()
        
    def window_lung(self):
        #SNUH version
        self.w_level = -700
        self.w_width = 1500
        #print('lung')
        return self.windowing()
        
    def window_custom(self):
        return self.windowing()
    
    def __call__(self, hu_img):
        self.img = hu_img
        self.opt = getattr(self, self.mode, lambda:'custom')
        return self.opt()  

    
def windowing(img, window):
    """ct windowing
    Parameters:
        img (np.array) -- ct image
        window(list or tuple) -- [window level, window width]
        
    Return:
        window_image (np.array) -- window setting image
    
    """
    w_level, w_width = window
    w_min = w_level - (w_width / 2)
    w_max = w_level + (w_width / 2)
    window_image = img.copy()
    window_image = np.clip(window_image, w_min, w_max)

    return window_image
