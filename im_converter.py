# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:13:00 2019

@author: imbroscb
"""

import numpy as np


def im_convert(tensor):
  image=tensor.clone().detach().numpy() 
  image=image.transpose(1,2,0)
  
  # denormalize the image (*std + mean)
  image=image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))
  
  # remove the third dimention (since it is a grayscale image)
  image=image[:,:,0]
  
  # transform it again in range (0,1)
  image=image.clip(0,1)
  
  return image
