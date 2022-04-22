import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2


def img_to_negative(img):
    img = 255-img    # Use numpy Broadcasting
    return img

def threshold_img(img, threshold_val):
    img[img>threshold_val] = 1    # Numpy Conditional Indexing
    return img

def mirror_image(img):
    mirror_img = img[:,::-1]
    return mirror_img

def shrink_image_by_half(img):
    
    return img[0:-1:2, 0:-1:2] # Return Image with only alternate rows and cols pixel values

def zoom_by_twice(img):
    """Uses Pixel Replication for zooming, which increase the size of image by twice,
        Works for color images as well as grayscale images
    """
    if len(img.shape)==3:
        zoomed_image = np.zeros(shape = (img.shape[0]*2, img.shape[1]*2, img.shape[2]), dtype=np.int)        
    else:
        zoomed_image = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2))  
    n_rows, n_cols = zoomed_image.shape[0:2]
    zoomed_image[0:-1:2,0:-1:2] = img  # Copy original Image
    zoomed_image[0:n_rows+1:2,1:n_cols+1:2,] = img # Copy Columns First
    zoomed_image[1:n_rows+1:2,] = zoomed_image[0:n_rows+1:2,] # Copy Rows
    
    return zoomed_image



def threshold_color_image(img, threshold_val):
    img_copy = np.zeros(img.shape)
    img_copy[img>threshold_val] = 255
    return img_copy

def threshold_grayscale_image(img, threshold_val):
    binary_img = np.zeros(img.shape)
    binary_img[img>threshold_val] = 1
    return binary_img

def calc_hist(img):
    bins = np.arange(0,255, dtype=np.int)
    count = np.zeros(shape=(256), dtype=np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            count[img[i,j]] += 1
    
    return count, bins

def gamma_correction(img, gamma, c_factor=2):
    pixel_value = np.arange(0,256, dtype=np.int)
    corrected_value = c_factor*np.power(pixel_value, gamma)
    # Scale Back to 0-255 range value
    corrected_value = 255*corrected_value/np.max(corrected_value)
    corrected_value = corrected_value.astype(np.int)
    img_copy = np.empty(shape=img.shape, dtype=np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_copy[i,j] = corrected_value[pixel_value[img[i,j]]]
    
    return img_copy

def stretch_contrast(img):
    """ 
        Perfrom Histogram Equalisation to boost the contrast
    """
    r_min = np.min(img)
    r_max = np.max(img)
    m = 255/(r_max-r_min)
    img_copy = m*(img-r_min)

    return img_copy

def histogram_equalizer(img):
    count, bins = calc_hist(img)
    equalizer_values = np.zeros(shape=count.shape, dtype=np.float32)
    scale_value = img.shape[0]*img.shape[1]
    for i in bins:
        for j in range(i):
            equalizer_values[i] +=  count[j]
    equalizer_values = ((equalizer_values*255)/scale_value).astype(np.int)
    img_copy = np.empty(shape=img.shape, dtype=np.int)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_copy[i,j] = equalizer_values[img[i,j]]
    return img_copy


