import numpy as np
import scipy as sp

def filter_image_in_frequency(img, h_uv):
    # Preprocessing: To 
    u, v = np.indices(img.shape)
    shift_mat = np.power(-1, u+v)     # Matrix of (-1)^(x+y)
    # Multipy the img by (-1)^(x+y)
    shifted_img = img*shift_mat
    
    # Calculate DFT of the image
    dft_img = sp.fft.fft2(shifted_img, s=h_uv.shape)
    # Perform Multiplication of filter and DFT of the image
    g_uv = dft_img*h_uv
    # Calculate the Inverse Fourier Tranform of the result
    inv_fft_g = sp.fft.ifft2(g_uv)
    # Only Take the Real Values
    inv_fft_g_real = np.real(inv_fft_g)
    # Crop the image to the original image size
    result = inv_fft_g_real[:img.shape[0], :img.shape[1]]
    # Undo the shifting
    return result*shift_mat

def ideal_freq_low_pass_filter(size, cutoff_freq):
    ideal_lp_filter = np.zeros(shape=size, dtype = np.int)
    M_div2, N_div2 = size[0]/2, size[1]/2
    u, v = np.indices(size, dtype=np.int)

    response = np.sqrt(np.power(u-M_div2, 2) + np.power(v-N_div2,2))
    ideal_lp_filter[response<cutoff_freq] = 1
    return ideal_lp_filter

def ideal_freq_high_pass_filter(size, cutoff_freq):
    ideal_hp_filter = 1-ideal_freq_low_pass_filter(size, cutoff_freq)
    return ideal_hp_filter
    
def gaussian_low_pass_filter(size, cutoff_freq):
    u, v = np.indices(size, dtype=np.int)
    M_div2, N_div2 = size[0]/2, size[1]/2
    lp_filter = np.exp((-2/np.power(cutoff_freq,2))*(np.power(u-M_div2, 2) + np.power(v-N_div2,2)))
    return lp_filter
    
def gaussian_high_pass_filter(size, cutoff_freq):
    gauss_hp_filter = 1- gaussian_low_pass_filter(size, cutoff_freq)
    return gauss_hp_filter

def butterworth_low_pass_filter(size, cutoff_freq, order):
    u, v = np.indices(size, dtype=np.int)
    M_div2, N_div2 = size[0]/2, size[1]/2
    
    d_uv2 = np.power(u-M_div2, 2) + np.power(v-N_div2,2)   # Distance Squared
    d_uv_div_d0 = d_uv2/np.power(cutoff_freq,2)            # Distance Squared/D0^2
    divisor = 1 + np.power(d_uv_div_d0, order)             # 1 + (Distance Squared/D0^2)^n
    
    return 1/divisor                                       # 1/(1 + (Distance Squared/D0^2)^n)

def butterworth_high_pass_filter(size, cutoff_freq, order):
    butterworth_lp_filter = 1 - butterworth_low_pass_filter(size, cutoff_freq, order)
    return butterworth_lp_filter

