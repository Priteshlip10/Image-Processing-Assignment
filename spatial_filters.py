import numpy as np
import scipy as sp

avg_filter = np.ones(shape=(3,3))*1/9

weighted_avg_filter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])/16

laplacian_filter_4 = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

laplacian_filter_8 = np.array([[-1, -1, -1],
                               [-1, 8,  -1],
                               [-1, -1, -1]])

sobel_horizontal_filter = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])

sobel_vertical_filter = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])

def correlation_filter(img, filter):
    """
    Uses Spatial correaltion function to perform filter using the mask
    This works best with the filter size with odd dimensions eg {(3*3), (5,5)}
    """
    f_row = filter.shape[0]//2
    f_col = filter.shape[1]//2
    n_rows, n_cols = img.shape
    corr_img = np.zeros(shape = (n_rows+f_row*2, n_cols + f_col*2))
    corr_img[f_row:n_rows+f_row, f_col:n_cols+f_col] = img    # Zero padded Image(Boundary filled with zeros)
    img_copy = np.empty(shape=(img.shape))
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            img_copy[i,j] = np.sum(corr_img[i: i+f_row*2+1, j: j+f_col*2+1]*filter)
    return img_copy

def median_filter(img, filter_size = 3):
    f_row = filter_size//2
    f_col = filter_size//2
    n_rows, n_cols = img.shape
    corr_img = np.zeros(shape = (n_rows+f_row*2, n_cols + f_col*2))
    corr_img[f_row:n_rows+f_row, f_col:n_cols+f_col] = img    # Zero padded Image(Boundary filled with zeros)
    img_copy = np.empty(shape=(img.shape))
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            img_copy[i,j] = np.median(corr_img[i: i+f_row*2+1, j: j+f_col*2+1])
    return img_copy