import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2


def show_two_images(img1, img2, figsize = (24,12), cmap = 'gray', vmin0=0, vmax0=255, vmin1=1, vmax1=255, title0=None, title1=None):
    figure, axes = plt.subplots(nrows=1,ncols=2, figsize=figsize)
    
    # If vmin and vmax isn't given, the function automatically adjusts the contrast
    # of the image from minimum value to maximum value of the image
    axes[0].imshow(img1, vmin=vmin0, vmax=vmax0,cmap=cmap)
    if title0 is not None:
        axes[0].set_title(title0)
    axes[1].imshow(img2, vmin=vmin1, vmax=vmax1,cmap=cmap)
    if title1 is not None:
        axes[1].set_title(title1)
    
    plt.show()
    
def imshow_gray(img, vmin=0, vmax=255):
    plt.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
    
def plot_image_histogram(img, figsize=(8,8)):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)
    # An "interface" to matplotlib.axes.Axes.hist() method
    counts, bins = np.histogram(img, bins=256, range=(0,256))    # Calculate the histograms counts and bins
    n, bins, patches = axes.hist(bins[:-1],bins, weights=counts, color='#05aa00',
                                alpha=0.8, rwidth=0.5)
    n_cumsum = np.cumsum(counts)
    n_cumsum = np.max(counts)*n_cumsum/n_cumsum[-1]
    axes.plot(bins[:-1], n_cumsum, '--', linewidth=1.5, label=' Scaled Cumulative Histogram')
    axes.legend(loc='upper right')
    
    axes.grid(axis='y', alpha=0.5, )
    axes.set_xlabel('Pixel Value')
    axes.set_ylabel('Frequency')
    axes.set_title('Image Histogram')
    maxfreq = n.max()
    y_dim_min = int(maxfreq/10)/2
    # Set a clean upper y-axis limit.
    # axes.set_ylim(ymax=np.ceil(maxfreq + y_dim_min))
    axes.set_xlim(0, 255)
    plt.show()

    
def plot_image_with_histogram(img, figsize=(24,12),cmap='gray',alpha=0.8, bar_width = 0.5, vmin=0, vmax=255):

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=figsize)
    axes[0].imshow(img,vmin=vmin, vmax=vmax, cmap=cmap)
    
    # An "interface" to matplotlib.axes.Axes.hist() method
    counts, bins = np.histogram(img, bins=256, range=(0,256))
    n, bins, patches = axes[1].hist(bins[:-1],bins, weights=counts, color='#05aa00',
                                alpha=alpha, rwidth=bar_width)
    n_cumsum = np.cumsum(counts)
    n_cumsum = np.max(counts)*n_cumsum/n_cumsum[-1]
    axes[1].plot(bins[:-1], n_cumsum, '--', linewidth=1.5, label=' Scaled Cumulative Histogram')
    axes[1].legend(loc='upper right')
    
    axes[1].grid(axis='y', alpha=0.5)
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Image Histogram')
    # Set a clean upper y-axis limit.
    maxfreq = n.max()
    y_dim_min = int(maxfreq/10)/2
    axes[1].set_ylim(ymax=np.ceil(maxfreq + y_dim_min))
    axes[1].set_xlim(0, 255)
    
    
    plt.show()
    
    

    
    