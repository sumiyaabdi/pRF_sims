import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(2020)

def load_data():
    """Load stimulus data and binarize
    
    Returns:
      stim (ndarray): stimulus data (frames, x-dim, y-dim)
      bin_stim (ndarray) : binarized stimulus data (frames, x-dim, y-dim)
    """
    
    # load data
    fname = 'images.npz'
    with np.load(fname) as dobj:
        data = dict(dobj)

    stim = data['ims'].T

    # get value of mean luminance 
    background = stim.mean().round()

    # binarize stimulus
    bin_stim = stim - background
    bin_stim[bin_stim != 0] = 1 
    
    return stim, bin_stim

def pRF_model(x0,y0,sig,im_shape=600,plot=True):
    """Creates 2D-gaussian population receptive field (pRF), with a centre
    x0, y0 and a spread of sig.

    Args:
      x0 (int) : centre of gaussian in x-direction
      y0 (int) : centre of gaussian in y-direction
      sig (int) : spread of the gaussian
      im_shape (int) : length of image
      plot (bool) : if True plot pRF model
      
    Returns:
      pRF (array) : 2D guassian pRF as defined in Dumoulin and Wandell (2008)
    """

    s = np.arange(1,im_shape+1,1) # number of pixels
    pRF = [] # initialize array of final pRF
    
    # calculate pRF value for each coordinate in visual space 
    for y in s:
        for x in s:
            g = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sig**2))
            pRF.append(g)

    pRF = np.array(pRF).reshape(im_shape,im_shape)
    
    if plot:
        plt.figure()
        plt.title('pRF model')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.imshow(pRF)
        plt.colorbar()
    
    return pRF

def predict_BOLD(pRF, plot=True):
    """ Predicts fMRI timeseries response from stimulus-driven pRF.
    Args:
      bin_stim (3d array) : array of binarized stimuli presented (time x x_axis x y_axis)
      pRF (2d array) : 2D guassian stimulus-driven pRF
      plot (bool) : if True plot predicted fMRI timeseries
      
    Returns:
      pred_bold (array) : predicted response at each timepoint of stim
    """    
    pred_bold = [] # intialize
    _, bin_stim = load_data()
    
    # predict response for each timepoint 
    for t in range(len(bin_stim)):
        b = np.sum(bin_stim[t]*pRF)
        pred_bold.append(b)
    
    pred_bold = np.array(pred_bold)
    
    if plot:
        plt.figure()
        plt.title('Predicted Response Timeseries')
        plt.xlabel('Time')
        plt.ylabel('Predicted BOLD')
        plt.plot(pred_bold)
        plt.show()
    
    return pred_bold