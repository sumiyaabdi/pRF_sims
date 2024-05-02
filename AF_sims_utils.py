import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
opj=os.path.join
import seaborn as sns
from yaml import load_all
opj=os.path.join
import glob
import scipy
import scipy.signal as signal
from scipy.stats import norm
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from copy import deepcopy
import yaml
from ipywidgets import *

sublist=['sub-001','sub-002','sub-004','sub-006','sub-007','sub-009']

with open('settings.yml', 'r') as f:
    settings=yaml.safe_load(f)

Z = norm.ppf

def d_prime(hits, misses, fas, crs):
    """
    calculate d' from hits(tp), misses(fn), false
    alarms (fp), and correct rejections (tn)

    returns: d_prime
    """

    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    hit_rate = hits / (hits + misses)
    fa_rate = fas / (fas + crs)

    # avoid d' infinity
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    elif hit_rate == 0:
        hit_rate = half_hit

    if fa_rate == 1:
        fa_rate = 1 - half_fa
    elif fa_rate == 0:
        fa_rate = half_fa

    d_prime = Z(hit_rate) - Z(fa_rate)
    c = -(Z(hit_rate) + Z(fa_rate)) / 2
    #     print(f'Hit rate: \t {hit_rate} \nFalse Alarm rate: {fa_rate}')
    return d_prime, c


def load_env(sub, op, hemi, ses=1,nruns=8,proj='pRF_attm'):
    env = dict()

    env['sub']=sub
    env['op'] = op
    env['ses'] =ses
    env['n_runs']=n_runs
    env['proj']=proj
    env['hemi']=hemi

    if env == 'mac':
        env['root'] = '/Users/sumiyaabdirashid/Spinoza-mnt/projects'
    elif env == 'spinoza':
        env['root'] = '/data1/projects/dumoulinlab/Lab_members/Sumiya/projects'

    env['pwd'] = f'{root}/{proj}/analysis/{sub}'
    env['fprep_output'] = f'{root}/{proj}/derivatives/fmriprep/{sub}/{ses}/func'
    env['fsurf_output']=f'{root}/{proj}/derivatives/freesurfer/{sub}'
    env['logs']=f'{root}/{proj}/sourcedata/{sub}/{ses}/prf'
    env['npys'] = f'{pwd}/npy'
    env['fits']=f'{pwd}/prf_fits'

    return env


def pRF_model(x0,y0,sig,im_shape=600,plot=True, add_labels=False,cmap=None,bin=False):
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
    s = np.arange(-int(im_shape/2),int(im_shape/2),1) # number of pixels
#     s = np.arange(1,im_shape+1,1) # number of pixels
    pRF = [] # initialize array of final pRF
    
    # calculate pRF value for each coordinate in visual space 
    for y in s:
        for x in s:
            g = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sig**2))
            if bin:
              g=np.sqrt((x-x0)**2 + (y-y0)**2) <= sig # binary pRF instead of probability
            pRF.append(g)

    pRF = np.array(pRF).reshape(im_shape,im_shape)
    
    if plot:
        fig,ax=plt.subplots(1,1,figsize=(4,4))

        ax.set_yticks([])
        ax.set_xticks([])

        ax.pcolormesh(pRF,cmap=cmap)
        ax.set_aspect('equal')
        if add_labels:
            ax.set_title('pRF model')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.colorbar()
    
    return pRF

def predict_bold(bin_stim,pRF,plot=True):
    """
    Predicts fMRI timeseries response from stimulus-driven pRF.
    Args:
      bin_stim (3d array) : array of binarized stimuli presented (time x x_axis x y_axis)
      pRF (2d array) : 2D guassian stimulus-driven pRF
      plot (bool) : if True plot predicted fMRI timeseries

    Returns:
      pred_bold (array) : predicted response at each timepoint of stim
    """
    pred_bold = [] # intialize

    # predict response for each timepoint
    for t in range(len(bin_stim)):
        b = np.sum(bin_stim[t]*pRF)
        pred_bold.append(b)

    pred_bold = np.array(pred_bold)/60

    if plot:
        plt.figure()
        plt.title('Predicted Response Timeseries')
        plt.xlabel('Time')
        plt.ylabel('Predicted BOLD')
        plt.plot(pred_bold)
        plt.show()

    return pred_bold


def convolve_stimulus_dm(stimulus, hrf):
    """convolve_stimulus_dm
    convolve_stimulus_dm convolves an N-D (N>=2) stimulus array with an hrf
    Parameters
    ----------
    stimulus : numpy.ndarray, N-D (N>=2) 
        stimulus experimental design, with the final dimension being time
    hrf : numpy.ndarray, 1D
        contains kernel for convolution
    """
    hrf_shape = np.ones(len(stimulus.shape), dtype=np.int)
    hrf_shape[-1] = hrf.shape[-1]

    return signal.fftconvolve(stimulus, hrf.reshape(hrf_shape), mode='full', axes=(-1))[..., :stimulus.shape[-1]]

def gauss_interaction(muSD, sigSD, muAF,sigAF):
    """Gaussian interaction model. Multiplying two gaussians to return the eccentricity 
    (mean) and size (standard deviation, sigma) of the resulting gaussian. Used in this 
    context as the 'Attention Field Model', to predict pRF position and size changes 
    resulting from attention. 
    SD = stimulus driven, AF = attention field, AD = attention driven.

    Args:
        muSD (float): eccentricity of stimulus driven pRF
        sigSD (float): size of stimulus driven pRF
        muAF (float): eccentricity of attention field
        sigAF (float): size of attention field

    Returns:
        muAD: eccentricity of attention driven pRF
        sigAD: size of attention driven pRF
        
    """
    
    muAD = ((muAF*sigSD**2) + (muSD*sigAF**2))/(sigAF**2+sigSD**2)
    sigAD = np.sqrt((sigAF**2 * sigSD**2)/(sigAF**2 + sigSD**2))
    
    return muAD,sigAD

def gaus_div_pos(x1,x2,sig2,sig3):
    """
    Rearranged from gaussian multiplication. Returns position of gaussian 
    which results from dividing two gaussians.
    """
    return (x1 + (x1*sig3**2 - x2*sig3**2)/sig2**2)


def AFsize_fromsize(sigSD,sigAD):
    """ Derived from the gaussian interaction model to return the size of 
    the attention field (sigma of the attention field gaussian)

    Args:
        sigSD (float): size (sigma) of the stimulus driven pRF gaussian
        sigAD (float): size (sigma) of the attention driven pRF gaussian

    Returns:
        float : size (sigma) of the attention field
    """
    return np.sqrt((sigSD**2)/((sigSD**2/sigAD**2)-1))

def calc_sigSD(sigAF,sigAD):
    """ Derived from the gaussian interaction model to return the size of 
    the attention field (sigma of the attention field gaussian).
    If sigAF > sigAD returns np.nan

    Args:
        sigSD (float): size (sigma) of the stimulus driven pRF gaussian
        sigAD (float): size (sigma) of the attention driven pRF gaussian

    Returns:
        float : size (sigma) of the attention field
    """
    sigAF=np.asarray(sigAF)
    sigAD=np.asarray(sigAD)

    assert all(sigAF > sigAD), 'make sure arg1 > arg2'
    assert all(sigAD != 0), 'make sure arg2 != 0'

    return np.sqrt((sigAF**2)/((sigAF**2/sigAD**2)-1))

def AFsize_frompos(sigSD,muAF,muAD,muSD):
    """_summary_

    Args:
        sigSD (float): size (sigma) of the stimulus driven pRF gaussian
        muAF (float): eccentricity of the attention field
        muAD (float): eccentricity of the attention driven pRF
        muSD (float): eccentricity of the stimulus driven pRF

    Returns:
        float: size (sigma) of the attention field
    """
    sigSD=np.asarray(sigSD)
    muAF=np.asarray(muAF)
    muAD=np.asarray(muAD)
    muSD=np.asarray(muSD)

    return np.sqrt((sigSD**2)*(muAF-muAD)/(muAD-muSD))


def load_exp(sub,experiment='attn'):
    """Loads experiments when utils page is loaded so individual functions don't have to.

    Args:
        sub (_type_): _description_
        experiment (_type_): _description_
    """

    if experiment == 'attn':
        settings=opj(os.environ['DIR_DATA_DERIV'],'analysis/attn_analysis/attn_analysis/settings.yml')
    elif experiment == 'std':
        settings=opj(os.environ['DIR_DATA_DERIV'],'analysis/attn_analysis/attn_analysis/settings_stdprf.yml')
    
    from attn_analysis.master_analysis import Analyse_Runs

    exp=Analyse_Runs(settings,sub)

    return exp

def load_allexp(sublist,hemilist=['L','R'],condlist=['attn','std']):
    master_exp={}
    
    for exp in condlist:
        for sub in sublist:
            for h in hemilist:
                dc={}
                dc={exp : { sub : {h : load_exp(sub,exp)}}}

                master_exp.update(dc)
    return master_exp

# master_exp = load_allexp(sublist)

def plot_vert_fromdf(df,exp=None,overlay=False,savefn=None):
    """Not yet implemented

    Args:
        df (_type_): _description_
        v (_type_): _description_

    Returns:
        _type_: _description_
    """

    fig=plot_vert_attn(df.subject,df.hemi,df.vertex,overlay_std=overlay)
    if not overlay:
        _ = plot_vert_std(df.subject,df.hemi,df.vertex)

    if savefn:
        fig.savefig(savefn)
        plt.close()
    return None

def plot_vert_std(sub,h,v):
    """Plot single vertex raw tc and standard pRF fit

    Args:
        sub (str): Subject number e.g.'sub-001'
        h (str): Hemisphere['L','R']
        v (int): Vertex number
    """

    exp=load_exp(sub,'std')
    fit_condition='stdprf'
    conds=[f'task-2R_hemi-{h}']


    raw_tc= np.array([np.load(f'{exp.clean_ts}/avgbld_{cond}.npy') for cond in conds])
    params=np.array([np.load(f'{exp.fits}/iter-params_{fit_condition}_{cond}.npy') for cond in conds])

    pred_tc = exp.gg_full.return_prediction(*params[0,v,:5]).T[:,0] # all_params: pooled, attnL, attnS

    fig, axs = plt.subplots(1,1,figsize=(12,4))
    x=np.arange(0,pred_tc.shape[0])
    axs.scatter(x,raw_tc[0,v], marker='.',alpha=0.2)    
    axs.plot(x,pred_tc)

    txt=f'params: {[round(i,3) for i in params[0,v]]}'

    axs.text(0.5,-0.15, txt, size=8, ha="center", transform=axs.transAxes)
    axs.set_title(f'{exp.sub} V {v}', fontsize=10)

    return axs


def plot_vert_attn(sub,h,v,experiment='attn',overlay_std=False):
    """Plot single vertex timecourse and fits for various conditions

    Args:
        sub (str): Subject number e.g.'sub-001'
        h (str): Hemisphere['L','R']
        v (int): Vertex number
        experiments (str, optional): 'attn','std', or 'all'. Defaults to 'all'.
    """

    exp=load_exp(sub,experiment)

    if experiment == 'attn':
        fit_condition='TR-194'
        conds=[f'attn-L_hemi-{h}', f'attn-S_hemi-{h}']
    elif experiment == 'std':
        fit_condition='stdprf'
        conds=[f'task-2R_hemi-{h}']
    
    h=h.upper()

    raw_tc= np.array([np.load(f'{exp.clean_ts}/avgbld_{cond}.npy') for cond in conds])
    params=np.array([np.load(f'{exp.fits}/iter-params_{fit_condition}_{cond}.npy') for cond in conds])
    eccs=np.array([np.load(f'{exp.derivs}/ecc_{fit_condition}_{cond}.npy') for cond in conds])

    # add on-off as regressor
    onoff=np.hstack((np.zeros(14),np.ones(260-14-21),np.zeros(21)))
    tskon = np.hstack((np.zeros(14),1,np.zeros(245)))
    tskoff = np.hstack((np.zeros(239),1,np.zeros(20)))
    onoff=scipy.signal.convolve(onoff,exp.hrf[0])[:260]
    tskon=scipy.signal.convolve(tskon,exp.hrf[0])[:260]
    tskoff=scipy.signal.convolve(tskoff,exp.hrf[0])[:260]


    p=np.array([np.load(opj(exp.glm_dir,f'full_{cond}_{fit_condition}_glm-beta.npy')) for cond in conds]) 

    pred_tc_S = exp.gg_full.return_prediction(*params[1,v,:5]).T[:,0] # all_params: pooled, attnL, attnS
    pred_tc_L = exp.gg_full.return_prediction(*params[0,v,:5]).T[:,0]
    ts_L=raw_tc[0,v,:]
    ts_S=raw_tc[1,v,:]
    
    yy=p[0,v,0]+ p[0,v,1]*pred_tc_L + p[0,v,2]*onoff + p[0,v,3]*tskon + p[0,v,4]*tskoff 
    yy2=p[1,v,0]+ p[1,v,1]*pred_tc_S + p[1,v,2]*onoff + p[1,v,3]*tskon + p[1,v,4]*tskoff 

    rsq_l = 1-(np.sum((ts_L-yy)**2) / np.sum((ts_L-np.mean(ts_L))**2))
    rsq_s = 1-(np.sum((ts_S-yy2)**2) / np.sum((ts_S-np.mean(ts_S))**2))

    txt=f'L: ecc {eccs[0,v]:.2f}, size {params[0,v,2]:.2f}, amp {params[0,v,3]:.3f},r2_pRF {params[0,v,-1]:.2f}, r2_GLM {rsq_l:.2f} \nS: ecc {eccs[1,v]:.2f}, size {params[1,v,2]:.2f}, amp {params[0,v,3]:.3f}, r2_pRF {params[1,v,-1]:.2f}, r2_GLM {rsq_s:.2f}'

    x = np.linspace(0,260,260, dtype=int)

    fig, axs = plt.subplots(1,1,figsize=(12,4))
    axs.scatter(x,ts_L-yy[0], marker='.',alpha=0.2)    
    axs.plot(x,yy-yy[0],label='attn L')
    axs.scatter(x,ts_S-yy2[0], marker='.',alpha=0.2)
    axs.plot(x,yy2-yy2[0],label='attn S')   
    axs.xaxis.set_visible(False)
    axs.text(0.5,-0.1, txt, size=8, ha="center", transform=axs.transAxes)
    axs.legend()
    axs.set_title(f'{exp.sub} V {v}', fontsize=10)

    if overlay_std:
        std=load_exp(sub,'std')
        fit_condition='stdprf'
        conds=[f'task-2R_hemi-{h}']

        raw_tc= np.array([np.load(f'{std.clean_ts}/avgbld_{cond}.npy') for cond in conds])
        raw_tc=np.concatenate((np.zeros((raw_tc.shape[0],raw_tc.shape[1],14)),raw_tc,np.zeros((raw_tc.shape[0],raw_tc.shape[1],21))),axis=2)
        params=np.array([np.load(f'{std.fits}/iter-params_{fit_condition}_{cond}.npy') for cond in conds])

        pred_tc = exp.gg_full.return_prediction(*params[0,v,:5]).T[:,0] # all_params: pooled, attnL, attnS

        x=np.arange(0,pred_tc.shape[0])
        # axs.scatter(x,raw_tc[0,v]-params[0,v,4], marker='.',alpha=0.1,color='gray',zorder=0)    
        axs.plot(x,pred_tc-params[0,v,4],zorder=0,c='gray',linestyle='--',alpha=0.5,label='std')

        txt=f'params: {[round(i,3) for i in params[0,v]]}'

        axs.text(0.5,-0.15, txt, size=8, ha="center", transform=axs.transAxes)
        axs.set_title(f'{std.sub} V {v}', fontsize=10)

    # return fig


def get_blend(Vertex, threshold=0, brightness=0.2,
                    contrast=0.2, smooth=1):
    """Blend the data with a curvature map depending on a transparency map.

    Vertex objects cannot use transparency as Volume objects. This method
    is a hack to mimic the transparency of Volume objects, blending the
    Vertex data with a curvature map. This method returns a VertexRGB
    object, and the colormap parameters (vmin, vmax, cmap, ...) of the
    original Vertex object cannot be changed later on.
    Parameters
    ----------
    Vertex : Vertex you want to plot
        Vertex you want to plot.
    alpha : array of shape (n_vertices, )
        Transparency map.
    threshold : float
        Threshold for the curvature map.
    brightness : float
        Brightness of the curvature map.
    contrast : float
        Contrast of the curvature map.
    smooth : float
        Smoothness of the curvature map.

    Returns
    -------
    blended : VertexRGB object
        The original map blended with a curvature map.
    """
    
    # alpha = (~np.isnan(Vertex.data)).astype("float")
    alpha = threshold.astype("float")    
    curvature = cx.db.get_surfinfo(Vertex.subject).data
    curvature = curvature.astype("float")
    curvature = curvature * contrast #+ brightness
    curvature_raw = cx.Vertex(curvature, subject=Vertex.subject, vmin=0, vmax=1, cmap='gray').raw
    alpha = np.clip(alpha, 0, 1)  
    blended = deepcopy(Vertex.raw)  # copy because VertexRGB.raw returns self
    blended.red.data = blended.red.data * alpha + (1 - alpha) * curvature_raw.red.data
    blended.green.data = blended.green.data * alpha + (1 - alpha) * curvature_raw.green.data
    blended.blue.data = blended.blue.data * alpha + (1 - alpha) * curvature_raw.blue.data
    blended.red.data = blended.red.data.astype("uint8")
    blended.green.data = blended.green.data.astype("uint8")
    blended.blue.data = blended.blue.data.astype("uint8")

    return blended

def mask_verts(verts,mask,val=np.nan):
    return np.where(mask,verts,val)

def list_rois(sub,hemi,folder=''):
    """
    Returns list of all custom ROIs in `freesurfer/subject/label` folder.
    Drops the 'custom' in the file name.
    """
    fsurf=opj(os.environ['SUBJECTS_DIR'],sub)
    return [i.split('.')[-2] for i in glob.glob(opj(fsurf,'label',folder,f'{hemi.lower()}*custom.label'))]


def return_binned(data,
                  binon,
                  bins=None,
                  start=None,
                  stop=None,
                  binsize=None,
                  nbins=None,
                  func=None):
    """
    Bins data according to provided bins. Can provide either a list of bins, 
    or start, stop and bin sizes or nbins.

    Equivalent to: scipy.stats.binned_statistic()
    
    data (1D array) : values to be averaged within a bin
    binon (1D array) : independent values to bin-on (separate from values being averaged)
    bins (list) : list of bins
    start (float) :
    stop (float) :
    binsize (float) :
    nbins (int) :
    func (func) : optional function to apply to bins
    
    Returns
    binned_data (arr)
    """
    
    data=np.asarray(data)
    binon=np.asarray(binon)
    
    assert data.ndim,'data must be 1D'
    
    if not bins and binsize:
        assert start and stop, 'provid list of bins or start and stop values'
        bins=np.arange(start,stop,binsize)
    elif not bins and nbins:
        assert start and stop, 'provid list of bins or start and stop values'
        bins=np.linspace(start,stop,nbins)
    else:
        assert bins, 'provide bins'
        
    if func:
        binned=[func(data[(binon >= bins[i])&(binon<bins[i+1])]) for i in range(len(bins)-1)]
    else:
        binned=[np.nanmean(data[(binon >= bins[i])&(binon<bins[i+1])]) for i in range(len(bins)-1)]

    return np.asarray(binned)

class Gauss():
    def __init__(self):
        pass

    def fit_line(self,b,mu,sig):
        self.mu=mu
        self.sig=sig
        self.bins=b
        self.line = ((1 / (np.sqrt(2 * np.pi) * self.sig)) * \
                    np.exp(-0.5 * (1 / self.sig * (self.bins - self.mu))**2))
        return self.line

    def plot(self,b):
        if not hasattr(self,'line'):
            self.line=self.fit_line(b)
        plt.plot(self.bins,self.line)

    def mul(self,mu2,sig2):
        """
        This function multiplies the current Gaussian by another (specifified by
        mu2, sig2) and returns the resulting gaussian (mu3,sig3)
        """
        mu3=((self.mu*sig2**2) + (mu2*self.sig**2))/(self.sig**2+sig2**2)
        sig3 = np.sqrt((self.sig**2 * sig2**2)/(self.sig**2 + sig2**2))
        print(mu3, sig3)
        return mu3,sig3
    
def cutTC(fpath,cuts,axis=1):
    """Cut timecourse to f[:,cuts[0],cuts[1]]

    Args:
        fpath (str): filepath
        cuts (tuple): start and end index of timecourse, used to cut timecourse
    """

    assert os.path.exists(fpath), f'File does not exist: {fpath}'
    assert fpath[-4:] == '.npy', f'File is not .npy, {fpath}'
    assert len(cuts) == 2, f'cuts must len 2, {cuts}'
    assert isinstance(cuts[0],int) & isinstance(cuts[1],int), f'cuts are not intergers, {cuts}'
    assert axis == 0 or axis == 1, f'axis must be 0 or 1, instead it is {axis}'

    tc=np.load(fpath,allow_pickle=True)
    if axis == 1:
        cuttc=tc[:,cuts[0]:cuts[1]]
    else:
        cuttc=tc[cuts[0]:cuts[1],:] 
    return cuttc
    
def detrendTC(tc,standardize='psc'):
    hp_set=np.ones(tc.shape[1])
    detrTC = signal.clean(tc.T,
                        detrend=True,
                        standardize=standardize,
                        confounds=hp_set).T
    return detrTC

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def pix2deg(pixels, scrWidthCm, dist, scrSizePix=[270, 270]):
    """Convert size in pixels to size in degrees for a given Monitor object""" 

    cmSize = pixels * float(scrWidthCm) / scrSizePix[0]
    return old_div(cmSize, (dist * 0.017455))

def deg2pix(degrees, scrWidthCm, dist, scrSizePix=[270, 270]):
    """Convert size in degrees to size in pixels for a given pRF object"""

    cmSize = np.array(degrees) * dist * 0.017455
    return round(cmSize * scrSizePix[0] / float(scrWidthCm), 0)

def del_h5(sub):
    sub_path=opj(os.environ['DIR_DATA_DERIV'],f'eyetracking/{sub}_training')
    dirs=[dir for dir in os.listdir(sub_path) if 'yesno' in dir]
    h5s=[opj(sub_path,dirs[i],'eye.h5') for i in range(len(dirs))]
    h5s=[h5 for h5 in h5s if os.path.exists(h5)]
    [os.remove(h5) for h5 in h5s]

def analyse_edf(edf_file,plot_gaze=True,plot_gaze_psc=True,verbose=False):
    eye_ = dataset.ParseEyetrackerFile(edf_file,
                                        subject='sub-001',
                                        use_bids=True,
                                        verbose=verbose,
                                        nr_vols=260,
                                        TR=1.5
                                        )
    if plot_gaze or plot_gaze_psc:
        if not os.path.exists(opj(opd(opd(edf_file)),'figs')):
            os.mkdir(opj(opd(opd(edf_file)),'figs'))
    
    if plot_gaze:
        # gaze x/y
        df_gaze = eye_.df_space_func.copy()
        input_l = [df_gaze[f"gaze_{i}_int"].values for i in ["x","y"]]
        avg = [float(input_l[i].mean()) for i in range(len(input_l))]
        std = [float(input_l[i].std()) for i in range(len(input_l))]
        plotting.LazyPlot(
            input_l,
            line_width=2,
            figsize=(24,5),
            color=["#1B9E77","#D95F02"],
            labels=[f"gaze {i} (M={round(avg[ix],2)}; SD={round(std[ix],2)}px)" for ix,i in enumerate(["x","y"])],
            x_label="volumes (TR-space)",
            y_label="position (pixels)",
            add_hline={"pos": avg},
            title="gaze position during run (raw pixels)",
            save_as=opj(opd(opd(edf_file)),'figs',edf_file.split('/')[-1].split('.')[0]+'_gaze-xy.pdf')
        )
    if plot_gaze_psc:
        # gaze x/y percent signal changed
        input_l = [df_gaze[f"gaze_{i}_int_psc"].values for i in ["x","y"]]
        avg = [float(input_l[i].mean()) for i in range(len(input_l))]
        std = [float(input_l[i].std()) for i in range(len(input_l))]
        plotting.LazyPlot(
            input_l,
            line_width=2,
            figsize=(24,5),
            color=["#1B9E77","#D95F02"],
            labels=[f"gaze {i} (M={round(avg[ix],2)}; SD={round(std[ix],2)}px)" for ix,i in enumerate(["x","y"])],
            x_label="volumes (TR-space)",
            y_label="position (%change pixels)",
            add_hline=0,
            title="gaze position during run (percent change)",
            save_as=opj(opd(opd(edf_file)),'figs',edf_file.split('/')[-1].split('.')[0]+'_gaze-xy-psc.pdf')
        )
    
    return eye_

def analyse_eyetrack(subs,del_files=False,task='yesno'):
    TR = 1.5

    for sub in subs:
        print(f'Running {sub}')

        if del_files:
            print(f'Deleting existing h5 files for {sub}')
            del_h5(sub)

        edf_files=[]
        sub_path=opj(os.environ['DIR_DATA_DERIV'],f'eyetracking/{sub}_training')
        dirs=[dir for dir in os.listdir(sub_path) if task in dir]
        edfs=[opj(sub_path,dirs[i],dirs[i].rsplit('_',1)[0]+'.edf') for i in range(len(dirs))]
        edfs=[edf for edf in edfs if os.path.exists(edf)]

        for edf_file in edfs:
            print(edf_file)
            e=analyse_edf(edf_file)


def load_subs(subs):
    fsubstr='summary_1rowvert.csv'
    # fn=f'{os.environ["DIR_DATA_HOME"]}/derivatives/analysis/backup/08032023_data/sub-006/ses-1/sub-006_{fsubstr}'
    fn=f'{os.environ["DIR_DATA_HOME"]}/derivatives/analysis/sub-006/ses-1/sub-006_{fsubstr}'

    setfn=opj(os.environ['ANALYSIS'],'attn_analysis/settings.yml')
    with open(setfn, 'r') as f:
        settings=yaml.safe_load(f)

    masterdf=pd.DataFrame()

    for sub in subs:
        f=fn.replace('sub-006',sub)
        # print(sub, ' ',os.path.exists(f))
        df=pd.read_csv(f,index_col='Unnamed: 0')
        for h in ['L','R']:
            epi=np.load(opj(f.rsplit('/',1)[0],f'prf_derivs/meanepi_total_hemi-{h}.npy'),allow_pickle=True)
            ids=np.where(epi <0.1)[0]
            df.drop(df[(df.vertex.isin(ids))&(df.hemi == h)].index,inplace=True)   
            masterdf = pd.concat([masterdf,df],ignore_index=True)
    masterdf.drop(masterdf[masterdf.std_r2 < 0.1].index,inplace=True)
    masterdf.drop(masterdf[masterdf.attnL_ecc > 5].index,inplace=True)
    masterdf.drop(masterdf[masterdf.attnS_ecc > 5].index,inplace=True)
    masterdf.drop(masterdf[masterdf.std_size <= 0].index, inplace=True)
    # masterdf.drop(masterdf.iloc[np.where(masterdf[['std_prf_ecc','std_size','attnS_ecc','attnS_prfsize','attnL_ecc','attnL_prfsize']] > 20 )[0]].index,inplace=True)
    masterdf.loc[(masterdf.roi == 'v4'),'roi']='hv4'
    masterdf.loc[(masterdf.roi == 'lowerIPS'),'roi']='LowerIPS'
    masterdf.loc[(masterdf.roi == 'upperIPS'),'roi']='UpperIPS'
    masterdf.drop(masterdf[masterdf.std_prf_ecc >= 5].index,inplace=True)
    masterdf.drop_duplicates(inplace=True)
    
    return masterdf            

def ecc_log_plot(par,
                 dfi,
                 sub=None,
                 per_sub=None,
                 roi=None,
                 per_roi=None,
                 legend='side',
                 ylim=None,
                 log=True, 
                 savefn=None):
    """
    Examples:
        ecc_log_plot('prfsize',per_sub=True,roi='LO',ylim=[-0.5,0.5],log=False)
        ecc_log_plot('onoff',per_roi=['v1','v2','v3'])

    """
    # dfi=masterdf
    c=sns.color_palette('nipy_spectral',20)
    
    if sub:
        dfi=dfi[dfi.subject==sub]
    if roi:
        dfi=dfi[dfi.roi==roi]
        
        
    b=np.logspace(0,5,25,base=2,endpoint=True)
    div=b.max()/5
    b=b/div
    x=[(b[i]+b[i+1])/2 for i in range(len(b)-1)]
    
    if not log:
        b=np.linspace(0,5,25,endpoint=True)
        div=b.max()/5
        x=[(b[i]+b[i+1])/2 for i in range(len(b)-1)]
        
    
    if per_sub and per_roi:
        return "Use different func"

    if per_sub:
        df_iter='subject'
        df_ls = per_sub if isinstance(per_sub,list) else dfi.subject.unique() 
        title=roi
        c=sns.color_palette('Set2',len(dfi.subject.unique()))
        niether=False

    elif isinstance(per_roi,(list,tuple,np.ndarray)) or per_roi ==True:
        df_iter='roi'
        df_ls = per_roi if isinstance(per_roi,list) else dfi.roi.unique()
        title=sub
        c=sns.color_palette('nipy_spectral',len(dfi.roi.unique()))
        niether=False
    else:
        df_iter=dfi.columns
        df_ls=['']
        niether=True
        
        
    fig,ax=plt.subplots(1,1, figsize=(12,6))
    
    ymin=[]
    ymax=[]
    
    for i,it in enumerate(df_ls):
        if niether:
            df=dfi[df_iter]
        else:
            df=dfi[dfi[df_iter]==it]
        binned_L=np.asarray([df[(df.std_prf_ecc > b[i])&(df.std_prf_ecc < b[i+1])][f'attnL_{par}'].mean() for i in range(len(b)-1)])
        L_sem=np.asarray([scipy.stats.sem(df[(df.std_prf_ecc > b[i])&
                                  (df.std_prf_ecc < b[i+1])][f'attnL_{par}']) for i in range(len(b)-1)])
        binned_S=np.asarray([df[(df.std_prf_ecc > b[i])&(df.std_prf_ecc < b[i+1])][f'attnS_{par}'].mean() for i in range(len(b)-1)])
        S_sem=np.asarray([scipy.stats.sem(df[(df.std_prf_ecc > b[i])&
                                  (df.std_prf_ecc < b[i+1])][f'attnS_{par}']) for i in range(len(b)-1)])
        sem_tot=np.sqrt(L_sem**2 + S_sem**2)
        
        
        diff_bin=binned_L-binned_S
#         print(diff_bin)
        ymin.append(np.nanmin(diff_bin-sem_tot))
        ymax.append(np.nanmax(diff_bin+sem_tot))

        ax.hlines(0,0,5,'black','dashed')
        ax.vlines(0.15,-10,10,'k','dashed')
        ax.vlines(0.5,-10,10,'k','dashed')

        # ax.errorbar(x,diff_bin,yerr=sem_tot,alpha=0.2,color=c[i])
#         print(diff_bin-sem_tot)
#         print(diff_bin+sem_tot)

        if not log:
            ax.scatter(x,diff_bin,label=it,lw=4,alpha=0.8,color=c[i],marker='.')
        else:
            ax.scatter(x,diff_bin,label=it,lw=4,alpha=0.8,color=c[i],marker='.')
            ax.set_xscale('log')

        ylab= '% SC' if par == 'onoff' else 'degrees VA'
        ax.set_ylabel(ylab) 
    
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(np.nanmin(ymin)-0.25,np.nanmax(ymax)+0.25)
    if legend == 'side':
        plt.legend(bbox_to_anchor=(1.05,1),loc=2)
    else:
        plt.legend(loc='lower right',prop={'size': 35},frameon=False)
    fig.suptitle(f'{par}')
    fig.tight_layout()
    
    if savefn:
        fig.savefig(savefn)

def false_discovery_control(ps, *, axis=0, method='bh'):
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)

def fit_gauss(m, s, bins=np.linspace(-20,20,200)):
    y = ((1 / (np.sqrt(2 * np.pi) * s)) * 
         np.exp(-0.5 * ((bins - m) / s )**2))
    return y


def DoG_attn(muSD, sigSD, muAF,sigAF,offset=1,scalingRF=1,amp_rng=[0,5],x=np.linspace(-20,20,100),sigSD2=None):
    # assert isinstance(sigAF,(list,np.ndarray)), "for DoG-AF sigAF must have 2 values"    
    sigAF1,sigAF2=sigAF[0],sigAF[1]
    # assert sigAF1<sigAF2, "sigAF2 must be larger than sigAF1"    
    if isinstance(muSD,(list,np.ndarray)):
        scalingRF_list=np.where((muSD > amp_rng[0])&(muSD < amp_rng[1]),scalingRF,1)
        xx=np.asarray([x]*len(muSD)).T
        vfun=np.vectorize(fit_gauss)
        if sigSD2:
            pass #popt=np.asarray([scipy.optimize.curve_fit(fit_gauss,x,(np.asarray([(offset+fit_gauss(muAF,sigAF1,x)-fit_gauss(muAF,sigAF1,x))]*len(muSD)).T * vfun(muSD,sigSD,xx))[:,i])[0] for i in range(len(muSD))])
        else:
            popt=np.asarray([scipy.optimize.curve_fit(fit_gauss,x,(np.asarray([(offset+fit_gauss(muAF,sigAF1,x)-fit_gauss(muAF,sigAF1,x))]*len(muSD)).T * vfun(muSD,sigSD,xx))[:,i])[0] for i in range(len(muSD))])
        mu,sig=popt[:,1],popt[:,0]
    else:    
        popt,_=scipy.optimize.curve_fit(fit_gauss,x,fit_gauss(muSD,sigSD,x)*(offset+fit_gauss(muAF,sigAF1,x)-fit_gauss(muAF,sigAF2,x)))
        mu,sig=popt[1],popt[0]

    return mu,sig


def gaus_attn(muSD, sigSD, muAF,sigAF,offsetSD=0,scalingSD=1,offsetAF=1,scalingAF=1,amp_rng=[0,5],x=np.linspace(-20,20,100),sigSD2=None,sigAF2=None):
    scalingRF=scalingSD.value if not isinstance(scalingSD,(int,float)) else scalingSD
    if (offsetAF == 0) & (offsetSD == 0):
        mu,sig=gauss_interaction(muSD,sigSD,muAF,sigAF)
    else:
        if isinstance(muSD,(list,np.ndarray)):
            print(f'Running gaus_attn with list of SD-pRFs: {muSD}')
            # selectively scale RFs in certain eccentricity positions
            scalingRF_list=np.where((muSD > amp_rng[0])&(muSD < amp_rng[1]),scalingRF,1)
            print(scalingRF_list)
            xx=np.asarray([x]*len(muSD))
            vfun=np.vectorize(fit_gauss)
            if isinstance(sigSD2,(list,float,np.ndarray,int)):
                #for DoG SD-pRF
                popt=np.asarray([scipy.optimize.curve_fit(fit_gauss,x,(offsetAF+scalingAF*vfun(muAF,sigAF,xx.T)).T[i]*(scalingRF_list[i])*((vfun(muSD,sigSD,xx.T)/vfun(muSD,sigSD2,xx.T)).T[i]))[0] for i in range(len(muSD))])
            else:
                popt=np.asarray([scipy.optimize.curve_fit(fit_gauss,x,(offsetAF+scalingAF*vfun(muAF,sigAF,xx.T)).T[i]*(scalingRF_list[i])*(vfun(muSD,sigSD,xx.T).T[i]))[0] for i in range(len(muSD))])
            mu,sig=popt[:,1],popt[:,0]
        else:
            # print(f'Running gaus_attn with single SD-pRF: ',muSD, type(muSD))
            if sigSD2: 
                # for DoG SD-pRF
                popt,_=scipy.optimize.curve_fit(fit_gauss,x,(offsetAF+scalingAF*fit_gauss(muAF,sigAF,x))*scalingRF*(fit_gauss(muSD,sigSD,x)/fit_gauss(muSD,sigSD2,x)))
            else:
                popt,_=scipy.optimize.curve_fit(fit_gauss,x,(offsetAF+scalingAF*fit_gauss(muAF,sigAF,x))*scalingRF*(fit_gauss(muSD,sigSD,x)))
            mu,sig=popt[1],popt[0]
    return mu,sig

def normRF_gaus_attn(muSD, sigSD1,sigSD2, muAF,sigAF,offset=0):    
    return fit_gauss(muSD,sigSD,200)*(offset+fit_gauss(muAF,sigAF,200))

def ecc_pl_format(ax,s_stim=[0,0.1],l_stim=[0.5,5],vline=False,**kwargs):
    for k,w in kwargs.items():
        ax.set(**{k:w})
    ax.hlines(0,0,5,'black',linestyles=(0,(5,5)))
    ax.fill_between(s_stim,[-50,-50],[50,50],alpha=0.1,color='lightgray',zorder=0,linewidth=0)
    ax.fill_between(l_stim,[-50,-50],[50,50],alpha=0.1,color='lightgray',zorder=0,linewidth=0)
    ax.hlines(ax.get_ylim()[1]-0.01,*s_stim,'black','solid',lw=4,color=settings.get('cmap').get('small'))
    ax.hlines(ax.get_ylim()[1]-0.01,*l_stim,'black','solid',lw=4,color=settings.get('cmap').get('large'))
    if vline:
        ax.vlines(s_stim[1],*ax.get_ylim(),color='black',linestyles=(0,(5,5)),zorder=0)
    # plotting.conform_ax_to_obj(ax=ax)