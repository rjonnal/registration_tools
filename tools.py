import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import numba
import sys,os,glob
from images import make_stack
from matrix_tools import equal_pad
import time
from scipy.ndimage import morphology

def disk(diameter):
    xx,yy = np.meshgrid(np.arange(diameter),np.arange(diameter))
    xx = xx - float(diameter-1)/2.0
    yy = yy - float(diameter-1)/2.0
    d = np.sqrt(xx**2+yy**2)
    out = np.zeros(xx.shape)
    out[np.where(d<=diameter/2.0)] = 1.0
    return out

def background_subtract(im,diameter=25):
    strel = disk(diameter)
    bg = morphology.grey_opening(im,structure=strel)
    return im-bg+bg.mean()

def xc(im1,im2,auto_resize=True):
    """This is the core of the cross-correlation:
    the inverse FT of the product of the FT of one and
    the complex conjugate of the FT of the other.
    This can be used to calculate xcorr and autocorr."""
    f1 = np.fft.fft2(im1)
    f2 = np.fft.fft2(im2)
    f2c = f2.conjugate()
    return np.abs(np.fft.ifft2(f1*f2c))

def acxc(im1,im2):
    """Slightly modified cross-correlation core; like
    xc, but should produce the convolution of the
    xc with the average single image autocorrelations.
    This is motivated by the fact that we're looking
    for a peak that looks like the individual ac's.
    NB: this totally doesn't work, probably because
    I'm an idiot."""
    f1 = np.fft.fft2(im1)
    f1c = f1.conjugate()
    f2 = np.fft.fft2(im2)
    f2c = f2.conjugate()
    t1 = f1**2*f1c*f2c/2.0
    t2 = f2c**2*f1*f2/2.0
    return np.abs(np.fft.ifft2(t1+t2))

def nxcorr_overlap_norm(im1,im2,correct_for_overlap=True,xcfunc=xc):
    """This is the main cross-correlation function:
    it normalizes the images (subtracts means and
    divides by standard deviation), calls xc for
    the fundamental correlation matrix, and then scales
    it according to the amount of overlap (which depends
    upon translation between the two) and also by the
    individual normalized image autocorrelations.

    Overlap correction can be skipped if translations
    are known to be small or if the numerical value
    of the cross-correlation is not important. On the
    other hand, overlap correction adds only 2%-3%
    computational overhead."""

    im1,im2 = normalize(im1),normalize(im2)
    xc_val = xcfunc(im1,im2)
    if correct_for_overlap:
        peaky,peakx = np.where(xc_val==xc_val.max())
        peaky = peaky[0]
        peakx = peakx[0]
        sy,sx = xc_val.shape
        if peaky>sy//2:
            peaky = peaky-sy
        if peakx>sx//2:
            peakx = peakx-sx
            #peakx,peaky = block_correct(im1,peakx,peaky)
        sy,sx = xc_val.shape
    
        overlap_pixels = float(sy-np.abs(peaky))*float(sx-np.abs(peakx))
        all_pixels = np.prod(im1.shape)
        corrector = all_pixels/overlap_pixels
    else:
        corrector = 1.0
        
    denom = np.sqrt(np.max(xcfunc(im1,im1)))*np.sqrt(np.max(xcfunc(im2,im2)))
    out = xc_val/denom*corrector
    return out

def nxcorr_ac_norm(im1,im2,correct_for_overlap=True,xcfunc=xc):
    """This is the main cross-correlation function:
    it normalizes the images (subtracts means and
    divides by standard deviation), calls xc for
    the fundamental correlation matrix, and then scales
    it according to the amount of overlap. Unlike nxcorr
    above, it scales for overlap by using only the overlapping
    regions of the two images to compute the autocorrelations
    used in the correlation denominator.

    Overlap correction can be skipped if translations
    are known to be small or if the numerical value
    of the cross-correlation is not important. On the
    other hand, overlap correction adds only 2%-3%
    computational overhead."""

    im1,im2 = normalize(im1),normalize(im2)
    xc_val = xcfunc(im1,im2)

    if correct_for_overlap:
        peaky,peakx = np.where(xc_val==xc_val.max())
        peaky = peaky[0]
        peakx = peakx[0]
        sy,sx = xc_val.shape
        if peaky>sy//2:
            peaky = peaky-sy
        if peakx>sx//2:
            peakx = peakx-sx
            #peakx,peaky = block_correct(im1,peakx,peaky)
        sy,sx = xc_val.shape

        # crop the images to overlapping region in order to compute
        # an accurate denominator
        temp1 = im1.copy()
        temp2 = im2.copy()
        if peaky>0:
            temp2 = temp2[:-peaky,:]
            temp1 = temp1[peaky:,:]
        elif peaky<0:
            temp2 = temp2[-peaky:,:]
            temp1 = temp1[:peaky,:]
        if peakx>0:
            temp2 = temp2[:,:-peakx]
            temp1 = temp1[:,peakx:]
        elif peakx<0:
            temp2 = temp2[:,-peakx:]
            temp1 = temp1[:,:peakx]

        denom = np.sqrt(np.max(xcfunc(temp1,temp1)))*np.sqrt(np.max(xcfunc(temp2,temp2)))
    else:
        denom = np.sqrt(np.max(xcfunc(im1,im1)))*np.sqrt(np.max(xcfunc(im2,im2)))
    out = xc_val/denom
    return out

def nxcorr(im1,im2,correct_for_overlap=False,xcfunc=xc):
    return nxcorr_ac_norm(im1,im2,correct_for_overlap=correct_for_overlap,xcfunc=xcfunc)

def normalize(im):
    return (im-im.mean())/im.std()

def block_correct(im,peakx,peaky):
    sy,sx = im.shape
    peakx -= sx//2
    peaky -= sy//2
    return peakx,peaky

class RegisteredPair:

    def __init__(self,im1,im2,background_correct=False,normalize_rows=False):

        im1,im2 = equal_pad(im1,im2)
        if (not np.count_nonzero(im1)) or (not np.count_nonzero(im2)):
            nxc = np.zeros(im1.shape)
            self.py = im1.shape[0]//2
            self.px = im1.shape[1]//2
            self.correlation = 0.0
        else:
            nxc = nxcorr(im1,im2,True)
            nxc = np.fft.fftshift(nxc)
            print 'got here'
            if normalize_rows:
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(nxc)
                plt.colorbar()
                vprof = np.mean(nxc,axis=1)
                nxc = nxc-vprof+vprof.mean()
                plt.subplot(1,2,2)
                plt.imshow(nxc)
                plt.colorbar()
                plt.show()
            if background_correct:
                nxc = background_subtract(nxc)
                #nxc = nxc/np.std(nxc)

            self.correlation = nxc.max()
            py,px = np.where(nxc==self.correlation)

            self.py = py[0]
            self.px = px[0]

        self.x,self.y = block_correct(nxc,self.px,self.py)
        self.im1 = im1
        self.im2 = im2
        self.nxc = nxc

    def show(self):
        plt.subplot(2,2,1)
        plt.cla()
        plt.imshow(self.im1,cmap='gray',interpolation='none',aspect='auto')
        plt.subplot(2,2,3)
        plt.cla()
        plt.imshow(self.nxc,cmap='jet',interpolation='none',aspect='auto',clim=(0,1.5))
        plt.autoscale(False)
        plt.plot(self.px,self.py,'ko')
        plt.subplot(2,2,2)
        plt.cla()
        plt.imshow(self.im2,cmap='gray',interpolation='none',aspect='auto')
        plt.subplot(2,2,4)
        plt.cla()
        plt.imshow(self.add(),cmap='gray',interpolation='none',aspect='auto')

    def add(self):
        x = self.x
        y = self.y
        im1 = self.im1.copy()
        im2 = self.im2.copy()

        if x<0:
            im1 = im1[:,:x]
            im2 = im2[:,-x:]
        if x>0:
            im1 = im1[:,x:]
            im2 = im2[:,:-x]

        if y<0:
            im1 = im1[:y,:]
            im2 = im2[-y:,:]

        if y>0:
            im1 = im1[y:,:]
            im2 = im2[:-y,:]
        
        return (im1+im2)/2.0

def strip_register(ref,tar,strip_width,step_size=None):

    if step_size is None:
        step_size = strip_width
        
    rsy,rsx = ref.shape
    tsy,tsx = tar.shape

    ymid_vec = np.arange(0,tsy,step_size)
    y1_vec = ymid_vec-strip_width//2
    y2_vec = ymid_vec+strip_width//2+1
    for y1,y2 in zip(y1_vec,y2_vec):
        y1 = max(0,y1)
        y2 = min(tsy,y2)
        temp = tar.copy()
        temp[:y1] = 0.0
        temp[y2:] = 0.0
        rp = RegisteredPair(ref,temp,normalize_rows=False)
        rp.show()
        plt.pause(.0001)
def main():

    #stack = make_stack(2,xrms=10.,yrms=10.,noise_rms=0.04,line_rms=0.0,source_file='rocks.npy',width=1200,height=1200)

    stack = np.load('infrared.npy')
    a = stack[0,:,:]
    b = stack[0,:,:]

    #b[:250,:] = 0.0
    #b[260:,:] = 0.0
    
    #rp = RegisteredPair(a,b)
    #rp.show()
    #plt.show()
    
    strip_register(a,b,33,16)
    
if __name__=='__main__':
    main()
