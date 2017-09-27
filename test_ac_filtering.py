import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import numba
import sys,os,glob
from images import make_stack


def xc(im1,im2):
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

def nxcorr(im1,im2,correct_for_overlap=True,xcfunc=xc):
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
        peakx,peaky = block_correct(im1,peakx,peaky)
        sy,sx = xc_val.shape
        overlap_pixels = float(sy-np.abs(peaky))*float(sx-np.abs(peakx))
        all_pixels = np.prod(im1.shape)
        corrector = all_pixels/overlap_pixels
    else:
        corrector = 1.0
    denom = np.sqrt(np.max(xcfunc(im1,im1)))*np.sqrt(np.max(xcfunc(im2,im2)))
    return xc_val/denom*corrector

def normalize(im):
    return (im-im.mean())/im.std()

def block_correct(im,peakx,peaky):
    sy,sx = im.shape
    if peakx>sx//2:
        peakx -= sx
    if peaky>sy//2:
        peaky -= sy
    return peakx,peaky

class RegisteredPair:

    def __init__(self,im1,im2):
        nxc = nxcorr(im1,im2,True)
        self.correlation = nxc.max()
        py,px = np.where(nxc==self.correlation)
        py = py[0]
        px = px[0]
        px,py = block_correct(nxc,px,py)
        self.x = px
        self.y = py
        self.im1 = im1
        self.im2 = im2
        self.nxc = nxc

    def show(self):
        plt.subplot(1,3,1)
        plt.imshow(self.im1,cmap='gray',interpolation='none')
        plt.subplot(1,3,2)
        plt.imshow(self.nxc,cmap='jet',interpolation='none')
        plt.colorbar()
        plt.autoscale(False)
        plt.plot(self.x,self.y,'ko')
        plt.subplot(1,3,3)
        plt.imshow(self.im2,cmap='gray',interpolation='none')
        plt.show()


def main():
    ab = make_stack(2,noise_rms=10.0,noise_edge=20)
    a = ab[0,:,:]
    b = ab[1,:,:]

    rp = RegisteredPair(a,b)
    rp.show()
    sys.exit()
    ac = nxcorr(a,a)
    XX,YY = np.meshgrid(np.arange(ac.shape[1]),np.arange(ac.shape[0]))
    py,px = np.where(ac==ac.max())
    XX = XX-px
    YY = YY-py
    d = np.sqrt(XX**2+YY**2)
    mask = np.zeros(d.shape)
    mask[np.where(d>5)] = 0
    plt.imshow(mask)
    plt.show()
    
if __name__=='__main__':
    main()
