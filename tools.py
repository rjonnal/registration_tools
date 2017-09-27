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

def normalize(im):
    return (im-im.mean())/im.std()

def block_correct(im,peakx,peaky):
    sy,sx = im.shape
    peakx -= sx//2
    peaky -= sy//2
    return peakx,peaky

class RegisteredPair:

    def __init__(self,im1,im2):
        im1,im2 = equal_pad(im1,im2)
        nxc = nxcorr(im1,im2,True)
        nxc = np.fft.fftshift(nxc)
        bsnxc = background_subtract(nxc)
        self.correlation = nxc.max()
        self.bscorrelation = bsnxc.max()

        py,px = np.where(nxc==self.correlation)
        bspy,bspx = np.where(nxc==self.correlation)

        # check to make sure the nxc and the background-subtraced
        # nxc have the same peak; this adds a bit of confidence
        err = np.sqrt((py-bspy)**2+(px-bspx)**2)
        #print err,self.correlation-self.bscorrelation
        
        self.py = py[0]
        self.px = px[0]

        self.x,self.y = block_correct(nxc,self.px,self.py)
        self.im1 = im1
        self.im2 = im2
        self.nxc = nxc

    def show(self):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(self.im1,cmap='gray',interpolation='none',aspect='auto')
        plt.subplot(2,2,3)
        plt.imshow(self.nxc,cmap='jet',interpolation='none',aspect='auto')
        plt.colorbar()
        plt.autoscale(False)
        plt.plot(self.px,self.py,'ko')
        plt.subplot(2,2,2)
        plt.imshow(self.im2,cmap='gray',interpolation='none',aspect='auto')
        plt.subplot(2,2,4)
        plt.imshow(self.add(),cmap='gray',interpolation='none',aspect='auto')
        plt.show()

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


def register_strip(ref,strip,sequential=True,y_guess=None):
    rsy,rsx = ref.shape
    ssy,ssx = strip.shape
    if sequential:
        y1vec = np.arange(0,rsy,ssy)
        y2vec = y1vec+ssy

        ymids = (y1vec+y2vec)/2.0
        if y_guess is not None:
            order = np.argsort(np.abs(ymids-y_guess))
        else:
            order = np.arange(len(ymids))
            
        cout = np.ones(len(ymids))*np.nan
        xout = np.ones(len(ymids))*np.nan
        yout = np.ones(len(ymids))*np.nan

        plt.subplot(len(cout)+1,2,1)
        plt.imshow(strip,cmap='gray')
        
        for idx in order:
            y1 = y1vec[idx]
            y2 = y2vec[idx]
            
            rstrip = ref[y1:y2,:]
            rp = RegisteredPair(strip,rstrip)
            cout[idx] = rp.correlation
            xout[idx] = rp.x
            yout[idx] = rp.y
            plt.subplot(len(cout)+1,2,2*(idx+2)-1)
            plt.imshow(rstrip,cmap='gray')
            plt.title(rp.correlation)
            plt.subplot(len(cout)+1,2,2*(idx+2))
            plt.imshow(rp.nxc,cmap='jet')
            plt.colorbar()
        plt.show()
        sys.exit()
            
    else:
        rp = RegisteredPair(strip,ref)

def main():

    noise_rms = np.linspace(0.0,2.0,128)
    shift = np.linspace(0.0,20.0,128)

    result = np.zeros((len(noise_rms),len(shift)))
    
    for nidx,nr in enumerate(noise_rms):
        for sidx,s in enumerate(shift):
            stack = make_stack(1,xrms=0.0,yrms=0.0,noise_rms=nr,line_rms=0.0,width=800,height=800)[0,:,:]
            s = int(round(s))
            a = stack[:200,:200]
            b = stack[s:200+s,:200]
            acrop = a[s:200,:200].ravel()
            bcrop = b[:200-s,:200].ravel()
            cc = np.corrcoef(acrop,bcrop)
            rp = RegisteredPair(a,b)
            result[nidx,sidx] = rp.correlation
            try:
                clim = np.percentile(result[:nidx-1,:],(0,100))
            except:
                clim = (0.0,1.0)
            plt.cla()
            plt.imshow(result,clim=clim)
            #plt.colorbar()
            plt.pause(.000001)
    np.save('noise.npy',noise_rms)
    np.save('shift.npy',shift)
    np.save('result.npy',result)
    sys.exit()
    stack = make_stack(50,xrms=10.,yrms=10.,noise_rms=0.04,line_rms=0.0,width=800,height=800)


    
    
    a = stack[0,:50,:500]
    b = stack[0,5:55,10:510]

    rp = RegisteredPair(a,b)
    rp.show()
    sys.exit()

    
    y_guess0 = rp.y
    
    t0 = time.time()
    for k in range(10):
        y0 = 125
        y1 = 150
        ymid = (y1+y0)//2
        y_guess = ymid+y_guess0
        register_strip(a,b[y0:y1,:],True,y_guess=y_guess)
    print time.time()-t0
if __name__=='__main__':
    main()
