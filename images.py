import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import sys,os,glob
from matrix_tools import deproud

def make_stack(N,xrms=2.0,yrms=2.0,noise_rms=5.0,noise_edge=0,line_rms=0.0,source_file='leaves.npy',width=None,height=None):

    
    
    src = np.load(os.path.join('images',source_file))

    sy,sx = src.shape
    yoff = int(3*yrms)
    xoff = int(3*xrms)
    
    if height is None:
        height = sy-2*yoff
    if width is None: 
        width = sx-2*xoff
    
    assert not (line_rms and noise_edge)
    out = np.zeros((N,height,width))
    xoff = xoff
    yoff = yoff
    for k in range(N):
        x1 = int(xoff)
        x2 = x1+width
        y1 = int(yoff)
        y2 = y1+height
        y1,y2,x1,x2 = deproud(src,y1,y2,x1,x2)
        
        im = src[y1:y2,x1:x2]
        if noise_edge:
            mask = np.ones(im.shape)
            mask[:noise_edge,:] = 0.0
            mask[-noise_edge:,:] = 0.0
            mask[:,:noise_edge] = 0.0
            mask[:,-noise_edge:] = 0.0
            arand = np.random.random(im.shape)*im.max()
            im = mask*im+(1-mask)*arand
            
        noise = np.random.randn(height,width)*noise_rms
        if line_rms:
            temp = np.zeros(im.shape)
            x_line_offset = np.random.randn()*line_rms
            y_line_offset = np.random.randn()*line_rms
            for h in range(height):
                lx1 = int(x1+x_line_offset)
                lx2 = lx1+width
                ly1 = int(y1+h+y_line_offset)
                ly2 = ly1
                ly1,ly2,lx1,lx2 = deproud(src,ly1,ly2,lx1,lx2)
                temp[h,:] = src[ly1,lx1:lx2]
                x_line_offset = x_line_offset+np.random.randn()*line_rms
                y_line_offset = y_line_offset+np.random.randn()*line_rms
            out[k,:,:] = temp+noise
        else:
            out[k,:,:] = im+noise
        xoff = xoff + np.random.randn()*xrms
        yoff = yoff + np.random.randn()*yrms

    return out
    
