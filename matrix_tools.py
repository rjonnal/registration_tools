import numpy as np

def deproud(im,y1,y2,x1,x2):
    '''Return bounding box coordinates that are not proud of the image.'''
    sy,sx = im.shape
    print sy,y1,y2
    print sx,x1,x2
    assert y2-y1<sy
    assert x2-x1<sx
    
    while y1<0:
        y1 = y1 + 1
        y2 = y2 + 1
    while y2>sy-1:
        y1 = y1 - 1
        y2 = y2 - 1
    while x1<0:
        x1 = x1 + 1
        x2 = x2 + 1
    while x2>sx-1:
        x1 = x1 - 1
        x2 = x2 - 1

    return y1,y2,x1,x2


def pad(im,new_shape,mode='zero'):
    sy,sx = im.shape
    nsy,nsx = new_shape
    assert nsy>=sy
    assert nsx>=sx
    if mode.lower()=='mean':
        value = im.mean()
    elif mode.lower()=='zero':
        value = 0.0
    else:
        value = np.nan
    out = np.ones(new_shape)*value
    out[:sy,:sx] = im
    return out

def equal_pad(im1,im2,mode='zero'):
    sy1,sx1 = im1.shape
    sy2,sx2 = im2.shape
    sy = max(sy1,sy2)
    sx = max(sx1,sx2)
    new_shape = (sy,sx)
    im1 = pad(im1,new_shape)
    im2 = pad(im2,new_shape)
    return im1,im2
    
