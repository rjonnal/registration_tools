import numpy as np
from matplotlib import pyplot as plt
import os,sys
from clicky import collector
from scipy.interpolate import griddata
from octopod.DataStore import Hive

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii


def transform(im1,im2,x1=[],y1=[],x2=[],y2=[],method='linear',fill_value=0.0,clim_percentile=(0,100)):

    # make a quasi-unique identifier for this image pair
    pair_key = '_'.join(['%0.3f'%k for k in [im1[0,0],im1[0,-1],im1[-1,0],im1[-1,-1],im2[0,0],im2[0,-1],im2[-1,0],im2[-1,-1]]])

    if not os.path.exists('.feature_cache'):
        os.mkdir('.feature_cache')
    
    hive = Hive(os.path.join('.feature_cache',pair_key))

    try:
        _x1 = hive.get('x1')
        _x2 = hive.get('x2')
        _y1 = hive.get('y1')
        _y2 = hive.get('y2')
        px = hive.get('px')
        py = hive.get('py')
        im2_transformed = hive.get('im2_transformed')
        output = {'x1':x1,'y1':y1,'x2':x2,'y2':y2,'px':px,'py':py,'im2_transformed':im2_transformed}

        assert type(_x1)==np.ndarray
        assert type(_x2)==np.ndarray
        assert type(_y1)==np.ndarray
        assert type(_y2)==np.ndarray
        assert type(px)==np.ndarray
        assert type(py)==np.ndarray
        assert type(im2_transformed)==np.ndarray
        x1 = _x1
        x2 = _x2
        y1 = _y1
        y2 = _y2
    except Exception as e:
        if not len(x1)*len(x2)*len(y1)*len(y2):
            xc,yc,idx = collector([im1,im2],clim_percentile=clim_percentile)
            x1 = xc[::2]
            x2 = xc[1::2]
            y1 = yc[::2]
            y2 = yc[1::2]

        px = np.polyfit(x1,x2,1)
        x_scale = px[0]
        x_shift = -px[1]/x_scale

        py = np.polyfit(y1,y2,1)
        y_scale = py[0]
        y_shift = -py[1]/y_scale

        # now we interpolate the second image into
        # the coordinate space of the first
        XX1,YY1 = np.meshgrid(np.arange(im1.shape[1]),np.arange(im1.shape[0]))
        XX2,YY2 = np.meshgrid(np.arange(im2.shape[1]),np.arange(im2.shape[0]))
        XX2 = XX2/x_scale-px[1]/px[0]
        YY2 = YY2/y_scale-py[1]/py[0]

        im2_transformed = np.reshape(griddata((YY2.ravel(),XX2.ravel()),im2.ravel(),(YY1.ravel(),XX1.ravel()),method=method,fill_value=fill_value),im1.shape)

        output = {'x1':x1,'y1':y1,'x2':x2,'y2':y2,'px':px,'py':py,'im2_transformed':im2_transformed}

        for k in output.keys():
            hive.put(k,output[k])
    
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(im2_transformed)
    plt.subplot(2,2,4)
    plt.imshow(im2_transformed-im1)
    plt.colorbar()
    plt.show()
    
    return output

if __name__=='__main__':
    im1 = np.load('images/kids.npy')
    im1 = im1[200:-481,200:-200]
    oldsy,oldsx = im1.shape
    newsy = newsx = 167
    yoff,xoff = 15,15
    im2 = im1[yoff:yoff+newsy,xoff:xoff+newsy]
    zf = float(oldsy)/float(newsy)
    im2 = ndii.zoom(im2,zf)
    x1=[31.657258064516128, 55.165322580645153, 41.415322580645153, 41.193548387096769, 39.41935483870968, 45.850806451612897, 74.237903225806463, 118.81451612903226, 153.85483870967741, 164.27822580645162, 174.03629032258064, 121.4758064516129, 91.314516129032256, 68.25]
    y1=[35.847479838709688, 40.061189516129048, 54.254737903225816, 75.323286290322585, 134.75877016129033, 145.40393145161289, 90.625705645161304, 106.37167338709679, 103.04506048387097, 99.94022177419356, 156.04909274193551, 159.15393145161289, 114.35554435483871, 158.71038306451612]
    x2=[20.528225806451644, 47.806451612903288, 30.95161290322585, 31.173387096774206, 29.177419354838776, 36.7177419354839, 71.314516129032313, 124.76209677419359, 166.01209677419359, 178.43145161290323, 189.52016129032262, 128.08870967741939, 91.495967741935488, 63.774193548387132]
    y2=[24.980544354838685, 30.081350806451582, 46.049092741935453, 73.105544354838685, 143.85151209677417, 156.93618951612899, 90.625705645161275, 109.0329637096774, 106.14989919354838, 101.71441532258063, 168.02489919354838, 172.68215725806448, 118.56925403225804, 172.23860887096771]
    #print transform(im1,im2,x1,y1,x2,y2)
    print transform(im2,im1,x2,y2,x1,y1)
