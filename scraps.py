def register_strip(ref,strip,min_sequential_corr=6.0,y_guess=None):
    rsy,rsx = ref.shape
    ssy,ssx = strip.shape
    clim = (0,1)
    if min_sequential_corr:
        #clim = (0,6)
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
            plt.imshow(rp.nxc,cmap='jet',clim=clim)
            plt.colorbar()
            if rp.correlation>=min_sequential_corr:
                break
        plt.show()
        sys.exit()
            
    else:
        rp = RegisteredPair(strip,ref)
