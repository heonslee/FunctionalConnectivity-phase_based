import numpy as np
import scipy.signal
def delayRecons(data, m, tau):
    nchs, data_len = data.shape
    ddata = np.zeros((nchs, data_len - tau*(m-1), m), dtype=data.dtype)
    for j in range(m): # j+=1
        i1 = j*tau
        i2 = data_len- (m-(j+1))*tau
        ddata[:,:,j] = data[:,i1:i2]
    return ddata

def phaselagentropy(sig, m, tau):
    asig = scipy.signal.hilbert(sig, axis=-1)
    nchs = sig.shape[0]
    
    # compute cross spectra        
    csig = [asig[c]*np.conj(asig[c+1:]) for c in range(nchs-1)]
    csig = np.concatenate(csig, axis=0)
    
    # phase difference
    pll = np.sign(np.imag(csig)) # phase lead-lag
    pll = np.heaviside(pll, 0) # phase lead-lag
    
    # convert phase-lag to unique patterns
    ddata = delayRecons(pll, m, tau)
    udata = np.stack([ddata[:,:,mm]*(2**mm) for mm in range(m)], axis=2) #[n_chpairs, ntime, m]
    udata = udata.sum(2).astype(int) # [n_chpairs, ntime]
    
    # get probability and compute PLE
    bins = np.arange(-0.5, 2**m+0.5)
    # uniq_num, prob = np.unique(udata,return_counts=True, axis=-1)
    # out = np.bincount(udata.flatten(), minlength=3*len(bins))
    prob = np.array([np.histogram(udat_, bins=bins)[0] for udat_ in udata]) # [n_chpairs, n_bins]
    prob = prob/udata.shape[1] # [n_chpairs, n_bins]
    ple = -np.sum(prob * np.log2(prob + 1e-6), -1)
    return ple/np.log2(len(bins)-1)

def get_chpair_names(ch_names):
    chpair_names = []
    for i,c1 in enumerate(ch_names[:-1]):
        for j,c2 in enumerate(ch_names[i+1:]):
            chpair_names.append(c1 + '-' + c2)
    return chpair_names
        