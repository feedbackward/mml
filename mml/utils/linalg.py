'''Utilities: helper functions for vector/array-related operations.'''

## External modules.
import numpy as np


###############################################################################


def pwd(A, B):
    '''
    Compute pairwise distances in l2 norm.
    Assumes that A (n,d) and B (m,d) are
    arrays of d-dimensional observations.
    
    Returns an array of (n,m) shape filled
    with pairwise distances. 
    This is clearly correct but slow, just
    for confirming that pwd_fast (below) is
    indeed correctly implemented.
    '''
    
    n, m = (A.shape[0], B.shape[0])
    out = np.zeros((n,m), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            out[i,j] = np.linalg.norm(A[i,:]-B[j,:])

    return out


def pwd_fast(A, B):
    '''
    Same as pwd(), computing pairwise distances between
    the row vectors of two matrices, except using a nice
    speed-up available when using the l2 norm.
    References:
    Alex Smola's blog post "In praise of the Second Binomial Formula".
    Also:
    https://www.r-bloggers.com/pairwise-distances-in-r/
    '''
    
    n, m = (A.shape[0], B.shape[0])
    out = np.zeros((n,m), dtype=A.dtype)
    out += (A**2).sum(axis=1, keepdims=True) # adds to one COL at a time.
    out += (B**2).sum(axis=1, keepdims=True).T # adds to one ROW at a time.
    out -= 2 * A.dot(B.T)
    
    # Correct for computational error:
    # pwd_Mtx is formally non-negative (all elements), but
    # tiny negative residuals can remain, thus we must remove these.
    idx_neg = (out < 0)
    out[idx_neg] = 0.0
    
    # Finally, take roots to put in proper scale, and return.
    return np.sqrt(out)


def ave_groups(a, num_subsets, sizes=False):
    '''
    Compute subset averages in a vectorized fashion.
    Nice reference online: https://stackoverflow.com/a/53178377
    '''
    n = len(a)
    m = n//num_subsets
    w = np.full(num_subsets,m)
    w[:n-m*num_subsets] += 1 # the subset sizes.
    sums = np.add.reduceat(a, np.r_[0,w.cumsum()[:-1]])

    if sizes:
        return (np.true_divide(sums,w), w)
    else:
        return np.true_divide(sums,w)


def soft_thres(u, mar):
    '''
    The so-called "soft threshold" function, which
    appears when evaluating the proximal operator
    for a smooth function plus l1-norm.
    
    Input "u" will be an array, and "mar" will be the
    margin of the soft-threshold, a non-negative real
    value.

    Output will be of the same shape as the input.
    '''
    return np.sign(u) * np.clip(a=(np.abs(u)-mar), a_min=0, a_max=None)


def array_meanstd(array, axis, dtype=np.float64):
    '''
    Computes the mean and standard deviation from a given array.
    Note that higher-precision settings can actually make a
    pretty big impact on the output of these stat computations.
    See example URLs below:
     https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
     https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
    '''
    return (np.mean(array, axis=axis, dtype=dtype),
            np.std(array, axis=axis, dtype=dtype, ddof=1))


def trim_2d(array, axis=0):
    '''
    Helper function which checks for NaNs.
    Assumes that array is a two-dimensional numpy array.
    It checks for the first appearance of NaNs, and
    then trims all rows (cols if axis=1) from that point onward.
    If nothing is found, the array is returned as-is.
    '''
    
    if axis == 0:
        tocheck = np.where(np.isnan(array[:,0]))[0]
        if tocheck.size > 0:
            badidx_start = tocheck[0]
            return array[0:badidx_start,:]
        else:
            return array
    elif axis == 1:
        tocheck = np.where(np.isnan(array[0,:]))[0]
        if tocheck.size > 0:
            badidx_start = tocheck[0]
            return array[:,0:badidx_start]
        else:
            return array
    else:
        raise ValueError("This function only accepts axis of 0 or 1.")


def softmax(x):
    return np.exp(x) / np.exp(x).sum()
            

def pmone(y, poslab=1):
    '''
    Here y is a vector of labels; this function
    converts them to plus/minus one, where the
    argument "poslab" specifies the label to be
    treated at the positive case (+1). Else, -1.
    '''
    
    if (y is None):
        return None
    else:
        out = np.where((y==poslab), 1, -1)
        return out.astype(np.int8)


def flip_pmone(y, prob):
    '''
    A simple function which flips labels,
    assuming that the labels included in
    "y" here are +1 and -1 only.
    Here "prob" is the probability of
    a given label being flipped, assumed
    independent of all others but all
    points have the same probability of
    being flipped.
    '''
    
    if (prob <= 0):
        return y
    elif (prob <= 1):
        randvec = np.random.uniform(size=y.shape)
        multvec = np.where((randvec <= prob), -1, 1)
        return y * multvec
    

def onehot(y, set_nc=None):
    '''
    A function for encoding y into a one-hot vector.
    Inputs:
    - y is a (k,1) array, taking values in {0,1,...,nc-1}.
    Returns:
    - An array of size (k,nc), called C.
    '''

    # Set the number of classes.
    if set_nc is None:
        nc = np.unique(y).size # compute based on data.
    else:
        nc = set_nc
        
    # Now get to work on the one-hot vectors.
    k = y.shape[0]
    C = np.zeros((k,nc), dtype=np.uint8)
    for i in range(k):
        j = y[i,0] # assuming y has only one column.
        C[i,j] = 1

    return C


def concat_ones(X):
    '''
    Concatenates a column vector of ones, in leftmost slot.
    '''
    return np.concatenate((np.ones((X.shape[0],1),dtype=X.dtype), X),
                          axis=1)


###############################################################################
