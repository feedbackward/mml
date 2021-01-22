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


def onehot(y, num_classes):
    '''
    Assumes y is (n,1) shaped array of labels
    taking values between 0 and (num_classes-1).
    
    Note: using reshape(-1) is the recommended way
    to flatten and get a view as often as possible.
    (Ref: numpy.ravel documentation)
    '''
    return np.eye(num_classes, dtype=y.dtype)[y.reshape(-1)]



###############################################################################
