'''Utilities: helper functions for vector/array-related operations.'''

## External modules.
import numpy as np


###############################################################################


def pwd(A, B, norm=np.linalg.norm):
    '''
    Compute pairwise distances in any norm
    between the "rows" of arrays A and B,
    where each row is the sub-array associated
    with each index in the first axis.
    
    Returns an array of (len(A),len(B)) shape,
    populated with pairwise distances.

    For the *special case* of the l2 norm,
    we can see that pwd_fast (implemented below)
    is much faster; this fn is a good sanity checker.
    '''
    n, m = (len(A),len(B))
    out = np.zeros((n,m), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            out[i,j] = norm(A[i,...]-B[j,...])
    return out


def pwd_fast(A, B):
    '''
    Same as pwd(), computing pairwise distances between
    the row vectors of two matrices, except using a nice
    speed-up available when using the l2 norm.
    References:
    - Alex Smola's blog post "In praise of the Second Binomial Formula".
    - Also, see https://www.r-bloggers.com/pairwise-distances-in-r/.
    '''
    n, m = (len(A),len(B))
    A_flatrows = A.reshape((n,-1))
    B_flatrows = B.reshape((m,-1))
    A_dim = A_flatrows.shape[1]
    B_dim = B_flatrows.shape[1]
    if A_dim != B_dim:
        raise RuntimeError(
            "A dim ({}) != B dim ({}).".format(A_dim,B_dim)
        )
    out = np.zeros((n,m), dtype=A.dtype)
    out += (A_flatrows**2).sum(axis=1,keepdims=True) # add one COL at a time.
    out += (B_flatrows**2).sum(axis=1,keepdims=True).T # add one ROW at a time.
    out -= 2 * A_flatrows.dot(B_flatrows.T)
    
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
    if y is None:
        return y
    else:
        return np.eye(num_classes, dtype=y.dtype)[y.reshape(-1)]


###############################################################################
