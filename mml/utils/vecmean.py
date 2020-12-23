'''Utilities: tools for estimating the location of a random vector.'''

## External modules.
import numpy as np

## Internal modules.
from mml.utils.linalg import pwd_fast


###############################################################################


def smallball(A):
    '''
    High-dimensional median via pairwise distances, returning the
    point which is contains over half the other points in the smallest
    possible ball. Ref: Hsu and Sabato (2016).
    Array A is of shape (k,d), taken as k points in d-dim space.
    Returns an array of shape (1,d).
    '''
    idx = np.argmin(np.median(pwd_fast(A=A,B=A), axis=0))
    return np.take(A, [idx], axis=0)


# Helper for "med_geomed" below.
def _get_diffs_recip(u, A):
    '''
    Safely computes the *reciprocal* of the differences (measured by l2 norm)
    between all the rows of A (shape (k,d)) and u, assumed to have shape (d,)
    or (d,1), either is fine. However, when the difference is very close to
    zero, then instead of 1/0, it returns 0, since we desire a zero-valued
    weight for such a pair.
    Returns array of shape (k,1).
    '''
    out = np.linalg.norm(x=(A-u), ord=None, axis=1, keepdims=True)
    
    # For near-zero values, set to 1 for safe inversion.
    idx = out < 1e-06
    out[idx] = 1.0
    out = 1/out
    # Now that inversion is done, set the tiny values to zero.
    out[idx] = 0.0
    return out


def geomed_set(A):
    '''
    Geometric median, but within the set of points available,
    rather than over the entire space.
    '''
    idx = np.argmin(np.sum(pwd_fast(A=A,B=A), axis=0))
    return np.take(A, [idx], axis=0)


def geomed(A, thres=1e-03, max_iter=100):
    '''
    Geometric median, following the algorithm of Vardi and Zhang (2000).
    Note however that in their paper, they have eta_i as per-example
    weights, whereas we have all points weighted the same, so eta_i=1.0
    for all i.
    Takes array A, shape (k,d), of k points in d-dim space.
    Returns array of shape (1,d).
    '''
    
    k, d = A.shape
    out_shape = (1,d)
    
    if k == 1:
        # If one point, trivially return it as-is.
        return np.take(A, [0], axis=0)
    
    elif k == 2:
        # If two points, any point between the two
        # is valid; use the midpoint as natural choice.
        return A.mean(axis=0, keepdims=True)
    
    else:
        # If all points are the same, returning any one is fine.
        if (np.linalg.norm((A-A[0,:])) < 1e-12):
            return np.take(A, [0], axis=0)
        
        # If make it this far, do iterative procedure.
        else:
            
            u_old = A.mean(axis=0, keepdims=False)
            thres_check = 1.0+thres # set large enough as default.
            t = 0

            # Iterative updates.
            while (thres_check > thres and t < max_iter):
            
                # Reciprocal of differences.
                dr = _get_diffs_recip(u=u_old, A=A)
                dr_sum = dr.sum()
                weisz = (A*dr).sum(axis=0, keepdims=False) / dr_sum
                r = np.linalg.norm((weisz-u_old)*dr_sum)
                
                # Here "rinv" is the eta(y)/r(y) quantity in the
                # paper of Vardi and Zhang. They use per-example
                # weights of eta_i, while we fix these to 1.0.
                # On the other hand, their routine assumes that
                # at most one point will overlap; here we extend
                # their update such that an arbitrary number can
                # overlap.
                hit_count = np.sum(dr == 0.0)
                if hit_count > 0:
                    rinv = hit_count / r
                else:
                    rinv = 0 # from their eqn [2.5], zero if no hits.
                
                u_new = np.copy(max(0,(1-rinv))*weisz + min(1,rinv)*u_old)
                t += 1
                thres_check = np.linalg.norm(u_new-u_old)
                u_old = np.copy(u_new)
            
            return u_new.reshape(out_shape)


def mest_bycoord(A, paras):
    '''
    M-estimation of each coordinate to form vector mean estimator.
    e.g., Catoni-type etc. Let it be simply passed an mest subroutine object.
    '''
    raise NotImplementedError


###############################################################################
