'''Utilities: tools for estimating the location of a random vector.'''

## External modules.
import numpy as np

## Internal modules.
from mml.utils.linalg import pwd_fast


###############################################################################


## For reference:
## All routines here take arrays (denoted A) assumed to be
## of shape (n,...), and returns an array of shape (...).


def smallball(A):
    '''
    High-dimensional median via pairwise distances, returning the
    point which is contains over half the other points in the smallest
    possible ball. Ref: Hsu and Sabato (2016).
    Array A is of shape (n,...), taken as n vectors of interest.
    Returns an array of shape (...).
    '''
    idx_argmin = np.argmin(np.median(pwd_fast(A=A,B=A), axis=0))
    return A[idx_argmin,...]


def geomed_set(A):
    '''
    Geometric median, but within the set of points available,
    rather than over the entire space.
    '''
    idx_argmin = np.argmin(np.sum(pwd_fast(A=A,B=A), axis=0))
    return A[idx_argmin,...]


## Helpers for "geomed" below.

def _row_norms(A):
    '''
    A natural default function for computing "row" norms
    for arbitrary ndarray objects than can have more than
    two axes.
    
    More concretely, this function does the following:
    - Treats first axis as indexing the "vectors" of interest.
    - For each i in len(A):
      - flatten A[i,...] into a 1-dim array
      - compute the l2 norm for that array
    - All these norms are returned as-is, with shape (len(A),1).
    '''
    return np.linalg.norm(A.reshape((len(A),-1)), axis=1, keepdims=True)


def _get_diffs_recip(u, A, row_norms=_row_norms):
    '''
    Safely computes the *reciprocal* of the distance between
    the vector u (shape (1,...)) and each row of A (shape (n,...)),
    noting that each row has a shape of (...).
    When the difference is very close to zero, then instead of 1/0,
    it returns 0, since we desire a zero-valued weight for such a pair.
    Returns array of shape (n,1).
    '''

    ## Fix the shape of the output.
    out_shape = (len(A),1)
    
    ## Check to ensure we can broadcast.
    if len(u) == 1 and u.ndim == A.ndim:
        out = row_norms(A-u).reshape(out_shape)
    elif u.shape == A[0,...].shape:
        u_exp = np.expand_dims(u, axis=0)
        out = row_norms(A-u_exp).reshape(out_shape)
    else:
        raise RuntimeError(
            "u shape () vs. A shape ()".format(u.shape,A.shape)
        )
    
    # For near-zero values, set to 1 for safe inversion.
    idx = out < 1e-06
    out[idx] = 1.0
    out = 1.0/out
    # Now that inversion is done, set the tiny values to zero.
    out[idx] = 0.0
    return out


def geomed(A, thres=1e-03, max_iter=100, row_norms=_row_norms):
    '''
    Geometric median, following the algorithm of Vardi and Zhang (2000).
    Note however that in their paper, they have eta_i as per-example
    weights, whereas we have all points weighted the same, so eta_i=1.0
    for all i.
    '''
    if len(A) == 1:
        # If one point, trivially return it as-is.
        return A[0,...]
    
    elif len(A) == 2:
        # If two points, any point between the two
        # is valid; use the midpoint as natural choice.
        return A.mean(axis=0, keepdims=False)
    
    else:
        
        if row_norms(A-A[0:1,...]).max() < 1e-10:
            # If all points are the same, returning any one is fine.
            return A[0,...]
        
        else:
            
            # If make it this far, do iterative procedure.
            u_old = A.mean(axis=0, keepdims=True) # initialize.
            thres_check = thres+1.0 # set large enough as default.
            t = 0
            
            # Iterative updates.
            while (thres_check > thres and t < max_iter):
                
                # Reciprocal of differences.
                dr = _get_diffs_recip(u=u_old, A=A)
                dr_sum = dr.sum()
                dr_exp = np.expand_dims(dr, axis=tuple(range(2,A.ndim)))
                weisz = (A*dr_exp).sum(axis=0, keepdims=True) / dr_sum
                r = row_norms((weisz-u_old)*dr_sum).item() # their eqn [2.8].
                
                # Here "rinv" is the eta(y)/r(y) quantity in the
                # paper of Vardi and Zhang. They use per-example
                # weights of eta_i, while we fix these to 1.0.
                # On the other hand, their routine assumes that
                # at most one point will overlap; here we extend
                # their update such that an arbitrary number can
                # overlap.
                hit_count = (dr == 0.0).sum()
                if hit_count > 0:
                    rinv = hit_count / r
                else:
                    rinv = 0 # from their eqn [2.5], zero if no hits.
                
                u_new = np.copy(max(0,1-rinv)*weisz + min(1,rinv)*u_old)
                t += 1
                thres_check = row_norms(u_new-u_old).item()
                u_old = np.copy(u_new)

            ## Finally, return vector of the desired shape.
            if u_new.ndim < A.ndim and u_new.shape == A[0,...].shape:
                return u_new
            elif u_new.ndim == A.ndim and len(u_new) == 1:
                return u_new[-1,...]
            else:
                raise RuntimeError(
                    "Unexpected output shape: {}".format(u_new.shape)
                )


###############################################################################
