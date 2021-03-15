'''Utilities: helper functions for M-estimation and related methods.'''

## External modules.
import numpy as np


###############################################################################


## Contents:
# Helper functions for M-estimators of location.
# Helper functions for M-estimators of scale.
# General-purpose routine(s) for M-estimators of location.
# General-purpose routine(s) for M-estimators of scale.
# Alternative scale estimation routines.
# Support for legacy code.


## Helper functions for M-estimators of location.

inf_fns = ["inf_algsq", "inf_atan", "inf_catnarrow", "inf_catwide",
           "inf_gudermann", "inf_fair", "inf_huber", "inf_hubermod",
           "inf_logistic", "inf_tanh"]

def inf_algsq(u):
    '''
    The "square root algebraic" function.
    '''
    return u / np.sqrt(1.+u**2/2.)           


def inf_atan(u):
    '''
    Standard computation of inverse tangent (arctangent).
    '''
    return np.arctan(u)


def inf_catnarrow(u):
    '''
    The narrowest of the influence functions of Catoni (2012).
    '''
    signs = np.sign(u)
    return np.where(np.absolute(u) <= 1.,
                    -signs*np.log1p(-signs*u + u**2/2),
                    signs*np.log(2))


def inf_catwide(u):
    '''
    The widest of the influence functions of Catoni (2012).
    '''
    return np.sign(u) * np.log1p(np.sign(u)*u + u**2/2)


def inf_gudermann(u):
    '''
    Gudermannian function.
    Ref: Abramowitz and Stegun (1964, Ch.4).
    '''
    return 2. * np.arctan(np.exp(u)) - np.pi/2.


def inf_fair(u, c=1.3998):
    '''
    The "fair" function, cited from Rey (1983, 6.4.5).
    Origin of "fair" is unknown, but was in well-known
    ROSEPACK library; see Holland and Welsch (1977).
    '''
    return u / (1.+np.absolute(u)/c)


def inf_huber(u, c=1.345):
    '''
    The Huber function, originally proposed in Huber (1964).
    '''
    return np.where(np.absolute(u) <= c, u, c*np.sign(u))


def inf_hubermod(u, c=1.2107):
    '''
    Modified Huber function from Rey (1983, 6.4.4).
    Has continuous second derivative.
    Note: we divide the original proposal by 2.
    '''
    return c * np.where(np.absolute(u) <= c*np.pi/2., np.sin(u/c), np.sign(u))


def inf_logistic(u, c1=4., c2=1.):
    '''
    Logistic function.
    '''
    return c1 / (1.+np.exp(-c2*u)) - c1/2.


def inf_tanh(u):
    '''
    Hyperbolic tangent function, the influence function
    derived from the log-hyperbolic cosine penalty function.
    '''
    return np.tanh(u)


## Helper functions for M-estimators of scale.

# These are the so-called "chi" functions and their classical
# default "beta" values.

betas = {"andrews": 0.8726193,
         "dw": 0.8579632,
         "geman_abs": 0.3851295,
         "geman_quad": 0.3443205,
         "huber2": 0.7784655,
         "tukey": 0.1994997}

chi_fns = [ a for a in betas.keys() ]

def chi_andrews(u, beta=betas["andrews"], c=1.3387):
    '''
    Andrews function from the famous Princeton study on
    robust statistics. This form is via Rey (1983, 6.4.9).
    '''
    return np.where(np.absolute(u) <= c*np.pi,
                    2*c**2*(1.-np.cos(u/c))-beta,
                    4*c**2-beta)


def chi_dw(u, beta=betas["dw"], c=2.9846):
    '''
    Dennis-Walsh function. See D and W (1978) or Rey (1983).
    '''
    return c**2 * (1.-np.exp(-(u/c)**2)) - beta


def chi_geman_abs(u, beta=betas["geman_abs"]):
    '''
    Geman function: absolute value type.
    See for example Black and Rangarajan (1996).
    '''
    return np.absolute(u) / (1.+np.absolute(u)) - beta


def chi_geman_quad(u, beta=betas["geman_quad"]):
    '''
    Geman function: quadratic type.
    See for example Black and Rangarajan (1996).
    '''
    return u**2 / (1+u**2) - beta


def chi_huber2(u, beta=betas["huber2"], c=1.5):
    '''
    Huber's proposal 2 (Huber 1964, section 11).
    Default c is from Venables and Ripley (2002, Ch.5).
    '''
    return np.clip(u**2, a_min=None, a_max=c**2)


def chi_tukey(u, beta=betas["tukey"], c=1.547):
    '''
    Tukey's biweight antiderivative function.
    Default value comes from Rousseeuw and Yohai (1984, p.261).
    '''
    return np.where(np.absolute(u) < c,
                    (u**6/(6*c**4))-(u**4/(2*c**2))+(u**2/2)-beta,
                    c**2/6-beta)


## General-purpose routine(s) for M-estimators of location.

def est_loc_fixedpt(X, s, inf_fn, thres=1e-03, iters=50):
    '''
    [Vectorized version]
    General purpose fixed-point routine for computing an
    M-estimate of location, using a generic influence function.
    Assumes X is (k,d), with k the number of observations.
    Assumes s is either a scalar or (1,d) shaped.
    
    Reference:
    Holland and Ikeda (2017), eqn. (3) and Prop. 17.
    '''
    new_theta = X.mean(axis=0, keepdims=True) # initialization.
    old_theta = None
    diff = 1.0

    for t in range(iters):
        old_theta = np.copy(new_theta)
        new_theta = s * np.mean(inf_fn((X-old_theta)/s),
                                axis=0, keepdims=True)
        new_theta += old_theta
        if np.all(np.absolute(old_theta-new_theta) <= thres):
            break

    return new_theta


## General-purpose routine(s) for M-estimators of scale.

_chi_min = 0.001

def est_scale_chi_fixedpt(X, chi_fn, thres=1e-03, iters=50):
    '''
    [Vectorized version]
    A general-purpose fixed-point routine for scale
    estimation using chi function M-estimators, which uses
    a fixed-point update.
    Assumes X is (k,d), with k the number of observations.
    
    Reference:
    Holland and Ikeda (2017), eqn. (4) and Prop. 17.
    '''
    
    # Initialize to sd, check for degeneracy.
    beta = -chi_fn(u=0.0)
    s_new = np.std(X, axis=0, keepdims=True)
    idx_bad = s_new <= 0.0
    s_new[idx_bad] = 1.0 # Just fix to keep stable.
    diff = 1.0
    for t in range(iters):
        s_old = np.copy(s_new)
        s_new = np.mean(chi_fn(u=(X/s_old)), axis=0, keepdims=True)
        s_new = np.sqrt(1+np.clip(s_new/beta, a_min=-1.0, a_max=None))
        s_new *= s_old
        s_new[s_new <= 1e-12] = _chi_min # in case of being too small.
        if np.all(np.abs(s_new-s_old) <= thres):
            break
    s_new[idx_bad] = _chi_min
    return s_new


## Alternative scale estimation routines.

def scale_madmean(X):
    '''
    Median absolute deviations (MAD) about the mean.
    Assumes X is (k,d), with k the number of observations.
    '''
    return np.median(np.absolute(X-np.mean(X, axis=0, keepdims=True)),
                     axis=0, keepdims=True)

def scale_madzero(X):
    '''
    Median absolute deviations (MAD) about zero.
    Assumes X is (k,d), with k the number of observations.
    '''
    return np.median(np.absolute(X), axis=0, keepdims=True)

def scale_madmed(X):
    '''
    Median absolute deviations (MAD) about the median.
    Assumes X is (k,d), with k the number of observations.
    '''
    return np.median(np.absolute(X-np.median(X, axis=0, keepdims=True)),
                     axis=0, keepdims=True)


## Support for legacy code.

cat12_gud = lambda X, s: est_loc_fixedpt(X=X, s=s, inf_fn=inf_gudermann)
_chi_fn_legacy = lambda u: chi_geman_quad(u=u)
scale_chi = lambda X: est_scale_chi_fixedpt(X=X, chi_fn=_chi_fn_legacy)


###############################################################################
