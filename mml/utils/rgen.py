'''Utilities: prepare a random data generator plus relevant stats.'''

## External modules.
import numpy as np
from scipy.stats import lognorm, norm


###############################################################################


def get_generator(name, rg=np.random.default_rng(), **kwargs):
    '''
    Takes a distribution name and paras,
    returns a data-generating function.

    Note: all data generation is done with
    numpy (not scipy), which uses the more
    modern Generator objects (from 1.17--).
    The kwargs here follow numpy namings.
    '''
    
    if name == "lognormal":
        gen = lambda n : rg.lognormal(mean=kwargs["mean"],
                                      sigma=kwargs["sigma"],
                                      size=n)
    elif name == "normal":
        gen = lambda n : rg.normal(loc=kwargs["loc"],
                                   scale=kwargs["scale"],
                                   size=n)
    else:
        raise ValueError("Please provide a proper distribution name.")

    return gen

    
def get_stats(name, **kwargs):
    '''
    Takes a distribution name and paras,
    and returns key statistics.

    Note: for stat-getting the only choice
    is to use *scipy*. We need to be careful
    to ensure the parametrization matches
    with that used in our get_generator fn,
    which runs on numpy Generator methods.
    Thus, kwargs here follow *numpy* namings.
    '''

    _sp = "mv" # moment specification for scipy.stats computations.
    
    if name == "lognormal":
        mean, var = lognorm.stats(s=kwargs["sigma"],
                                  scale=np.exp(kwargs["mean"]),
                                  moments=_sp)
    elif name == "normal":
        mean, var = norm.stats(loc=kwargs["loc"],
                               scale=kwargs["scale"],
                               moments=_sp)
    else:
        raise ValueError("Please provide a proper distribution name.")

    return {"mean": mean, "var": var}
    


###############################################################################
