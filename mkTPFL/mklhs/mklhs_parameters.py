"""A module to keep track of parameters for the MKLHS scheme."""

from MKLHS import *

mklhs_init_res = None
mklhs_params = None
mklhs_scheme = None

class MKLHSParameters():
    """An instance of parameters for the MKLHS scheme.
    Attributes:
        MKLHS.Params
    """
    def __init__(self, params=None, dataset_name=None, label=None):
        """ Inits Parameters with the given parameters. """
        global mklhs_init_res, mklhs_params, mklhs_scheme
        if not mklhs_init_res:
            mklhs_init_res = init_mklhs()

        if isinstance(params, LHSParams):
            mklhs_scheme = LHSScheme(params)
            mklhs_params = mklhs_scheme.params
        else:
            params = LHSParams()
            mklhs_scheme = LHSScheme(params)
            # n_str = str(hex(2**300))[2:] # n=q (mklhs_params.n is message mod)
            # mklhs_scheme.params.setN(n_str)
            mklhs_params = mklhs_scheme.params

        if dataset_name:
            mklhs_scheme.params.setDataset(dataset_name)
        if label:
            mklhs_scheme.params.setLabel(label)
        self.scheme = mklhs_scheme

    @property
    def n_str(self):
        return mklhs_params.getNStr()
    @property
    def dataset(self):
        return mklhs_params.dataset
    @property
    def label(self):
        return mklhs_params.label


def MKLHS_init():
    global mklhs_init_res, mklhs_params, mklhs_scheme
    if not mklhs_scheme:
        mklhs_scheme = MKLHSParameters().scheme
    return mklhs_scheme

def print_mklhs_parameters(mklhs_params):
    """  Prints parameters. """
    print("[ MKLHS parameters ] ")
    mklhs_params.print()