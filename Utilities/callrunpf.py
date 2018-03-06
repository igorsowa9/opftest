"""
Created on Thu Sep 28 2017

Simple runpf caller with a solution Pypower case that includes Pf Qf Pt Qt in the branch array

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

from pypower.runpf import runpf
from pypower.ppoption import ppoption


def callrunpf(ppc_sol, verbose=0):
    if ppc_sol['branch'].shape[1] > 13:                     # check if more columns than expected are included
        branchcopy = ppc_sol['branch']                          # create copy of the original branch
        branch = ppc_sol['branch'][:, 0:13]                     # copy solution branch data except Pf Qf Pt Qt
        ppc_sol['branch'] = branch                              # replace the original branch

    opt = ppoption(VERBOSE=verbose, OUT_ALL=verbose)        # set verbose value (0 by default)
    r = runpf(ppc_sol, opt)                                 # execute power flow

    if 'branchcopy' in locals():                            # check if branch copy was needed
        ppc_sol['branch'] = branchcopy                          # return the original branch

    # returns the power flow solution struct
    return r[0]
