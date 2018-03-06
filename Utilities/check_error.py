"""
Created on Thu Sep 28 2017

Checks the average voltage error of an OPF solution

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

from numpy import average, abs
from callrunpf import callrunpf
from pypower.idx_bus import VM
from decimal import Decimal


def check_error(ppc_sol, verbose=1):
    r = callrunpf(ppc_sol)                                  # run power flow
    vdif = abs(ppc_sol['bus'][:, VM] - r['bus'][:, VM])     # calculate |Vopf - Vpf|
    avgerror = average(vdif)                                # get the average value
    if verbose > 0:
        print('- Average voltage error: %.2e\n' % Decimal(avgerror))

    return avgerror
