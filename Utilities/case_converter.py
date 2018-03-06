"""
Created on Tue Sep 29 2017

Script to convert Matpower cases from .MAT files to Pypower cases

NOTE: Converted cases need the import ** from numpy import array ** manually added at the header

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

import scipy.io as sio
from pypower.savecase import savecase
from numpy import zeros


# Name of the .MAT file (and of the saved file)
casename = 'case33bw'
# Name the struct had in Matlab ('mpc' or 'ans' as default)
structname = 'mpc'
# Path for .MAT and .PY files
casespath = 'U:/ACS-Public/58_lhe/lhe-jte/HiWi/Python/Python3/cases/'

# Load the .MAT file
# -------------------
mat = sio.loadmat(casespath + casename + '.mat', struct_as_record=False, squeeze_me=True)

# Read from the struct and create a ppc dict
# -------------------------------------------
ppc = {'baseMVA': mat[structname].baseMVA, 'bus': mat[structname].bus, 'branch': mat[structname].branch,
       'gen': mat[structname].gen}

# Following code detects the existance of only one generator and adds and additional one to prevent a Pypower savecase
# error. It must then be deleted in the created ppc case
onegen = False
try:
    twoindices = ppc['gen'][0, 0]
except IndexError:
    print('Adding extra generator that must be deleted from the converted case to avoid Pypower conversion error...\n')
    onegen = True
    gen = zeros((2, ppc['gen'].shape[0]))
    gen[0, :] = ppc['gen']
    gen[1, :] = ppc['gen']
    ppc['gen'] = gen

# Indicate if there is no gencost matrix
try:
    ppc['gencost'] = mat[structname].gencost
    if onegen:
        print('Adding extra generator cost that must be deleted from the converted case...\n')
        cost = zeros((2, ppc['gencost'].shape[0]))
        cost[0, :] = ppc['gencost']
        cost[1, :] = ppc['gencost']
        ppc['gencost'] = cost
except Exception:
    print('Warning: case does not include gencost matrix')

# Save file
# ----------
savecase(casespath + casename, ppc)
