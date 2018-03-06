"""
Created on Tue Sep 05 2017

Script to execute OPF selecting the Pypower case

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

from pypower.case6ww import case6ww
from pypower.case14 import case14
from case_ieee30 import case_ieee30
from pypower.case39 import case39
from pypower.case57 import case57
from case85wcost import case85wcost
from pypower.case118 import case118
from pypower.case300 import case300
from case33bw_sau import case33bw_sau
from pypower.case14 import case14

from pypower.printpf import printpf
from solveropfnlp_1 import solveropfnlp_1
from solveropfnlp_2 import solveropfnlp_2
from solveropfnlp_3 import solveropfnlp_3
from solveropfnlp_4 import solveropfnlp_4
from solveropfnlp_5 import solveropfnlp_5
from callrunpf import callrunpf
from check_error import check_error


# Load a PyPower Case
# --------------------
ppc = case14()

# Call the solver function
# -------------------------
# solvers:                 NLP
#                 solveropfnlp_X(ppc)
# X values:
#               - 1: Generation cost minimization (continuous variables from gen.)
#               - 2: Active power losses minimization (continuous variables from gen.)
#               - 3: Active power losses minimization (continuous variables from taps and shunts)
#               - 4: Active power losses minimization (continuous approx. to discrete variables from taps and shunts)
#               - 5: Active power losses minimization (discrete variables from taps and shunts)

sol = solveropfnlp_4(ppc)

# Print OPF results
# -------------------
# printpf(sol)

try:
    print('- Modeling time: %.4f s\n' % sol['mt'])          # Print modeling time (before solver) if included (NLP)
except KeyError:
    print('')
print('- Solver time: {:.4f} s\n'.format(sol['et']))        # Print elapsed time after solve command
print('- Power losses: {:.4f} MW\n'.format(sol['ploss']))
print('- Objective function value: {:.4f}\n'.format(sol['obj']))


# Run power flow with the solution
# ---------------------------------
check_error(sol)

# Check PF solution
r = callrunpf(sol)

print("Check PF...")

print(ppc['branch'][0,:])
print(sol['tap'])


