"""
Created on Thu Oct 12 2017

Nonlinear Programming (NLP) Optimal Power Flow solver with continuous variables

Objective: Active power generation cost minimization

Input:      PyPower Case (ppc)
Output:     Solved ppc

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

import time
import numpy as np
from nose.util import tolist
from numpy import flatnonzero as find, abs, angle as ang
from numpy import zeros, pi
from pyomo.environ import *
from pyomo.core.base import Var, Objective, ConcreteModel, Constraint, minimize, Set, cos, sin, value
from pyomo.opt import SolverFactory
from pypower.ext2int import ext2int
from pypower.idx_brch import F_BUS, T_BUS, TAP, BR_R, BR_X, BR_B, RATE_A, PF, QF, PT, QT
from pypower.idx_bus import BUS_TYPE, REF, PD, QD, VM, VA, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, VG
from pypower.int2ext import int2ext
from pypower.makeYbus import makeYbus


def solveropfnlp_1(ppc, solver="ipopt"):
    if solver == "ipopt":
        opt = SolverFactory("ipopt", executable="/home/iso/PycharmProjects/opfLC_python3/Python3/py_solvers/ipopt-linux64/ipopt")
    if solver == "bonmin":
        opt = SolverFactory("bonmin", executable="/home/iso/PycharmProjects/opfLC_python3/Python3/py_solvers/bonmin-linux64/bonmin")
    if solver == "knitro":
        opt = SolverFactory("knitro", executable="D:/ICT\Artelys/Knitro 10.2.1/knitroampl/knitroampl")

    ppc = ext2int(ppc)      # convert to continuous indexing starting from 0

    # Gather information about the system
    # =============================================================
    baseMVA, bus, gen, branch, cost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]

    nb = bus.shape[0]       # number of buses
    ng = gen.shape[0]       # number of generators
    nl = branch.shape[0]    # number of lines

    # generator buses
    gb = tolist(np.array(gen[:, GEN_BUS]).astype(int))

    sb = find((bus[:, BUS_TYPE] == REF))    # slack bus index
    fr = branch[:, F_BUS].astype(int)       # from bus indices
    to = branch[:, T_BUS].astype(int)       # to bus indices

    tr = branch[:, TAP]     # transformation ratios
    tr[find(tr == 0)] = 1   # set to 1 transformation ratios that are 0

    r = branch[:, BR_R]     # branch resistances
    x = branch[:, BR_X]     # branch reactances
    b = branch[:, BR_B]     # branch susceptances

    start_time = time.clock()

    # Admittance matrix computation
    # =============================================================
    y = makeYbus(baseMVA, bus, branch)[0]   # admittance matrix

    yk = 1./(r+x*1j)                        # branch admittance
    yft = yk + 0.5j*b                       # branch admittance + susceptance
    gk = yk.real                            # branch resistance
    yk = yk/tr                              # include /tr in yk

    # Optimization
    # =============================================================
    branch[find(branch[:, RATE_A] == 0), RATE_A] = 9999     # set undefined Sflow limit to 9999
    Smax = branch[:, RATE_A] / baseMVA                      # Max. Sflow

    # Power demand parameters
    Pd = bus[:, PD] / baseMVA
    Qd = bus[:, QD] / baseMVA

    # Max and min Pg and Qg
    Pg_max = zeros(nb)
    Pg_max[gb] = gen[:, PMAX] / baseMVA
    Pg_min = zeros(nb)
    Pg_min[gb] = gen[:, PMIN] / baseMVA
    Qg_max = zeros(nb)
    Qg_max[gb] = gen[:, QMAX] / baseMVA
    Qg_min = zeros(nb)
    Qg_min[gb] = gen[:, QMIN] / baseMVA

    # Vmax and Vmin vectors
    Vmax = bus[:, VMAX]
    Vmin = bus[:, VMIN]

    vm = bus[:, VM]
    va = bus[:, VA]*pi/180

    # create a new optimization model
    model = ConcreteModel()

    # Define sets
    # ------------
    model.bus = Set(ordered=True, initialize=range(nb))     # Set of all buses
    model.gen = Set(ordered=True, initialize=gb)                # Set of buses with generation
    model.line = Set(ordered=True, initialize=range(nl))    # Set of all lines

    # Define variables
    # -----------------
    # Voltage magnitudes vector (vm)
    model.vm = Var(model.bus)

    # Voltage angles vector (va)
    model.va = Var(model.bus)

    # Reactive power generation, synchronous machines(SM) (Qg)
    model.Qg = Var(model.gen)
    Qg0 = zeros(nb)
    Qg0[gb] = gen[:, QG]/baseMVA

    # Active power generation, synchronous machines(SM) (Pg)
    model.Pg = Var(model.gen)
    Pg0 = zeros(nb)
    Pg0[gb] = gen[:, PG] / baseMVA

    # Active and reactive power from at all branches
    model.Pf = Var(model.line)
    model.Qf = Var(model.line)

    # Active and reactive power to at all branches
    model.Pt = Var(model.line)
    model.Qt = Var(model.line)

    # Warm start the problem
    # ------------------------
    for i in range(nb):
        model.vm[i] = vm[i]
        model.va[i] = va[i]
        if i in gb:
            model.Pg[i] = Pg0[i]
            model.Qg[i] = Qg0[i]
    for i in range(nl):
        model.Pf[i] = vm[fr[i]] ** 2 * abs(yft[i]) / (tr[i] ** 2) * np.cos(-ang(yft[i])) -\
                      vm[fr[i]] * vm[to[i]] * abs(yk[i]) * np.cos(va[fr[i]] - va[to[i]] - ang(yk[i]))
        model.Qf[i] = vm[fr[i]] ** 2 * abs(yft[i]) / (tr[i] ** 2) * np.sin(-ang(yft[i])) -\
                      vm[fr[i]] * vm[to[i]] * abs(yk[i]) * np.sin(va[fr[i]] - va[to[i]] - ang(yk[i]))
        model.Pt[i] = vm[to[i]] ** 2 * abs(yft[i]) * np.cos(-ang(yft[i])) -\
                      vm[to[i]] * vm[fr[i]] * abs(yk[i]) * np.cos(va[to[i]] - va[fr[i]] - ang(yk[i]))
        model.Qt[i] = vm[to[i]] ** 2 * abs(yft[i]) * np.sin(-ang(yft[i])) -\
                      vm[to[i]] * vm[fr[i]] * abs(yk[i]) * np.sin(va[to[i]] - va[fr[i]] - ang(yk[i]))

    # Define constraints
    # ----------------------------

    # Equalities:
    # ------------

    # Active power flow equalities
    def powerflowact(model, i):
        if i in gb:
            return model.Pg[i]-Pd[i] == sum(model.vm[i]*model.vm[j]*abs(y[i, j]) *
                                            cos(model.va[i] - model.va[j] - ang(y[i, j])) for j in range(nb))
        else:
            return sum(model.vm[i]*model.vm[j]*abs(y[i, j]) * cos(model.va[i] - model.va[j] -
                                                                  ang(y[i, j])) for j in range(nb)) == -Pd[i]

    model.const1 = Constraint(model.bus, rule=powerflowact)

    # Reactive power flow equalities
    def powerflowreact(model, i):
        if i in gb:
            return model.Qg[i]-Qd[i] == sum(model.vm[i]*model.vm[j]*abs(y[i, j]) *
                                            sin(model.va[i] - model.va[j] - ang(y[i, j])) for j in range(nb))
        else:
            return sum(model.vm[i]*model.vm[j]*abs(y[i, j]) * sin(model.va[i] - model.va[j] -
                                                                  ang(y[i, j])) for j in range(nb)) == -Qd[i]

    model.const2 = Constraint(model.bus, rule=powerflowreact)

    # Active power from
    def pfrom(model, i):
        return model.Pf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / (tr[i] ** 2) * np.cos(-ang(yft[i])) - \
                              model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) * \
                              cos(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))

    model.const3 = Constraint(model.line, rule=pfrom)

    # Reactive power from
    def qfrom(model, i):
        return model.Qf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / (tr[i] ** 2) * np.sin(-ang(yft[i])) - \
                              model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) * \
                              sin(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))

    model.const4 = Constraint(model.line, rule=qfrom)

    # Active power to
    def pto(model, i):
        return model.Pt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.cos(-ang(yft[i])) - \
                              model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) * \
                              cos(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))

    model.const5 = Constraint(model.line, rule=pto)

    # Reactive power to
    def qto(model, i):
        return model.Qt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.sin(-ang(yft[i])) - \
                              model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) * \
                              sin(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))

    model.const6 = Constraint(model.line, rule=qto)

    # Slack bus phase angle
    model.const7 = Constraint(expr=model.va[sb[0]] == 0)

    # Inequalities:
    # ----------------

    # Active power generator limits Pg_min <= Pg <= Pg_max
    def genplimits(model, i):
        return Pg_min[i] <= model.Pg[i] <= Pg_max[i]

    model.const8 = Constraint(model.gen, rule=genplimits)

    # Reactive power generator limits Qg_min <= Qg <= Qg_max
    def genqlimits(model, i):
        return Qg_min[i] <= model.Qg[i] <= Qg_max[i]

    model.const9 = Constraint(model.gen, rule=genqlimits)

    # Voltage constraints ( Vmin <= V <= Vmax )
    def vlimits(model, i):
        return Vmin[i] <= model.vm[i] <= Vmax[i]

    model.const10 = Constraint(model.bus, rule=vlimits)

    # Sfrom line limit
    def sfrommax(model, i):
        return model.Pf[i]**2 + model.Qf[i]**2 <= Smax[i]**2

    model.const11 = Constraint(model.line, rule=sfrommax)

    # Sto line limit
    def stomax(model, i):
        return model.Pt[i]**2 + model.Qt[i]**2 <= Smax[i]**2

    model.const12 = Constraint(model.line, rule=stomax)

    # Set objective function
    # ------------------------
    def obj_fun(model):
        if cost[0, 3] == 2:
            return sum(model.Pg[gb[i]] * baseMVA * cost[i, 4] + cost[i, 5] for i in range(ng))
        if cost[0, 3] == 3:
            return sum((model.Pg[gb[i]] * baseMVA)**2 * cost[i, 4] + model.Pg[gb[i]] *
                       baseMVA * cost[i, 5] + cost[i, 6] for i in range(ng))
    model.obj = Objective(rule=obj_fun, sense=minimize)

    mt = time.clock() - start_time                  # Modeling time

    # Execute solve command with the selected solver
    # ------------------------------------------------
    start_time = time.clock()
    results = opt.solve(model, tee=True)
    et = time.clock() - start_time                  # Elapsed time
    print(results)

    # Update the case info with the optimized variables
    # ==================================================
    for i in range(nb):
        bus[i, VM] = model.vm[i].value              # Bus voltage magnitudes
        bus[i, VA] = model.va[i].value*180/pi       # Bus voltage angles
    # Include Pf - Qf - Pt - Qt in the branch matrix
    branchsol = zeros((nl, 17))
    branchsol[:, :-4] = branch
    for i in range(nl):
        branchsol[i, PF] = model.Pf[i].value * baseMVA
        branchsol[i, QF] = model.Qf[i].value * baseMVA
        branchsol[i, PT] = model.Pt[i].value * baseMVA
        branchsol[i, QT] = model.Qt[i].value * baseMVA
    # Update gen matrix variables
    for i in range(ng):
        gen[i, PG] = model.Pg[gb[i]].value * baseMVA
        gen[i, QG] = model.Qg[gb[i]].value * baseMVA
        gen[i, VG] = bus[gb[i], VM]
    # Calculate active power losses
    Ploss = sum(gk[i] * ((model.vm[fr[i]].value / tr[i])**2 + model.vm[to[i]].value**2 -
                         2/tr[i] * model.vm[fr[i]].value * model.vm[to[i]].value *
                         np.cos(model.va[fr[i]].value - model.va[to[i]].value)) for i in range(nl)) * baseMVA
    # Convert to external (original) numbering and save case results
    ppc = int2ext(ppc)
    ppc['bus'][:, 1:] = bus[:, 1:]
    branchsol[:, 0:2] = ppc['branch'][:, 0:2]
    ppc['branch'] = branchsol
    ppc['gen'][:, 1:] = gen[:, 1:]
    ppc['obj'] = value(obj_fun(model))
    ppc['ploss'] = Ploss
    ppc['et'] = et
    ppc['mt'] = mt
    ppc['success'] = 1

    # ppc solved case is returned
    return ppc
