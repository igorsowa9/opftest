"""
Created on Mon Oct 16 2017

Nonlinear Programming (NLP) Optimal Power Flow solver with continuous variables
including variables from tap changers and shunt elements

Objective: Active power losses minimization

Input:      PyPower Case (ppc)
Output:     Solved ppc

"""
__author__ = "Javier Tendillo Ruiz"
__copyright__ = "Copyright 2017, RWTH Aachen University"
__Python_version__ = "3.6.3"

import time

import numpy as np
from nose.util import tolist
from numpy import flatnonzero as find, abs, angle as ang, real, imag
from numpy import zeros, pi
from pyomo.environ import *
from pyomo.core.base import Var, Objective, ConcreteModel, Constraint, minimize, Set, cos, sin, value
from pyomo.opt import SolverFactory
from pypower.ext2int import ext2int
from pypower.idx_brch import F_BUS, T_BUS, TAP, BR_R, BR_X, BR_B, RATE_A, PF, QF, PT, QT
from pypower.idx_bus import BUS_I, BUS_TYPE, REF, BS, PD, QD, VM, VA, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, VG
from pypower.int2ext import int2ext
from pypower.makeYbus import makeYbus
from copy import copy


def solveropfnlp_3(ppc, solver="ipopt"):
    if solver == "ipopt":
        opt = SolverFactory("ipopt", executable="/home/iso/PycharmProjects/opfLC_python3/Python3/py_solvers/ipopt-linux64/ipopt")
    if solver == "bonmin":
        opt = SolverFactory("bonmin", executable="/home/iso/PycharmProjects/opfLC_python3/Python3/py_solvers/bonmin-linux64/bonmin")
    if solver == "knitro":
        opt = SolverFactory("knitro", executable="D:/ICT/Artelys/Knitro 10.2.1/knitroampl/knitroampl")

    ppc = ext2int(ppc)      # convert to continuous indexing starting from 0

    # Gather information about the system
    # =============================================================
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    nb = bus.shape[0]       # number of buses
    ng = gen.shape[0]       # number of generators
    nl = branch.shape[0]    # number of lines

    # generator buses
    gb = tolist(np.array(gen[:, GEN_BUS]).astype(int))

    sb = find((bus[:, BUS_TYPE] == REF))    # slack bus index
    fr = branch[:, F_BUS].astype(int)       # from bus indices
    to = branch[:, T_BUS].astype(int)       # to bus indices

    tr0 = copy(branch[:, TAP])              # transformation ratios
    tr0[find(tr0 == 0)] = 1                 # set to 1 transformation ratios that are 0
    tp = find(branch[:, TAP] != 0)          # lines with tap changers
    ntp = find(branch[:, TAP] == 0)         # lines without tap changers

    # Tap changer settings
    dudtap = 0.01                           # Voltage per unit variation with tap changes
    tapmax = 10                             # Highest tap changer setting
    tapmin = -10                            # Lowest tap changer setting

    # Shunt element options
    stepmax = 1                             # maximum step of the shunt element

    Bs0 = bus[:, BS] / baseMVA              # shunt elements susceptance
    sd = find(bus[:, BS] != 0)              # buses with shunt devices

    r = branch[:, BR_R]     # branch resistances
    x = branch[:, BR_X]     # branch reactances
    b = branch[:, BR_B]     # branch susceptances

    start_time = time.clock()

    # Admittance matrix computation
    # =============================================================
    # Set tap ratios and shunt elements to neutral position
    branch[:, TAP] = 1
    bus[:, BS] = 0

    y = makeYbus(baseMVA, bus, branch)[0]   # admittance matrix
    yk = 1./(r+x*1j)                        # branch admittance
    yft = yk + 0.5j*b                       # branch admittance + susceptance
    gk = yk.real                            # branch resistance

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
    model.taps = Set(ordered=True, initialize=tp)               # Set of all lines with tap changers
    model.shunt = Set(ordered=True, initialize=sd)              # Set of buses with shunt elements

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

    # Transformation ratios
    model.tr = Var(model.taps)

    # Tap changer positions + their bounds
    model.tap = Var(model.taps, bounds=(tapmin, tapmax))

    # Shunt susceptance
    model.Bs = Var(model.shunt)

    # Shunt positions + their bounds
    model.s = Var(model.shunt, bounds=(0, 1))

    # Warm start the problem
    # ------------------------
    for i in range(nb):
        model.vm[i] = vm[i]
        model.va[i] = va[i]
        if i in gb:
            model.Pg[i] = Pg0[i]
            model.Qg[i] = Qg0[i]
    for i in range(nl):
        model.Pf[i] = vm[fr[i]] ** 2 * abs(yft[i]) / (tr0[i] ** 2) * np.cos(-ang(yft[i])) -\
            vm[fr[i]] * vm[to[i]] * abs(yk[i]) / tr0[i] * np.cos(va[fr[i]] - va[to[i]] - ang(yk[i]))
        model.Qf[i] = vm[fr[i]] ** 2 * abs(yft[i]) / (tr0[i] ** 2) * np.sin(-ang(yft[i])) -\
            vm[fr[i]] * vm[to[i]] * abs(yk[i]) / tr0[i] * np.sin(va[fr[i]] - va[to[i]] - ang(yk[i]))
        model.Pt[i] = vm[to[i]] ** 2 * abs(yft[i]) * np.cos(-ang(yft[i])) -\
            vm[to[i]] * vm[fr[i]] * abs(yk[i]) / tr0[i] * np.cos(va[to[i]] - va[fr[i]] - ang(yk[i]))
        model.Qt[i] = vm[to[i]] ** 2 * abs(yft[i]) * np.sin(-ang(yft[i])) -\
            vm[to[i]] * vm[fr[i]] * abs(yk[i]) / tr0[i] * np.sin(va[to[i]] - va[fr[i]] - ang(yk[i]))
    for i in tp:
        model.tr[i] = tr0[i]
    for i in sd:
        model.Bs[i] = Bs0[i]

    # Define constraints
    # ----------------------------

    # Equalities:
    # ------------

    # Active power flow equalities
    def powerflowact(model, i):
        bfrom_i = tp[find(fr[tp] == i)]         # branches from bus i with transformer
        bto_i = tp[find(to[tp] == i)]           # branches to bus i with transformer
        allbut_i = find(bus[:, BUS_I] != i)     # Set of other buses
        if i in gb:
            return model.Pg[i]-Pd[i] == sum(model.vm[i] * model.vm[j] * abs(y[i, j]) *
                                            cos(model.va[i] - model.va[j] - ang(y[i, j])) for j in allbut_i) - \
                   sum(model.vm[i] * model.vm[to[j]] * abs(yk[j]) * cos(model.va[i] -
                       model.va[to[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bfrom_i) - \
                   sum(model.vm[i] * model.vm[fr[j]] * abs(yk[j]) * cos(model.va[i] -
                       model.va[fr[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bto_i) + \
                   model.vm[i] ** 2 * (sum(abs(yk[j]) * (1 / model.tr[j]**2 - 1) *
                                           np.cos(- ang(yk[j])) for j in bfrom_i) + real(y[i, i]))
        else:
            return sum(model.vm[i] * model.vm[j] * abs(y[i, j]) *
                       cos(model.va[i] - model.va[j] - ang(y[i, j])) for j in allbut_i) - \
                   sum(model.vm[i] * model.vm[to[j]] * abs(yk[j]) * cos(model.va[i] -
                       model.va[to[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bfrom_i) - \
                   sum(model.vm[i] * model.vm[fr[j]] * abs(yk[j]) * cos(model.va[i] -
                       model.va[fr[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bto_i) + \
                   model.vm[i] ** 2 * (sum(abs(yk[j]) * (1 / model.tr[j]**2 - 1) *
                                           np.cos(- ang(yk[j])) for j in bfrom_i) + real(y[i, i])) == -Pd[i]

    model.const1 = Constraint(model.bus, rule=powerflowact)

    # Reactive power flow equalities
    def powerflowreact(model, i):
        bfrom_i = tp[find(fr[tp] == i)]         # branches from bus i with transformer
        bto_i = tp[find(to[tp] == i)]           # branches to bus i with transformer
        allbut_i = find(bus[:, BUS_I] != i)     # Set of other buses
        sh = sd[find(sd == i)]                  # Detect shunt elements
        if i in gb:
            return model.Qg[i]-Qd[i] == \
                   sum(model.vm[i] * model.vm[j] * abs(y[i, j]) *
                       sin(model.va[i] - model.va[j] - ang(y[i, j])) for j in allbut_i) - \
                   sum(model.vm[i] * model.vm[to[j]] * abs(yk[j]) * sin(model.va[i] - model.va[to[j]] - ang(yk[j]))
                       * (1 / model.tr[j] - 1) for j in bfrom_i) - \
                   sum(model.vm[i] * model.vm[fr[j]] * abs(yk[j]) * sin(model.va[i] - model.va[fr[j]] - ang(yk[j]))
                       * (1 / model.tr[j] - 1) for j in bto_i) + \
                   model.vm[i] ** 2 * (sum(abs(yk[j]) * (1 / model.tr[j] ** 2 - 1) * np.sin(- ang(yk[j]))
                                           for j in bfrom_i) - imag(y[i, i]) - sum(model.Bs[j] for j in sh))
        else:
            return sum(model.vm[i] * model.vm[j] * abs(y[i, j]) *
                       sin(model.va[i] - model.va[j] - ang(y[i, j])) for j in allbut_i) - \
                   sum(model.vm[i] * model.vm[to[j]] * abs(yk[j]) * sin(model.va[i] -
                       model.va[to[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bfrom_i) - \
                   sum(model.vm[i] * model.vm[fr[j]] * abs(yk[j]) * sin(model.va[i] -
                       model.va[fr[j]] - ang(yk[j])) * (1 / model.tr[j] - 1) for j in bto_i) + \
                   model.vm[i] ** 2 * (sum(abs(yk[j]) * (1 / model.tr[j] ** 2 - 1) * np.sin(- ang(yk[j]))
                                           for j in bfrom_i) - imag(y[i, i]) - sum(model.Bs[j] for j in sh)) == - Qd[i]

    model.const2 = Constraint(model.bus, rule=powerflowreact)

    # Active power from
    def pfrom(model, i):
        if i in tp:
            return model.Pf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / (model.tr[i] ** 2) * np.cos(-ang(yft[i])) - \
                                  model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) / model.tr[i] * \
                                  cos(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))
        else:
            return model.Pf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / tr0[i] ** 2 * np.cos(-ang(yft[i])) - \
                                  model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) / tr0[i] * \
                                  cos(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))

    model.const3 = Constraint(model.line, rule=pfrom)

    # Reactive power from
    def qfrom(model, i):
        if i in tp:
            return model.Qf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / (model.tr[i] ** 2) * np.sin(-ang(yft[i])) - \
                                  model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) / model.tr[i] * \
                                  sin(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))
        else:
            return model.Qf[i] == model.vm[fr[i]] ** 2 * abs(yft[i]) / tr0[i] ** 2 * np.sin(-ang(yft[i])) - \
                                  model.vm[fr[i]] * model.vm[to[i]] * abs(yk[i]) / tr0[i] * \
                                  sin(model.va[fr[i]] - model.va[to[i]] - ang(yk[i]))

    model.const4 = Constraint(model.line, rule=qfrom)

    # Active power to
    def pto(model, i):
        if i in tp:
            return model.Pt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.cos(-ang(yft[i])) - \
                                  model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) / model.tr[i] * \
                                  cos(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))
        else:
            return model.Pt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.cos(-ang(yft[i])) - \
                                  model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) / tr0[i] * \
                                  cos(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))

    model.const5 = Constraint(model.line, rule=pto)

    # Reactive power to
    def qto(model, i):
        if i in tp:
            return model.Qt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.sin(-ang(yft[i])) - \
                                  model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) / model.tr[i] * \
                                  sin(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))
        else:
            return model.Qt[i] == model.vm[to[i]] ** 2 * abs(yft[i]) * np.sin(-ang(yft[i])) - \
                                  model.vm[to[i]] * model.vm[fr[i]] * abs(yk[i]) / tr0[i] * \
                                  sin(model.va[to[i]] - model.va[fr[i]] - ang(yk[i]))

    model.const6 = Constraint(model.line, rule=qto)

    # Slack bus phase angle
    model.const7 = Constraint(expr=model.va[sb[0]] == 0)

    # Transformation ratio equalities
    def trfunc(model, i):
        return model.tr[i] == 1 + dudtap * model.tap[i]

    model.const8 = Constraint(model.taps, rule=trfunc)

    # Shunt susceptance equality
    def shuntfunc(model, i):
        return model.Bs[i] == model.s[i] / stepmax *Bs0[i]

    model.const9 = Constraint(model.shunt, rule=shuntfunc)

    # Inequalities:
    # ----------------

    # Active power generator limits Pg_min <= Pg <= Pg_max
    def genplimits(model, i):
        return Pg_min[i] <= model.Pg[i] <= Pg_max[i]

    model.const10 = Constraint(model.gen, rule=genplimits)

    # Reactive power generator limits Qg_min <= Qg <= Qg_max
    def genqlimits(model, i):
        return Qg_min[i] <= model.Qg[i] <= Qg_max[i]

    model.const11 = Constraint(model.gen, rule=genqlimits)

    # Voltage constraints ( Vmin <= V <= Vmax )
    def vlimits(model, i):
        return Vmin[i] <= model.vm[i] <= Vmax[i]

    model.const12 = Constraint(model.bus, rule=vlimits)

    # Sfrom line limit
    def sfrommax(model, i):
        return model.Pf[i]**2 + model.Qf[i]**2 <= Smax[i]**2

    model.const13 = Constraint(model.line, rule=sfrommax)

    # Sto line limit
    def stomax(model, i):
        return model.Pt[i]**2 + model.Qt[i]**2 <= Smax[i]**2

    model.const14 = Constraint(model.line, rule=stomax)

    # Set objective function
    # ------------------------
    def obj_fun(model):
        return sum(gk[i] * ((model.vm[fr[i]] / model.tr[i])**2 + model.vm[to[i]]**2 -
                            2 / model.tr[i] * model.vm[fr[i]] * model.vm[to[i]] *
                            cos(model.va[fr[i]] - model.va[to[i]])) for i in tp) + \
               sum(gk[i] * ((model.vm[fr[i]] / tr0[i]) ** 2 + model.vm[to[i]] ** 2 -
                            2 / tr0[i] * model.vm[fr[i]] * model.vm[to[i]] *
                            cos(model.va[fr[i]] - model.va[to[i]])) for i in ntp)

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
        if i in sd:
            bus[i, BS] = model.Bs[i].value * baseMVA
        bus[i, VM] = model.vm[i].value              # Bus voltage magnitudes
        bus[i, VA] = model.va[i].value*180/pi       # Bus voltage angles
    # Include Pf - Qf - Pt - Qt in the branch matrix
    branchsol = zeros((nl, 17))
    branchsol[:, :-4] = branch
    for i in range(nl):
        if i in tp:
            branchsol[i, TAP] = model.tr[i].value
        branchsol[i, PF] = model.Pf[i].value * baseMVA
        branchsol[i, QF] = model.Qf[i].value * baseMVA
        branchsol[i, PT] = model.Pt[i].value * baseMVA
        branchsol[i, QT] = model.Qt[i].value * baseMVA
    # Update gen matrix variables
    for i in range(ng):
        gen[i, PG] = model.Pg[gb[i]].value * baseMVA
        gen[i, QG] = model.Qg[gb[i]].value * baseMVA
        gen[i, VG] = bus[gb[i], VM]
    # Convert to external (original) numbering and save case results
    ppc = int2ext(ppc)
    ppc['bus'][:, 1:] = bus[:, 1:]
    branchsol[:, 0:2] = ppc['branch'][:, 0:2]
    ppc['branch'] = branchsol
    ppc['tap'] = zeros((tp.shape[0], 1))
    for i in range(tp.shape[0]):
        ppc['tap'][i] = model.tap[tp[i]].value
    ppc['shunt'] = zeros((sd.shape[0], 1))
    for i in range(sd.shape[0]):
        ppc['shunt'][i] = model.s[sd[i]].value
    ppc['gen'][:, 1:] = gen[:, 1:]
    ppc['obj'] = value(obj_fun(model))
    ppc['ploss'] = value(obj_fun(model)) * baseMVA
    ppc['et'] = et
    ppc['mt'] = mt
    ppc['success'] = 1

    # ppc solved case is returned
    return ppc
