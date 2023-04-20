
import motulator as mt
import numpy as np



base = mt.BaseValues(
    U_nom=120, I_nom=120, tau_nom=7, w_nom=314)
motor = mt.DcMotor()
mech = mt.Mechanics()
conv = mt.DC_DC_Converter()
mdl = mt.DcMotorDrive(motor, mech, conv)

pars = mt.DcMotorCtrlPars()
ctrl = mt.DcMotorCtrl(pars)

# Speed reference
times = np.array([0, .125, .25, .375, .5, .625, .75, .875, 1])*4
values = np.array([0, 0, 1, 1, 0, -1, -1, 0, 0])*base.w_nom
ctrl.w_M_ref = mt.Sequence(times, values)
# External load torque
times = np.array([0, .125, .125, .875, .875, 1])*4
values = np.array([0, 0, 1, 1, 0, 0])*base.tau_nom
mdl.mech.tau_L_t = mt.Sequence(times, values)

sim = mt.Simulation(mdl, ctrl,delay=1)
# sim.delay = Delay(1,1)
sim.simulate(t_stop=4)

mt.plot_dc(sim, t_span=(0,4), base=base)
