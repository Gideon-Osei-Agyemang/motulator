"""
Continuous-time models for dc motor drives.

The default values correspond to a 2.2-kW dc motor.

"""
import numpy as np
from motulator.helpers import Bunch


# %%
class DcMotorDrive:
    """
    Continuous-time model for a DC motor drive.

    This interconnects the subsystems of a DC motor drive and provides
    an interface to the solver. More complicated systems could be modeled using
    a similar template.

    Parameters
    ----------
    motor : DcMotor
        DC Motor model.
    mech  : Mechanics
        Mechanics model.
    conv  : Converter
        DC-DC Converter Model
    """

    def __init__(self, motor=None, mech=None,conv=None):
        self.motor = motor
        self.mech = mech
        self.conv=conv

        # Initial time
        self.t0 = 0

        # Store the solution in these lists
        self.data= Bunch()
        self.data.t, self.data.q = [],[]
        self.data.i_a, self.data.w_M = [], []

    def get_initial_values(self):
        """
        Get the initial values.

        Returns
        -------
        x0 : list, length 2
            Initial values of the state variables.

        """
        x0 = [
            self.motor.i_a0,
            self.mech.w_M0
        ]
        return x0

    def set_initial_values(self,t0,x0):
        """
        Set the initial values.

        Parameters
        ----------
        x0 : ndarray
            Initial values of the state variables.

        """
        self.t0 = t0
        self.motor.i_a0 = x0[0]
        self.mech.w_M0 = x0[1]

    def f(self,t, x):
        """
        Compute the complete state derivative list for the solver.

        Parameters
        ----------
        t : float
            Time.
        x : ndarray
            State vector

        Returns
        -------
            list
            State derivatives.

        """
        # Unpack the states
        i_a, w_M=x

        # Interconnections: output for computing the state derivatives
        u_a = self.conv.dc_voltage(self.conv.q, self.conv.u_dc0)

        # State derivatives
        motor_f,tau_M= self.motor.f(i_a,w_M,u_a)
        mech_f = self.mech.f(t, w_M, tau_M)

        # List of state derivatives
        return motor_f + mech_f[0]

    def save(self, sol):
        """
        Save the solution.

        Parameters
        ----------
        sol : Bunch object
            Solution from the solver.

        """
        self.data.t.extend(sol.t)
        self.data.q.extend(sol.q)
        self.data.i_a.extend(sol.y[0])
        self.data.w_M.extend(sol.y[1])

    def post_process(self):
        """Transform the lists to the ndarray format and post-process them."""
        # From lists to the ndarray
        for key in self.data:
            self.data[key] = np.asarray(self.data[key])

        # Compute some useful quantities
        self.data.u_a = self.conv.dc_voltage(self.data.q, self.conv.u_dc0)
        self.data.i_a, self.data.tau_M= self.motor.f(self.data.i_a,self.data.w_M,self.data.u_a)
        self.mech.f(self.data.t, self.data.w_M, self.data.tau_M)
        self.data.tau_L = (
            self.mech.tau_L_t(self.data.t) + self.mech.tau_L_w(self.data.w_M))
