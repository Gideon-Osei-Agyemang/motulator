"""
Voltage control method for a dc motor drive.

The supply voltage to the motor is varied to control the
speed of the dc motor using PWM.

"""
from typing import Callable
from dataclasses import dataclass, field
import numpy as np
from common import  Ctrl
from motulator.helpers import  Bunch, complex2abc


# %%
@dataclass
class DcMotorCtrlPars:
    """Control parameters for dc motor drive."""

    w_M_ref: Callable[[float], float] = field(
        repr=False, default=lambda t: (t > .2)*(2*np.pi*4))

    # Sampling period
    T_s: float = 200e-6

    # Bandwidths
    alpha_c: float = 2*np.pi*100
    alpha_s: float = 0.1*2*np.pi*100

    # Maximum values
    u_a_max: float = 140
    tau_M_max: float =14

    # Motor parameter estimates
    R: float = 0.5
    L: float = 0.0025
    J: float = 0.001
    k_f: float= 0.35


# %%
class DcMotorCtrl(Ctrl):
    """
    This class interconnects the subsystems of the control system and
    provides the interface to the solver.

    Parameters
    ----------
    pars : DCMotorCtrlPars
        Control parameters.

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, pars):
        super().__init__()
        self.k_f=pars.k_f
        self.T_s = pars.T_s
        self.w_M_ref=pars.w_M_ref
        self.CurrentCtrl = CurrentCtrl(pars)
        self.SpeedCtrl= SpeedCtrl(pars)
        self.pwm = PWM(pars)

    def __call__(self, mdl):
        """
        Run the main control loop.

        Parameters
        ----------
        mdl : DCMotorDrive
            Continuous-time model of a dc motor drive for getting the
            feedback signals.

        Returns
        -------
        T_s : float
            Sampling period.
        d_abc_ref : ndarray, shape (3,)
            Duty ratio references.

        """

        # Get the speed reference
        w_M_ref = self.w_M_ref(self.t)

        # Measure the feedback signals
        i_a = mdl.motor.meas_current()
        w_M = mdl.mech.meas_speed()
        u_dc = mdl.conv.meas_dc_voltage()

        #Get the armature voltage and dc-bus voltage
        u_a= u_dc* mdl.conv.q

        # Outputs
        tau_M_ref=self.SpeedCtrl.output(w_M_ref,w_M)
        i_a_ref = tau_M_ref/self.k_f
        tau_M_ref_lim=self.SpeedCtrl.realized_lim_torque(w_M_ref,w_M)
        u_a_ref=self.CurrentCtrl.output(i_a_ref,i_a)
        d_abc_ref, u_a_ref_lim=self.pwm.output(u_a_ref,u_dc)

        # Data logging
        data = Bunch(
            d_ab_ref=d_abc_ref,
            u_a_ref=u_a_ref,
            u_dc=u_dc,
            u_a=u_a,
            i_a=i_a,
            i_a_ref=i_a_ref,
            t=self.t,
            tau_M_ref=tau_M_ref,
            w_M=w_M,
            w_M_ref=w_M_ref,
            tau_M_ref_lim=tau_M_ref_lim,
            u_a_ref_lim=u_a_ref_lim,
        )
        self.save(data)

        # Update states
        self.CurrentCtrl.update(i_a_ref,i_a,u_a_ref_lim)
        self.SpeedCtrl.update(w_M_ref,w_M)
        self.pwm.update(u_a_ref_lim)
        self.update_clock(self.T_s)

        return self.T_s, d_abc_ref

class CurrentCtrl:
    """
    2DOF PI current controller.

    This controller is implemented using the disturbance-observer structure.
    The controller is mathematically identical to the 2DOF PI current controller.

    Parameters
    ----------
    pars : data object
        Control parameters.

    """

    def __init__(self,pars):
        self.L=pars.L
        self.R=pars.R
        self.alpha_c=pars.alpha_c
        self.T_s=pars.T_s
        self.k_f=pars.k_f
        self.u_a_max=pars.u_a_max

        #Controller gains for the 2DOF PI Current Controller
        self.kp=self.alpha_c*self.L
        self.ki=self.alpha_c*self.alpha_c*self.L
        self.r=self.alpha_c*self.L - self.R

        #Integral state
        self.I_c=0

    def output(self,i_a_ref,i_a):
        """
        Compute the unlimited voltage reference.

        Parameters
        ----------
        i_a_ref : float
            Current reference.
        i_a : float
            Measured current.
        u_a_max: float
            Maximum Armature Voltage

        Returns
        -------
        u_a_ref : float
            Unlimited voltage reference.
        """

        #Error signal
        e=i_a_ref-i_a

        u_a_ref=self.kp*e + self.ki*self.I_c - self.r*i_a
        if np.abs(u_a_ref) > self.u_a_max:
             u_a_ref= np.sign(u_a_ref)*self.u_a_max

        return u_a_ref


    def update(self,i_a_ref,i_a,u_a_ref_lim):
        """
        Update the Integral State

        Parameters
        ----------
        i_a_ref : float
            Current Reference
        i_a : float
            Armature Current
        u_a_ref_lim: float
            Limited Armature Voltage Reference

        """
        #Error signal
        e=i_a_ref-i_a

        #Update integral state
        u_a_ref=self.output(i_a_ref,i_a)
        self.I_c = self.I_c + self.T_s*(e + (u_a_ref_lim - u_a_ref)/self.kp)


class SpeedCtrl:
    """
    2DOF PI speed controller.

    This controller is implemented using the disturbance-observer structure.
    The controller is mathematically identical to the 2DOF PI speed controller.

    Parameters
    ----------
    pars : data object
        Control parameters.

    """

    def __init__(self,pars):
        self.J=pars.J
        self.alpha_s=pars.alpha_s
        self.T_s=pars.T_s
        self.tau_M_max=pars.tau_M_max
        self.k_f=pars.k_f

        #Controller gains for the 2DOF PI speed controller
        self.b=self.alpha_s*self.J
        self.kps=self.alpha_s*self.J
        self.kis=self.alpha_s*self.b

        #integral state
        self.I_s=0

    def output(self,w_M_ref, w_M):
        """
        Compute the unlimited torque reference.

        Parameters
        ----------
        w_M_ref : float
            Speed reference.
        w_M : float
            Measured speed.

        Returns
        -------
        tau_M_ref : float
            Unlimited torque reference.
        """
        #Error signal
        e=w_M_ref-w_M

        tau_M_ref=self.kps*e + self.kis*self.I_s - self.b*w_M

        return tau_M_ref

    def realized_lim_torque(self,w_M_ref,w_M):
        """
        Compute the limited torque reference.

        Parameters
        ----------
        w_M_ref : float
            Speed reference.
        w_M : float
            Measured speed.

        Returns
        -------
        tau_M_ref_lim : float
            Unlimited torque reference.
        """

        tau_M_ref=self.output(w_M_ref,w_M)
        tau_M_ref_lim=tau_M_ref
        if np.abs(tau_M_ref) > self.tau_M_max:
            tau_M_ref_lim= np.sign(tau_M_ref)*self.tau_M_max

        return tau_M_ref_lim

    def update(self,w_M_ref,w_M):
        """
        Update the Integral State

        Parameters
        ----------
        w_M_ref : float
            Speed Reference
        w_M : float
            Measured speed
        """
        #Error signal
        e=w_M_ref-w_M

        #Update integral state
        tau_M_ref=self.output(w_M_ref,w_M)
        tau_M_ref_lim=self.realized_lim_torque(w_M_ref,w_M)
        self.I_s = self.I_s + self.T_s*(e + (tau_M_ref_lim - tau_M_ref)/self.kps)



class PWM:
    """
    Duty ratio references and realized voltage for three-phase PWM.

    This contains the computation of the duty ratio references and the realized
    voltage. The digital delay effects are taken into account in the realized
    voltage.

    Parameters
    ----------
    pars : data object
        Control parameters.

    """

    def __init__(self, pars):
        self.T_s = pars.T_s
        self.realized_voltage=0
        self.u_a_ref_lim_old=0
        self.u_a_max=pars.u_a_max

    @staticmethod
    def duty_ratios(u_a_ref, u_dc):
        """
        Compute the duty ratios for a single phase PWM.

        This computes the duty ratios using a symmetrical suboscillation
        method. This method is identical to the standard space-vector PWM.

        Parameters
        ----------
        u_a_ref : float
            Armature Voltage reference.
        u_dc : float
            DC-bus voltage.

        Returns
        -------
        d_abc_ref : ndarray, shape (3,)
            Duty ratio references.
        """

        # Duty ratios
        d_ab= (0.5*(1+u_a_ref/u_dc)) - (0.5*(1-u_a_ref/u_dc))
        d_abc_ref=complex2abc(complex(d_ab))

        return d_abc_ref


    def __call__(self, u_a_ref, u_dc):
        """
        Compute the duty ratios and update the state.

        Parameters
        ----------
        u_ref : complex
            Voltage reference in synchronous coordinates.
        u_dc : float
            DC-bus voltage.

        Returns
        -------
        d_abc_ref : ndarray, shape (3,)
            Duty ratio references.

        """
        d_abc_ref, u_a_ref_lim = self.output(u_a_ref, u_dc)
        self.update(self, u_a_ref_lim)

        return d_abc_ref

    def output(self, u_a_ref, u_dc):
        """Compute the duty ratio .
        """

        # Duty ratios
        d_abc_ref = self.duty_ratios(u_a_ref, u_dc)
        u_a_ref_lim = u_a_ref
        if u_a_ref > self.u_a_max:
            u_a_ref_lim = d_abc_ref*u_dc

        return d_abc_ref, u_a_ref_lim

    def update(self, u_a_ref_lim):
        """
        Update the voltage estimate for the next sampling instant.

        Parameters
        ----------
        u_ref_lim : complex
            Limited voltage reference in synchronous coordinates.

        """
        self.realized_voltage = 0.5*(self.u_a_ref_lim_old + u_a_ref_lim)
        self.u_a_ref_lim_old = u_a_ref_lim