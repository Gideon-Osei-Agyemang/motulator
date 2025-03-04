"""
Vector control methods for induction motor drives.

The algorithms are written based on the inverse-Γ model.

"""
from typing import Callable
from dataclasses import dataclass, field
import numpy as np

from motulator.helpers import abc2complex, Bunch
from motulator.control.common import Ctrl, CurrentCtrl, SpeedCtrl, PWM


# %%
@dataclass
class InductionMotorVectorCtrlPars:
    """Vector control parameters for induction motor drives."""

    # pylint: disable=too-many-instance-attributes
    # Speed reference (in electrical rad/s)
    w_m_ref: Callable[[float], float] = field(
        repr=False, default=lambda t: (t > .2)*(2*np.pi*50))
    # Mode
    sensorless: bool = True
    # Sampling period
    T_s: float = 250e-6
    # Bandwidths
    alpha_c: float = 2*np.pi*200
    alpha_o: float = 2*np.pi*40  # Used only in the sensorless mode
    alpha_s: float = 2*np.pi*4
    # Sensored observer
    g = .2  # Used only in the sensored mode
    # Maximum values
    tau_M_max: float = 1.5*14.6
    i_s_max: float = 1.5*np.sqrt(2)*5
    # Nominal values
    psi_R_nom: float = .9
    u_dc_nom: float = 540
    # Motor parameter estimates (inverse-Γ model)
    R_s: float = 3.7
    R_R: float = 2.1
    L_sgm: float = .021
    L_M: float = .224
    n_p: int = 2
    J: float = .015


# %%
class InductionMotorVectorCtrl(Ctrl):
    """
    Vector control for an induction motor drive.

    This class interconnects the subsystems of the control system and
    provides the interface to the solver.

    Parameters
    ----------
    pars : InductionMotorVectorControlPars
        Control parameters.

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, pars):
        super().__init__()
        self.T_s = pars.T_s
        self.w_m_ref = pars.w_m_ref
        self.sensorless = pars.sensorless
        self.n_p = pars.n_p
        self.speed_ctrl = SpeedCtrl(pars)
        self.current_ref = CurrentRef(pars)
        self.current_ctrl = CurrentCtrl(pars)
        if self.sensorless:
            self.observer = SensorlessObserver(pars)
        else:
            self.observer = Observer(pars)
        self.pwm = PWM(pars)

    def __call__(self, mdl):
        """
        Run the main control loop.

        Parameters
        ----------
        mdl : InductionMotorDrive
            Continuous-time model of an induction motor drive for getting the
            feedback signals.

        Returns
        -------
        T_s : float
            Sampling period.
        d_abc_ref : ndarray, shape (3,)
            Duty ratio references.

        """
        # Get the speed reference
        w_m_ref = self.w_m_ref(self.t)

        # Measure the feedback signals
        i_s_abc = mdl.motor.meas_currents()  # Phase currents
        u_dc = mdl.conv.meas_dc_voltage()  # DC-bus voltage

        if not self.sensorless:
            w_m = self.n_p*mdl.mech.meas_speed()  # Rotor speed
        else:
            w_m = self.observer.w_m  # Get the estimated speed

        # Get the states
        u_s = self.pwm.realized_voltage
        psi_R = self.observer.psi_R
        theta_s = self.observer.theta_s

        # Space vector and coordinate transformation
        i_s = np.exp(-1j*theta_s)*abc2complex(i_s_abc)

        # Outputs
        tau_M_ref = self.speed_ctrl.output(w_m_ref/self.n_p, w_m/self.n_p)
        i_s_ref, tau_M_ref_lim = self.current_ref.output(tau_M_ref, psi_R)
        w_s = self.observer.output(u_s, i_s, w_m)  # w_m not used if sensorless
        u_s_ref = self.current_ctrl.output(i_s_ref, i_s, w_s)
        d_abc_ref, u_s_ref_lim = self.pwm.output(u_s_ref, u_dc, theta_s, w_s)

        # Save data
        data = Bunch(
            i_s=i_s,
            i_s_ref=i_s_ref,
            psi_R=psi_R,
            t=self.t,
            tau_M_ref_lim=tau_M_ref_lim,
            theta_s=theta_s,
            u_dc=u_dc,
            u_s=u_s,
            w_m=w_m,
            w_m_ref=w_m_ref,
            w_s=w_s,
        )
        self.save(data)

        # Update the states
        self.pwm.update(u_s_ref_lim)
        self.speed_ctrl.update(tau_M_ref_lim)
        self.current_ref.update(u_s_ref, u_dc)
        self.current_ctrl.update(u_s_ref_lim, w_s)
        self.observer.update(i_s, w_s)
        self.update_clock(self.T_s)

        return self.T_s, d_abc_ref


# %%
class CurrentRef:
    """
    Current reference calculation.

    This method includes field-weakenting operation based on the unlimited
    voltage reference feedback. The breakdown torque and current limits are
    taken into account.

    Parameters
    ----------
    pars : InductionMotorVectorCtrlPars
        Control parameters.

    Notes
    -----
    The field-weakening method and its tuning corresponds roughly to [1]_.

    References
    ----------
    .. [1] Hinkkanen, Luomi, "Braking scheme for vector-controlled induction
       motor drives equipped with diode rectifier without braking resistor,"
       IEEE Trans. Ind. Appl., 2006, https://doi.org/10.1109/TIA.2006.880852

    """

    def __init__(self, pars):
        self.T_s = pars.T_s
        self.i_s_max = pars.i_s_max
        self.L_sgm = pars.L_sgm
        self.n_p = pars.n_p
        # Local parameters for initializing the constants
        psi_R_nom = pars.psi_R_nom
        L_M = pars.L_M
        R_R = pars.R_R
        u_dc_nom = pars.u_dc_nom
        # Nominal d-axis current
        self.i_sd_nom = psi_R_nom/L_M
        # Field weakening
        self.gain_fw = 3*R_R*psi_R_nom/(pars.L_sgm*u_dc_nom)**2
        # State variable
        self.i_sd_ref = self.i_sd_nom

    def output(self, tau_M_ref, psi_R):
        """
        Compute the stator current reference.

        Parameters
        ----------
        tau_M_ref : float
            Torque reference.
        psi_R : float
            Rotor flux magnitude.

        Returns
        -------
        i_s_ref : complex
            Stator current reference.
        tau_M : float
            Limited torque reference.

        """

        def q_axis_current_limit(i_sd_ref, psi_R):
            # Priority given to the d component
            i_sq_max1 = np.sqrt(self.i_s_max**2 - i_sd_ref**2)
            # Breakdown torque limit
            i_sq_max2 = psi_R/self.L_sgm + i_sd_ref
            # q-axis current limit
            i_sq_max = np.min([i_sq_max1, i_sq_max2])
            return i_sq_max

        # q-axis current reference
        i_sq_ref = tau_M_ref/(1.5*self.n_p*psi_R) if psi_R > 0 else 0

        # Limit the current
        i_sq_max = q_axis_current_limit(self.i_sd_ref, psi_R)
        if np.abs(i_sq_ref) > i_sq_max:
            i_sq_ref = np.sign(i_sq_ref)*i_sq_max

        # Current reference
        i_s_ref = self.i_sd_ref + 1j*i_sq_ref

        # Limited torque (for the speed controller)
        tau_M_ref_lim = 1.5*self.n_p*psi_R*i_sq_ref

        return i_s_ref, tau_M_ref_lim

    def update(self, u_s_ref, u_dc):
        """
        Field-weakening based on the unlimited reference voltage.

        Parameters
        ----------
        u_s_ref : complex
            Unlimited stator voltage reference.
        u_dc : float
            DC-bus voltage.

        """
        u_s_max = u_dc/np.sqrt(3)
        self.i_sd_ref += self.T_s*self.gain_fw*(
            u_s_max**2 - np.abs(u_s_ref)**2)
        if self.i_sd_ref > self.i_sd_nom:
            self.i_sd_ref = self.i_sd_nom
        elif self.i_sd_ref < -self.i_s_max:
            self.i_sd_ref = -self.i_s_max


# %%
class SensorlessObserver:
    """
    Sensorless reduced-order flux observer.

    This observer corresponds to [3]_. The observer gain decouples the
    electrical and mechanical dynamics and allows placing the poles of the
    corresponding linearized estimation error dynamics. This implementation
    operates in estimated rotor flux coordinates.

    Parameters
    ----------
    pars : InductionMotorVectorCtrlPars
        Control parameters.

    Notes
    -----
    This implementation corresponds to (26)-(30) in [3]_ with the choice
    c = w_s**2 in (17). The closed-loop poles, cf. (40), can still be
    affected via the coefficient b > 0.

    References
    ----------
    .. [3] Hinkkanen, Harnefors, Luomi, "Reduced-order flux observers with
       stator-resistance adaptation for speed-sensorless induction motor
       drives," IEEE Trans. Power Electron., 2010,
       https://doi.org/10.1109/TPEL.2009.2039650

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, pars):
        self.T_s = pars.T_s
        self.R_s = pars.R_s
        self.R_R = pars.R_R
        self.L_sgm = pars.L_sgm
        self.L_M = pars.L_M
        self.alpha_o = pars.alpha_o
        self.zeta_inf = .2
        # Initial states
        self.theta_s, self.psi_R, self.i_s_old, self.w_m = 0, 0, 0, 0
        # Store for the update method to avoid recalculation
        self.dpsi_R, self.w_r = 0, 0

    def output(self, u_s, i_s, *_):
        """
        Compute the output.

        Parameters
        ----------
        u_s : complex
            Stator voltage in estimated rotor flux coordinates.
        i_s : complex
            Stator current in estimated rotor flux coordinates.

        Returns
        -------
        w_s : float
            Angular frequency of the rotor flux.

        """
        alpha = self.R_R/self.L_M

        # Observer gain (17) with c = w_s**2 (without the orthogonal projection
        # which is embedded into the state update)
        b = alpha + 2*self.zeta_inf*np.abs(self.w_m)
        g = b/(alpha - 1j*self.w_m)

        # Induced voltage from stator quantities, cf. (7)
        e_s = u_s - self.R_s*i_s - self.L_sgm*(i_s - self.i_s_old)/self.T_s
        # Induced voltage (8) from the rotor quantities
        e_r = self.R_R*i_s - (alpha - 1j*self.w_m)*self.psi_R

        # Angular frequency of the rotor flux vector
        den = self.psi_R + self.L_sgm*(i_s.real + g.imag*i_s.imag)
        if den > 0:
            w_s = (e_s.imag + g.imag*(e_r - e_s).real)/den
        else:
            w_s = self.w_m

        # Slip angular frequency (stored for the update method)
        self.w_r = e_r.imag/self.psi_R if self.psi_R > 0 else 0

        # Increment of the flux magnitude (stored for the update method)
        self.dpsi_R = ((1 - g.real)*(e_s.real + w_s*self.L_sgm*i_s.imag) +
                       g.real*e_r.real)

        return w_s

    def update(self, i_s, w_s):
        """Update the states for the next sampling period."""
        self.w_m += self.T_s*self.alpha_o*(w_s - self.w_r)
        self.psi_R += self.T_s*self.dpsi_R
        self.theta_s += self.T_s*w_s  # Next line: limit into [-pi, pi)
        self.theta_s = np.mod(self.theta_s + np.pi, 2*np.pi) - np.pi
        self.i_s_old = i_s


# %%
class Observer:
    """
    Sensored reduced-order flux observer.

    This reduced-order flux observer [4]_ uses the measured rotor speed. The
    selected default gain allows smooth transition from the current model at
    zero speed to the (damped) voltage model at higher speeds.

    Parameters
    ----------
    pars : InductionMotorVectorCtrlPars
        Control parameters.

    Notes
    -----
    This implementation places the pole in synchronous coordinates at::

        s = -R_R/L_M - g*abs(w_m) - 1j*(w_s - w_m)

    References
    ----------
    .. [4] Verghese, Sanders, “Observers for flux estimation in induction
       machines,” IEEE Trans. Ind. Electron., 1988,
       https://doi.org/10.1109/41.3067

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, pars):
        self.T_s = pars.T_s
        self.R_s = pars.R_s
        self.R_R = pars.R_R
        self.L_sgm = pars.L_sgm
        self.alpha = pars.R_R/pars.L_M
        # Nonnegative gain for damping
        self.g = pars.g
        # Initial states
        self.theta_s, self.psi_R, self.i_s_old = 0, 0, 0
        # Store for the update method to avoid recalculation
        self.dpsi_R = 0

    def output(self, u_s, i_s, w_m):
        """
        Compute the output of the observer.

        Parameters
        ----------
        u_s : complex
            Stator voltage in estimated rotor flux coordinates.
        i_s : complex
            Stator current in estimated rotor flux coordinates.
        w_m : float
            Rotor angular speed (in electrical rad/s)

        Returns
        -------
        w_s : float
            Angular frequency of the rotor flux.

        """
        # The observer pole could be placed arbitrarily by means of the
        # observer gain k. The current model would be obtained using k = 1,
        # resulting in the pole at s = -R_R/L_M - 1j*(w_s - w_m). The pure
        # voltage model corresponds to k = 0, resulting in the marginally
        # stable pole at s = -1j*w_s.

        # This gain leads to s = -alpha - g*abs(w_m) - 1j*(w_s - w_m)
        k = 1 + self.g*np.abs(w_m)/(self.alpha - 1j*w_m)

        # Induced voltage from the stator quantities
        e_s = u_s - self.R_s*i_s - self.L_sgm*(i_s - self.i_s_old)/self.T_s
        # Induced voltage from the rotor quantities
        e_r = self.R_R*i_s - (self.alpha - 1j*w_m)*self.psi_R

        # Angular frequency of the rotor flux vector
        den = self.psi_R + self.L_sgm*((1 - k)*i_s).real
        v = (1 - k)*e_s + k*e_r
        w_s = v.imag/den if den > 0 else w_m

        # Increment of the flux magnitude (stored for the update method)
        self.dpsi_R = v.real + w_s*self.L_sgm*((1 - k)*i_s).imag

        return w_s

    def update(self, i_s, w_s):
        """Update the states for the next sampling period."""
        self.psi_R += self.T_s*self.dpsi_R
        self.theta_s += self.T_s*w_s  # Next line: limit into [-pi, pi)
        self.theta_s = np.mod(self.theta_s + np.pi, 2*np.pi) - np.pi
        self.i_s_old = i_s
