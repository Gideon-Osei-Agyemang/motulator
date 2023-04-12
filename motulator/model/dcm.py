"""
Continous-time  model for a DC Motor.

The current is the state variable for this model and
the default values correspond to a 2.2-kW DC motor.

"""

class DcMotor:


    """
    DC motor model.


    Parameters
    ----------

    R: float
        Armature resistance.
    k_f: float
        Flux factor
    L: float
        Armature inductance

    """

    def __init__(self, R=0.5, k_f=0.35, L=0.0025):
        self.R = R
        self.L = L
        self.k_f = k_f
        #initial value
        self.i_a0=0

    def magnetic(self, i_a):

        """
        Parameters
        ----------
        i_a: Float
            Armature Current

        Returns
        ----------
        tau_M: Float
            Electromagnetic torque
        """

        tau_M = self.k_f*i_a
        return tau_M


    def f(self,i_a,w_M,u_a):

        """
        Compute the state derivative.

        Parameters
        ----------

        u_a : Float
            Terminal voltage.
        w_M : Float
            Rotor angular speed (in mechanical rad/s).
        i_a :   Float
            Armature Current

        Returns
        -------
        di_a : Float
            Time Derivative of the Armature Current
        tau_M: Float
            Electromagnetic Torque.


        Notes
        -----
        In addition to the state derivative, this method also returns the
        output signal (torque `tau_M`) needed for interconnection with
        other subsystems. This avoids overlapping computation in simulation.

        """

        tau_M = self.magnetic(i_a)
        di_a = (u_a - self.R*i_a - self.k_f*w_M)/self.L
        return [di_a], tau_M


    def meas_current(self):
        """
        Measure the armature current.

        This returns the armature current at the end of the sampling period.

        Returns
        -------
        i_a0 : float
            Armature Current

        """

        return self.i_a0