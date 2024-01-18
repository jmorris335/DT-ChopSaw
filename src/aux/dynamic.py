from scipy import signal
import numpy as np

class DynamicBlock():
    """
    A dynamic model with simulation and display properties built in. Should be inherited by
    another class as `class someclass(DynamicBlock):` to take advantage of all methods. The __init__
    function can be called in the __init__ function to set the A, B, C, and D matrices of the model
    (according to a state space system). This is done as: `super().__init__(A, B, C, D)`.

    Parameters:
    ----------
    A : array_like
        The state matrix (n x n, where n is the number of states)
    B : array_like
        The input matrix (n x p, where p is the number of inputs)
    C : array_like, optional
        The output matrix (q x n, where q is the number of outputs). Default is to output each state.
    D : array_like, optional
        The feedforward matrix (q x p). Default is zero matrix (no feedforward).
    """
    def __init__(self, A, B, C=None, D=None):
        n_states = self.getNumStates(A)
        n_inputs = self.getNumInputs(B)
        if C is None: C = np.identity(n_states)
        n_outputs = 1 if len(np.shape(C)) == 1 else np.shape(C)[0] 
        if D is None: D = np.zeros((n_outputs, n_inputs))
        self.ss = signal.StateSpace(A, B, C, D)

    def update(self, A=None, B=None, C=None, D=None):
        """Sets the matrices of the internal state space class to the inputed values"""
        if A is not None: self.ss.A = A
        if B is not None: self.ss.B = B
        if C is not None: self.ss.C = C
        if D is not None: self.ss.C = D

    def getStates(self):
        """Should be overriden in inherited class."""
        return np.zeros((self.getNumStates, 1))
    
    def getInputs(self):
        """Should be overriden in inherited class."""
        return np.zeros((self.getNumInputs, 1))

    def step(self, U=None, X0=None, dt=0.1, ):
        """Returns the outputs for a single step of the simulation.
        
        Parameters
        ----------
        U : array_like, optional
            px1 array describing the inputs to the system over the step
        X0 : array_like, optional
            nx1 array specifying the values for the state variables at the start of the sim
        T0 : float, default=0.
            Initial starting time for the simulation
        dt : float, default=0.1
            Timestep for the simulation. The simulation will return the state values at T0 + dt

        Outputs
        ----------
        yout : array_like
            nx1 array describing the state values at the time T0 + dt
        """
        if U is None: U = np.zeros((2, self.getNumInputs()))
        elif not hasattr(U, "__len__"): U = [[U], [U]]
        elif len(np.shape(U)) == 1 or np.shape(U)[1] == 1: U = [U, U]
        (_, _, xout) = signal.lsim(self.ss, U=U, X0=X0, T=[0, dt])
        return xout[-1]
    
    def simulate(self, U, X0, T):
        """Simulates the dynamic block for the times and inputs provided"""
        return signal.lsim(self.ss, U=U, X0=X0, T=T)
    
    def getNumStates(self, A=None):
        """Returns the number of states in a state space system, calculated as 
        the number of "rows" in A.

        Parameters:
        ----------
        A : array_like | scipy.signal.lti | scipy.signal.dlti | None, optional
            The state matrix ("A") in a state space system. 
            - If array_like: Either a 1D or 2D matrix is allowed.
            - If lti or dlti: The A matrix is extracted from the system.
            - If None: The A matrix is extracted from the internal state space system
        """
        if (isinstance(A, (signal.lti, signal.dlti))): 
            ss = A._as_ss()
            A = ss.A
        elif A is None: A = self.ss.A
        return np.shape(A)[0]

    def getNumInputs(self, B=None):
        """Returns the number of inputs in a state space system, calculated as 
        the number of "columns" in B.

        Parameters:
        ----------
        B : array_like | scipy.signal.lti | scipy.signal.dlti, optional
            The input matrix ("B") in a state space system. 
            - If array_like: Either a 1D or 2D matrix is allowed.
            - If lti or dlti: The B matrix is extracted from the system.
            - If None: The B matrix is extracted from the internal state space system
        """
        if (isinstance(B, (signal.lti, signal.dlti))): 
            ss = B._as_ss()
            B = ss.B
        elif B is None: B = self.ss.B
        return 1 if len(np.shape(B)) == 1 else np.shape(B)[1] 