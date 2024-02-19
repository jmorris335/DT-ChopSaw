"""
| File: motor.py 
| Info: Presents the state-model for a DC motor, along with associated classes
| Author: John Morris, jhmrrs@clemson.edu  
| Organizaux: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| References: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling

| Version History:
| - 0.0, 6 Dec 2023: Initialized
"""

from src.auxiliary.support import findDefault
from db.logger import Logger
from src.auxiliary.dynamic import DynamicBlock

class Motor(DynamicBlock):
    """
    A primitive state model for a DC motor.

    Parameters:
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        id : str, default="0"
            The identification number of the motor.
        V_M : float, default=18.
            The typical supply voltage for the motor.
        K_M : float, default=0.01
            Motor constant (AKA torque constant, back-EMF constant), in N*m/A.
        J_M : float, default=0.01
            The moment of inertia for the rotor, in kg*m^2.
        B_M : float, default=0.1
            The coefficient of viscous friction, proportional to the speed of the rotor, in N*m*s.
        R_M : float, default=1.
            The electrical resistance of the armature, in ohms.
        L_M : float, default=0.5
            The electrical inductance of the copper winding, in henries        
        theta : float, default=0.0
            Angular position of motor shaft, measured CCW from the abscissa, in rad
        omega : float, default=0.0
            Angular velocity of motor shaft, with CCW as positive, in rad/s
        current : float, default=0.
            Current inherent in motor, in amps
        load : float, default=0.0
            Torque applied to the motor from the payload, in N*m
        voltage : float, default=0.0
            Voltage applied to the motor, in volts
    """
    def __init__(self, **kwargs):
        self.id = findDefault("0", "id", kwargs)
        self.name = f'Motor_{self.id}'
        self.log = Logger(self)

        # Physical Constants
        self.V_M = findDefault(18., "V_M", kwargs)
        self.K_M = findDefault(0.01, "K_M", kwargs)
        self.J_M = findDefault(0.01, "J_M", kwargs)
        self.B_M = findDefault(0.1, "B_M", kwargs)
        self.R_M = findDefault(1., "R_M", kwargs)
        self.L_M = findDefault(0.5, "L_M", kwargs)

        # Dynamic Values
        self.theta = findDefault(0., "theta", kwargs)
        self.omega = findDefault(0., "omega", kwargs)
        self.current = findDefault(0., "current", kwargs)

        # Inputs
        self.load = findDefault(0., "load", kwargs)
        self.voltage = findDefault(0., "voltage", kwargs)

        # Set up State Space model
        self.A = [[0, 1, 0],
                  [0, -self.B_M / self.J_M, self.K_M / self.J_M],
                  [0, -self.K_M / self.L_M, -self.R_M / self.L_M]]
        self.B = [[0, 0], [-1 / self.J_M, 0], [0, 1 / self.L_M]]
        super().__init__(A=self.A, B=self.B)

    def set(self, **kwargs):
        """Determines if any passed keyword arguments are attributes of the entity, and 
        sets them if so."""
        for key, val in kwargs.items():
            attr = getattr(self, key, None)
            if attr is not None:
                setattr(self, key, val)

    def getStates(self)-> list:
        """Returns a array of the current values for the dynamic state variables."""
        return [self.theta, self.omega, self.current]
    
    def getInputs(self)-> list:
        """Returns an array of the current values for the inputs."""
        return [self.load, self.voltage]
    
    def setStates(self, states: list=[0., 0., 0.]):
        """Sets the state variables for the object in order: theta, omega, phi, phidot."""
        if len(states) == super().getNumStates():
            self.theta, self.omega, self.current = states
            self.log.setData('load', states[0])
            self.log.setData('omega', states[1])
            self.log.setData('current', states[2])
        else: 
            raise Exception("Wrong number of states set for blade object (ID="+str(self.id) + ")")

    def setInputs(self, inputs: list=[0., 0.]):
        """Sets the input variables for the object in order: torque"""
        if len(inputs) == super().getNumInputs():
            self.load, self.voltage = inputs
        else: 
            raise Exception("Wrong number of inputs set for blade object (ID="+str(self.id) + ")")

    def step(self, dt: float=0.1):
        """Updates the dynamic values of the object over a single time step."""
        U = self.getInputs()
        X0 = self.getStates()
        self.setStates(super().step(U=U, X0=X0, dt=dt))
    
    def applyVoltage(self, voltage: float=0.):
        self.voltage = voltage

    def applyLoad(self, load: float=0.):
        self.load = load

    def calcTorque(self):
        """Returns the torque based off the current in the motor."""
        return self.K_M * self.current

    def __str__(self):
        return "DC Motor (ID=" + str(self.id) + ")"