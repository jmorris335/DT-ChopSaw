"""
| File: blade.py 
| Info: Presents the state-model for a saw blade, along with associated classes
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 6 Dec 2023: Initialized
| - 0.1, 2 Jan 2023: Basic dynamics formalized
"""

from enum import Enum
import numpy as np

from src.aux.support import findDefault
from src.aux.dynamic import DynamicBlock

class Blade(DynamicBlock):
    '''
    A primitive state model of a saw blade.

    Parameters
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        id : str, default="0"
            The identification number of the blade.
        age : int, default=0 days
            The time since the blade was first used.
        radius : float, default=0.092 meters
            The radius of the blade.
        num_teeth : int, default=56
            The number of teeth on the blade.
        weight : float, default=0.01 kg
            The weight of the blade.
        thickness : float, default=0.0022 meters
            The thickness of the blade plate.
        kerf : float. default=0.0027 meters
            The width of the slot left by the blade, or width of material removed.
        arbor_dia : float, default=0.015875 meters
            The diameter of the arbor hole on the blade.
        hook : float, default=-0.104 radians
            The hook angle of the blade teeth.
        rake : float, default=0 radians
            The rake angle of the blade teeth.
        tooth_type : ToothType, default=ToothType.FTG
            The geometry and configuration of the teeth on the blade, inputted as a enumerated object. 
        rotational_friction : float, default=0.01 N*m
            Average force of friction resisting rotation.
        moi : float; kg*m^2
            The blade's moment of inertia; if not passed then the MoI is a caluclated value.
        theta : float, default=0 radians
            The angular position of the blade, measured from an arbitrary starting point.
        phi : float, default=pi/2 radians
            The vertical orientation of the blade measured relative to the cutting surface, so that upright is at pi/2.
        phidot : float, default=0 rad/s
            The rotation of the blade around its secondary axis.
        omega : float, default=0 rad/s
            The angular velocity of the blade. 
        alpha : float, default=0 rad/s
            The angular acceleration of the blade.
    ''' 
    def __init__(self, **kwargs):
        self.id = findDefault("0", "id", kwargs)

        # Physical Constants
        self.age = findDefault(0, "age", kwargs)
        self.radius = findDefault(.092, "radius", kwargs)
        self.num_teeth = findDefault(56, "num_teeth", kwargs)
        self.weight = findDefault(.01, "weight", kwargs)
        self.thickness = findDefault(0.001, "thickness", kwargs)
        self.kerf = findDefault(0.0027, "kerf", kwargs)
        self.arbor_dia = findDefault(0.015875, "arbor_dia", kwargs)
        self.hook = findDefault(-0.104, "hook", kwargs)
        self.rake = findDefault(-0.104, "rake", kwargs)
        self.tooth_type = findDefault(ToothType['FTG'], "tooth_type", kwargs)
        self.rotational_friction = findDefault(.01, "rotational_friction", kwargs)
        self.moi = findDefault(self.calcMomentOfInertia(), "moi", kwargs)

        # Dynamic Values
        self.theta = findDefault(0, "theta", kwargs)
        self.phi = findDefault(np.pi/2, "phi", kwargs)
        self.phidot = findDefault(0, "phidot", kwargs)
        self.omega = findDefault(0, "omega", kwargs)

        # Inputs
        self.torque = findDefault(0, "torque", kwargs)

        # Set up state-space model
        self.A = [[0, 1, 0, 0],
                  [0,  -self.rotational_friction / self.moi, 0, 0,],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]
        self.B = [[0], [1 / self.moi], [0], [0]]
        super().__init__(A=self.A, B=self.B)

    def getStates(self):
        """Returns a array of the current values for the dynamic state variables."""
        return [self.theta, self.omega, self.phi, self.phidot]
    
    def getInputs(self):
        """Returns an array of the current values for the inputs."""
        return [self.torque]
    
    def setStates(self, states: list=[0., 0., 0., 0.]):
        """Sets the state variables for the object in order: theta, omega, phi, phidot."""
        if len(states) == super().getNumStates():
            self.theta, self.omega, self.phi, self.phidot = states
        else: 
            raise Exception("Wrong number of states set for blade object (ID="+str(self.id) + ")")

    def setInputs(self, inputs: list=[0.]):
        """Sets the input variables for the object in order: torque"""
        if len(inputs) == super().getNumInputs():
            self.torque = inputs[0]
        else: 
            raise Exception("Wrong number of inputs set for blade object (ID="+str(self.id) + ")")

    def step(self, dt: float=0.1):
        """Updates the dynamic values of the object over a single time step."""
        U = self.getInputs()
        X0 = self.getStates()
        self.setStates(super().step(U=U, X0=X0, dt=dt))
    
    def calcMomentOfInertia(self):
        """Calculates the Moment of Inertia (assuming a disc), in kg*m^2 about the primary axis."""
        return 1/2 * self.weight * self.radius**2 - (1/2 * self.weight * (self.arbor_dia/2)**2)
    
    def applyTorque(self, torque: float=0):
        """Updates the input based on an applied torque about the primary axis."""
        self.setInputs([torque])

    def __str__(self):
        """Returns a string describing the object."""
        return "Saw blade (ID=" + str(self.id) + ")"

class ToothType(Enum):
    """
    The type of tooth used in terms of in-line configuration and shape. 
    From: https://circularsawblade.net/ftg
    
    Values
    ----------
    - `FTG` (Flat Top Grind): Triangular, in-line geometry.
    - `TCG` (Triple Chip Grind): Alternates FTG and relieved tooth shapes.
    - `ATB` (Alternate Top Bevel): Alternating beveled teeth.
    - `ATBR` (Alternate Top Bevel with Raker): Alternates ATB and FTG teeth (usually 4:1 ratio).
    - `HATB` (High Alternate Top Bevel): ATB with a steeper bevel angle.
    """
    FTG = 1
    TCG = 2
    ATB = 3
    ATBR = 4
    HATB = 5

#TODO: Make Tooth object, with function that returns the tooth located at a specific angle