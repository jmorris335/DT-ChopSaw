"""
| File: blade.py 
| Info: Presents the state-model for a saw blade, along with associated classes
| Author: John Morris, jhmrrs@clemson.edu  
| OrganizauxProduct Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.1, 6 Dec 2023: Initialized
"""

from enum import Enum
import numpy as np

from src.aux.support import findDefault

class Blade:
    '''
    A primitive state model of a saw blade.

    Parameters
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        radius : float, default=0.092
            The radius of the blade, in meters.
        num_teeth : int, default=56
            The number of teeth on the blade.
        weight : float, default=0.01
            The weight of the blade, in kilograms
        thickness : float, default=0.0022
            The thickness of the blade plate, in meters.
        kerf : float. default=0.0027
            The width of the slot left by the blade, or width of material removed, in meters.
        arbor_dia : float, default=0.015875
            The diameter of the arbor hole on the blade, in meters.
        hook : float, default=-0.104
            The hook angle of the blade teeth, in radians.
        tooth_type : ToothType, default=ToothType.FTG
            The geometry and configuration of the teeth on the blade, inputted as a enumerated object. 
        rotational_friction : float, default=0.01
            Average force of friction resisting rotation, in N*m.
        moi : float
            The blade's moment of inertia, in kg*m^2; if not passed then the MoI is a caluclated value.
        theta : float, default=0
            The angular position of the blade, measured from an arbitrary starting point, in radians.
        phi : float, default=pi/2
            The vertical orientation of the blade measured relative to the cutting surface, so that upright is at pi/2, in radians.
        omega : float, default=0
            The angular velocity of the blade, in rad/s. 
        alpha : float, default=0
            The angular acceleration of the blade, in rad/s.
    ''' 
    def __init__(self, **kwargs):
        # Physical Constants
        self.radius = findDefault(.092, "radius", kwargs)
        self.num_teeth = findDefault(56, "num_teeth", kwargs)
        self.weight = findDefault(.01, "weight", kwargs)
        self.thickness = findDefault(0.001, "thickness", kwargs)
        self.kerf = findDefault(0.0027, "kerf", kwargs)
        self.arbor_dia = findDefault(0.015875, "arbor_dia", kwargs)
        self.hook = findDefault(-0.104, "hook", kwargs)
        self.tooth_type = findDefault(ToothType['FTG'], "tooth_type", kwargs)
        self.rotational_friction = findDefault(.01, "rotational_friction", kwargs)
        self.moi = findDefault(self.calcMomentOfInertia(), "moi", kwargs)

        # Dynamic Values
        self.theta = findDefault(0, "theta", kwargs)
        self.phi = findDefault(0, "phi", kwargs)
        self.omega = findDefault(0, "omega", kwargs)
        self.alpha = findDefault(0, "alpha", kwargs)

    def calcMomentOfInertia(self):
        """Calculates the Moment of Inertia (assuming a disc), in kg*m^2"""
        return 1 / 2 * self.weight * self.radius**2
    
    def tilt(self, delta_phi: float=0):
        """Updates the vertical orientation of the blade"""
        self.phi += delta_phi
    
    def applyTorque(self, torque: float=0):
        """Updates the state based on an applied torque."""
        self.alpha = (torque - np.sign(self.omega) * self.rotational_friction) / self.moi

    def _step(self, delta: float=0.1):
        """Updates the dynamic values of the object over a single time step."""
        self.theta += self.omega * delta
        self.omega += self.alpha * delta
        self.applyTorque(torque=0)

    def toString(self):
        """Returns a string describing the object."""
        return "Blade object, radius: " + str(self.radius)

class ToothType(Enum):
    """
    The type of tooth used in terms of in-line configuration and shape. 
    From: https://circularsawblade.net/ftg
    
    Values
    ----------
    - `FTG` (Flat Top Grind): Triangular, in-line geometry
    - `TCG` (Triple Chip Grind): Alternates FTG and relieved tooth shapes
    - `ATB` (Alternate Top Bevel): Alternating beleved teeth
    - `ATBR` (Alternate Top Bevel with Raker): Alternates ATB and FTG teeth (usually 4:1 ratio)
    - `HATB` (High Alternate Top Bevel): ATB with a steeper bevel angle
    """
    FTG = 1
    TCG = 2
    ATB = 3
    ATBR = 4
    HATB = 5