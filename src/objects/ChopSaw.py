"""
| File: chopsaw.py 
| Info: Presents the state-model for a radial-arm chop saw. 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
|- 0.1, 6 Dec 2023: Initialized
"""
import numpy as np
from matplotlib.patches import Circle

from src.aux.support import findDefault
from src.objects.blade import Blade
from src.objects.motor import Motor
from src.objects.structure import Arm, Table
from src.aux.dynamic import DynamicBlock

class ChopSaw:
    """
    Aggregate state model of a ChopSaw
    
    Parameters:
    ----------
    blade : Blade, default=Blade()
        A Blade object, constructs a new Blade object by default.
    motor : Motor, default=Motor()
        A Motor object, constructs a new Motor object by default.
    arm : Arm, default=Arm()
        Arm of the saw, constructs a new Arm object by default.
    table : Table, default=Table()
        Table of the saw (not the workbench), constructs a new Table object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        id : str, default="0"
            The identification number of the saw.
        age : int, default=0
            Age of the saw, in days.
        power_on : bool, default=False
            Indicates input has been given to enable power to the saw.

    Notes on Coordinate Systems
    ---------------------------
    1. Saw coordinates, or global coordinates, are defined with respect to the cutting path
    of the the saw. The lowest point that the workpiece can be cut at is defined as (0,0,0).
    This point is typically defined by the guards and table of the saw. The saw should be able
    to pass below this point, though the workpiece should never be located lower than this.
    2. Workpiece coordinates are the same as saw coordiantes, but in two dimensions. The plane
    of the coordinate space is tangent to the blade (and thus the miter angle). 
    
    """

    def __init__(self, blade: Blade=Blade(), motor: Motor=Motor(), arm: Arm=Arm(), 
                 table: Table=Table(), **kwargs):
        self.blade = blade
        self.motor = motor
        self.arm = arm
        self.table = table

        self.id = findDefault("0", "id", kwargs)
        self.age = findDefault(0, "age", kwargs)
        self.power_on = findDefault(False, "power_on", kwargs)

    def powerSwitchOn(self, power_on: bool=True):
        self.power_on = power_on

    def step(self):
        if self.power_on: self.motor.applyVoltage(18)
        self.motor.applyLoad()
        pass

    # Maybe delete this?
    def bladePosition(self):
        """Returns the position of center of the blade in workpiece coordinates where (0,0) indicates 
        the center, lowest point on the cutting path, defined by the saw table. See module notes."""
        x = self.arm.x_arm
        y = self.arm.h0 + self.arm.l0 * (np.sin(self.arm.theta_arm) * np.cos(self.arm.phi_arm))
        return x, y

    def __str__(self):
        """Returns a string describing the object."""
        return "ChopSaw Object, age: " + str(self.age) + "; \nContains: \n\t" + self.blade.toString()
    
    def plot(self):
        """Returns matplotlib patch object of blade."""
        return Circle(self.bladePosition(), self.blade.radius)
