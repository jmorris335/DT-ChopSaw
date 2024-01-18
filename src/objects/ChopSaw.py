"""
| File: chopsaw.py 
| Info: Presents the state-model for a radial-arm chop saw. 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
|- 0.1, 6 Dec 2023: Initialized
"""

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

    def __str__(self):
        """Returns a string describing the object."""
        return "ChopSaw Object, age: " + str(self.age) + "; \nContains: \n\t" + self.blade.toString()
