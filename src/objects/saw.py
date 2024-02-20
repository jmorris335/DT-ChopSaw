"""
| File: saw.py 
| Info: Presents the state-model for a radial-arm miter saw. 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
|- 0.1, 6 Dec 2023: Initialized
"""
import numpy as np

from src.auxiliary.support import findDefault
from src.db.logger import Logger
from src.objects.twin import Twin
from src.objects.blade import Blade
from src.objects.motor import Motor
from src.objects.structure import Arm, Table

class Saw(Twin):
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

        name : str, default="Saw"
            The name of the saw entity.
        powerswitch_on : bool, default=False
            Indicates input has been given to enable power to the saw.
        supplied_voltage : float, default=18.0
            Voltage level supplied to the saw, presumably from the grid.

    Notes on Coordinate Systems
    ---------------------------
    1. Saw coordinates, or global coordinates, are defined with respect to the cutting path
    of the the saw. The lowest point that the workpiece can be cut at is defined as (0,0,0).
    This point is typically defined by the guards and table of the saw. The saw should be able
    to pass below this point, though the workpiece should never be located lower than this.
    2. Workpiece coordinates are the same as saw coordiantes, but in two dimensions. The plane
    of the coordinate space is tangent to the blade (and thus the miter angle). 
    
    """

    def __init__(self, blade: Blade=None, motor: Motor=None, arm: Arm=None, 
                 table: Table=None, **kwargs):
        Twin.__init__(self, **kwargs)

        # Components
        self.blade = blade if blade is not None else Blade()
        self.motor = motor if motor is not None else Motor()
        self.arm = arm if arm is not None else Arm()
        self.table = table if table is not None else Table()

        # Static Values
        self.name = findDefault("Saw", "name", kwargs)
        self.powerswitch_on = findDefault(False, "powerswitch_on", kwargs)
        self.supplied_voltage = findDefault(18., "supplied_voltage", kwargs)

        # Twin inherited methods/attributes overloading
        self.logger = Logger(self)
        self.objects = [self.blade, self.motor, self.arm, self.table]
        self.patches = self.blade.patches + self.arm.patches
        self.updatePatches()

    def toggleSwitch(self, power_on: bool=None) -> bool:
        """Sets the toggle switch for the saw.
        
        Parameters
        ----------
        power_on : bool, optional
            Value to set the toggle switch to; True maps to power flowing.
        
        Returns
        -------
        bool : True if switch is set to closed circuit, False otherwise.
        """
        if power_on is None:
            self.powerswitch_on = not self.powerswitch_on
        else:
            self.powerswitch_on = power_on
        if self.powerswitch_on:
            self.motor.voltage = self.supplied_voltage
        return self.powerswitch_on

    def step(self):
        """Updates all information with the saw based on any given parameters."""
        if self.powerswitch_on: 
            self.motor.voltage = self.supplied_voltage
        self.logData("blade_position_x", self.bladePosition[0])
        self.logData("blade_position_y", self.bladePosition[1])
        
        self.blade.torque += (self.motor.calcTorque())
        super().step()
        self.motor.load += self.blade.torque

    def updatePatches(self):
        """Overloads function from `Twin` to update blade patch with specific values."""
        self.arm.updatePatches()
        self.blade.updatePatches(*self.bladePosition)

    @property
    def bladePosition(self):
        """Returns the position of center of the blade in workpiece coordinates where (0,0) indicates 
        the center, lowest point on the cutting path, defined by the saw table. See module notes."""
        x = self.arm.x_arm - self.arm.gap_arm + self.arm.l0_rotating_arm * np.cos(self.arm.theta_arm)
        y = self.arm.h0_arm + self.arm.l0_rotating_arm * np.sin(self.arm.theta_arm) * np.cos(self.arm.phi_arm)
        return x, y
