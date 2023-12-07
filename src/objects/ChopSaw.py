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

class ChopSaw:
    """
    Aggregate state model of a ChopSaw
    
    Parameters:
    ----------
    blade : Blade, default=Blade()
        A Blade object, constructs a new Blade object by default.
    motor : Motor, default=Motor()
        A Motor object, constructs a new Motor object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        id : int, default=1
            The Identification Number of the blade.
        age : int, default=0
            Age of the saw, in days.
        power_on : bool, default=False
            Indicates input has been given to enable power to the saw.
    """
    def __init__(self, blade: Blade=Blade(), motor: Motor=Motor(), **kwargs):
        self.blade = blade
        self.motor = motor

        self.id = findDefault(1, "id", kwargs)
        self.age = findDefault(0, "age", kwargs)
        self.power_on = findDefault(False, "power_on", kwargs)
        
    def toString(self):
        """Returns a string describing the object."""
        return "ChopSaw Object, age: " + str(self.age) + "; \nContains: \n\t" + self.blade.toString()
