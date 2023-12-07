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

class ChopSaw:
    """
    Aggregate state model of a ChopSaw
    
    Parameters:
    ----------
    blade : Blade, default=Blade()
        A Blade object, constructs a new Blade object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:
        age : int, default=0
            Age of the saw in years.
    """
    def __init__(self, blade=Blade(),**kwargs):
        self.age = findDefault(0, "age", kwargs)
        self.blade = blade
        
    def toString(self):
        """Returns a string describing the object."""
        return "ChopSaw Object, age: " + str(self.age) + "; \nContains: \n\t" + self.blade.toString()
