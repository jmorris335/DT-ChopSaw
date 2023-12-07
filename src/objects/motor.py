"""
| File: motor.py 
| Info: Presents the state-model for a DC motor, along with associated classes
| Author: John Morris, jhmrrs@clemson.edu  
| Organizaux: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.1, 6 Dec 2023: Initialized
"""

from src.aux.support import findDefault

class Motor:
    """
    A primitive state model for a DC motor.

    Parameters:
    ----------
    **kwargs : dict, optional
    Optional editing of state variables during initialization. Possible arguments are:

        id : int, default=1
            The identification number of the motor.
        typ_voltage : float, default=18.0
            Typical operating voltage for the motor, in Volts.
    """
    def __init__(self, **kwargs):
        self.id = findDefault(1, "id", kwargs)
        self.typ_voltage = findDefault(18, "typ_voltage", kwargs)