"""
| File: structure.py 
| Info: Presents the state-model for a the structure of a radial-arm saw. 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - 0.1, 15 Jan 2024: Initialized
"""
import numpy as np

from src.aux.support import findDefault

class Arm:
    """
    Primitive state model of the arm for a radial arm saw.
    
    Parameters:
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        h0 : float, default=0.08 meters
            The resting height for the center of the blade holder.
        l0 : float, default=0.125 meters
            The length between the hinge point (for theta_arm) and the center of the blade holder.
        x_arm : float, default=0.0 meters
            The linear displacement of the arm in/towards (positive) and out/away from the operator.
            0 (default) is all the way towards the workpiece (fully engaged).
        theta_arm : float, default=0.0 radians
            The angular position of the blade holder wrt to the workbench, with 0 providing full 
            engagement of the blade and pi/2 being perpendicular to the workbench.
        phi_arm : float, default=0.0 radians
            The angular tilt of the blade (allowing for miter cuts), measured CCW from vertical.
    """
    def __init__(self, **kwargs):
        self.id = findDefault("0", "id", kwargs)

        # Physical Constants
        self.h0 = findDefault(0.08, "h0", kwargs)
        self.l0 = findDefault(.125, "l0", kwargs)

        # Dynamic Values
        self.x_arm = findDefault(0., "x_arm", kwargs)
        self.theta_arm = findDefault(0., "theta_arm", kwargs)
        self.phi_arm = findDefault(0., "phi_arm", kwargs)

    def setValues(self, **kwargs):
        """Basic setter."""
        for key, val in kwargs:
            if key == "x_arm": self.x_arm = val
            elif key == "theta_arm": self.theta_arm = val
            elif key == "phi_arm": self.phi_arm = val

    # def getHeightOfBladeCenter(self):
    #     """Returns the height of the blade center relative to the blade resting on the surface
    #     of the table."""
    #     return self.h0 + self.l0 * np.sin(self.theta_arm) * np.cos(self.phi_arm)

    def __str__(self):
        """Returns a string describing the object."""
        return "Arm (ID=" + str(self.id) + ")"
        
class Table:
    """
    Primitive state model of the table of a radial arm saw.
    
    Parameters:
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        theta_table : float, default=0.0 radians
            The angular position of the table in reference to the workpiece, as seen from bird's eye 
            view, with positive defined CCW from default (straight) cuts.
    """
    def __init__(self, **kwargs):
        self.id = findDefault("0", "id", kwargs)

        # Dynamic Values
        self.theta_table = findDefault(0., "theta_table", kwargs)

    def setValues(self, **kwargs):
        for key, val in kwargs:
            if key == "theta_tab": self.theta_tab = val

    def __str__(self):
        """Returns a string describing the object."""
        return "Table (ID=" + str(self.id) + ")"
