"""
| File: cut.py 
| Info: Presents the state-model for a cut made with a chopsaw on a homogeneous workpiece 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
|- 0.1, 15 Jan 2024: Initialized
"""

import numpy as np

from src.aux.support import findDefault
from src.objects.chopsaw import ChopSaw
from src.objects.workpiece import Workpiece

class Cut():
    '''
    Models the act of cutting a workpiece with a saw blade.

    Parameters
    ----------
    saw : ChopSaw, default=ChopSaw()
        The saw performing the cut, constructs a new ChopSaw object by default.
    wkp : Workpiece, default=Workpiece()
        The workpiece being cut, constructs a new Workpiece object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        h : float; meters
            Current workpiece height at place of cut.
        V : float; rad/s
            Cutting speed (tangential speed of the saw blade)
        t_chip : float; meters
            The undeformed chip thickness (average depth of the saw blade cut).

    References
    ----------
    1. Kapoor, S. G., R. E. DeVor, R. Zhu, R. Gajjela, G. Parakkal, and D. Smithey. “Development of Mechanistic Models for the Prediction of Machining Performance: Model Building Methodology.” Machining Science and Technology 2, no. 2 (December 1, 1998): 213-38. https://doi.org/10.1080/10940349808945669.
    '''
    def __init__(self, saw=ChopSaw(), workpiece=Workpiece(), **kwargs):
        self.id = findDefault("0", "id", kwargs)

        # Components
        self.saw = saw
        self.wkp = workpiece

        # Physical Values
        self.alpha_n = findDefault(self.saw.blade.rake, "alpha_n", kwargs)
        self.t_c = findDefault(self.saw.blade.kerf, "t_c", kwargs)

        # Dyanmic Values
        self.h = findDefault(self.H, "h", kwargs)

        # Calibration Constants
        self.a0 = findDefault(0., "a0", kwargs)
        self.a1 = findDefault(0., "a1", kwargs)
        self.a2 = findDefault(0., "a2", kwargs)
        self.a3 = findDefault(0., "a3", kwargs)
        self.b0 = findDefault(0., "b0", kwargs)
        self.b1 = findDefault(0., "b1", kwargs)
        self.b2 = findDefault(0., "b2", kwargs)
        self.b3 = findDefault(0., "b3", kwargs)
        temp_K = self.calcPressureCoef(self.a0, self.a1, self.a2, self.a3, self.a4)
        self.K_fric = findDefault(temp_K, "K_fric", kwargs)
        temp_K = self.calcPressureCoef(self.b0, self.b1, self.b2, self.b3, self.b4)
        self.K_norm = findDefault(temp_K, "K_norm", kwargs)

        # Inputs
        self.V = findDefault(self.saw.blade.omega*self.saw.blade.radius, "V", kwargs)
        self.theta_arm = findDefault(self.saw.arm.theta_arm, "theta_arm", kwargs)
        self.phi_arm = findDefault(self.saw.arm.phi_arm, "phi_arm", kwargs)

        # Outputs
        self.F_radial = findDefault(0., "F_radial", kwargs)
        self.torque_cut = findDefault(0., "torque_cut", kwargs)

    def calibrate(self, **kwargs):
        for key, arg in kwargs:
            if key == "a0": self.a0 = arg
            elif key == "a1": self.a1 = arg
            elif key == "a2": self.a2 = arg
            elif key == "a3": self.a3 = arg
            elif key == "a4": self.a4 = arg
            elif key == "b0": self.b0 = arg
            elif key == "b1": self.b1 = arg
            elif key == "b2": self.b2 = arg
            elif key == "b3": self.b3 = arg
            elif key == "b4": self.b4 = arg

    def calcPressureCoef(self, c0, c1, c2, c3, c4):
        """Calculates the pressure coefficients for use in Merchant's model using the regression 
        model found in Kapoor et al. (2024)."""
        temp = c0 + c1*np.log(self.t) + c2*np.log(self.V) + c3*self.alpha_n + c4*np.log(self.V)*np.log(self.t)
        return np.exp(temp)
    
    def calcCutArea(self):
        """Returns the area of the chip cross section."""
        arm_h = self.saw.arm.getHeightOfBladeCenter()
        blade_tip_h = self.saw.blade.radius*np.cos(self.saw.arm.phi_arm)
        cut_h = self.h - (arm_h - blade_tip_h)
        return self.t_c * cut_h
    
    def setInputs(self):
        pass

    def definePath(self):
        pass
    
    def __str__(self):
        """Returns a string describing the object."""
        return "Cut (ID=" + str(self.id) + ")"