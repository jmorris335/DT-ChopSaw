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
from matplotlib.patches import Rectangle

from src.auxiliary.support import findDefault
from src.objects.twin import Twin
from db.logger import Logger

class Arm(Twin):
    """
    Primitive state model of the arm for a radial arm saw.
    
    Parameters:
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        h0_arm : float, default=0.08 meters
            The resting height for the center of the blade holder.
        l0_rotating_arm : float, default=0.125 meters
            The length between the hinge point (for theta_arm) and the center of the blade holder.
        width_arm : float, default=0.05 meters
            The diameter of the circular arm, primarily used for plotting purposes.
        l0_arm : float, default=0.15 meters
            The minimum length of the linear arm (unextended).
        gap_arm : float, default=0.12 meters
            The distance between the unextended arm and the rear guard.
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
        Twin.__init__(self, **kwargs)

        # Static Values
        self.name = findDefault("Arm", "name", kwargs)
        self.h0_arm = findDefault(0.04, "h0_arm", kwargs)
        self.l0_rotating_arm = findDefault(.125, "l0_rotating_arm", kwargs)
        self.width_arm = findDefault(.05, "width_arm", kwargs)
        self.l0_arm = findDefault(.15, "l0_slider", kwargs)
        self.gap_arm = findDefault(.12, "gap_arm", kwargs)

        # Dynamic Values
        self.x_arm = findDefault(0., "x_arm", kwargs)
        self.theta_arm = findDefault(0., "theta_arm", kwargs)
        self.phi_arm = findDefault(0., "phi_arm", kwargs)

        # Twin inherited methods/attributes overloading
        self.logger = Logger(self)
        self.patches = self.plot()

    def plot(self, x=None, y=None):
        """Returns a list of matplotlib.patches.Patch objects that represent the entity."""
        if x is None: x = - self.l0_arm - self.gap_arm
        if y is None: y = self.h0_arm - .5 * self.width_arm
        
        patches = list()
        patches.append(self.plotLinearArmPatch(x, y))
        patches.append(self.plotSliderArmPatch(x, y))
        patches.append(self.plotAngularArmPatch(x, y))
        return patches
    
    def plotLinearArmPatch(self, x, y):
        """Returns matplotlib patch object for linear arm."""
        return Rectangle(xy=(x, y), width=self.l0_arm, height=self.width_arm, 
                         animated=True, fc="black", ec='k', lw=1, label='Static Arm')
    
    def plotSliderArmPatch(self, x, y):
        """Returns matplotlib patch object for slider arm."""
        return Rectangle(xy=(x + self.l0_arm, y + self.width_arm * 0.1), 
                         width=self.x_arm, height=self.width_arm * 0.8, 
                         animated=True, fc="white", ec='k', lw=1, label='Sliding Arm')
    
    def plotAngularArmPatch(self, x, y):
        """Returns matplotlib patch object for angular arm."""
        x_ang = x + self.l0_arm + self.x_arm
        y_ang = y + self.h0_arm
        patch = Rectangle(xy=(x_ang, y_ang), width=self.l0_rotating_arm, 
                          height=self.width_arm, rotation_point=(x_ang, y_ang + .5*self.width_arm),
                          animated=True, fc="yellow", ec="k", lw=1, label='Rotating Arm')
        patch.set_angle(self.theta_arm * 180 / np.pi)
        return patch
    
    def updatePatches(self, x=None, y=None):
        """Updates patch objects of entity."""
        if x is None: x = -self.l0_arm - self.gap_arm
        if y is None: y = self.h0_arm - .5 * self.width_arm
        
        self.updateLinearArmPatch(x, y)
        self.updateSliderPatch(x, y)
        self.updateAngularArmPatch(x, y)

    def updateLinearArmPatch(self, x, y):
        """Updates xy position of linear arm patch."""
        self.patches[0].set(xy = (x, y))

    def updateSliderPatch(self, x, y):
        """Updates xy position, length of slider patch."""
        x_slider = x + self.l0_arm
        y_slider = y + self.width_arm * 0.1
        self.patches[1].set(xy = (x_slider, y_slider), width=self.x_arm)

    def updateAngularArmPatch(self, x, y):
        """Updates xy position, rotation angle of angular arm patch."""
        x_ang = x + self.l0_arm + self.x_arm
        self.patches[2].set(xy = (x_ang, y))
        self.patches[2].rotation_point = (x_ang, y + .5*self.width_arm)
        self.patches[2].set_angle(self.theta_arm * 180 / np.pi)
        
class Table(Twin):
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
        Twin.__init__(self)

        #Static Values
        self.name = findDefault("Table", "name", kwargs)

        # Dynamic Values
        self.theta_table = findDefault(0., "theta_table", kwargs)

        # Twin inherited methods/attributes overloading
        self.logger = Logger(self)
