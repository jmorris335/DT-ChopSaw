"""
Random note
"""

from src.support.support import findDefault

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
        thickness : float, default=0.0022
            The thickness of the blade plate, in meters.
        kerf : float. default=0.0027
            The width of the slot left by the blade, or width of material removed, in meters.
        arbor_dia : float, default=0.015875
            The diameter of the arbor hole on the blade, in meters.
        hook : float, default=-6.0
            The hook angle of the blade teeth, in degrees.
        tooth_type : enum_type string, default='FTG'
            From: https://circularsawblade.net/ftg
            The type of tooth used in terms of in-line configuration and shape. Possible values are:
            
            - `FTG` (Flat Top Grind): Triangular, in-line geometry
            - `TCG` (Triple Chip Grind): Alternates FTG and relieved tooth shapes
            - `ATB` (Alternate Top Bevel): Alternating beleved teeth
            - `ATBR` (Alternate Top Bevel with Raker): Alternates ATB and FTG teeth (usually 4:1 ratio)
            - `HATB` (High Alternate Top Bevel): ATB with a steeper bevel angle
    ''' 
    def __init__(self, **kwargs):
        self.radius = findDefault(.092, "radius", kwargs)
        self.num_teeth = findDefault(56, "num_teeth", kwargs)
        self.thickness = findDefault(0.001, "thickness", kwargs)
        self.kerf = findDefault(0.0027, "kerf", kwargs)
        self.arbor_dia = findDefault(0.015875, "arbor_dia", kwargs)
        self.hook = findDefault(-6., "hook", kwargs)
        self.tooth_type = findDefault('FTG', "tooth_type", kwargs)

    def toString(self):
        """Returns a string describing the object."""
        return "Blade object, radius: " + str(self.radius)