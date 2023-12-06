from src.support.support import findDefault

class Blade:
    '''
    A primitive state model of a saw blade.
    '''
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        **kwargs : dict, optional
            Optional editing of state variables during initialization. Possible arguments are:
            radius : float, default=10
                The radius of the blade in inches
        '''
        radius = findDefault(10, "radius", kwargs)