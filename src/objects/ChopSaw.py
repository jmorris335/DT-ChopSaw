"""
File: ChopSaw.py 

Info: Presents the state-model for a radial-arm chop saw. 

Author: John Morris, jhmrrs@clemson.edu  

Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  

Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

Version History:
- 0.1, 6 Dec 2023: Initialized
"""

import numpy as np

from src.support.support import findDefault


class ChopSaw:
    """Aggregate state model of a ChopSaw"""
    def __init__(self, **kwargs):
        '''
        Parameters:
        ----------
        **kwargs : dict, optional
            Optional editing of state variables during initialization. Possible arguments are:
            age : int, default=0
                Age of the saw in years.
        '''
        self.age = findDefault(0, "age", kwargs)

    


