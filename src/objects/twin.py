"""
| File: twin.py 
| Info: Inherited parent class of all digital twin objects.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 19 Feb 2024: Initialization
"""

from src.auxiliary.support import findDefault
from db.logger import Logger

class Twin:
    """
    A template parent class inherited by all digital twin objects.

    Parameters
    ----------
    **kwargs : dict, optional
    Optional editing of state variables during initialization. Possible arguments are:
        time_step : float, default=.01 seconds
            The difference in time between each step in the simulation.
        sim_time : float, default=0.0 seconds
            The current time in the simulation. A common use is to set to EPOCH time.

    Methods
    -------
    id -> int
        Property; Gets the DB ID for the twin.
    set(**kwargs) -> None
        Determines if any passed keyword arguments are attributes of the entity, and 
        sets them if so.
    log(*args) -> None:
        Wrapper for log function of `Logger` object.
    step(**kwargs) -> None:
        Updates all information with the twin based on any given parameters.
    """
    def __init__(self, **kwargs):
        self.name = findDefault("Twin", 'name', kwargs)
        self.time_step = findDefault(0.01, 'time_step', kwargs)
        self.sim_time = findDefault(0., 'sim_time', kwargs)

        self.logger = 'Logger type should be implemented by inheriting class.'
        self.objects = list() #list of objects aggregated (but not identifical) to twin
        self.patches = list() #list of patches belonging to the twin

    @property
    def id(self) -> int:
        """Gets the DB ID for the twin."""
        return self.logger.entity_id

    def set(self, **kwargs) -> None:
        """Determines if any passed keyword arguments are attributes of the class, and 
        sets them if so.
        
        See the class definition for a specific list of keyword arguments that can be 
        called.
        """
        for key, val in kwargs.items():
            attr = getattr(self, key, None)
            if attr is not None:
                setattr(self, key, val)
        for object in self.objects:
            object.set(**kwargs)

    def step(self) -> None:
        """Updates all information with the twin based on any given parameters.
        
        Note that the function also increases the time step.

        Notes
        -----
        The only properties that need to be updated during the step routine are those
        with relationships to other properties. Each relationship needs to be checked
        and updated in a non-arbitrary order during the step routine, so that no values 
        are out of relationship by the end of the simulation step.
        """
        self.sim_time += self.time_step
        for object in self.objects:
            object.step()

    def updatePatches(self):
        """Updates all the patches for the entity and any aggregated twins."""
        for object in self.objects:
            object.updatePatches()

    def logMessage(self, *args) -> None:
        """Wrapper for log function for `Logger` object.
        
        Parameters
        ----------
        msg : str
            The message to add to the `Logger` object.
        """
        self.logger.log(*args)

    def logData(self, *args) -> None:
        """Wrapper for addData function for `Logger` object.
        
        Parameters
        ----------
        label : str
            The name of the data to enter into the database.
        val : Any
            The value of the data being entered into the database.
        """
        self.logger.addData(*args)

    def __str__(self):
        """Returns a string describing the object."""
        return f'{self.name} object'
