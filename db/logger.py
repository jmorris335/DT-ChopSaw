"""
| File: logger.py 
| Info: Presents class for keeping a time history of a digital twin instance
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 15 Feb 2023: Initialized
"""

import time
from unittest.mock import Mock
from config import DB_OFF, SUDO

import db.db_ops as db

class Logger:
    """General object for gathering, storing, and distributing variables for a digital twin
    instance. Can be used on discrete digital twins and aggregate digital twins. The `Logger`
    is intended to exist as a member of another class (*has-a* relationship).

    Parameters
    ----------
    entity : Any
        The digital twin instance attached to the `Logger`.

    Attributes
    ----------
    cnx : CMySQLConnection
        Connection to the database.
    csr : CMySQLCursor
        Cursor object for interacting with the database
    db_opts : list
        A list containing `csr` and `table_name`, used for convenience as *self.db_opts
    logs : list
    
    
    Notes
    -----
    1. The structure of the `Logger` is a dictionary with lists for each passed parameter. The
    last entry in each list is assumed to be the most current, so that the list acts as a chronological
    history (though time may not be the variable of change in the `Logger`).

    1. The default use case is for each instance of a digital twin to possess it's own instance
    of `Logger`. This makes it easy for `Logger` instances to hold duplicate information unless the 
    following best practice is followed: **Only parameters that are unique to the digital twin 
    instance should be added to the `Logger`.** 
    
        A common poor use case is using a `Logger` on an aggregate digital twin to store data points
        for inherited digital twin instances. Instead, the `Logger` instance for each managed 
        digital twin should store the information for the child twins.
    """
    def __init__(self, entity):
        self.cnx = db.connectToDB()
        self.cnx.autocommit = True
        self.csr = self.cnx.cursor()

        self.setupDB()
        self.entity = entity
        self.addEntityToDB()

    def __del__(self):
        self.cnx.close()

    def setupDB(self):
        """Sets up the database if not already setup."""
        self.setupTwinTable()
        self.setupDataTable()
        self.setupLogTable()

    def setupTwinTable(self):
        """Createds the twin table in the database if it does not already exist."""
        self.addTable('Twins', 'name', 'VARCHAR(50)')

    def setupDataTable(self):
        """Creates the data table in the database if it does not already exist."""
        cols = ['time', 'entity_id', 'label', 'value']
        var_types = ['DOUBLE', 'INT', 'VARCHAR(50)', 'DOUBLE']
        self.addTable('SimData', cols, var_types)

    def setupLogTable(self):
        """Creates the log table in the database if it does not already exist."""
        cols = ['time', 'entity_id', 'value']
        var_types = ['DOUBLE', 'INT', 'DOUBLE']
        self.addTable('Log', cols, var_types)

    def addTable(self, name: str, columns: list, var_types: list):
        """Adds a table to the database.
        
        Parameters
        ----------
        name : str
            The name of the table to be added to the database.
        columns : list | str
            A list of strings referencing column names to be added to the table.
        var_types : list | str
            A list of strings referencing the variable types of each column. 

        Notes
        -----
        To avoid adding any columns to the table (besides a primary key), set `columns`
        and `var_types` to be an empty list.
        """
        if db.checkIfTableExists(self.csr, name): return
        db.createTable(self.csr, name)
        db.addColumns(self.csr, name, columns, var_types)

    def addEntityToDB(self):
        """Adds the entity to the twin table and adds the entity_id to the class."""
        values = [self.entity.name]
        cols = ['name']
        db.addEntry(self.csr, 'Twins', values, cols)
        self.entity_id = self.findEntityID()
        
    def findEntityID(self):
        """Returns the ID for the most recently added entity with the name associated with the
        `Logger`."""
        entities_with_name = db.getEntriesWhere(self.csr, 'Twins', 'name', self.entity.name)
        return entities_with_name[-1][0]

    def addData(self, label: str, val: float, time_val: float=None):
        """Adds a data point to the collection.
        
        Parameters
        ----------
        label : str
            The name of the data to enter into the database.
        val : Any
            The value of the data being entered into the database.
        """
        if time_val is None:
            time_val = time.time()
        values = [time_val, self.entity_id, label, val]
        columns = ['time', 'entity_id', 'label', 'value']
        db.addEntry(self.csr, 'SimData', values, columns)

    def setData(self, name, val):
        """Adds the data point and sets the value in the entity.

        Parameters
        ----------
        name
            The name for the data point (typically a string)
        val
            The value of the data point to add
        
        Exceptions
        ----------
        NameError
            If the name is not an attribute of the entity.
        """
        setattr(self.entity, name, val)
        self.addData(name, val)
        
    def getLatestValue(self, label: str):
        """Returns the most recently added value from the `SimData` table for the entry
        with the given label.
        
        Assumes the following column order: [~, time, entity_id, label, value] for the returned 
        entries."""
        entries = db.getEntries(self.csr, 'SimData')
        sorted(entries, key=lambda e: e[1]).reverse()
        i = [e[3] for e in entries].index(label)
        return entries[i][4]
    
    def getLatestValues(self) -> dict:
        """Returns all the most recently added values and their names from the `SimData`
        table where the entity is consistent with the `Logger`.
        """
        entries = self.getAllValues()
        sorted(entries, key=lambda e: e[1]).reverse()
        out = dict()
        for e in entries:
            if e[3] not in dict:
                out[e[3]] = e[4]
        return out
    
    def getAllValues(self, label: str=None) -> list:
        """Returns all data points from the `SimData` table corresponding to the associated
        entity and label (if passed)."""
        entries = db.getEntries(self.csr, 'SimData')
        def fun(entry):
            if not entry[2] == self.entity_id: return False
            if label is not None:
                if not entry[3] == label: return False
            return True
        entries = filter(fun, entries)
        return entries
    
    def log(self, log_entry: str):
        """Adds the message to the log along with the time stamp."""
        values = [time.time(), self.entity_id, log_entry]
        cols = ['time', 'entity_id', 'log_entry']
        db.addEntry(self.csr, 'Log', values, cols)

    def resetTables(self):
        """Clears the tables associated with the logger and resets the database so that entries
        in all tables associated with the entity are removed.
        
        Note that the function can only be called if global configuration SUDO is set to True.
        """
        if not SUDO: return
        db.removeFromTable(self.csr, 'Twins', 'entity_id', self.entity_id)
        db.removeFromTable(self.csr, 'Log', 'entity_id', self.entity_id)
        db.removeFromTable(self.csr, 'SimData', 'entity_id', self.entity_id)

    def resetDB(self):
        """Removes all tables from the database affiliated with the Logger. SUDO must be true.
        """
        if not SUDO: return
        db.dropTable(self.csr, 'Twins')
        db.dropTable(self.csr, 'Log')
        db.dropTable(self.csr, 'SimData')

if DB_OFF:
    Logger = Mock(Logger)