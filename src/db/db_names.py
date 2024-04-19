"""
| File: db_names.py 
| Info: Single reference for names of tables in the database and column types.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 19 Feb 2023: Initialized
"""

############################################
"""Twin table:
    Table of all digital twin objects in database.
"""
TWIN_TBL_NAME = 'Twins'
TWIN_TBL_COLS = ['name']
TWIN_TBL_VARTYPES = ['VARCHAR(50)']
############################################

############################################
"""SimData table:
    Table of all data outputted by the simulation.
"""
SIMDATA_TBL_NAME = 'SimData'
SIMDATA_TBL_COLS = ['time', 'entity_id', 'label', 'value']
SIMDATA_TBL_VARTYPES = ['DOUBLE', 'INT', 'VARCHAR(50)', 'DOUBLE']
############################################

############################################
"""Log table:
    Table of all log messages.
"""
LOG_TBL_NAME = 'Log'
LOG_TBL_COLS = ['time', 'entity_id', 'log_entry']
LOG_TBL_VARTYPES = ['DOUBLE', 'INT', 'TEXT']
############################################

############################################
"""Mocap table:
    Table of all Mocap sensor readings.
"""
MOCAP_TBL_NAME = 'Mocap'
MOCAP_TBL_COLS = ['mocap_pk', 'label', 'value', 'sequence', 'marker_pk']
MOCAP_TBL_VARTYPES = ['INT', 'VARCHAR', 'FLOAT', 'INT', 'INT']
############################################

############################################
"""MocapSequences table:
    Table of all Mocap coordinate sequences and their timestampes.
"""
MOCAPSEQ_TBL_NAME = 'MocapSequences'
MOCAPSEQ_TBL_COLS = ['sequence_pk', 'timestamp']
MOCAPSEQ_TBL_VARTYPES = ['INT', 'TIMESTAMP']
############################################

############################################
"""MocapMarkers table:
    Table of all Mocap markers and their labels.
"""
MOCAPMKR_TBL_NAME = 'MocapMarkers'
MOCAPMKR_TBL_COLS = ['marker_pk', 'label']
MOCAPMKR_TBL_VARTYPES = ['INT', 'VARCHAR']
############################################


