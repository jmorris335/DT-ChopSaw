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
twin_tbl_name = 'Twins'
twin_tbl_cols = ['name']
twin_tbl_vartypes = ['VARCHAR(50)']
############################################

############################################
"""SimData table:
    Table of all data outputted by the simulation.
"""
simdata_tbl_name = 'SimData'
simdata_tbl_cols = ['time', 'entity_id', 'label', 'value']
simdata_tbl_vartypes = ['DOUBLE', 'INT', 'VARCHAR(50)', 'DOUBLE']
############################################

############################################
"""Log table:
    Table of all log messages.
"""
log_tbl_name = 'Log'
log_tbl_cols = ['time', 'entity_id', 'log_entry']
log_tbl_vartypes = ['DOUBLE', 'INT', 'TEXT']
############################################


