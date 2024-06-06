"""
| File: config.py 
| Info: Includes options that can be inherited throughout the scope as needed.
|
| Version History:
| - 0.0, 15 Feb 2023: Initialized
"""

DB_OFF = False
"""
DB_OFF : bool, default = False
    The database should not be connected to or written to. Helpful for development or
    testing the code when a DB cannot (or should not) be connected.
"""

SUDO = True
"""
SUDO : bool, default = False
    Allows extreme actions, like clearing the database.
"""