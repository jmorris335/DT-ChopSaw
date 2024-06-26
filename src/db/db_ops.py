"""
| File: db_ops.py 
| Info: Provides basic operations for working with the database.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 15 Feb 2023: Initialized
"""
import mysql.connector
from mysql.connector.connection_cext import CMySQLConnection as Connection
from mysql.connector.cursor_cext import CMySQLCursor as Cursor

from config import SUDO

def connectToDB(db_name: str=None, DEV_OPS: bool=True) -> Connection:
    """Connects to the database using the options in sql.config file.
    
    Note that autocommit is turned on by default.
    """
    options = {
        'option_files': './src/db/mysql/sql.config',
        'option_groups': 'dev' if DEV_OPS else 'client'
    }
    if db_name is not None:
        options['database'] = db_name

    cnx = mysql.connector.connect(**options)
    cnx.autocommit = True
    return cnx

def createDB(csr: Cursor, db_name: str='test_DEV'):
    """Creates the database in the schema."""
    csr.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

def createTable(csr: Cursor, table_name):
    """Creates a table in the database.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table to be added to the database.
    """
    id_name = f'{table_name.lower()}_id'

    cmd = f"CREATE TABLE `{table_name}` (" 
    cmd += f"{id_name} INT AUTO_INCREMENT PRIMARY KEY);"

    csr.execute(cmd)

def addColumn(csr: Cursor, table_name: str, column_name: str, var_type: str="TEXT"):
    """Adds a column to a table in the database.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table in the database.
    column_name : str
        The name of the column to be added to the table.
    var_type : str
        The type of variable to be sored in the column.
    """
    cmd = f'ALTER TABLE `{table_name}` ADD {column_name} {var_type}'
    csr.execute(cmd)

def addColumns(csr: Cursor, table_name: str, column_names, var_types):
    """Adds columns to a table in the database.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table in the database.
    column_names : list | str
        A list of names for the column to be added to the table. Can be added 
        as a single string.
    var_types : list | str
        The type of variable to be sored in the column. Can be added as a single
        string.
    """
    if isinstance(column_names, str): 
        column_names = [column_names]
    if isinstance(var_types, str): 
        var_types = [var_types]
    
    for col in zip(column_names, var_types):
        addColumn(csr, table_name, col[0], col[1])

def addEntry(csr: Cursor, table_name: str, values, columns):
    """Adds the values into the table_name.
     
    Parameters
    ----------
    cnx : mysql.connector.connection_cext.CMySQLConnection
        Connection to the database.
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table in the database.
    values : list | str
        An mxn list of the values to be added to the column, with each row corresponding to
        a single entry. Can be entered as a single str.
    columns : list | str
        An 1xn list of the names of the columns in the table corresponding to the values.
        Can be entered as a single variable.
    """
    if isinstance(values, str): values = [values]
    if isinstance(columns, str): columns = [columns]
         
    cmd = f"INSERT INTO `{table_name}` "
    if columns is not None:
        cmd += "("
        for col in columns:
            cmd += f"{col}, "
        cmd = cmd[:-2] + ") " #replace trailing comma
        cmd += "VALUES ("
    else:
        cmd += "VALUES (NULL, " 
    for i in range(len(values)):
         cmd += "%s, "
    cmd = cmd[:-2] + ")"

    csr.execute(cmd, values)

def getEntries(csr: Cursor, table_name: str, column_name: str=None) -> list:
    """Returns all the entries in a column in the database. Returns all the entries 
    in the table if the column is not provided.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table in the database.
    column_name : str, optional
        The name of the column for the entries to be accessed. If not provided,
        the function returns all the entries in the table.
    """ 
    if column_name is not None:
        cmd = f"SELECT {column_name} FROM `{table_name}`"
    else:
        cmd = f"SELECT * FROM `{table_name}`"
    csr.execute(cmd)
    results = csr.fetchall()

    if column_name is not None:
        return [r[:] for r in results]
    return results

def getEntriesWhere(csr: Cursor, table_name: str, column_name: str, check_val: str) -> list:
    """Returns a list of entries from the `table_name` table where value in `column_name`
    equals `check_val`."""
    cmd = f"SELECT * FROM `{table_name}` WHERE {column_name} = %(val)s"
    
    csr.execute(cmd, {'val' : check_val})
    results = csr.fetchall()
    return [r[:] for r in results]

def getLatestEntry(csr: Cursor, table_name: str, column_name: str=None) -> list:
    """Returns the latest entry in the table under the specific column. Returns all
    entries in the table if column is not provided."""
    entries = getEntries(csr, table_name, column_name)
    return entries[-1]

def checkIfTableExists(csr: Cursor, table_name: str) -> bool:
    """Returns true if the table exists in the database.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table to be checked.
    """
    cmd = "SHOW TABLES LIKE %(table_name)s"

    csr.execute(cmd, {'table_name': table_name})
    results = csr.fetchall()
    return len(results) > 0

def checkIfColExists(csr: Cursor, table_name: str, column_name: str) -> bool:
    """Returns true if the column exists.
    
    Parameters
    ----------
    csr : mysql.connector.cursor_cext.CMySQLCursor
        Cursor object for executing SQL statements to the database.
    table_name : str
        The name of the table in the database.
    column_name : str
        The name of the column to be checked in the table.
    """
    cmd = f"SHOW COLUMNS FROM `{table_name}` LIKE %(col)s"

    csr.execute(cmd, {'col' : column_name})
    results = csr.fetchall()
    return len(results) > 0

def getColumnIndex(csr: Cursor, column_name: str) -> int:
    """Returns the index for a given column."""
    col_names = [i[0] for i in csr.description]
    index = col_names.index(column_name)
    return index

def getMaxPrimaryKey(csr: Cursor, table_name: str) -> int:
    """Returns the maximum primary key for the given table, or 0 if not pks found."""
    cmd = f"SHOW KEYS FROM `{table_name}` WHERE Key_name = 'PRIMARY';"
    csr.execute(cmd, {'table_name' : table_name})
    idx = getColumnIndex(csr, "Column_name")
    pk_column_name = csr.fetchall()[0][idx]
    entries = getEntries(csr, table_name, pk_column_name)
    if len(entries) == 0: return 0
    max_pk = max([i[0] for i in entries])
    return max_pk

def setupForeignKey(csr: Cursor, table_name: str, column_name: str, fk_table: str, fk_column: str):
    """Adds the designation of a foreign key to the column in the table."""
    cmd = f"ALTER TABLE `{table_name}` ADD FOREIGN KEY (%(col)s) REFERENCES `{fk_table}`(%(fk)s);"

    csr.execute(cmd, {'col' : column_name, 'fk' : fk_column})

def removeFromTable(csr: Cursor, table_name: str, column_name: str, value: str):
    """Removes all entries from the table where the value in `column` equals `value`."""
    cmd = f"DELETE FROM `{table_name}` WHERE {column_name} %(val)s;"

    csr.execute(cmd, {'val' : value})

def clearTable(csr: Cursor, table_name: str):
    """Clears the table in the database. SUDO must be set to True."""
    if not SUDO: return
    cmd = f"DELETE FROM `{table_name}`;"

    csr.execute(cmd)
    
def dropTable(csr: Cursor, table_name: str):
    """Drops the table from the Database. SUDO must be set to True."""
    if not SUDO: return
    cmd = f"DROP TABLES `{table_name}`;"

    csr.execute(cmd)