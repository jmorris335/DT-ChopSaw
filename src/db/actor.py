import src.db.db_ops as db
import src.db.db_names as db_name
    
class Connection:
    def __init__(self):
        self.cnx = db.connectToDB()
        self.conn = self.cnx.cursor(buffered=True)

    def getCursor(self):
        return self.conn
    
    def __del__(self):
        self.cnx.close()
    
class DBActor:
    """Class for general read/write access to the DB.
    
    Provides binding for db_ops as a callable class.
    """
    def __init__(self):
        self.name = db_name

    def sql(self, cmd: str, values=None) -> list:
        """Executes the MySQL statement with the given values (`cmd` must be formatted as 
        an fstring)
        """
        conn = Connection()
        csr = conn.getCursor()
        if values is None:
            csr.execute(cmd)
        else:
            csr.execute(cmd, values)
        results = list(csr)
        return results

    def createDB(self, db_name: str='test_DEV')->None:
        """Creates the database in the schema."""
        conn = Connection()
        db.createDB(conn.getCursor(), db_name)

    def createTable(self, table_name)-> None:
        """Creates a table in the database.
        
        Parameters
        ----------
        table_name : str
            The name of the table to be added to the database.
        """
        conn = Connection()
        db.createTable(conn.getCursor(), table_name)

    def addColumn(self, table_name: str, column_name: str, var_type: str="TEXT")-> None:
        """Adds a column to a table in the database.
        
        Parameters
        ----------
        table_name : str
            The name of the table in the database.
        column_name : str
            The name of the column to be added to the table.
        var_type : str
            The type of variable to be sored in the column.
        """
        conn = Connection()
        db.addColumn(conn.getCursor(), table_name, column_name, var_type)

    def addColumns(self, table_name: str, column_names, var_types) -> None:
        """Adds columns to a table in the database.
        
        Parameters
        ----------
        table_name : str
            The name of the table in the database.
        column_names : list | str
            A list of names for the column to be added to the table. Can be added 
            as a single string.
        var_types : list | str
            The type of variable to be sored in the column. Can be added as a single
            string.
        """
        conn = Connection()
        if isinstance(column_names, str): 
            column_names = [column_names]
        if isinstance(var_types, str): 
            var_types = [var_types]
        
        for col in zip(column_names, var_types):
            db.addColumn(conn.getCursor(), table_name, col[0], col[1])

    def addEntry(self, table_name: str, values, columns) -> None:
        """Adds the values into the table_name.
        
        Parameters
        ----------
        table_name : str
            The name of the table in the database.
        values : list | str
            An mxn list of the values to be added to the column, with each row corresponding to
            a single entry. Can be entered as a single str.
        columns : list | str
            An 1xn list of the names of the columns in the table corresponding to the values.
            Can be entered as a single variable.
        """
        conn = Connection()
        db.addEntry(conn.getCursor(), table_name, values, columns)

    def getEntries(self, table_name: str, column_name: str=None) -> list:
        """Returns all the entries in a column in the database. Returns all the entries 
        in the table if the column is not provided.
        
        Parameters
        ----------
        table_name : str
            The name of the table in the database.
        column_name : str, optional
            The name of the column for the entries to be accessed. If not provided,
            the function returns all the entries in the table.
        """ 
        conn = Connection()
        return db.getEntries(conn.getCursor(), table_name, column_name)

    def getEntriesWhere(self, table_name: str, column_name: str, check_val: str) -> list:
        """Returns a list of entries from the `table_name` table where value in `column_name`
        equals `check_val`."""
        conn = Connection()
        return db.getEntriesWhere(conn.getCursor(), table_name, column_name, check_val)

    def getLatestEntry(self, table_name: str, column_name: str=None) -> list:
        """Returns the latest entry in the table under the specific column. Returns all
        entries in the table if column is not provided."""
        conn = Connection()
        return db.getLatestEntry(conn.getCursor(), table_name, column_name)

    def checkIfTableExists(self, table_name: str) -> bool:
        """Returns true if the table exists in the database.
        
        Parameters
        ----------
        table_name : str
            The name of the table to be checked.
        """
        conn = Connection()
        return db.checkIfTableExists(conn.getCursor(), table_name)

    def checkIfColExists(self, table_name: str, column_name: str) -> bool:
        """Returns true if the column exists.
        
        Parameters
        ----------
        table_name : str
            The name of the table in the database.
        column_name : str
            The name of the column to be checked in the table.
        """
        conn = Connection()
        return db.checkIfColExists(conn.getCursor(), table_name, column_name)
    
    def getMaxPrimaryKey(self, table_name: str) -> int:
        """Returns the maximum primary key for the given table. Returns 0 if
        no primary keys found."""
        conn = Connection()
        key =  db.getMaxPrimaryKey(conn.getCursor(), table_name)
        if isinstance(key, int):
            return key
        return 0

    def setupForeignKey(self, table_name: str, column_name: str, fk_table: str, fk_column: str) -> None:
        """Adds the designation of a foreign key to the column in the table."""
        conn = Connection()
        db.setupForeignKey(conn.getCursor(), table_name, column_name, fk_table, fk_column)

    def removeFromTable(self, table_name: str, column_name: str, value: str) -> None:
        """Removes all entries from the table where the value in `column` equals `value`."""
        conn = Connection()
        db.removeFromTable(conn.getCursor(), table_name, column_name)

    def clearTable(self, table_name: str) -> None:
        """Clears the table in the database. SUDO must be set to True."""
        conn = Connection()
        db.clearTable(conn.getCursor(), table_name)
        
    def dropTable(self, table_name: str) -> None:
        """Drops the table from the Database. SUDO must be set to True."""
        conn = Connection()
        db.dropTable(conn.getCursor(), table_name)