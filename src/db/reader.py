import src.db.db_ops as db

class Reader:
    """Class for accessing and reading from a database, without intention of editing.
    
    Provides binding for db_ops as a callable class.
    """
    def __init__(self):
        self.cnx = db.connectToDB()
        self.cnx.autocommit = True
        self.csr = self.cnx.cursor()

    def getEntries(self, table_name: str, column_name: str=None) -> list:
        """Returns all the entries in a column in the database. Returns all the entries 
        in the table if the column is not provided.
        """
        db.getEntries(self.csr, table_name, column_name)

    def getEntriesWhere(self, table_name: str, column_name: str, check_val: str) -> list:
        """Returns a list of entries from the `table_name` table where value in `column_name`
        equals `check_val`.
        """
        return db.getEntriesWhere(self.csr, table_name, column_name, check_val)

    def getLatestEntry(self, table_name: str, column_name: str=None) -> list:
        """Returns the latest entry in the table under the specific column. Returns all
        entries in the table if column is not provided.
        """
        return db.getLatestEntry(self.csr, table_name, column_name)

    def checkIfTableExists(self, table_name: str) -> bool:
        """Returns true if the table exists in the database.
        """
        return db.checkIfTableExists(self.csr, table_name)
    
    def sql(self, cmd: str, values) -> list:
        """Executes the MySQL statement with the given values (`cmd` must be formatted as 
        an fstring)
        """
        self.csr.execute(cmd, values)
        results = self.csr.fetchall()
        return results
