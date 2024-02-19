import unittest

from db.db_ops import *

class DBTest(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.conn = connectToDB('MiterSaw_DEV')
        self.csr = self.conn.cursor()

    def tearDown(self) -> None:
        self.conn.close()

    def test_db(self):
        entries = getEntries(self.csr, 'TestTable', 'num_chickens')
        self.assertIsNotNone(entries, "No entries found")