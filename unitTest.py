import unittest

from mainCode import *
import os

# define the unit tests
class my_unit_tests(unittest.TestCase):

    def test_csv_file_exists(self):
        self.assertTrue(os.path.isfile('children-born-per-woman.csv'))


# make sure user cant input value greater than 7 for days

#i wanna add a try thingy so maybe a self.assert(equal) to an error message of "Insert a number no larger than (days)"

    # run the tests
if __name__ == "__main__":
    unittest.main()




