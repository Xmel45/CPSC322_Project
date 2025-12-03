from mysklearn import myutils
from mysklearn import myutils
"""
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025
Programming Assignment #6
Description: This program is a helper program for PA6 copied over from previous PAs
"""
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.column_names) # Returns shape by getting the length of the data (number of rows) and length of the header (number of columns)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        col = []
        if (type(col_identifier) == str): col_num=self.column_names.index(col_identifier) # Gets the column index from the column identifier
        else: col_num = col_identifier
        if (include_missing_values):        # If it includes mising values, adds all values
            for row in range(len(self.data)):
                col.append(self.data[row][col_num ])
        else:
            for row in range(len(self.data)):   # If it doesn't include missing values, skip NAs
                if(self.data[row][col_num] == "NA"): pass
                else: col.append(self.data[row][col_num ])

        
        return col 

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for row in range(len(self.data)):
            for col in range(len(self.column_names)):
                try:
                    self.data[row][col] = float(self.data[row][col]) # Converts to numeric if possible
                except ValueError as e:                             # Otherwise it skips over
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        for row in sorted(row_indexes_to_drop,reverse = True): # Reverses pop order to go from largest to smallest to prevent indexing errors/problems
            self.data.pop(row)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        self.data = []

        with open(filename, "r") as infile:
            contents = csv.reader(infile) 
            for content in contents:
                self.data.append(content)       # Creates a table with the data from the csv file
        
                                                
        self.column_names = self.data.pop(0)  # Gets the first row of the table as the column names
        self.convert_to_numeric() # Converts all data to numeric if possible 
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open (filename,"w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names) # Writes the column names as the header
            for row in self.data:              # Writes the rest of the rows
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        cols = [self.column_names.index(col) for col in key_column_names] # Gets the indexes for the key_column_names
        seen_rows = []
        duplicate_indexes = []
        
        for index, row in enumerate(self.data):
            key = [row[i] for i in cols]
            if key in seen_rows:                # If the row has been seen before, add its index to the duplicate indexes list
                duplicate_indexes.append(index)
            else:                               # Otherwise the row is a new/unique row, add it to the seen rows list
                seen_rows.append(key)

        return duplicate_indexes 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        na_indexes = []
        for row in range(len(self.data)):
                for col in range(len(self.column_names)):
                    if(self.data[row][col]=="NA"):          # If the row contains an NA value, it is marked for removal
                        na_indexes.append(row)
        
        for ind in sorted(na_indexes, reverse=True):        # Pops the rows with NA values in reverse order to avoid indexing errors
            self.data.pop(ind)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        total = 0
        count = 0
        col = self.column_names.index(col_name)
        for row in range(len(self.data)): 
            if self.data[row][col]=="NA":       # Checks to make sure it doesn't try and add NAs to the average
                pass
            else: 
                total+=self.data[row][col]
                count+=1
        average = total/count                   # calculate average
        for row in range(len(self.data)):       # replace missing values with average
            if self.data[row][col]=="NA":
                self.data[row][col] = average

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_list =[]
        if(len(self.data)==0):          # Check for empty data
            return MyPyTable()
        
        for col in col_names:
            min = float('inf')
            max = avg = 0
            values_list =[]

            for row in range(len(self.data)):
                if(self.data[row][self.column_names.index(col)]!='NA'):
                    if(self.data[row][self.column_names.index(col)] < min): # Check for min
                        min = self.data[row][self.column_names.index(col)]

                    if(self.data[row][self.column_names.index(col)] > max): # Check for max
                        max = self.data[row][self.column_names.index(col)]
                    
                    values_list.append(self.data[row][self.column_names.index(col)]) # Add values for median
                    avg+=self.data[row][self.column_names.index(col)]       # Add total sum for average
                
            
            if(len(values_list)%2==0):          # If the list is of even length, get the two middle values and get the average
                values_list = sorted(values_list)
                mid1 = values_list[((len(values_list) // 2)-1)]
                mid2 = values_list[(len(values_list) // 2)]
                median = (mid1+mid2)/2
            else:                               # Otherwise get the middle value
                values_list = sorted(values_list)
                median = values_list[(len(values_list) // 2)]
            summary_list.append([col,min,max,(min+max)/2,avg/len(self.data),median]) # Add column summary


        return MyPyTable(data=summary_list, column_names=["attribute", "min", "max", "mid", "avg", "median"])

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_idx = [self.column_names.index(c) for c in key_column_names]
        other_key_idx = [other_table.column_names.index(c) for c in key_column_names]

        # Build lookup dictionary for other_table
        other_lookup = {}
        for row in other_table.data:
            key = tuple(row[i] for i in other_key_idx)
            other_lookup.setdefault(key, []).append(row)

        # Build combined column names
        other_non_keys = [c for c in other_table.column_names if c not in key_column_names]
        combined_columns = self.column_names + other_non_keys

        result = []

        for s_row in self.data:
            key = tuple(s_row[i] for i in self_key_idx)

            if key not in other_lookup:
                continue  # no match

            for o_row in other_lookup[key]:
                merged = s_row + [o_row[other_table.column_names.index(col)] for col in other_non_keys]
                result.append(merged)

        return MyPyTable(combined_columns, result)



    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """


        self_key_idx = [self.column_names.index(c) for c in key_column_names]
        other_key_idx = [other_table.column_names.index(c) for c in key_column_names]

        # Non-key cols on right
        other_non_keys = [c for c in other_table.column_names if c not in key_column_names]
        other_non_key_idx = [other_table.column_names.index(c) for c in other_non_keys]

        combined_columns = self.column_names + other_non_keys

        # Build dictionaries of key → rows
        self_dict = {}
        for row in self.data:
            key = tuple(row[i] for i in self_key_idx)
            self_dict.setdefault(key, []).append(row)

        other_dict = {}
        for row in other_table.data:
            key = tuple(row[i] for i in other_key_idx)
            other_dict.setdefault(key, []).append(row)

        all_keys = self_dict.keys() | other_dict.keys()
        result = []

        for key in all_keys:
            self_rows = self_dict.get(key, [None])
            other_rows = other_dict.get(key, [None])

            for s_row in self_rows:
                for o_row in other_rows:

                    # Build left side
                    if s_row is None:
                        key_map = dict(zip(key_column_names, key))
                        left_part = [
                            key_map.get(col, "NA") for col in self.column_names
                        ]
                    else:
                        left_part = s_row

                    # Build right side
                    if o_row is None:
                        right_part = ["NA"] * len(other_non_keys)
                    else:
                        right_part = [o_row[i] for i in other_non_key_idx]

                    result.append(left_part + right_part)

        return MyPyTable(combined_columns, result)


