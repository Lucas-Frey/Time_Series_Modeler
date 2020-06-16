import pandas as pd
import numpy as np

from Utils.MemoryReducer import memory_magic
from Base.BaseReader import BaseReader


class SalesTrainValidationReader(BaseReader):

    def __init__(self, file_path, create_sample=False):

        super(SalesTrainValidationReader, self).__init__(file_path, create_sample)

    def process(self):

        self.df = self.df.sample(15).iloc[:, np.r_[0:6, len(self.df.columns) - 750:len(self.df.columns)]]

        self.printer.print_begin('Processing the dataframe...')

        # Code that calls the memory_magic function to immediately cut down on memory consumption.

        self.df = memory_magic(self.df, self.printer)

        self.printer.print_begin('Formatting the d column...')
        self.df['id'] = self.df['id'].apply(lambda val:
                                            str(val).rsplit('_', 1)[0])
        self.printer.print_end('Formatting the d column!')

        # Code that removes unneeded columns for statistical testing. Statistical testing requires only
        # the target variable for feature engineering. This will likely be removed from the final notebook.

        self.printer.print_begin('Dropping unnecessary columns...')
        self.df.drop('item_id', axis=1, inplace=True)
        self.df.drop('dept_id', axis=1, inplace=True)
        self.df.drop('cat_id', axis=1, inplace=True)
        self.df.drop('store_id', axis=1, inplace=True)
        self.df.drop('state_id', axis=1, inplace=True)
        self.printer.print_end('Dropped unnecessary columns!')

        # Code for the melting the sales_train_df dataframe. This will take the 1969 d_# columns and turn them
        # into one column. This creates a drastically larger dataframe.

        self.printer.print_begin('Melting the dataframe...')
        melt_id_vars = ['id']
        self.df = pd.melt(self.df, id_vars=melt_id_vars, var_name='d', value_name='demand')
        self.printer.print_end('Melted the dataframe!')

        # Code that will format the d column. It will convert the d_# to just the #.

        self.printer.print_begin('Formatting the d column...')
        self.df['d'] = pd.to_numeric(self.df['d'].apply(lambda x: str(x).split('_')[1]))
        self.printer.print_end('Formatting the d column!')

        self.df = memory_magic(self.df, self.printer)

        self.printer.print_end('Processed the dataframe!')

    def create_sample_df(self):
        return self.df.sort_values('id').iloc[:, np.r_[0:6, len(self.df.columns) - 750:len(self.df.columns)]]
