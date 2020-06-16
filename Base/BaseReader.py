import pandas as pd
from abc import ABC, abstractmethod
from Utils.SerialPrinter import SerialPrinter


class BaseReader(ABC):

    def __init__(self, file_path, create_sample):

        self.printer = SerialPrinter()

        self.printer.print_begin('Instantiating BaseReader...')

        self.file_path = file_path
        self.create_sample = create_sample

        self.file_directory = self.get_file_directory(self.file_path)
        self.file_name = self.get_file_name(self.file_path)

        self.df = None

        self.printer.print_end('Instantiated BaseReader!')

    @staticmethod
    def get_file_directory(file_path):
        return file_path.rsplit('/')[0] + '/'

    @staticmethod
    def get_file_name(file_path):
        return file_path.split('/')[-1]

    def read(self):
        # Code that will read in the csv file using pandas's pd.read_csv function.

        self.printer.print_begin('Reading in sales_train_validation.csv...')

        self.df = pd.read_csv(self.file_path)

        if self.create_sample:
            self.printer.print_begin('Creating sample...')
            sample_df = self.create_sample_df()
            self.printer.print_end('Created sample!')

            self.printer.print_begin('Saving sample...')
            sample_df.to_csv(self.file_directory + 'sample_' + self.file_name, index=False)
            self.printer.print_end('Saved sample!')

        self.printer.print_end('Read in sales_train_validation.csv!')

        self.process()

        return self.df

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def create_sample_df(self):
        pass
