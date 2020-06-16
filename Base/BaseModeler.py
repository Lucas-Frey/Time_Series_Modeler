import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import managers
import os
from abc import ABC, abstractmethod

from Utils.SerialPrinter import SerialPrinter
from Utils.ParallelPrinter import ParallelPrinter

class BaseModeler(ABC):
    """
    Base class for modelling time series. It has capabilities to model
    in parallel or model iteratively. It is extendable for each model.
    """

    def __init__(self, multiseries: bool = False, parallelize: bool = False):
        """
        Constructor to initialize the model class.

        :param multiseries: Are there multiple time series predicted?
        :param parallelize: Should the time series be predicted in
        parallel? If false, then predicted iteratively.
        """

        # Initializes the printer object in this class.
        self.serial_printer = SerialPrinter()

        self.serial_printer.print_begin('Instantiating BaseModeler...')

        self.parallelize = parallelize
        self.multiseries = multiseries

        self.processors = os.cpu_count()

        # Code that supposedly fixes a lot of multiprocessing problems.
        # Cite: https://pythonspeed.com/articles/python-multiprocessing/
        # mp.set_start_method("spawn")

        self.serial_printer.print_end('Instantiated BaseModeler!')

    def multi_fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method determing what method to use when fitting and predicting
        a time series.

        :param df: The dataframe containing the time series data.
        :return: A dataframe containing the predictions.
        """

        self.serial_printer.print_begin('Determining fit predict method...')

        if self.parallelize:
            self.serial_printer.print_end('Determined fit predict method...')
            predictions = self.parallel_fit_predict(df)
        else:
            self.serial_printer.print_end('Determined fit predict method...')
            predictions = self.iterative_fit_predict(df)

        return predictions

    def parallel_fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to fit and predict time series in parallel.
        :param df: The dataframe containing the time series data.
        :return: A dataframe containing the predictions.
        """

        self.serial_printer.print_begin('Preparing data for parallelization...')

        # The groupby function is used to group each of the time series
        # within the dataframe together. Therefore, the time series are
        # not split up in the following operations.

        self.serial_printer.print_begin('Grouping the dataframe...')
        groups = df.groupby('id')
        self.serial_printer.print_end('Grouped the dataframe!')

        # The array_split function will split the groups up into
        # approximately equal chunks. The amount of chunks is equal to
        # the amount of processors.

        self.serial_printer.print_begin('Splitting the dataframe...')
        list_of_splits = np.array_split(groups, self.processors)
        self.serial_printer.print_end('Split the dataframe!')

        # The array_split function returns a list of ndarrays of groups.
        # So, we convert it to a list of dataframes.

        self.serial_printer.print_begin('Concatenating data in each split...')
        list_of_splits = [pd.concat(split[:, 1]).reset_index(drop=True)
                          for split in list_of_splits]
        self.serial_printer.print_end('Concatenated data in each split!')

        self.serial_printer.print_end('Prepared data for parallelization!')

        # Creates a dictionary whose keys are the number index of each
        # process and the values are the amount of time series that
        # process has to predict.

        self.serial_printer.print_begin('Creating a process-job dictionary...')
        process_job_dict = dict(zip(range(0, self.processors, 1),
                                    [s['id'].nunique() for s in list_of_splits]))
        self.serial_printer.print_end('Created a process-job dictionary!')

        # Creates a parallel printing object that can be accessed from
        # each of the processes. It creates a list of references to the
        # printer object. Each one of these printers will be sent to a
        # different process.
        # Cite: https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-python-processes

        self.serial_printer.print_begin('Creating a shareable printer...')
        managers.BaseManager.register('ParallelPrinter', ParallelPrinter)
        manager = managers.BaseManager()
        manager.start()
        sharable_printer = manager.ParallelPrinter(self.processors,
                                                   process_job_dict)
        printers = np.repeat([sharable_printer], self.processors)
        self.serial_printer.print_end('Created a shareable printer!')

        # Creates a process pool. This action can take a while.

        self.serial_printer.print_begin('Creating the process pool...')
        pool = mp.Pool(self.processors)
        self.serial_printer.print_end('Created the process pool!')

        # This code actually begins the parallel processing. The starmap
        # method will send a split in the list of splits, a process id
        # in the range, and a printer to the iterative_fit_predict
        # method.

        self.serial_printer.print_notification('Beginning parallel process. '
                                               'Going to parallel printing.')

        sharable_printer.print_notify_parallel_begin()
        predictions_df_list = pool.starmap(self.iterative_fit_predict,
                                           zip(list_of_splits,
                                               range(0, self.processors, 1),
                                               printers))
        pool.close()
        pool.join()
        sharable_printer.print_notify_parallel_end()

        self.serial_printer.print_notification('Completed parallel processing. '
                                               'Going back to serial printing.')

        predictions_df = pd.concat(predictions_df_list)

        return predictions_df

    def iterative_fit_predict(self,
                              df: pd.DataFrame,
                              proc_num: int,
                              shared_printer: ParallelPrinter) -> pd.DataFrame:
        """
        Method to iteratively fit and predict a time series.
        :param df: The dataframe containing the time series data.
        :param proc_num: The integer id of the process.
        :param shared_printer: The parallel printer.
        :return: A dataframe containing the predictions.
        """

        # Groups the dataframe by id again to prepare to apply the
        # the actual fit_predict function. Note, the ids are sometimes
        # categorical. Therefore, the observed=True is necessary.

        groups = df.groupby('id', observed=True)

        # Creates an ndarray of the ids. This is to identify where
        # the iterating process is within the wider sense of itself.

        ids = df['id'].unique().to_numpy()

        # apply function that applies the fit_predict wrapper function
        # to each group within the dataframe.

        predictions = groups.apply(lambda x:
                                   self.fit_predict_wrapper(x, proc_num,
                                                            np.where(ids == x.name)[0][0],
                                                            shared_printer))

        predictions_df = pd.DataFrame.from_dict(dict(zip(predictions.index,
                                                         predictions.values))).T

        return predictions_df

    def fit_predict_wrapper(self, df: pd.DataFrame,
                            proc_num: int,
                            job_num: int,
                            shared_printer: ParallelPrinter):
        """
        Wrapper function for the fit_predict function. Used primary for
        the shared printer capabilties.
        :param df: The dataframe containing the time series data.
        :param proc_num: The integer id of the process.
        :param job_num: The integer id of the job.
        :param shared_printer: The parallel printer.
        :return: A dataframe containing the predictions.
        """

        predictions = self.fit_predict(df)

        # Important parallel printing capability.

        shared_printer.update(proc_num, job_num)

        return predictions

    @abstractmethod
    def fit_predict(self, df):
        pass

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def predict(self, model):
        pass