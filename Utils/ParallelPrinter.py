import datetime
import numpy as np
import pandas as pd
from itertools import repeat


class ParallelPrinter:

    program_started = False

    def __init__(self, total_procs, proc_job_dict):
        """
        Constructor method to initialize the parallel printer. It will
        initialize the printing format based on the amount of processes.

        :param total_procs: The total number of processes.
        :param proc_job_dict: A dictionary the number of jobs
        associated with each processes.
        """

        self.prev_print_carriage = False

        # Assigns the parameters for later reference.

        self.total_procs = total_procs
        self.proc_job_dict = proc_job_dict
        self.max_jobs = max(self.proc_job_dict.values())
        self.min_jobs = min(self.proc_job_dict.values())
        self.current_job = 0

        self.proc_job_completed_dict = {proc: 0
                                        for proc in self.proc_job_dict.keys()}

        # Creates a dataframe of the time it takes to complete each job
        # within each process.

        self.start_time = None
        self.job_time_tracker = pd.DataFrame(np.nan,
                                             index=range(0, self.max_jobs),
                                             columns=self.proc_job_dict.keys())

        # Initializes the printables to be edited in the
        # configure_prints methods.

        self.regular_divider = ''
        self.parallel_title = ''
        self.parallel_message = ''

        self.configure_prints()

    def configure_prints(self):
        """
        Method to configure the printable statements based on a variable
        number of processes and jobs.

        """

        # Creates a tuple to be referenced in the divider string. The
        # + 2 accomodates for the time and job completion columns.
        tup = tuple(repeat((''), self.total_procs + 2))

        time_col_len = 12
        job_col_len = 17

        # Determines the process column length from whats left after the
        # time and job completion columns are allocated. Note, the
        # process columns are of equal length. Therefore, any extra
        # space is allocated to the time column.

        process_col_len = (120 - time_col_len - job_col_len) // self.total_procs
        time_col_len += (120 - time_col_len - job_col_len) % self.total_procs

        # Creates a formattable string based on the lengths of each
        # column for the divider.

        time_col_div_str = '+{:-<' + str(time_col_len - 2) + '}+'
        job_col_div_str = '{:-<' + str(job_col_len - 1) + '}+'
        process_col_div_str = '{:-<' + str(process_col_len - 1) + '}+'

        # Concatenates the formattable strings and formats them with an
        # empty tuple.

        parallel_divider_fmt = time_col_div_str + \
                               job_col_div_str + \
                               process_col_div_str * self.total_procs
        self.parallel_divider = parallel_divider_fmt.format(*tup)

        # Creates a formattable string based on the lengths of each
        # column for the messages.

        time_col_msg_str = '| {:>' + str(time_col_len - 4) + '} |'
        job_col_msg_str = ' {:>' + str(job_col_len - 3) + '} |'
        process_col_msg_str = ' {:>' + str(process_col_len - 3) + '} |'
        self.parallel_message = time_col_msg_str + \
                                job_col_msg_str + \
                                process_col_msg_str * self.total_procs

        # Creates a tuple of titles for each process. Then, assigns the
        # Time, job completion, and process title tuple to a regular
        # parallel message string to create the title.

        process_titles = tuple(['P_' + str(i)
                                for i in range(1, self.total_procs + 1)])
        self.parallel_title = self.parallel_message.format('Time',
                                                           'Job # / Total',
                                                           *process_titles)

    def update(self, proc_num: int, job_num: int):
        """
        Method to update the information the parallel printer is
        tracking. It will only print if that job is completed for all
        processes.
        :param proc_num: The integer id of the process.
        :param job_num: The integer id of the job.
        """

        # Assigns the time completed to the job_tracker.

        time_completed = datetime.datetime.now().replace(microsecond=0)
        self.job_time_tracker.iloc[job_num, proc_num] = time_completed

        self.proc_job_completed_dict[proc_num] += 1

        # self.print_jobs_completed()

        # Checks to see if the jobs in that index are also completed
        # in the other processes.
        non_min_procs = [proc for proc in self.proc_job_dict.keys()
                         if self.proc_job_dict[proc] > job_num]

        if not self.job_time_tracker.iloc[job_num, non_min_procs].isnull().values.any():
            self.print_parallel_message(job_num)

    def print_jobs_completed(self):
        current_time = datetime.datetime.now().time().replace(microsecond=0)
        total_jobs = sum(self.proc_job_dict.values())
        jobs_completed = sum(self.proc_job_completed_dict.values())

        job_str = '{:d} / {:d}'.format(jobs_completed, total_jobs)

        job_strs = ['{:} / {:}'.format(str(self.proc_job_completed_dict[proc]),
                                     str(self.proc_job_dict[proc])) for proc in self.proc_job_dict.keys()]

        if self.prev_print_carriage:
            print('\r', end='')
            print(self.parallel_message.format(str(current_time),
                                               job_str, *job_strs),
                  end='', flush=True)

        else:
            print(self.parallel_message.format(str(current_time),
                                               job_str, *job_strs),
                  end='', flush=True)
            self.prev_print_carriage = True

    def print_parallel_message(self, job_num: int):
        """
        Method to print a message of information gained about the
        parallel processes.
        :param job_num: The integer id of the job.
        """

        # Gets the current time.

        current_time = datetime.datetime.now().time().replace(microsecond=0)

        # Creates and formats the job completion string.

        job_str = '{:d} / {:d}'.format(job_num + 1, self.max_jobs)

        # Checks to see if this is the first job completed, if it is,
        # then it does not try to print a time delta. If it is not, then
        # it will print the differences in time completed for the jobs.

        print('\r', end='')
        self.prev_print_carriage = False

        if job_num == 0:
            prev_job_times = self.start_time
            current_job_times = self.job_time_tracker.iloc[job_num, :].values

            cond = current_job_times != -1
            splits = np.where(cond, current_job_times - prev_job_times, '-')

            job_times = tuple([str(split) for split in splits])

            print(self.parallel_message.format(str(current_time),
                                               job_str,
                                               *job_times))
        else:

            prev_job_times = self.job_time_tracker.iloc[job_num - 1, :].values
            current_job_times = self.job_time_tracker.iloc[job_num, :].values

            cond = current_job_times != -1
            splits = np.where(cond, current_job_times - prev_job_times, '-')

            job_times = tuple([str(split) for split in splits])

            print(self.parallel_message.format(str(current_time),
                                               job_str,
                                               *job_times))

        self.print_jobs_completed()

    def print_notify_parallel_begin(self):
        """
        Method to print the notification of the beginning of a parallel
        printing sequence.
        """

        self.start_time = datetime.datetime.now().replace(microsecond=0)

        print(self.parallel_divider)
        print(self.parallel_title)
        print(self.parallel_divider)

    def print_notify_parallel_end(self):
        """
        Method to print the notification of the ending of a parallel
        printing sequence.
        """

        print(self.parallel_divider)

