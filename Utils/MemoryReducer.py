import pandas as pd
import numpy as np


DTYPE_LIST = ['float64', 'float32', 'float16',
              'int64', 'int32', 'int16', 'int8',
              'uint64', 'uint32', 'uint16', 'uint8',
              'object', 'category', 'datetime64', 'bool']


def get_dtype_memory(df):
    mem_list = []

    # For loop that will iterate through each dtype in the DTYPE_LIST and save its memory usage.

    for dtype in DTYPE_LIST:
        selected_dtype = df.select_dtypes(include=[dtype])

        # The selected_dtype.memory_usage(deep=True) line returns a series where each row is the memory
        # usage per column with that dtype. Therefore, we use the .sum() to get the total for that dtype.

        mem_usage_b = selected_dtype.memory_usage(deep=True).sum()

        # Note, the mem_usage_b is the amount of bits. Then, the mem_usage_b / 1024 ** 2 converts
        # bits to megabits.

        mem_usage_mb = mem_usage_b / 1024 ** 2

        mem_list.append(mem_usage_mb)

    # We save out list as a numpy array. Then, we append to the end of the array the total memory consumed
    # by summing each value in the list.

    mem_list = np.array(mem_list)
    mem_list = np.append(mem_list, mem_list.sum())

    return mem_list

def get_dtype_count(df):
    count_list = []

    # For loop that will iterate through each dtype in the DTYPE_LIST and save its memory usage.

    for dtype in DTYPE_LIST:
        # Selects the columns that have the dtype.

        selected_dtype_df = df.select_dtypes(include=[dtype])

        # Simply takes the length of the array holding the columns to get the amount of columns.

        count_list.append(len(selected_dtype_df.columns.to_list()))

    # We save out list as a numpy array. Then, we append to the end of the array the total columns
    # by summing each value in the list.

    count_list = np.array(count_list)
    count_list = np.append(count_list, count_list.sum())

    return count_list

def memory_magic(df, printer):

    printer.print_begin('Running memory_magic...')

    # Code to get the initial memory statistics. It will call a function that will return a list of
    # the memory consumed per column and it will call another function that will get the amount of
    # columns per datatype. Since this function will change datatypes, this will be useful for comparison.

    printer.print_begin('Getting initial memory statistics...')
    before_memory = get_dtype_memory(df)
    before_count = get_dtype_count(df)
    printer.print_end('Got initial memory statistics!')

    # Code to downcast all integer type columns. It will take a column that is int64 and check its maximum
    # value. Then, it will assign a datatype appropriate for excapsulating that number.

    printer.print_begin('Converting and downcasting integer columns...')
    df_int_col = df.select_dtypes(include=['int'])
    df[df_int_col.columns] = df[df_int_col.columns].apply(pd.to_numeric, downcast='unsigned')
    printer.print_end('Converted and downcasted integer columns!')

    # Code to downcast all float type columns. It will take a column that is float64 and check its maximum
    # value. Then, it will assign a datatype appropriate for excapsulating that number.

    printer.print_begin('Converting and downcasting float columns...')
    df_flt_col = df.select_dtypes(include=['float'])
    df[df_flt_col.columns] = df[df_flt_col.columns].apply(pd.to_numeric, downcast='float')
    printer.print_end('Converted and downcasted float columns!')

    # Code to convert object columns to categorical columns. It will check to see if at least half a
    # column's values are repeated. If they are, then it will turn it into a categorical column.

    printer.print_begin('Converting object columns...')
    for col in df.select_dtypes(include=['object']):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])

        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    printer.print_end('Converted object columns!')

    # Code to get the post memory statistics. It will call a function that will return a list of
    # the memory consumed per column and it will call another function that will get the amount of
    # columns per datatype. Since this function will change datatypes, this will be useful for comparison.
    # These are going to be contrasted with the initial memory statistics.

    printer.print_begin('Getting post memory statistics...')
    after_memory = get_dtype_memory(df)
    after_count = get_dtype_count(df)

    dtype_list_tot = np.append(DTYPE_LIST, ['total'])
    mem_matrix = np.column_stack((before_memory, after_memory, after_memory - before_memory,
                                  before_count, after_count, after_count - before_count))
    printer.print_end('Got post memory statistics!')

    # Code to print out the final memory statistics. First, it will print a preformatted table header. Then,
    # it will iterate through each dtype and print the memory consumed by that datatype and the amount of
    # columns using that dtype before and after downcasting and conversions are done.

    # print_begin(step1, step2, step3 + 10, 'Printing post memory statistics...', tabs)
    # current_time = time.time()
    #
    # print(TABLE_DIV_00)
    # print(TABLE_LABELS)
    # print(TABLE_DIV_01)
    # print(TABLE_HEADER.format('dtype', 'start', 'end', '+/-', 'start', 'end', '+/-'))
    # print(TABLE_DIV_01)

    # for i, row in enumerate(mem_matrix):
    #
    #     if not int(row[3]) == 0 | int(row[4]) == 0:
    #         if dtype_list_tot[i] in ['total']:
    #             print(TABLE_DIV_01)
    #
    #         lbl_str_tpl = dtype_list_tot[i], row[0], row[1], row[2], int(row[3]), int(row[4]), int(row[5])
    #         print(TABLE_VALUES.format(*lbl_str_tpl))
    #
    # print(TABLE_DIV_01)
    # print_end(step1, step2, step3 + 11, current_time, 'Printed post memory statistics!', tabs)

    printer.print_end('Ran memory_magic!')

    # Returns the reformmatted dataframe.

    return df