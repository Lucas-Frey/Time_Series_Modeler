import datetime


class SerialPrinter:

    serial_printer_started = False

    def __init__(self):
        """
        Constructor method to instantiate the serial printer class.
        It will initialize the printing messages.
        """

        # Instantiates the class variables.

        self.current_chapter = 0
        self.current_section = 0
        self.current_step = 0

        self.start_times_stack = []
        self.indentation_stack = 0

        tup = tuple(['', '', '', '', ''])
        title_tup = tuple(['Time', 'Symbol', 'Index', 'Delta', 'Message'])

        self.regular_divider = '+{:-<10}+{:-<8}+{:-<10}+{:-<12}+{:-<74}+'.format(*tup)
        self.title = '| {:<8} | {:<6} | {:<8} | {:<10} | {:<72} |'.format(*title_tup)
        self.regular_message = '| {:>8} | {:<6} | {:>8} | {:>10} | {:<72} |'

        self.begin_message_symbol = '->'
        self.end_message_symbol = '<-'
        self.notification_symbol = '!'

        self.print_serial_printer_start()

    def print_serial_printer_start(self):
        """
        Method to print the beginning of the program. It will only print
        once and then set the program_started static variable to false.
        """
        if not SerialPrinter.serial_printer_started:

            # Gets the current time and stores it on the stack. In the
            # end it will be popped off to determine how long the
            # program took.

            now = datetime.datetime.now()
            self.start_times_stack.append(now)

            # Prints the title.

            print(self.regular_divider)
            print(self.title)
            print(self.regular_divider)

            # Sets the bool to false.

            SerialPrinter.serial_printer_started = True

    def print_begin(self, message: str):
        """
        Method to print the beginning of a process of processes.
        :param message: The message to be printed.
        """

        # Gets the current time. It pushes it on to the stack. When this
        # process is complete, it will be popped off to determine how
        # long the process took.

        now = datetime.datetime.now().replace(microsecond=0)
        self.start_times_stack.append(now)
        now = str(now.time())

        # Creates the arguments for the print message. Note, that the
        # symbol and message arguments are tabbed in based on how many
        # processes deep the process is. Delta is set to '-' because
        # this is the beginning of a process.

        symbol = ('  ' * self.indentation_stack) + self.begin_message_symbol
        index = '{:02d}:{:02d}:{:02d}'.format(self.current_chapter,
                                              self.current_section,
                                              self.current_step)
        delta = '-'
        message = ('  ' * self.indentation_stack) + message

        print(self.regular_message.format(now, symbol, index, delta, message))

        # Increments the indentation stack. So, any new processes
        # created will be tabbed in one.

        self.indentation_stack += 1

    def print_end(self, message):
        """
        Method to print the end of a process of processes.
        :param message: The message to be printed.
        """

        # Immediately decrements the indentation stack since we are no
        # longer in a process.

        self.indentation_stack -= 1

        # Gets the current time and the previous time from the stack.
        # It will take the difference to find the time delta.

        now = datetime.datetime.now().replace(microsecond=0)
        prev = self.start_times_stack.pop()
        delta = '+ ' + str(now - prev)
        now = str(now.time())

        # Creates the arguments for the print message. Note, that the
        # symbol and message arguments are tabbed in based on how many
        # processes deep the process is.

        symbol = ('  ' * self.indentation_stack) + self.end_message_symbol
        index = '{:02d}:{:02d}:{:02d}'.format(self.current_chapter,
                                              self.current_section,
                                              self.current_step)
        message = ('  ' * self.indentation_stack) + message

        print(self.regular_message.format(now, symbol, index, delta, message))

    def print_notification(self, message):
        """
        Method for printing notifications that aren't related to any
        specific process.
        :param message: The message to be printed.
        """

        # Gets the current time.

        now = str(datetime.datetime.now().time().replace(microsecond=0))

        # Creates the arguments for the print message. Note, that the
        # symbol and message arguments are tabbed in based on how many
        # processes deep the process is. Delta and index are set to '-'
        # because this is not related to any other process.

        symbol = self.notification_symbol
        index = '-'
        delta = '-'
        message = message

        print(self.regular_divider)
        print(self.regular_message.format(now, symbol, index, delta, message))
        print(self.regular_divider)
