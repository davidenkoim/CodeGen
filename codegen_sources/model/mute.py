"""
This module contains four methods for muting file descriptors.
You can mute the standard file descriptors:
    with mute_stdout():
        ...
    with mute_stderr():
        ...
    with mute_stdout_stderr():
        ...
Or mute a list of known file descriptors:
    with mute_fd(sys.stdout, sys.stderr, ...)
        ...
Please see more help on this last method.
"""
from contextlib import contextmanager
import os
import sys

"""
Module's main output should look like:
$ python2 nostderr.py
Will print and log without mute:
                        This is a print statement on stdout
2020-08-21 05:22:09,069 This is a  log  statement on stderr
Will print and log with mute_stderr:
                        This is a print statement on stdout
Will print and log with mute_stdout:
2020-08-21 05:22:09,070 This is a  log  statement on stderr
Will print and log with mute_stdout_stderr:
Will print and log with mute_fd(sys.stdout, sys.stderr, capture_path="out.txt"):
DONE, Here are the contents of "out.txt":
                        This is a print statement on stdout
2020-08-21 05:22:09,071 This is a  log  statement on stderr
Will print and log with mute_fd(sys.stdout, sys.stderr, capture_fd=TemporaryFile(mode="w+"))
DONE, Here are the contents of the TemporaryFile:
                        This is a print statement on stdout
2020-08-21 05:22:09,073 This is a  log  statement on stderr
FINALLY print and log AGAIN without mute:
                        This is a print statement on stdout
2020-08-21 05:22:09,074 This is a  log  statement on stderr
$ python3 nostderr.py
Will print and log without mute:
                        This is a print statement on stdout
2020-08-21 05:22:18,959 This is a  log  statement on stderr
Will print and log with mute_stderr:
                        This is a print statement on stdout
Will print and log with mute_stdout:
2020-08-21 05:22:18,959 This is a  log  statement on stderr
Will print and log with mute_stdout_stderr:
Will print and log with mute_fd(sys.stdout, sys.stderr, capture_path="out.txt"):
DONE, Here are the contents of "out.txt":
                        This is a print statement on stdout
2020-08-21 05:22:18,960 This is a  log  statement on stderr
Will print and log with mute_fd(sys.stdout, sys.stderr, capture_fd=TemporaryFile(mode="w+"))
DONE, Here are the contents of the TemporaryFile:
                        This is a print statement on stdout
2020-08-21 05:22:18,961 This is a  log  statement on stderr
FINALLY print and log AGAIN without mute:
                        This is a print statement on stdout
2020-08-21 05:22:18,961 This is a  log  statement on stderr
"""


def mute_stderr(*args, **kwargs):
    """
    Mutes error output of a block when used like:
    with mute_stderr():
        ...
    Also accepts more file descriptors as args and the kwargs 'capture_path'
    and 'capture_fd'. See help on mute_fd for their uses.
    """
    return mute_fd(sys.stderr, *args, **kwargs)


def mute_stdout(*args, **kwargs):
    """
    Mutes standard output of a block when used like:
    with mute_stdout():
        ...
    Also accepts more file descriptors as args and the kwargs 'capture_path'
    and 'capture_fd'. See help on mute_fd for their uses.
    """
    return mute_fd(sys.stdout, *args, **kwargs)


def mute_stdout_stderr(*args, **kwargs):
    """
    Mutes standard output and error of a block when used like:
    with mute_stdout_stderr():
        ...
    Also accepts more file descriptors as args and the kwargs 'capture_path'
    and 'capture_fd'. See help on mute_fd for their uses.
    """
    return mute_fd(sys.stdout, sys.stderr, *args, **kwargs)


def mute_fd(*args, **kwargs):
    """
    Writes into /dev/null, a capture_fd, or a capture_path, the contents
    of muted file descriptors given as args of a with block.
    Will close a given capture_path, but not a given capture_fd.
    To be used like:
    with mute_fd(sys.stderr, ...):
        ...
    Or
    with mute_fd(sys.stderr, ..., capture_path="/path/to/file"):
        ...
    Or
    with mute_fd(sys.stderr, ..., capture_fd=TemporaryFile()):
        ...
    """
    if 'capture_fd' in kwargs and kwargs['capture_fd']:
        return __mute_fd(*args, **kwargs)

    capture_path = os.devnull
    if 'capture_path' in kwargs and kwargs['capture_path']:
        capture_path = kwargs['capture_path']
    return __mute_fd(*args, capture_fd=open(capture_path, 'w'), close=True)


@contextmanager
def __mute_fd(*args, **kwargs):
    """
    Writes into fd output of a block, leaving fd open, when used like:
    with mute_fd(sys.stderr, ..., capture_fd=an_open_fd):
        ...
    """
    capture_fd = None
    if 'capture_fd' in kwargs and kwargs['capture_fd']:
        capture_fd = kwargs['capture_fd']
    else:
        raise Exception("No keyword argument named 'capture_fd'")
    try:
        fileno = [fd.fileno() for fd in args]
        oldfd = [os.dup(fno) for fno in fileno]
        try:
            [os.dup2(capture_fd.fileno(), fno) for fno in fileno]
            yield
        finally:
            [os.dup2(old, fno) for old, fno in zip(oldfd, fileno)]
            if 'close' in kwargs and kwargs['close']:
                capture_fd.close()
                # return None  # not py 2 compatible
            # return capture_fd  # not py 2 compatible
    except AttributeError:
        # If stderr not a file descriptor, but something that won't copy
        yield


def __print_and_log(logger):
    print('                        This is a print statement on stdout')
    logger.info('This is a  log  statement on stderr')


if __name__ == '__main__':
    import logging
    from tempfile import TemporaryFile

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(message)s')
    logger = logging.getLogger(__name__)
    print('\nWill print and log without mute:')
    __print_and_log(logger)

    print('\nWill print and log with mute_stderr:')
    with mute_stderr():
        __print_and_log(logger)

    print('\nWill print and log with mute_stdout:')
    with mute_stdout():
        __print_and_log(logger)

    print('\nWill print and log with mute_stdout_stderr:')
    with mute_stdout_stderr():
        __print_and_log(logger)

    print('\nWill print and log with '
          'mute_fd(sys.stdout, sys.stderr, capture_path="out.txt"):')
    with mute_fd(sys.stdout, sys.stderr, capture_path="out.txt"):
        __print_and_log(logger)
    print('DONE, Here are the contents of "out.txt":')
    with open("out.txt", "r") as f:
        print(f.read())

    print('\nWill print and log with mute_fd(sys.stdout, sys.stderr, '
          'capture_fd=TemporaryFile(mode="w+"))')
    with TemporaryFile(mode="w+") as fd:
        with mute_fd(sys.stdout, sys.stderr, capture_fd=fd):
            __print_and_log(logger)
        print('DONE, Here are the contents of the TemporaryFile:')
        fd.seek(0)
        print(fd.read())

    print('\nFINALLY print and log AGAIN without mute:')
    __print_and_log(logger)
