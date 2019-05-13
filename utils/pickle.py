"""
File containing helper functions to load the data set and to save/load any kind of object :
- function save_obj
- function load_obj
"""

import os
import errno
import pickle


def save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: Object to save.
    :param name: Name of the file.
    :return: Nothing.
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads an object from a pickle file.
    :param name: File name.
    :return: The loaded object.
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_directory(directory):
    """
    Safe way to make a directoryif it doesn't already exist
    :param directory: Directory's name
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def make_file_directory(filename):
    """
    Safely makes the directory that will contain the file if it doesn't already exist
    :param filename: File's name
    """
    directory = os.path.dirname(filename)
    make_directory(directory)
