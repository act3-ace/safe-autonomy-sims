"""
This module provides a simple example for researchers in how to load and begin to parse the results of the
evaluation framework.

Author: John McCarroll
"""

import pickle


def main():
    """
    The main function of this script, responsible for unserializing evaluation output and printing custom metrics.
    """
    file_path = "/tmp/omg_save_me/metrics.pkl"
    # file_path = "/tmp/data/inspection/metrics.pkl"

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    print("data")
    print(data)
    print()


if __name__ == "__main__":
    main()
