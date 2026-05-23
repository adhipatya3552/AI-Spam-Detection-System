import os


def create_directory(path):

    """
    Create directory if it does not exist.
    """

    if not os.path.exists(path):

        os.makedirs(path)

        print(f"Directory created: {path}")

    else:

        print(f"Directory already exists: {path}")


def print_separator():

    """
    Print separator line.
    """

    print("\n" + "=" * 50 + "\n")


def print_title(title):

    """
    Print formatted title.
    """

    print_separator()

    print(title.upper())

    print_separator()