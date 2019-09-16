from sklearn.preprocessing import OneHotEncoder


def encode(classes):
    """One hot encoder of given classes

    Args:
        classes (np.ndarray): Array of different integer encoded classes
    Returns:
        encoder (OneHotEncoder): Trained on given classes encoder
    """

    encoder = OneHotEncoder()
    encoder.fit(classes)

    return encoder


def num2class(num_classes):
    """Convert integer encoded classes to string

    Args:
        num_classes (numpy.ndarray): Array of integer encoded classes
    Returns:
        str_classes (List): List of string encoded classes
    """

    str_classes = []
    for x in num_classes:
        if x == 0:
            str_classes.append('bmp-1')
        elif x == 1:
            str_classes.append('btr-80')
        elif x == 2:
            str_classes.append('background')
        elif x == 3:
            str_classes.append('t-55')
        elif x == 4:
            str_classes.append('t-72b')
    return str_classes


def class2num(str_classes):
    """Convert string encoded classes to integer

    Args:
        str_classes (numpy.ndarray): Array of string encoded classes
    Returns:
        num_classes (List): List of integer encoded classes
    """

    num_classes = []
    for x in str_classes:
        if x == "bmp-1":
            num_classes.append(0)
        elif x == 'btr-80':
            num_classes.append(1)
        elif x == 'background':
            num_classes.append(2)
        elif x == 't-55':
            num_classes.append(3)
        elif x == 't-72b':
            num_classes.append(4)
    return num_classes


if __name__ == "__main__":
    pass
