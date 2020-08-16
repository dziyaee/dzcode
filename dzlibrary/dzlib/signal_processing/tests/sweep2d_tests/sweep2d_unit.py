

def generate_shape_test_params(settings):
    '''Function to parse and return Sweep2d input args from settings dict loaded from a settings yaml'''
    settings = settings['Sweep2d']['Unit']

    modes = settings['modes']
    tests = settings['tests']

    # this can be done in a one-liner, but it's less readable that way (imo)
    inputs = [test.values() for test in tests.values()]  # returns list of dict_values of list of input args
    inputs = [[tuple(input_arg) for input_arg in input_args] for input_args in inputs]  # returns list of list of tuples of input args
    return inputs, modes
