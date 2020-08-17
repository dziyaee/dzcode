

def generate_shape_test_params(settings):
    '''Function to parse and return Sweep2d input args from settings dict loaded from a settings yaml'''
    settings = settings['Sweep2d']['Unit']

    # input arg shapes
    tests = list(settings['tests'].values())  # returns list of dicts of input arg lists per test
    inputs = [list(test.values()) for test in tests]  #  converts input args dicts to lists
    inputs = [[tuple(arg) for arg in args] for args in inputs]  # converts input arg lists to tuples

    # input arg modes
    modes = settings['modes']
    return inputs, modes
