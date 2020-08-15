import numpy as np
import torch


def get_shapes(settings):
    test_inputs = settings['tests']

    shapes = [[*test_shapes.values()] for _, test_shapes in test_inputs.items()]
    shapes = [[tuple(shape) for shape in test_shapes] for test_shapes in shapes]
    return shapes




