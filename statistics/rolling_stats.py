import numpy as np


class Rolling():
    ''' The Rolling Class calculates and stores the rolling statistics of a dataset, including size, mean, sum-of-squares, variance (var), and standard deviation (std). The previous mean is also stored and used in the calculation. The rolling statistics are calculated via an implementation of Welford's online algorithm.

        Attributes:
            size: int, size of dataset (total number of elements)
            floats:
                mean: rolling mean of entire dataset
                sumsquares: rolling sum-of-squares of the entire dataset
                prev_mean: rolling mean of previous dataset
                var: rolling variance of entire dataset
                std: rolling standard deviation of entire dataset
    '''

    def __init__(self, agg=(0, 0, 0)):
        ''' The Rolling Class is initialized using the aggregated size, mean, and sum-of-squares of a previous dataset. If no previous dataset exists, these values default to zero. The previous mean is initialized to the same value as the aggregated mean.

            Arguments:
                agg: tuple, size, mean, sum-of-squares of previous dataset
        '''

        self.size, self.mean, self.sumsquares = agg
        self.prev_mean = self.mean

        if self.size != 0:
            self.var = self.sumsquares / self.size

        else:
            self.var = 0

        self.std = np.sqrt(self.var)

    def stats(self, data):
        ''' The stats method is an implementation of Welford's online algorithm to calculate the rolling statistics of a dataset.

            Arguments:
                data: Array-like object

            Returns:
                floats:
                    mean: rolling mean of entire dataset
                    std: rolling standard deviation of entire dataset
        '''

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        # Add current dataset's size to previous rolling size
        self.size += data.size

        # Add contribution of current dataset's mean to previous rolling mean
        self.mean += np.sum(data - self.mean) / self.size

        # Add contribution of current dataset's sum-of-squares to previous rolling sum-of-squares
        self.sumsquares += np.sum((data - self.prev_mean) * (data - self.mean))

        # After being used in the sum-of-squares calculation, update previous rolling mean to current rolling mean
        self.prev_mean = self.mean

        # Calculate rolling variance and std of entire dataset
        self.var = self.sumsquares / self.size
        self.std = np.sqrt(self.var)

        return self.mean, self.std
