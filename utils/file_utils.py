import numpy as np

class file_utils():
        
    def get_rand_filename(root="", seed=4, num_dig=9):
        lower_bound = 10 ** (num_dig - 1)
        return root + np.random.rand(lower_bound, high=(10 * lower_bound))