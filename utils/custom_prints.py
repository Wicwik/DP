import numpy as np

def set_print_opt(formatter={'float_kind':"{:.6f}".format}):
    np.set_printoptions(formatter=formatter)