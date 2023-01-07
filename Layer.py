import numpy as np

from activation import ActivationFunction


class Layer_i:
    """Define  a layer"""
    def __init__(self, N: int, N_1: int, l: int,
                 activation_fun: ActivationFunction):
        """
        :param N: dimension of input
        :param N_1: dimension of output
        :param l: position of layer in a NN

        """
        self.N = N
        self.N_1 = N_1
        self.l = l
        self.w = np.random.rand(self.N, self.N_1) - 0.5
        self.delta = np.zeros(self.N_1 * self.N)
        self.activation_fun = activation_fun

    def __str__(self):
        description = 'Layer with {} neurons. Dimensions: {}x{}'.format(self.N, self.N, self.N_1)
        return description

    def eval_a_z(self, z_1, last=0):
        self.a = np.matmul(self.w, z_1)
        self.z = self.activation_fun.eval_act_fun(self.a)

        if last == 1:
            self.a = self.a.T
            self.z = self.a
