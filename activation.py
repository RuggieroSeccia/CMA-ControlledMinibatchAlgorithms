import numpy as np
import scipy.special


class ActivationFunctionFactory:
    SUPPORTED_ACTIVATION_FUNS = ['sigmoid', 'relu']

    def get_activation_function(self,
                                activation_fun: str):
        if activation_fun not in self.SUPPORTED_ACTIVATION_FUNS:
            raise ValueError(f'{activation_fun=} not supported. Supported values are: {self.SUPPORTED_ACTIVATION_FUNS}')
        if activation_fun == 'sigmoid':
            return SigmoidActivationFunction()
        else:
            return ReluActivationFunction()


class ActivationFunction:

    def eval_act_fun(self, a):
        pass

    def eval_act_fun_prime(self,
                           a):
        pass

    def eval_act_fun_prime_z(self,
                             z):
        pass


class SigmoidActivationFunction(ActivationFunction):
    def eval_act_fun(self, a):
        return scipy.special.expit(a)

    def eval_act_fun_prime(self,
                           a):  # g'(a)=g(a)*(1-g(a))
        G = self.eval_act_fun(a)
        return G * (np.ones(G.shape) - G)

    def eval_act_fun_prime_z(self,
                             z):  # g'(a)=z*(1-z)
        return z * (np.ones(z.shape) - z)


class ReluActivationFunction(ActivationFunction):
    def eval_act_fun(self, a):
        return np.maximum(a, 0)

    def eval_act_fun_prime(self,
                           a):
        return np.maximum(np.sign(a), 0)

    def eval_act_fun_prime_z(self,
                             z):  #
        return np.maximum(np.sign(z), 0)
