import time
from typing import List, Tuple

import numpy as np
import scipy.optimize
from numpy.linalg import norm

from Dataset import Dataset
from Layer import Layer_i
# import activation as act
from activation import ActivationFunctionFactory


class NN:
    def __init__(self,
                 list_neurons: List[int],
                 activation_fun: str,
                 data_set: Dataset,
                 _rho: float):
        """
        :param list_neurons: list of with the number of neurons per layers:
                        [10,10,1] defines a NN with 2 hidden neurons with 10 neurons each and an output layer with 1 dim
        :param data_set: dataset in input
        :param _rho: penalization term for the L2 regularization
        """

        self.L = len(list_neurons)
        self.rho = _rho
        self.activation_fun = ActivationFunctionFactory().get_activation_function(activation_fun)
        self.num_var_lista, self.lista_Layer = self.compute_num_var(list_neurons, data_set)
        self.num_var = sum(self.num_var_lista)
        self.w = self.get_w()

    def __str__(self):
        description = 'Network with {} layers.'.format(self.L)
        description = description + '\nNeurons in each layer: {}.'.format([n.N for n in self.lista_Layer])
        description = description + '\nTotal number of variables: {}.'.format(self.num_var)
        return description

    def compute_num_var(self, list_neurons: List[int], data_set: Dataset) -> Tuple[List[int], List[Layer_i]]:
        """
        Computes number of variables and defines the list of layers objects.
        Returns a list with the number of variables per layer and a list with the layers obj defining the NN
        """
        num_var_lista = []
        lista_Layer = []
        for l in range(self.L):
            if l == 0:
                layer = Layer_i(list_neurons[l], data_set.n, l, self.activation_fun)
                nv = list_neurons[l] * data_set.n

            else:
                layer = Layer_i(list_neurons[l], list_neurons[l - 1], l, self.activation_fun)
                nv = list_neurons[l] * list_neurons[l - 1]

            num_var_lista.append(nv)
            lista_Layer.append(layer)

        return num_var_lista, lista_Layer

    def get_w(self):
        '''
        function to compute the weights vector w
        '''
        w = np.zeros(self.num_var)
        k = 0
        for i in (self.lista_Layer):
            w[k:k + i.N_1 * i.N] = i.w.ravel()
            k += i.N_1 * i.N
        return w

    def split_w(self):
        '''
        function to assign the values of w to each layer
        '''
        n = self.lista_Layer[0].N_1
        k = 0
        count = 0
        for i in self.lista_Layer:
            if count == 0:
                i.w = self.w[k:k + i.N * n].reshape(i.N, n)
                count += 1

            else:
                i.w = self.w[k:k + i.N * i.N_1].reshape(i.N, i.N_1)

            k += i.N * i.N_1
        return

    def insert_w_l(self, w_l, w_rest, index):
        """Insert w_l in position index of w_rest"""
        return np.insert(w_rest, index, w_l)

    def forward(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        forward propagation
        :param w: weights to optimize
        :param x: input features
        :param y: labels
        """
        k = 0
        # forward propagation
        for i in range(self.L):  # collego w con i layers

            if i == 0:
                self.lista_Layer[i].w = w[0:self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N, self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(x)

            if i > 0 & i < self.L - 1:
                self.lista_Layer[i].w = w[k:k + self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N,
                    self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(self.lista_Layer[i - 1].z)

            if i == self.L - 1:
                self.lista_Layer[i].w = w[k:k + self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N,
                    self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(self.lista_Layer[i - 1].z, 1)
            k += self.lista_Layer[i].N * self.lista_Layer[i].N_1

        y_pred = self.lista_Layer[self.L - 1].z

        e = y_pred - y
        f = 1 / x.shape[1] * (np.sum((e) ** 2) + self.rho * np.sum(w ** 2))

        return f

    def forward_backward(self,
                         w: np.ndarray,
                         x: np.ndarray,
                         y: np.ndarray,
                         check_time: bool = False,
                         time_0: int = 1,
                         limit_time: float = 1e6,
                         fun_obj_and_time_values: List[str] = ['Errore']):
        k = 0
        # forward propagation
        for i in range(self.L):  # riassegno w ai layers e calcolo le z

            if i == 0:
                self.lista_Layer[i].w = w[0:self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N, self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(x)

            if i > 0 & i < self.L - 1:
                self.lista_Layer[i].w = w[k:k + self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N, self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(self.lista_Layer[i - 1].z)

            if i == self.L - 1:
                self.lista_Layer[i].w = w[k:k + self.lista_Layer[i].N * self.lista_Layer[i].N_1].reshape(
                    self.lista_Layer[i].N, self.lista_Layer[i].N_1)
                self.lista_Layer[i].eval_a_z(self.lista_Layer[i - 1].z, last=1)
            k += self.lista_Layer[i].N * self.lista_Layer[i].N_1

        y_pred = self.lista_Layer[self.L - 1].z
        e = y_pred - y

        f = 1 / x.shape[1] * (np.sum((e) ** 2) + self.rho * np.sum(w ** 2))

        # backward propagation
        # ultimo layer
        derivata = None
        self.lista_Layer[self.L - 1].delta = e
        d_ultimo_layer = np.sum(e * self.lista_Layer[self.L - 2].z.T, axis=0)  # g'(a_L)=1

        # penultimo layer
        for l in range(self.L - 2, -1, -1):  # we need to cycle up to 0
            self.lista_Layer[l].delta = np.transpose(self.lista_Layer[l].activation_fun.eval_act_fun_prime_z(self.lista_Layer[l].z)) * np.matmul(
                self.lista_Layer[l + 1].delta, self.lista_Layer[l + 1].w)
            if l == 0:
                deriv = np.dot(self.lista_Layer[l].delta.T, x.T)
            else:
                deriv = np.dot(np.transpose(self.lista_Layer[l].delta), np.transpose(self.lista_Layer[l - 1].z))
            if l == self.L - 2:
                derivata = np.concatenate((deriv.ravel(), d_ultimo_layer.ravel()))
            else:
                derivata = np.concatenate((deriv.ravel(), derivata.ravel()))

        derivata = 2 / x.shape[1] * (derivata + self.rho * w)

        if check_time == True:
            if time.time() - time_0 > limit_time:
                derivata = derivata * 0
        fun_obj_and_time_values.append((f, time.time() - time_0))

        return f, derivata

    def optimize_NN_with_LBFGS(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               check_time: bool,
                               time_0: float,
                               limit_time: int,
                               fun_obj_and_time_values: List[Tuple[float, float]],
                               maxiter: float,
                               maxfun: float,
                               gtol: float,
                               disp: bool):
        """Apply LBFGS to all the weights of the NN"""

        res = scipy.optimize.minimize(self.forward_backward, self.w,
                                      args=(x, y, check_time, time_0, limit_time, fun_obj_and_time_values),
                                      method='L-BFGS-B', jac=True,
                                      options={'maxiter': maxiter, 'maxfun': maxfun, 'gtol': gtol, 'disp': disp,
                                               'ftol': 0.1 * np.finfo(float).eps})
        self.w = res.x
        self.split_w()
        return res

    def check_grad(self,
                   w: np.ndarray,
                   x: np.ndarray,
                   y: np.ndarray):
        """Function to check gradient is computed correctly"""
        f, jacobian = self.forward_backward(w, x, y)
        eps = np.sqrt(np.finfo(float).eps)
        epsilon = [eps] * self.num_var
        z = scipy.optimize.approx_fprime(w, self.forward, epsilon, x, y)
        print('------------------------------------------------------------------------------------------------')
        print(np.linalg.norm(z))
        print(np.linalg.norm(jacobian))
        print('difference between derivatives: ', np.sqrt(sum((jacobian - z) ** 2)))
        print('------------------------------------------------------------------------------------------------')
        diff = np.sqrt(sum((jacobian - z) ** 2))
        return diff

    def EDFL(self,
             d: np.ndarray,
             w: np.ndarray,
             f_k: float,
             f_new: float,
             x: np.ndarray,
             y: np.ndarray,
             Delta: float,
             delta: float,
             gamma: float,
             time1: float,
             limit_time: float):
        """
        Implements the Extrapolation Derivative Free Linesearch

        :param d: moving direction
        :param w: starting point
        :param f_k: starting obj fun value
        :param f_new: f(w_k+zeta*d) first point to evaluate the obj fun in the first iteration of LS
        :param x,y: data point to compute obj fun
        :param Delta: starting value of the stepsize
        :param delta: update of the stepsize in the LS
        :param gamma: factor for "enough decrease"
        :param time1: starting time
        :param limit_time: limit time
        """

        alpha = Delta
        f_eval = 0
        d_norm = norm(d)
        if f_new > f_k - gamma * alpha * d_norm ** 2:
            print(f'Linesearch failed with {alpha=:.3e}.')
            alpha = 0

            return alpha, d, f_eval, f_k

        cpu_time = time.time() - time1

        if limit_time - cpu_time < 0:
            return alpha, d, f_eval, f_new

        f_try = self.forward(w + alpha / delta * d, x, y)
        f_eval += 1

        while f_try <= min((f_k - gamma * (alpha / delta) * d_norm ** 2, f_new)):
            print(f'trying with alpha: {alpha / delta:.3e}')
            f_new = f_try
            alpha = alpha / delta
            f_try = self.forward(w + alpha / delta * d, x, y)
            f_eval += 1

            if limit_time - cpu_time < 0:
                return alpha, d, f_eval, f_new

        return alpha, d, f_eval, f_new

    def NMEDFL(self,
               d: np.ndarray,
               w: np.ndarray,
               f_k: float,
               f_new: float,
               R: float,
               x: np.ndarray,
               y: np.ndarray,
               Delta: float,
               delta: float,
               gamma: float,
               time1: float,
               limit_time: float):
        """
        Implements the NonMonotone Extrapolation Derivative Free Linesearch
        :param d: moving direction
        :param w: starting point
        :param f_k: starting obj fun value
        :param f_new: f(w_k+zeta*d) first point to evaluate the obj fun in the first iteration of LS
        :param R: max value of f in the last M iterations
        :param x,y: data point to compute obj fun
        :param Delta: starting value of the stepsize
        :param delta: update of the stepsize in the LS
        :param gamma: factor for "enough decrease"
        :param time1: starting time
        :param limit_time: limit time
        """
        alpha = Delta
        f_eval = 0
        f_eval += 1
        d_norm = norm(d)

        if f_new > R - gamma * alpha ** 2 * d_norm ** 2:
            print(f'Linesearch failed with {alpha=:.3e}.')
            alpha = 0
            return alpha, d, f_eval, f_k

        cpu_time = time.time() - time1

        if limit_time - cpu_time < 0:
            return alpha, d, f_eval, f_new

        f_try = self.forward(w + alpha / delta * d, x, y)
        f_eval += 1

        while f_try <= min((f_k - gamma * (alpha / delta) ** 2 * d_norm ** 2, f_new)):
            f_new = f_try
            alpha = alpha / delta
            f_try = self.forward(w + alpha / delta * d, x, y)
            f_eval += 1
            if limit_time - cpu_time < 0:
                return alpha, d, f_eval, f_new

        return alpha, d, f_eval, f_new
