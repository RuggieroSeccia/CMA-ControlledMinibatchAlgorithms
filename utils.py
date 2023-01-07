import os
from typing import Tuple, List

import numpy as np
from numpy.linalg import norm

from Dataset import Dataset
from NeuralNet import NN


def compute_new_row(neural_net: NN,
                    dataset: Dataset,
                    csv: str,
                    method: str, list_neurons: List[int],
                    partial_dict: dict,
                    column_names: List[str]):
    """Compute a new row to add to the dataframe results"""
    new_row_dict = {}
    new_row_dict['Dataset'] = csv
    new_row_dict['Algorithm'] = method
    new_row_dict['Network'] = list_neurons
    new_row_dict['Stop: cpu_time'] = 1

    new_row_dict.update(partial_dict)

    f_train, f_test, grad = compute_performance(neural_net, dataset, list_neurons, csv, new_row_dict['Running_Time'])

    new_row_dict['Train'] = f_train
    new_row_dict['Test'] = f_test
    new_row_dict['Grad'] = grad

    for c in column_names:
        if c not in new_row_dict.keys():
            new_row_dict[c] = None

    return new_row_dict


def create_dataset_network(dataset: str,
                           net: List[int],
                           config: dict):
    """
    Given the dataset, the NN and the algorithm configuration, initializes the NN and the dataset
    """
    np.random.seed(config['seed'])

    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'dataset/{dataset}.csv'))
    dataset = Dataset(csv_file_path, scaling=True)
    neural_net = NN(net, config['activation_fun'], dataset, config['rho'])
    print(dataset)
    print('*' * 100)
    print(neural_net)

    f, g = neural_net.forward_backward(neural_net.w, dataset.x_train, dataset.y_train)
    print('f(w_0): {:.3e}. ||g(w_0)||: {:.3e}'.format(f, norm(g)))

    return dataset, neural_net


def unpack_config(config: dict):
    '''
    unpack the parameters used within each optimization routine
    '''

    # shared parameters
    epochs = config['epochs']
    limit_time = config['limit_time']
    print_every_k_iter = config['print_every_k_iter']
    dim_mini_batch = config['dim_mini_batch']

    # algorithm specific parameters
    if config['method'] == "QN":
        options = config['algorithm']['QN']
        maxiter, maxfun, gtol, disp = options['maxiter'], options['maxfun'], options['gtol'], options['disp']
        return epochs, limit_time, maxiter, maxfun, gtol, disp

    elif config['method'] == "BLD":
        options = config['algorithm']['BLD']
        maxiter, maxfun, gtol, disp, gtol_threshold, maxiter_threshold = options['maxiter'], options['maxfun'], options['gtol'], options['disp'], options['gtol_threshold'], \
                                                                         options['maxiter_threshold']
        return epochs, limit_time, maxiter, maxfun, gtol, disp, gtol_threshold, maxiter_threshold, print_every_k_iter

    elif config['method'] == "IG":
        options = config['algorithm'][config['method']]
        alpha, riduci_alpha = options['alpha'], options['epsilon']

        return epochs, limit_time, alpha, riduci_alpha, dim_mini_batch, print_every_k_iter

    elif config['method'] == "gradient_armijo":
        options = config['algorithm'][config['method']]
        gamma, delta, Delta = options['gamma'], options['delta'], options['Delta']

        return epochs, limit_time, gamma, delta, Delta

    elif config['method'] == "CMA":
        options = config['algorithm'][config['method']]
        zeta, min_zeta, gamma, delta, theta, tau = options['zeta'], options['min_zeta'], options['gamma'], options['delta'], options['theta'], options[
            'tau']
        return epochs, limit_time, zeta, min_zeta, gamma, delta, theta, tau, dim_mini_batch, print_every_k_iter


    elif config['method'] == "NMCMA":
        options = config['algorithm'][config['method']]
        zeta, min_zeta, gamma, delta, theta, tau, M = options['zeta'], options['min_zeta'], options['gamma'], options['delta'], options['theta'], options[
            'tau'], options['M']
        return epochs, limit_time, zeta, min_zeta, gamma, delta, theta, tau, M, dim_mini_batch, print_every_k_iter


def compute_performance(neural_net: NN,
                        dataset: Dataset,
                        list_neurons: List[int],
                        csv: str,
                        cpu_time: float) -> Tuple[float, float, float]:
    """Given a trained NN and a dataset, computes the train and test error and prints some metrics"""

    f_train, g = neural_net.forward_backward(neural_net.w, dataset.x_train, dataset.y_train)
    f_test = neural_net.forward(neural_net.w, dataset.x_test, dataset.y_test)

    print('---------------------------------------------------------')
    print('-' * 23 + 'RESULTS' + '-' * 23)
    print('---------------------------------------------------------')
    print('Network: {}. Dataset: {}'.format(list_neurons, csv))
    print('Train: {:.3e}. Test: {:.3e}. Running Time: {:.2f}'.format(f_train, f_test, cpu_time))
    print('Norm of the gradient: {:.3e}\n'.format(norm(g)))

    return f_train, f_test, norm(g)
