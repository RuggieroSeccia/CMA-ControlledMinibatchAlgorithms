from typing import Union, Dict

import algorithms
from utils import *


def specify_config(method: Union[str, None]) -> Dict[str, Union[str, int, float, Dict]]:
    """configuration with all the parameters for the optimization method wanted"""
    config = {'method': method,
              'seed': 100,
              'limit_time': 10,
              'epochs': 2000,
              'rho': 1e-6,
              'activation_fun': 'sigmoid',
              'dim_mini_batch': 32,
              'print_every_k_iter': 1,
              'algorithm': {'QN': {'gtol': 1e-6,
                                   'disp': False,
                                   'maxfun': 1e8,
                                   'maxiter': 1e8
                                   },
                            'IG': {'alpha': 0.5,
                                   'epsilon': 0.001,
                                   },
                            'CMA': {'zeta': 0.5,
                                     'gamma': 1e-6,
                                     'delta': 0.5,
                                     'theta': 0.5,
                                     'tau': 1e-2,
                                     'min_zeta': 0,
                                     'limit_time_min_zeta': 1
                                     },
                            'NMCMA': {'zeta': 0.5,
                                      'gamma': 1e-6,
                                      'delta': 0.5,
                                      'theta': 0.5,
                                      'tau': 1e-2,
                                      'min_zeta': 0,
                                      'M': 10,
                                      'limit_time_min_zeta': 1
                                      },
                            }
              }
    return config




if __name__ == "__main__":
    # specify all the parameters for the experiment to run
    csv_file_name = 'Mv'
    method = 'IG'
    list_neurons = [20, 20, 20, 1]

    config = specify_config(method)
    # define the dataset and the neural network to optimize
    dataset, nn = create_dataset_network(csv_file_name, list_neurons, config)
    fun_obj_and_time_values, partial_dict = algorithms.call_optim_algo(method, config, dataset, nn)

    compute_performance(nn, dataset, list_neurons, csv_file_name, partial_dict['Running_Time'])
    print(partial_dict)
