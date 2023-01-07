from typing import Union, Dict

import pandas as pd

import algorithms
from utils import *


def specify_config(method: Union[str, None]) -> Dict[str, Union[str, int, float, Dict]]:
    """Specify default values for the numerical results"""
    config = {'method': method,
              'seed': 100,
              'limit_time': 2,
              'epochs': int(1e7),
              'activation_fun': 'sigmoid',
              'rho': 1e-6,
              'dim_mini_batch': 32,
              'print_every_k_iter': 1,
              'algorithm': {'IG': {'alpha': 0.5,
                                   'epsilon': 0.0001,
                                   },
                            'QN': {'gtol': 1e-6,
                                   'disp': False,
                                   'maxfun': 1e8,
                                   'maxiter': 1e8
                                   },
                            'CMA': {'zeta': 0.5,
                                    'gamma': 1e-6,
                                    'delta': 0.5,
                                    'theta': 0.5,
                                    'tau': 1e-2,
                                    'min_zeta': 0,
                                    },
                            'NMCMA': {'zeta': 0.5,
                                      'gamma': 1e-6,
                                      'delta': 0.5,
                                      'theta': 0.5,
                                      'tau': 1e-2,
                                      'min_zeta': 0,
                                      'M': 5,
                                      },
                            }
              }
    return config


def define_iterators() -> Tuple[List, List, List, int]:
    """Define the lists with all the combinations of numerical experiments to run"""
    # datasets
    csv_list = [
        'Ailerons',
        'Bejing Pm25',
        # 'Bikes Sharing',
        # 'BlogFeedback',
        # 'California',
        # 'Covtype',
        # 'Mv',
        # 'Protein',
        # 'Sido',
        # 'Skin NonSkin',
        # 'YearPredictionMSD'
    ]

    # methods
    method_list = [
        'QN',
        'IG',
        'CMA',
        'NMCMA'
    ]
    # Networks structure
    LIST_N = [
        [50, 1],
        [20, 20, 20, 1],
        # [50, 50, 50, 50, 50, 1],
        # [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1]
    ]
    # number of repetitions of the experiment
    num_runs = 2

    return csv_list, method_list, LIST_N, num_runs


if __name__ == "__main__":

    # Define all the paramters for the experiment to be run
    csv_list, method_list, LIST_N, num_runs = define_iterators()
    config = specify_config(None)
    column_names = [
        'Dataset',
        'Algorithm',
        'Network',
        'Run',
        'Stop: cpu_time',
        'Train',
        'Test',
        'Grad',
        'Running_Time',
        'Epochs',
        'Minibatches done',
        'Final_stepsize',
        'CMA: final alpha_tilde',
        'CMA: min_zeta',
        'CMA: f_eval',
        'CMA: #w_tilde_accepted',
        'CMA: #linesearch_failed',
        'CMA: #linesearch_accepted ',
        'CMA: #norm_d_tilde_small ',
        'CMA: norm_d_tilde_small',
        'CMA: out_level_curves',
        'CMA: gamma',
        'CMA: delta',
        'CMA: theta',
        'CMA: zeta_0',
        'NMCMA: M',
        'IG: alpha_0',
        'IG: epsilon']

    df = pd.DataFrame(columns=column_names)
    counter = 0

    final_results_dict = {}
    for method in method_list:
        for csv in csv_list:
            csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'dataset/{csv}.csv'))
            dataset = Dataset(csv_file_path, scaling=True)

            for list_neurons in LIST_N:
                for seed in range(1, num_runs + 1):
                    config['method'] = method
                    config['seed'] = seed * 100
                    print('------------------------------------------------------------------------------------------------------------')
                    method_print = method
                    print(f'Dataset: {csv}. Method: {method}. Rete: {list_neurons}. Run: {config["seed"]}')
                    print('------------------------------------------------------------------------------------------------------------')

                    np.random.seed(config['seed'])
                    nn = NN(list_neurons, 'sigmoid', dataset, config['rho'])

                    lista_f_t, partial_dict = algorithms.call_optim_algo(method, config, dataset, nn)

                    new_row_dict = compute_new_row(nn, dataset, csv, method, list_neurons, partial_dict, column_names)
                    df = df.append(new_row_dict, True)
                    df.to_csv('result_time' + str(config['limit_time']) + '.csv')
