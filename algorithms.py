import time

import numpy as np
from numpy.linalg import norm

import NeuralNet
from Dataset import Dataset
from NeuralNet import NN
from utils import unpack_config


def call_optim_algo(method: str,
                    config: dict,
                    dataset: Dataset,
                    nn: NeuralNet):
    """helper function to call the correct optimization method
    :param method: optimization method to call. Supported values are: QN. IG, CMA, NMCMA
    :param config: dictionary with the configuration parameters
    :param dataset: dataset object on which calling the optimization method
    :param nn: neural network to optimize
    """
    if method == 'QN':
        return QN(config, dataset, nn)
    elif method == 'IG':
        return IG(config, dataset, nn)
    elif method == 'CMA':
        return CMA(config, dataset, nn)
    elif method == 'NMCMA':
        return NMCMA(config, dataset, nn)


def QN(config: dict,
       dataset: Dataset,
       neural_net: NN):
    """Implement Quasi-Newton method from scipy

    :param config: configuration file with all the parameters to set up the experiment
    :param dataset: object with all the information aobut the data: values, size etc...
    :param neural_net: neural networks to train
    :returns
     - fun_obj_and_time_values: list with tuples: (objective function value, running time)
     - dict with parameters obtained from the optimization
    """
    fun_obj_and_time_values = []
    f_w_k = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)
    fun_obj_and_time_values.append((f_w_k, 0))

    epochs, limit_time, maxiter, maxfun, gtol, disp = unpack_config(config)
    time_0 = time.time()
    stop_cpu_time = 0
    res = neural_net.optimize_NN_with_LBFGS(dataset.x_train, dataset.y_train, True, time_0, limit_time, fun_obj_and_time_values, maxiter, maxfun, gtol, disp)

    print(f'Train: {res.fun:.2e}. Running time {time.time() - time_0:.3e}')
    cpu_time = time.time() - time_0
    stop_cpu_time = int(cpu_time > limit_time)
    partial_dict = {
        'Running_Time': cpu_time,
        'Stop: cpu_time': stop_cpu_time,
        'Run': config['seed'],
        'Epochs': res.nfev}  # for QN we consider the number of function evaluations as epochs

    return fun_obj_and_time_values, partial_dict


def IG_iteration(dataset: Dataset,
                 neural_net: NN,
                 dim_mini_batch: int,
                 alpha: float,
                 time1: float,
                 limit_time: float,
                 epsilon: float,
                 reduce_alpha: float):
    """Perform an IG iteration
    :param dataset: dataset passed in input
    :param neural_net: neural network on which we want to apply the minibatch algorithm
    :param dim_mini_batch: dimension of the minibatch we want to use for training
    :param alpha: stepsize used while applying the IG iteration
    :param time1: time when we start running the main algorithm (not IG_iteration). Used to determine if we exceed the limit_time
    :param limit_time: maximum running time. If exceeded the algorithm terminates
    :param epsilon: value used to update the stepsize within each minibatch iteration. Used only if reduce_alpha == True
    :param reduce_alpha: if True, the stepsize alpha is reduced at each minibatch iteration
    :returns
        - alpha: final value of alpha obained
        - nn: neural network after the IG iteration
        - cpu_time: final running time
        - t: number of minibatches run (needed to know if we stop the algorithm while running some epochs
    """
    cpu_time, t = None, None
    num_iterations = int(np.ceil(dataset.P / dim_mini_batch))
    if reduce_alpha:
        for t in range(num_iterations):

            dataset.minibatch(t * dim_mini_batch, (t + 1) * dim_mini_batch)
            f, g = neural_net.forward_backward(neural_net.w, dataset.x_train_mb, dataset.y_train_mb)

            neural_net.w = neural_net.w - alpha * g

            alpha = alpha * (1 - epsilon * alpha)
            cpu_time = time.time() - time1

            if limit_time - cpu_time < 0:
                return alpha, neural_net, cpu_time, t
        return alpha, neural_net, cpu_time, t
    else:
        # to compute the final direction without approximation errors we need to sum up all the directions
        d = np.zeros((num_iterations, neural_net.w.size))
        for t in range(num_iterations):

            dataset.minibatch(t * dim_mini_batch, (t + 1) * dim_mini_batch)
            f, g = neural_net.forward_backward(neural_net.w, dataset.x_train_mb, dataset.y_train_mb)
            d[t, :] = g
            neural_net.w = neural_net.w - alpha * g

            cpu_time = time.time() - time1

            if limit_time - cpu_time < 0:
                d = d.sum(axis=0)
                return alpha, neural_net, cpu_time, t, d
        d = d.sum(axis=0)
        return alpha, neural_net, cpu_time, t, d


def IG(config: dict,
       dataset: Dataset,
       neural_net: NN):
    """Implement Incremental Gradient method

    :param config: configuration file with all the parameters to set up the experiment
    :param dataset: object with all the information aobut the data: values, size etc...
    :param neural_net: neural networks to train
    :returns
     - fun_obj_and_time_values: list with tuples: (objective function value, running time)
     - dict with parameters obtained from the optimization
    """

    fun_obj_and_time_values = []
    fun_obj_and_time_values.append((neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train), 0))

    epochs, limit_time, alpha_0, epsilon, dim_mini_batch, print_every_k_iter = unpack_config(config)
    time1 = time.time()
    k, t = 0, 0
    alpha = alpha_0
    for k in range(epochs):

        alpha, neural_net, cpu_time, t = IG_iteration(dataset, neural_net, dim_mini_batch, alpha, time1, limit_time, epsilon, reduce_alpha=True)
        butta_cpu_time = time.time()
        fun_obj_and_time_values.append((neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train), cpu_time))
        time1 = time1 + time.time() - butta_cpu_time

        if limit_time - cpu_time < 0:
            break

        if print_every_k_iter:
            print('=' * 40)
            print(' ' * 15 + 'Epoca: ' + str(k + 1))
            print('=' * 40)
            print('alpha: {:.3e}'.format(alpha))
            f = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)

            print('f vale: {:.2e}. Epoca: {}. Running time: {}.'.format(f, k, time.time() - time1))

    cpu_time = time.time() - time1

    partial_dict = {
        'Running_Time': cpu_time,
        'Run': config['seed'],
        'Final_stepsize': alpha,
        'Epochs': k + 1,
        'Minibatches done': t,
        'IG: alpha_0': alpha_0,
        'IG: epsilon': epsilon}

    return fun_obj_and_time_values, partial_dict


def CMA(config: dict,
        dataset: Dataset,
        neural_net: NN):
    """Implement Comtrolled Minibatch method

    :param config: configuration file with all the parameters to set up the experiment
    :param dataset: object with all the information aobut the data: values, size etc...
    :param neural_net: neural networks to train
    :returns
     - fun_obj_and_time_values: list with tuples: (objective function value, running time)
     - dict with parameters obtained from the optimization
    """
    epochs, limit_time, zeta, min_zeta, gamma, delta, theta, tau, dim_mini_batch, print_every_k_iter = unpack_config(config)

    fun_obj_and_time_values = []
    w_tilde_accepted = 0  # numero di volte che uso w_tilde per aggionrare w
    linesearch_failed = 0  # numero di volte che la linesearch non mi aiuta a migliorare la soluzione
    linesearch_accepted = 0  # numero di volte che accetto alpha_tilde proposto da linesearch
    norm_d_tilde_small = 0  # numero di ovlte che d_tilde è troppo piccola per chiamare EDFL
    out_level_curves = 0  # numero di volte che sono fuori dalle curve di livello
    f_eval = 0
    alpha_tilde = -np.inf
    f_w_k = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)
    print(f'f(w_0): {f_w_k:.3e}')
    fun_obj_and_time_values.append((f_w_k, 0))
    time1 = time.time()
    f_w_0 = f_w_k.copy()
    f_eval += 1
    zeta_0 = zeta
    k, t = 0, 0
    stop_cpu_time = 1

    for k in range(epochs):
        print('=' * 40)
        print(' ' * 15 + 'Epoca: ' + str(k + 1))
        print('=' * 40)
        if zeta <= min_zeta:
            stop_cpu_time = 0
            break

        w_k = neural_net.w.copy()

        zeta, neural_net, cpu_time, t, d_tilde = IG_iteration(dataset, neural_net, dim_mini_batch, zeta, time1, limit_time, epsilon=None, reduce_alpha=False)
        if limit_time - cpu_time < 0:
            break
        f_w_tilde = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)
        f_eval += 1

        if f_w_tilde <= f_w_k - gamma * zeta:
            print(f'w_tilde viene accettato. Riduzione ottenuta {f_w_tilde - (f_w_k - gamma * zeta):.3e}')
            w_tilde_accepted += 1
            alpha = zeta
            y = neural_net.w.copy()
            f_y = f_w_tilde

        else:
            if norm(d_tilde) <= tau * zeta:
                print(f'norma di d_tilde troppo piccola. {norm(d_tilde)=:.3e}<= {(tau * zeta)=:.3e}')
                norm_d_tilde_small += 1
                if f_w_tilde <= f_w_0:  # todo secondo me qua serve f_w_k al posto di f_w_0
                    print('Accetto w_tilde poiche migliora f_w_0')  # todo cambia se cambi f_w_0 sopra
                    w_tilde_accepted += 1
                    alpha = zeta
                    y = neural_net.w.copy()
                    f_y = f_w_tilde
                else:
                    print('w_tilde rifiutato')
                    print('Non aggiorno w_k poiche sono fuori dalle curve di livello L_0')  # todo cambia se cambi f_w_0 sopra
                    out_level_curves += 1
                    alpha = 0
                    y = w_k
                    f_y = f_w_k
                print('Riduco la stepsize zeta')
                zeta = theta * zeta
            else:
                print('Effettuo Linesearch')
                alpha_tilde, d, f_eval_i, f_y = neural_net.EDFL(d_tilde, w_k, f_w_k, f_w_tilde, dataset.x_train, dataset.y_train, zeta, delta, gamma, time1, limit_time)
                print(f'alpha_tilde ottenuto: {alpha_tilde:.3e}')
                f_eval += f_eval_i
                if alpha_tilde * norm(d_tilde) ** 2 <= tau * zeta:
                    print(f'Passo troppo piccolo. {alpha_tilde * norm(d_tilde) ** 2:.3e}<= {tau * zeta}')
                    if alpha_tilde > 0:
                        print('Aggiorno alpha ugualmente poiche la linesearch non è fallita')
                        linesearch_accepted += 1
                        alpha = alpha_tilde
                        y = w_k + alpha * d_tilde
                        # f_y is already returned by the LS
                    elif alpha_tilde == 0 and f_w_tilde <= f_w_0:
                        print('Accetto w_tilde')
                        w_tilde_accepted += 1
                        alpha = zeta
                        linesearch_failed += 1
                        y = neural_net.w.copy()
                        f_y = f_w_tilde
                    else:  # if alpha_tilde ==0 and f_w_tilde <= f_w_0
                        out_level_curves += 1
                        linesearch_failed += 1
                        alpha = 0
                        y = w_k
                        f_y = f_w_k
                    zeta = theta * zeta
                else:
                    print('Linesearch effettuata. Aggiorno alpha')
                    linesearch_accepted += 1
                    alpha = alpha_tilde
                    y = w_k + alpha * d_tilde
                    # f_y is already returned by the LS

        if print_every_k_iter:
            print('_' * 10 + ' Recap ' + '_' * 10)
            print('f_eval: {}'.format(f_eval))
            print(f"{zeta=}")
            print(f"{alpha_tilde=}")
            print(f"{alpha=}")
            print(f'{"f_w_k:":<15}{f_w_k:>5.5f}')
            print(f'{"f_w_tilde:":<15}{f_w_tilde:>5.5f}')
            print(f'{"f_y:":<15}{f_y:>5.5f}')
            if f_w_k - f_y > 0:
                print(f'Miglioramento funzione obiettivo:{f_w_k - f_y=:.3e}')
            if f_w_tilde - f_y > 0:
                print(f'Miglioramento rispetto w_tilde: {f_w_tilde - f_y=:.3e}')
            _, g = neural_net.forward_backward(neural_net.w, dataset.x_train, dataset.y_train)
            print(f"{norm(g)=:.3e}")
            print(f"{norm(d_tilde)=:.3e}")

        # Update w_k and f_w_k
        neural_net.w = y
        f_w_k = f_y
        fun_obj_and_time_values.append((f_w_k, time.time() - time1))

        cpu_time = time.time() - time1
        if cpu_time > limit_time:
            break
        if zeta < min_zeta:  # alpha potrebbe diventare troppo piccolo. Devo stoppare le iterazioni e segnarmi il perche si è bloccato
            stop_cpu_time = 0
            break

    cpu_time = time.time() - time1

    partial_dict = {
        'Running_Time': cpu_time,
        'Stop: cpu_time': stop_cpu_time,
        'Run': config['seed'],
        'Epochs': k + 1,
        'Minibatches done': t,
        'CMA: #linesearch_failed': linesearch_failed,
        'CMA: #w_tilde_accepted': w_tilde_accepted,
        'CMA: #linesearch_accepted ': linesearch_accepted,
        'CMA: #norm_d_tilde_small ': norm_d_tilde_small,
        'Final_stepsize': zeta,
        'CMA: final alpha_tilde': alpha_tilde,
        'CMA: f_eval': f_eval,
        'CMA: gamma': gamma,
        'CMA: delta': delta,
        'CMA: theta': theta,
        'CMA: zeta_0': zeta_0,
        'CMA: min_zeta': min_zeta,
        'CMA: norm_d_tilde_small': norm_d_tilde_small,
        'CMA: out_level_curves': out_level_curves}

    return fun_obj_and_time_values, partial_dict


def NMCMA(config: dict,
          dataset: Dataset,
          neural_net: NN):
    """Implement Nonmonotone Comtrolled Minibatch method

    :param config: configuration file with all the parameters to set up the experiment
    :param dataset: object with all the information aobut the data: values, size etc...
    :param neural_net: neural networks to train
    :returns
     - fun_obj_and_time_values: list with tuples: (objective function value, running time)
     - dict with parameters obtained from the optimization
    """

    epochs, limit_time, zeta, min_zeta, gamma, delta, theta, tau, M, dim_mini_batch, print_every_k_iter = unpack_config(config)

    fun_obj_and_time_values = []
    w_tilde_accepted = 0  # numero di volte che uso w_tilde per aggionrare w
    linesearch_failed = 0  # numero di volte che la linesearch non mi aiuta a migliorare la soluzione
    linesearch_accepted = 0  # numero di volte che accetto alpha_tilde proposto da linesearch
    norm_d_tilde_small = 0  # numero di volte che d_tilde è troppo piccola per chiamare EDFL
    alpha_tilde = -np.inf
    f_eval = 0
    f_w_k = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)
    # print(f'f(w_0): {f_w_k:.3e}')
    fun_obj_and_time_values.append((f_w_k, 0))
    time1 = time.time()
    f_eval += 1
    zeta_0 = zeta
    k, t = 0, 0
    stop_cpu_time = 1
    list_f = np.zeros(M)

    for k in range(epochs):
        # print('=' * 40)
        # print(' ' * 15 + 'Epoca: ' + str(k + 1))
        # print('=' * 40)

        if zeta <= min_zeta:
            stop_cpu_time = 0
            break

        w_k = neural_net.w.copy()

        list_f[np.mod(k, M)] = np.max(f_w_k)
        R = np.max(list_f)
        zeta, neural_net, cpu_time, t, d_tilde = IG_iteration(dataset, neural_net, dim_mini_batch, zeta, time1, limit_time, epsilon=None, reduce_alpha=False)
        if limit_time - cpu_time < 0:
            break
        f_w_tilde = neural_net.forward(neural_net.w, dataset.x_train, dataset.y_train)
        f_eval += 1

        # print(f'f(w_tilde): {f_w_tilde:.3e}')

        if f_w_tilde <= R - gamma * np.max([zeta, zeta * norm(d_tilde)]):
            # print(f"{f_w_tilde=:.5f} <= {R - gamma * np.max([zeta, zeta * norm(d_tilde)])=:.5f}")
            # print(f"f(w_tilde) viene accettato!")
            w_tilde_accepted += 1
            alpha = zeta
            y = neural_net.w.copy()
            f_y = f_w_tilde
        else:
            # print(f"w_tilde non riduce sufficientemente la funzione obiettivo!")
            if norm(d_tilde) <= tau * zeta:
                # print(f'norma di d_tilde troppo piccola. {norm(d_tilde):.3e} <={tau * zeta=:.3e}\n '
                #       f'Riduco la stepsize zeta e non aggiorno w_k')
                norm_d_tilde_small += 1
                zeta = theta * zeta
                y = w_k
                f_y = f_w_k
                alpha = 0
            else:
                # print('Effettuo Linesearch:')
                alpha_tilde, d, f_eval_i, f_y = neural_net.NMEDFL(d_tilde, w_k, f_w_k, f_w_tilde, R, dataset.x_train, dataset.y_train, zeta, delta, gamma, time1, limit_time)
                # print(f'alpha_tilde ottenuto: {alpha_tilde:.3f}')
                f_eval += f_eval_i
                if alpha_tilde > 0:
                    linesearch_accepted += 1
                if alpha_tilde ** 2 * norm(d_tilde) ** 2 <= tau * zeta:
                    # print(f'Passo troppo piccolo. {alpha_tilde ** 2 * norm(d_tilde) ** 2:.3e}<= {tau * zeta}')
                    # print('Riduco zeta ed aggiorno w_k con il nuovo passo alpha_tilde trovato')
                    zeta = theta * zeta
                alpha = alpha_tilde
                y = w_k + alpha * d_tilde

        if print_every_k_iter:
            print('_' * 10 + ' Recap ' + '_' * 10)
            print('f_eval: {}'.format(f_eval))
            print(f"{zeta=}")
            print(f"{alpha_tilde=}")
            print(f"{alpha=}")
            print(f'{"f_w_k:":<15}{f_w_k:>5.5f}')
            print(f'{"f_w_tilde:":<15}{f_w_tilde:>5.5f}')
            print(f'{"f_y:":<15}{f_y:>5.5f}')
            if f_w_k - f_y > 0:
                print(f'Miglioramento funzione obiettivo:{f_w_k - f_y=:.3e}')
            if f_w_tilde - f_y > 0:
                print(f'Miglioramento rispetto w_tilde: {f_w_tilde - f_y=:.3e}')
            _, g = neural_net.forward_backward(neural_net.w, dataset.x_train, dataset.y_train)
            print(f"{norm(g)=:.3e}")

        # Update w_k and f_w_k
        neural_net.w = y
        f_w_k = f_y
        fun_obj_and_time_values.append((f_w_k, time.time() - time1))

        if zeta < min_zeta:  # zeta potrebbe diventare troppo piccolo. Devo stoppare le iterazioni e segnarmi il perche si è bloccato
            stop_cpu_time = 0
            break
        cpu_time = time.time() - time1
        if cpu_time > limit_time:
            break

    cpu_time = time.time() - time1

    partial_dict = {
        'Running_Time': cpu_time,
        'Stop: cpu_time': stop_cpu_time,
        'Run': config['seed'],
        'Epochs': k + 1,
        'Minibatches done': t,
        'CMA: #linesearch_failed': linesearch_failed,
        'CMA: #w_tilde_accepted': w_tilde_accepted,
        'CMA: #linesearch_accepted ': linesearch_accepted,
        'CMA: #norm_d_tilde_small ': norm_d_tilde_small,
        'Final_stepsize': zeta,
        'CMA: final alpha_tilde': alpha_tilde,
        'CMA: f_eval': f_eval,
        'CMA: gamma': gamma,
        'CMA: delta': delta,
        'CMA: theta': theta,
        'CMA: zeta_0': zeta_0,
        'CMA: min_zeta': min_zeta,
        'CMA: norm_d_tilde_small': norm_d_tilde_small,
        'NMCMA: M': M}

    return fun_obj_and_time_values, partial_dict
