

''' 
Bayesian optimisation for tuning the batch size 

stitched together from other repos, with my own objective function (ie validation error)

@latest: 18/8/22
@author: calmac

'''

import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, maximise=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values of the objective function for the previously
            evaluated hyperparameters.
        maximise: Boolean.
            Boolean flag that indicates whether the objective function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if maximise:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not maximise)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, maximise=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        maximise: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, maximise, n_params)
        )

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def objective(params):
    ''' This is where we pass our new hyperparameters into the dataloader and optimiser, 
        run a complete training routine (e.g. 100 epochs), and return the loss.
    '''
    
    print(params)

    # get hyperparameters from sample_next_hyperparameter()
    lr_sample = 0.1              # get learning rate  
    S_sample  = int(10**params[0])                  # get batch size
    print('Next sample:')
    print('lr:    {}'.format(lr_sample))
    print('S:     {}'.format(S_sample))
    print('lr/S:  {}'.format(lr_sample/S_sample))

    # Create fresh stuff for the next iteration 
    net = get_network(arch, dataset).cuda()
    print('--New network sorted.')

    new_optimiser = torch.optim.SGD(net.parameters(), lr=lr_sample, momentum=momentum, weight_decay=wd)
    print('--New optimiser sorted.')
    new_dataloaders = get_dataloaders(dataset, batch_size=S_sample)  
    print('--New dataloader sorted.')
    loss_function = nn.CrossEntropyLoss().cuda()

    # Complete desired number of train/val epochs with new params.
    num_train = int(train_percent*len(new_dataloaders['train'].dataset))
    num_val = int(np.round(1-train_percent, 1)*len(new_dataloaders['train'].dataset))
    train_stats = train_bayes(net, new_dataloaders['train'], loss_function, new_optimiser, num_epochs)
    val_stats = validation_bayes(net, new_dataloaders['val'], loss_function)
    
    train_acc = 100. * (train_stats['correct'] / num_train)
    print(
          'Training accuracy after {} epochs: {}/{} ({}%)'.format(
          num_epochs, train_stats['correct'], num_train, train_acc)
    )
    print('Training loss: {}'.format(train_stats['loss']))

    val_acc = 100. * (val_stats['correct'] / num_val)
    print(
          'Validation accuracy after {} training epochs: {}/{} ({}%)'.format(
          num_epochs, val_stats['correct'], num_val, val_acc)
    )
    print('Validation loss: {}\n'.format(val_stats['loss']))

    return val_stats['loss']


def bayesian_optimisation(n_iters, num_epochs, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, maximise=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the `objective function`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_obj: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        maximise: bool.
            Decide whether we are trying to maximise (True) or minimise (False) the objective function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    # n_params = bounds.shape[0]
    n_params=1

    timefile = open(os.path.join(root_path, 'totalTime.txt'), 'w')

    ''' Generate initial points (ie prior belief): should this be a burn-in i.e. N trials over 1 epoch, rather than waiting for 100 epochs?
    '''
    print('Creating prior...')
    t_start = time.time()

    if x0 is None:
        print('generating random hyperparameters...')
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
          x_list.append(params)
          y_list.append(objective(params))
    else:
        print('using initial hyperparameters...')
        for params in x0:
          x_list.append(params)
          y_list.append(objective(params))

    print('iteration took {}s'.format(time.time() - t_start))
    write_results('{}'.format(time.time() - t_start), timefile)
  
    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.RBF()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)
    print('Updating prior...\n')

    for n in range(n_iters):
        print('iteration...{}'.format(n+1))

        t_start = time.time()

        # fit covariance function to input data (ie params, acc)
        model.fit(xp, yp)

        # Sample next hyperparameters
        if random_search:
            x_random = np.random.uniform(bounds, size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, maximise=maximise, n_params=n_params)
            next_params = x_random[np.argmax(ei), :]
        else:
            # use acquisition fcn to predict params most likely to improve objective function
            next_params = sample_next_hyperparameter(expected_improvement, model, yp, maximise=maximise, bounds=bounds, n_restarts=25)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_params - xp) <= epsilon):
            next_params = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample performance for the new parameters
        sample_perf = objective(next_params)
        if np.isnan(sample_perf):
          # safety check: sometimes sampled hypers give mental results, where
          # loss goes to infty (NaN). If this happens, make loss 10.
          sample_perf = 10.0

        # Update lists
        x_list.append(next_params)
        y_list.append(sample_perf)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)
        # print(xp,yp)

        print('iteration took {}s'.format(time.time() - t_start))
        write_results('{}'.format(time.time() - t_start), timefile)
  
    return xp, yp

''' 
loops for training NN to set epoch (train_bayes) and then evaluating 
objective (validation_bayes), which will be stored for fitting our GP 
regression to for BOoptimiser.

'''

def train_bayes(net, dataloader, loss_function, optimiser, num_epochs):
    for epoch in range(num_epochs):
      loss_list = []
      net.train()
      correct = 0
      for data, target in dataloader:
          data, target = data.cuda(), target.cuda()
          optimiser.zero_grad()
          output = net(data)
          loss = loss_function(output, target)
          loss.backward()
          optimiser.step()
          loss_list.append(loss.detach().cpu().numpy())
          probs = F.softmax(output, dim=1)
          _,pred = torch.max(probs.data, 1)
          correct += (pred==target).sum().item()
      loss_epoch = np.mean(loss_list)
      print('epoch {} train loss: {}'.format(epoch+1,loss_epoch))
    return {'correct':correct, 'loss':loss_epoch}

def validation_bayes(net, dataloader, loss_function):
    net.eval()
    correct = 0
    loss_list = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            output = net(data) 
            loss = loss_function(output, target)
            loss_list.append(loss.detach().cpu().numpy())
            probs = F.softmax(output,dim=1)
            _, pred = torch.max(probs.data, 1)
            correct += (pred == target).sum().item()
        loss_epoch = np.mean(loss_list)
    return {'correct':correct, 'loss':loss_epoch}




def main():

    ''' 
    Set up prior.

    Since we know from theory that larger lr/S improves gen, this is our prior belief.
    In other words, we should provide the algorithm with this knowledge in the 
    form of initial points; we know that the noise regime is optimal, so we need 
    to give it more points within this region for it to sample from in order 
    to help it find the best region within the noise regime.
    Otherwise, the search space will be more random, thus limiting the algorithms 
    ability to look in the most likely places for the noise-curvature boundary.

    ''' 

    arch='vgg'
    dataset='cifar_10'
    optim_type='sgd'

    num_epochs=50
    n_iters = 25
    batch_size=256
    eta=0.1
    momentum=0.9
    wd=5e-4
    train_percent=0.8

    # give the algorithm some initial params to use: choose to represent our prior beliefs (ie more points at large eta/S)
    S_init = [64, 128, 256, 512]
    initial_params = [
                      [np.log10(S_init[0])],
                      [np.log10(S_init[1])],
                      [np.log10(S_init[2])],
                      [np.log10(S_init[3])],
    ] 

    S_bounds = [1, np.log10(512)]  # from s=10^(1)=10 to s=10^(bs)=S 
    bounds = np.array([S_bounds])


    # Run Bayesian optimisation algorithm
    xp, yp = bayesian_optimisation(
                                n_iters=n_iters,
                                num_epochs=num_epochs, 
                                bounds=bounds,
                                n_pre_samples=len(initial_params),
                                x0=initial_params,
                                random_search=False)


    best_params, min_loss = xp[np.argmin(yp)], yp[np.argmin(yp)]
    best_S  = int(10**best_params[0])  

    print('best params: {} at iter {}'.format(best_S, np.argmin(yp)))
    print(min_loss)

    save_results=True

    if save_results:
      i = 0
      files = ['S','loss','best']
      for i in range(len(files)):
        file = open(os.path.join(root_path, 'bayesoptim_{}.txt'.format(files[i])), 'w')
        for j in range(len(yp)):
          if i==0:
            info = str(10**xp[j, 0])+'\n'
            file.write(info)
          # elif i==1:
          #   info = str(xp[j, 1])+'\n'
          #   file.write(info)
          elif i==1:
            info = str(yp[j])+'\n'
            file.write(info)
        if i==2:
          file.write('best params: {} at iter {}, with loss={}'.format(best_S, np.argmin(yp), min_loss))
        file.close()




