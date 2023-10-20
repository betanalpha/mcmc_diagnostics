################################################################################
#
# The code is copyright 2003 Michael Betancourt and licensed under the
# new BSD (3-clause) license:
#  https://opensource.org/licenses/BSD-3-Clause
#
# For more information see https://github.com/betanalpha/mcmc_diagnostics.
#
################################################################################

# Load required libraries
import matplotlib
import matplotlib.pyplot as plot
from matplotlib.colors import LinearSegmentedColormap

light = "#DCBCBC"
light_highlight = "#C79999"
mid = "#B97C7C"
mid_highlight = "#A25050"
dark = "#8F2727"
dark_highlight = "#7C0000"

light_teal = "#6B8E8E"
mid_teal = "#487575"
dark_teal = "#1D4F4F"

import numpy
import math
import textwrap

import pystan
import pickle
import re

def compile_model(filename, model_name=None, **kwargs):
  """This will automatically cache models.
  
    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""
  from hashlib import md5
  
  with open(filename) as f:
    model_code = f.read()
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
      cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
      cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash) 
    try:
      sm = pickle.load(open(cache_fn, 'rb'))
    except:
      sm = pystan.StanModel(model_code=model_code)
      with open(cache_fn, 'wb') as f:
        pickle.dump(sm, f)
    else:
       print("Using cached StanModel")
    return sm

# Extract unpermuted expectand values from a StanFit object and format 
# them for convenient access
# @param stan_fit A StanFit object
# @return A dictionary of two-dimensional arrays for each expectand in 
#         the StanFit object.  The first dimension of each element 
#         indexes the Markov chains and the second dimension indexes the 
#         sequential states within each Markov chain. 
def extract_expectands(stan_fit):
  nom_params = stan_fit.extract(permuted=False, inc_warmup=False)
  params = { name: numpy.transpose(nom_params[:,:,k]) 
             for k, name in enumerate(stan_fit.flatnames) }
  return params

# Extract Hamiltonian Monte Carlo diagnostics values from a StanFit
# object and format them for convenient access
# @param stan_fit A StanFit object
# @return A dictionary of two-dimensional arrays for each expectand in 
#         the StanFit object.  The first dimension of each element 
#         indexes the Markov chains and the second dimension indexes the 
#         sequential states within each Markov chain. 
def extract_hmc_diagnostics(stan_fit):
  diagnostic_names = ['divergent__', 'treedepth__', 'n_leapfrog__', 
                      'stepsize__', 'energy__', 'accept_stat__' ]
  
  nom_params = stan_fit.get_sampler_params(inc_warmup=False)
  C = len(nom_params)
  params = { name: numpy.vstack([ nom_params[c][name] 
                                  for c in range(C) ])
           for name in diagnostic_names }
  return params

# Check all Hamiltonian Monte Carlo Diagnostics 
# for an ensemble of Markov chains
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param adapt_target Target acceptance proxy statistic for step size 
#                     adaptation.
# @param max_treedepth The maximum numerical trajectory treedepth
# @param max_width Maximum line width for printing
def check_all_hmc_diagnostics(diagnostics,
                              adapt_target=0.801,
                              max_treedepth=10,
                              max_width=72):
  """Check all Hamiltonian Monte Carlo Diagnostics for an 
     ensemble of Markov chains"""
     
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  no_warning = True
  no_divergence_warning = True
  no_treedepth_warning = True
  no_efmi_warning = True
  no_accept_warning = True
  
  messages = []
  
  C = diagnostics['divergent__'].shape[0]
  S = diagnostics['divergent__'].shape[1]
  
  for c in range(C):
    local_messages = []
    
    # Check for divergences
    n_div = sum(diagnostics['divergent__'][c])
    
    if n_div > 0:
      no_warning = False
      no_divergence_warning = False
      local_messages.append(f'  Chain {c + 1}: {n_div:.0f} of {S} '
                            f'transitions ({n_div / S:.2%}) diverged.')
    
    # Check for tree depth saturation
    n_tds = sum([ td >= max_treedepth 
                  for td in diagnostics['treedepth__'][c] ])
    
    if n_tds > 0:
      no_warning = False
      no_treedepth_warning = False
      local_messages.append(f'  Chain {c + 1}: {n_tds:.0f} of {S} '
                            f'transitions ({n_tds / S:.2%}) saturated '
                            f'the maximum treedepth of {max_treedepth}.')
    
    # Check the energy fraction of missing information (E-FMI)
    energies = diagnostics['energy__'][c]
    numer = sum( [ (energies[i] - energies[i - 1])**2 
                   for i in range(1, len(energies)) ] ) / S
    denom = numpy.var(energies)
    if numer / denom < 0.2:
      no_warning = False
      no_efmi_warning = False
      local_messages.append(f'  Chain {c + 1}: '
                            f'E-FMI = {numer / denom:.3f}.')
    
    # Check convergence of the stepsize adaptation
    ave_accept_proxy = numpy.mean(diagnostics['accept_stat__'][c])
    if ave_accept_proxy < 0.9 * adapt_target:
      no_warning = False
      no_accept_warning = False
      local_message = (f'  Chain {c + 1}: Average proxy acceptance '
                       f'statistic ({ave_accept_proxy:.3f}) is smaller '
                       f'than 90% of the target ({adapt_target:.3f}).')
      local_message = textwrap.wrap(local_message, max_width)
      local_messages += local_message
    
    if len(local_messages) > 0:
      messages.append(local_messages)
      messages.append([' '])
  
  if no_warning:
    desc = ('All Hamiltonian Monte Carlo diagnostics are consistent '
            'with accurate Markov chain Monte Carlo.')
    desc = textwrap.wrap(desc, max_width)
    messages.append(desc)
    messages.append([' '])
  
  if not no_divergence_warning:
    desc = ('Divergent Hamiltonian transitions result from '
            'unstable numerical trajectories.  These '
            'instabilities are often due to degenerate target '
            'geometry, especially "pinches".  If there are '
            'only a small number of divergences then running '
            'with adept_delta larger '
            f'than {adapt_target:.3f} may reduce the '
            'instabilities at the cost of more expensive '
            'Hamiltonian transitions.')
    desc = textwrap.wrap(desc, max_width)
    messages.append(desc)
    messages.append([' '])
  
  if not no_treedepth_warning:
    desc = ('Numerical trajectories that saturate the '
            'maximum treedepth have terminated prematurely.  '
            f'Increasing max_depth above {max_treedepth} '
            'should result in more expensive, but more '
            'efficient, Hamiltonian transitions.')
    desc = textwrap.wrap(desc, max_width)
    messages.append(desc)
    messages.append([' '])
  
  if not no_efmi_warning:
    desc = ('E-FMI below 0.2 arise when a funnel-like geometry '
            'obstructs how effectively Hamiltonian trajectories '
            'can explore the target distribution.')
    desc = textwrap.wrap(desc, max_width)
    messages.append(desc)
    messages.append([' '])
  
  if not no_accept_warning:
    desc = ('A small average proxy acceptance statistic '
            'indicates that the adaptation of the numerical '
            'integrator step size failed to converge.  This is '
            'often due to discontinuous or imprecise '
            'gradients.')
    desc = textwrap.wrap(desc, max_width)
    messages.append(desc)
    messages.append([' '])
  
  print('\n'.join([ '\n'.join(m) for m in messages ]))

# Plot outcome of inverse metric adaptation
# @param stan_fit A StanFit object
# @params B The number of bins for the inverse metric element histograms.
def plot_inv_metric(stan_fit, B=25):
  """Plot outcome of inverse metric adaptation"""
  chain_info = stan_fit.get_adaptation_info()
  C = len(chain_info)
  
  inv_metric_elems = [None] * C
  for c, raw_info in enumerate(chain_info):
    clean1 = re.sub("# Adaptation terminated\n# Step size = [0-9.]*\n#",
                    "", raw_info)
    clean2 = re.sub(" [a-zA-Z ]*:\n# ", "", clean1)
    clean3 = re.sub("\n$", "", clean2)
    inv_metric_elems[c] = [float(s) for s in clean3.split(',')]
  
  min_elem = min([ min(a) for a in inv_metric_elems ])
  max_elem = max([ max(a) for a in inv_metric_elems ])
  
  delta = (max_elem - min_elem) / B
  min_elem = min_elem - delta
  max_elem = max_elem + delta
  bins = numpy.arange(min_elem, max_elem + delta, delta)
  B = B + 2
  
  max_y = max([ max(numpy.histogram(a, bins=bins)[0]) for a in inv_metric_elems ])
  
  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[idx + delta] for idx in range(B) for delta in [0, 1]]
  
  N_plots = C
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0
  
  sampler_params = stan_fit.get_sampler_params(inc_warmup=False)
  sci_formatter = matplotlib.ticker.FuncFormatter(lambda x, lim: f'{x:.1e}')
  
  for c in range(C):
    counts = numpy.histogram(inv_metric_elems[c], bins=bins)[0]
    ys = counts[idxs]
    
    eps = sampler_params[c]['stepsize__'][0]
    
    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1
    
    axarr[idx1, idx2].plot(xs, ys, dark)
    axarr[idx1, idx2].set_title(f'Chain {c + 1}\n(Stepsize = {eps:.3e})')
    axarr[idx1, idx2].set_xlabel("Inverse Metric Elements")
    axarr[idx1, idx2].set_xlim([min_elem, max_elem])
    axarr[idx1, idx2].get_xaxis().set_major_formatter(sci_formatter)
    axarr[idx1, idx2].set_ylabel("")
    axarr[idx1, idx2].get_yaxis().set_visible(False)
    axarr[idx1, idx2].set_ylim([0, 1.05 * max_y])
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["left"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)
  
  plot.show()

# Display adapted symplectic integrator step sizes
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
def display_stepsizes(diagnostics):
  """Display adapted symplectic integrator step sizes"""
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  stepsizes = diagnostics['stepsize__']
  C = stepsizes.shape[0]
  
  for c in range(C):
    stepsize = stepsizes[c, 1]
    print(f'Chain {c + 1}: Integrator Step Size = {stepsize:.2e}')

# Display symplectic integrator trajectory lengths
# @ax Matplotlib axis object
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param nlim Optional histogram range
def plot_num_leapfrogs(ax, diagnostics, nlim=None):
  """Display symplectic integrator trajectory lenghts"""
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  lengths = diagnostics['n_leapfrog__']
  
  C = lengths.shape[0]
  colors = [dark_highlight, dark, mid_highlight, mid, light_highlight]
  cmap = LinearSegmentedColormap.from_list("reds", colors, N=C)
  
  vals_counts = [ numpy.unique(lengths[c], return_counts=True) 
                  for c in range(C) ] 
  max_n = max([ max(a[0]) for a in vals_counts ]).astype(numpy.int64) + 1
  max_counts = max([ max(a[1]) for a in vals_counts ])
  
  if nlim is None:
    nlim = [0.5, max_n + 0.5]
  
  idxs = [ idx for idx in range(max_n) for r in range(2) ]
  xs = [ idx + delta for idx in range(max_n) for delta in [-0.5, 0.5]]
  
  for c in range(C):
    counts = numpy.histogram(lengths[c], 
                             bins=numpy.arange(0.5, max_n + 1.5, 1))[0]
    ys = counts[idxs]
    
    ax.plot(xs, ys, colors[c])
  
  ax.set_xlabel("Numerical Trajectory Lengths")
  ax.set_xlim(nlim)
  ax.set_ylabel("")
  ax.get_yaxis().set_visible(False)
  ax.set_ylim([0, 1.1 * max_counts])
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

# Display symplectic integrator trajectory lengths by Markov chain
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
def plot_num_leapfrogs_by_chain(diagnostics):
  """Display symplectic integrator trajectory lengths"""
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  lengths = diagnostics['n_leapfrog__']
  C = lengths.shape[0]
  
  vals_counts = [ numpy.unique(lengths[c], return_counts=True) 
                  for c in range(C) ] 
  max_n = max([ max(a[0]) for a in vals_counts ]).astype(numpy.int64)
  max_counts = max([ max(a[1]) for a in vals_counts ])
  
  idxs = [ idx for idx in range(max_n) for r in range(2) ]
  xs = [ idx + delta for idx in range(max_n) for delta in [-0.5, 0.5]]
  
  N_plots = C
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0
  
  for c in range(C):
    counts = numpy.histogram(lengths[c], 
                             bins=numpy.arange(0.5, max_n + 1.5, 1))[0]
    ys = counts[idxs]
    
    eps = diagnostics['stepsize__'][c][0]
    
    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1
    
    axarr[idx1, idx2].plot(xs, ys, dark)
    axarr[idx1, idx2].set_title(f'Chain {c + 1}\n(Stepsize = {eps:.3e})')
    axarr[idx1, idx2].set_xlabel("Numerical Trajectory Lengths")
    axarr[idx1, idx2].set_xlim([0.5, max_n + 0.5])
    axarr[idx1, idx2].set_ylabel("")
    axarr[idx1, idx2].get_yaxis().set_visible(False)
    axarr[idx1, idx2].set_ylim([0, 1.1 * max_counts])
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)
  
  plot.show()

# Display symplectic integrator trajectory times
# @ax Matplotlib axis object
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param B The number of histogram bins
# @param nlim Optional histogram range
def plot_int_times(ax, diagnostics, B, tlim=None):
  """Display symplectic integrator trajectory times"""
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  lengths = diagnostics['n_leapfrog__']
  C = lengths.shape[0]
  eps = [ diagnostics['stepsize__'][c][0] for c in range(C) ]
  
  if tlim is None:
    # Automatically adjust histogram binning to range of outputs
    min_t = 0
    max_t = max([ eps[c] * max(diagnostics['n_leapfrog__'][c]) 
                  for c in range(C) ])
    
    # Add bounding bins
    delta = (max_t - min_t) / B
    min_t = min_t - delta
    max_t = max_t + delta
    tlim = [min_t, max_t]
    
    bins = numpy.arange(min_t, max_t + delta, delta)
    B = B + 2
  else:
    delta = (tlim[1] - tlim[0]) / B
    bins = numpy.arange(tlim[0], tlim[1] + delta, delta)
  
  colors = [dark_highlight, dark, mid_highlight, mid, light_highlight]
  cmap = LinearSegmentedColormap.from_list("reds", colors, N=C)
  
  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[b + o] for b in range(B) for o in range(2) ]
  
  max_counts = 0
  
  for c in range(C):
    counts = numpy.histogram(eps[c] * lengths[c], bins=bins)[0]
    ys = counts[idxs]
    max_counts = max(max_counts, max(counts))
    
    ax.plot(xs, ys, colors[c])
  
  ax.set_xlabel("Trajectory Integration Times")
  ax.set_xlim(tlim)
  ax.set_ylabel("")
  ax.get_yaxis().set_visible(False)
  ax.set_ylim([0, 1.1 * max_counts])
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

# Display empirical average of the proxy acceptance statistic across 
# each Markov chain
# @param diagnostics A dictionary of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
def display_ave_accept_proxy(diagnostics):
  """Display empirical average of the proxy acceptance statistic
     across each Markov chain"""
     
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  proxy_stats = diagnostics['accept_stat__']
  C = proxy_stats.shape[0]
  
  for c in range(C):
    proxy_stat = numpy.mean(proxy_stats[c,:])
    print(  f'Chain {c + 1}: Average proxy acceptance '
          + f'statistic = {proxy_stat:.3f}')

# Apply transformation identity, log, or logit transformation to
# named samples and flatten the output.  Transformation defaults to 
# identity if name is not included in `transforms` dictionary.  A 
# ValueError is thrown if samples are not properly constrained.
# @param name Expectand name.
# @param samples A named list of two-dimensional arrays for 
#                each expectand.  The first dimension of each element 
#                indexes the Markov chains and the second dimension 
#                indexes the sequential states within each Markov chain.
# @param transforms A dictionary with expectand names for keys and
#                   transformation flags for values.
# @return The transformed expectand name and a one-dimensional array of
#         flattened transformation outputs.
def apply_transform(name, samples, transforms):
  t = transforms.get(name, 0)
  transformed_name = ""
  transformed_samples = 0
  if t == 0:
    transformed_name = name
    transformed_samples = samples[name].flatten()
  elif t == 1:
    if numpy.amin(samples[name]) <= 0:
      raise ValueError( 'Log transform requested for expectand '
                       f'{name} but expectand values are not strictly ' 
                        'positive.')
    transformed_name = f'log({name})'
    transformed_samples = [ math.log(x) for x in 
                            samples[name].flatten() ]
  elif t == 2:
    if (numpy.amin(samples[name]) <= 0 or
          numpy.amax(samples[name]) >= 1):
      raise ValueError( 'Logit transform requested for expectand '
                       f'{name} but expectand values are not strictly '
                        'confined to the unit interval.')
    transformed_name = f'logit({name})'
    transformed_samples = [ math.log(x / (1 - x)) for x in
                            samples[name].flatten() ]
  return transformed_name, transformed_samples

# Plot pairwise scatter plots with non-divergent and divergent 
# transitions separated by color
# @param x_names A list of expectand names to be plotted on the x axis.
# @param y_names A list of expectand names to be plotted on the y axis.
# @param expectand_samples A named list of two-dimensional arrays for 
#                          each expectand to be plotted on the y axis.
#                          The first dimension of each element indexes 
#                          the Markov chains and the second dimension 
#                          indexes the sequential states within each 
#                          Markov chain.
# @param diagnostics A named list of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param xlim       Optional global x-axis bounds for all pair plots.
#                   Defaults to dynamic bounds for each pair plot.
# @param ylim       Optional global y-axis bounds for all pair plots.
#                   Defaults to dynamic bounds for each pair plot.
# @param transforms An optional dictionary with expectand names for keys 
#                   and transformation flags for values.  Valid flags 
#                   are
#                     0: identity
#                     1: log
#                     2: logit
#                   Defaults to empty dictionary.
# @params plot_mode Optional plotting style configuration: 
#                     0: Non-divergent transitions are plotted in 
#                        transparent red while divergent transitions are
#                        plotted in transparent green.
#                     1: Non-divergent transitions are plotted in gray 
#                        while divergent transitions are plotted in 
#                        different shades of teal depending on the 
#                        trajectory length.  Transitions from shorter
#                        trajectories should cluster somewhat closer to 
#                        the neighborhoods with problematic geometries.
#                   Defaults to 0.
# @param max_width Maximum line width for printing
def plot_div_pairs(x_names, y_names, expectand_samples, 
                   diagnostics, transforms={},
                   xlim=None, ylim=None, 
                   plot_mode=0, max_width=72):
  """Plot pairwise scatter plots with non-divergent and divergent 
     transitions separated by color"""
  if type(x_names) is not list:
    print(('Input variable `x_names` is not a list!'))
    return
  
  if type(y_names) is not list:
    print(('Input variable `y_names` is not a list!'))
    return
    
  if type(expectand_samples) is not dict:
    print(('Input variable `expectand_samples` '
           'is not a standard dictionary!'))
    return
  
  if type(diagnostics) is not dict:
    print('Input variable `diagnostics` is not a standard dictionary!')
    return
  
  if type(transforms) is not dict:
    print('Input variable `transforms` is not a standard dictionary!')
    return
  
  # Check transform flags
  for t_name, t_value in transforms.items():
    if t_value < 0 or t_value > 2:
      desc = (f'The transform flag {t_value} for '
              f'expectand {t_name} is invalid.  '
              'Plot will default to no tranformation.')
      desc = textwrap.wrap(desc, max_width)
      print('\n'.join(desc))
  
  # Check plot mode
  if plot_mode < 0 or plot_mode > 1:
    print(f'Invalid `plot mode` value {plot_mode}.')
    return
    
  # Transform expectand samples
  transformed_samples = {}
  
  transformed_x_names = []
  for name in x_names:
    try: 
      t_name, t_samples = apply_transform(name, 
                                          expectand_samples, 
                                          transforms)
    except ValueError as error:
      desc = textwrap.wrap(error, max_width)
      print('\n'.join(desc))
      return
    
    transformed_x_names.append(t_name)
    if not t_name in transformed_samples:
      transformed_samples[t_name] = t_samples
      
  transformed_y_names = []
  for name in y_names:
    try: 
      t_name, t_samples = apply_transform(name, 
                                          expectand_samples, 
                                          transforms)
    except ValueError as error:
      desc = textwrap.wrap(error, max_width)
      print('\n'.join(desc))
    
    transformed_y_names.append(t_name)
    if not t_name in transformed_samples:
      transformed_samples[t_name] = t_samples
      
  # Create pairs of transformed expectands, dropping duplicates
  pairs = []
  for x_name in transformed_x_names:
    for y_name in transformed_y_names:
      if x_name == y_name: 
        continue
      if [x_name, y_name] in pairs or [y_name, x_name] in pairs: 
        continue
      pairs.append([x_name, y_name])
  
  # Extract diagnostic information
  divergences = diagnostics['divergent__'].flatten()
  
  if plot_mode == 1:
    div_nlfs = [ x for x, d in 
                 zip(diagnostics['n_leapfrog__'].flatten(), divergences)
                 if d == 1  ]
    if len(div_nlfs) > 0:
      max_nlf = max(div_nlfs)
    else:
      max_nlf = 0
    nom_colors = [light_teal, mid_teal, dark_teal]
    cmap = LinearSegmentedColormap.from_list("teals", nom_colors, 
                                                      N=max_nlf)
  
  # Set plot layout dynamically
  N_pairs = len(pairs)
  
  if N_pairs == 1:
    N_cols = 1
    N_rows = 1
    N_plots = 1
  else:
    N_cols = 3
    N_rows = math.ceil(N_pairs / N_cols)
    
  if N_rows <= 3:
    N_plots = 1
  else:
    N_plots = math.ceil(N_rows / 3)
    N_rows = 3
    
  # Plot!
  k = 0
  
  for pair in pairs:
    if k == 0:
      f, axarr = plot.subplots(N_rows, N_cols, layout="constrained",
                               squeeze=False)
      
    x_name = pair[0]
    x_nondiv_samples = [ x for x, d in 
                         zip(transformed_samples[x_name], divergences) 
                         if d == 0  ]
    x_div_samples    = [ x for x, d in 
                         zip(transformed_samples[x_name], divergences) 
                         if d == 1  ]
    
    if xlim is None:
      xmin = min(numpy.concatenate((x_nondiv_samples, x_div_samples)))
      xmax = max(numpy.concatenate((x_nondiv_samples, x_div_samples)))
      local_xlim = [xmin, xmax]
    else:
      local_xlim = xlim
    
    y_name = pair[1]
    y_nondiv_samples = [ x for x, d in 
                         zip(transformed_samples[y_name], divergences) 
                         if d == 0  ]
    y_div_samples    = [ x for x, d in 
                         zip(transformed_samples[y_name], divergences) 
                         if d == 1  ]
    
    if ylim is None:
      ymin = min(numpy.concatenate((y_nondiv_samples, y_div_samples)))
      ymax = max(numpy.concatenate((y_nondiv_samples, y_div_samples)))
      local_ylim = [ymin, ymax]
    else:
      local_ylim = ylim
     
    idx1 = k // N_cols
    idx2 = k % N_cols
    
    if plot_mode == 0:
      axarr[idx1, idx2].scatter(x_nondiv_samples, y_nondiv_samples, s=5,
                                color=dark_highlight, alpha=0.05)
      axarr[idx1, idx2].scatter(x_div_samples, y_div_samples, s=5,
                                color="#00FF00", alpha=0.25)
    elif plot_mode == 1:
      axarr[idx1, idx2].scatter(x_nondiv_samples, y_nondiv_samples, 
                                s=5, color="#DDDDDD")
      if len(x_div_samples) > 0:
        axarr[idx1, idx2].scatter(x_div_samples, y_div_samples, s=5,
                                  cmap=cmap, c=div_nlfs)
                                
    axarr[idx1, idx2].set_xlabel(x_name)
    axarr[idx1, idx2].set_xlim(local_xlim)
    axarr[idx1, idx2].set_ylabel(y_name)
    axarr[idx1, idx2].set_ylim(local_ylim)
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)
    
    k += 1
    if k == N_rows * N_cols:
      # Flush current plot
      plot.show()
      k = 0
  
  # Turn off any remaining subplots
  if k > 0: 
    for kk in range(k, N_rows * N_cols):
      idx1 = kk // N_cols
      idx2 = kk % N_cols
      axarr[idx1, idx2].axis('off')
    plot.show()

# Compute hat{xi}, an estimate for the shape of a generalized Pareto 
# distribution from a sample of positive values using the method 
# introduced in "A New and Efficient Estimation Method for the 
# Generalized Pareto Distribution" by Zhang and Stephens 
# https://doi.org/10.1198/tech.2009.08017.
# 
# Within the generalized Pareto distribution family all moments up to 
# the mth order are finite if and only if 
#  xi < 1 / m.
#
# @params fs A one-dimensional array of positive values.
# @return Shape parameter estimate.
def compute_xi_hat(fs):
  """Compute empirical Pareto shape configuration for a positive sample"""
  N = len(fs)
  sorted_fs = sorted(fs)
  
  if sorted_fs[0] == sorted_fs[-1]:
    return -2
  
  if (sorted_fs[0] < 0):
    print("Sequence values must be positive!")
    return NaN
  
  # Estimate 25% quantile
  q = sorted_fs[math.floor(0.25 * N + 0.5)]
  if q == sorted_fs[0]:
    return -2
    
  # Heurstic Pareto configuration
  M = 20 + math.floor(math.sqrt(N))
  
  b_hat_vec = [None] * M
  log_w_vec = [None] * M
  
  for m in range(M):
    b_hat_vec[m] =   1 / sorted_fs[-1] \
                   + (1 - math.sqrt(M / (m + 0.5))) / (3 * q)
    if b_hat_vec[m] != 0:
      xi_hat = numpy.mean( [ math.log(1 - b_hat_vec[m] * f) 
                             for f in sorted_fs ] )
      log_w_vec[m] = N * (   math.log(-b_hat_vec[m] / xi_hat) 
                           - xi_hat - 1)
    else:
      log_w_vec[m] = 0
  
  # Remove terms that don't contribute to improve numerical stability 
  # of average
  log_w_vec = [ lw for lw in log_w_vec if lw != 0 ]
  b_hat_vec = [ b for b in b_hat_vec if b != 0 ]
  
  max_log_w = max(log_w_vec)
  b_hat = sum( [ b * math.exp(lw - max_log_w) 
               for b, lw in zip(b_hat_vec, log_w_vec) ] ) /\
          sum( [ math.exp(lw - max_log_w) for lw in log_w_vec ] )
  
  return numpy.mean( [ math.log(1 - b_hat * f) for f in sorted_fs ] )

# Compute empirical generalized Pareto shape for upper and lower tails
# for an arbitrary sample of expectand values, ignoring any 
# autocorrelation between the values.
# @param fs A one-dimensional array of expectand values.
# @return Left and right shape estimators.
def compute_tail_xi_hats(fs):
  """Compute empirical Pareto shape configuration for upper and lower tails"""
  f_center = numpy.median(fs)
  
  # Isolate lower and upper tails which can be adequately modeled by a 
  # generalized Pareto shape for sufficiently well-behaved distributions
  fs_left = [ math.fabs(f - f_center) for f in fs if f <= f_center ]
  N = len(fs_left)
  M = int(min(0.2 * N, 3 * 3 * math.sqrt(N)))
  fs_left = fs_left[M:N]
  
  fs_right = [ f - f_center for f in fs if f > f_center ]
  N = len(fs_right)
  M = int(min(0.2 * N, 3 * 3 * math.sqrt(N)))
  fs_right = fs_right[M:N] 
  
  # Default to NaN if left tail is ill-defined
  xi_hat_left = math.nan
  if len(fs_left) > 40:
    xi_hat_left = compute_xi_hat(fs_left)
  
  # Default to NaN if right tail is ill-defined
  xi_hat_right = math.nan
  if len(fs_right) > 40:
    xi_hat_right = compute_xi_hat(fs_right)
    
  return [xi_hat_left, xi_hat_right]

# Check upper and lower tail behavior of a given expectand output 
# ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
def check_tail_xi_hats(samples, max_width=72):
  """Check empirical Pareto shape configuration for upper and lower 
     tails of a given expectand output ensemble"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
  
  C = samples.shape[0]
  no_warning = True
  
  for c in range(C):
    xi_hats = compute_tail_xi_hats(samples[c,:])
    xi_hat_threshold = 0.25
    if math.isnan(xi_hats[0]) and math.isnan(xi_hats[1]):
      no_warning = False
      print(f'  Chain {c + 1}: Both left and right tail '
            'hat{{xi}}s are Nan!\n')
    elif math.isnan(xi_hats[0]):
      no_warning = False
      print(f'  Chain {c + 1}: Left tail '
            'hat{{xi}} is Nan!\n')
    elif math.isnan(xi_hats[1]):
      no_warning = False
      print(f'  Chain {c + 1}: Right tail '
            'hat{{xi}} is Nan!\n')
    elif (    xi_hats[0] >= xi_hat_threshold 
         and xi_hats[1] >= xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Both left and right tail '
            f'hat{{xi}}s ({xi_hats[0]:.3f}, '
            f'{xi_hats[1]:.3f}) exceed '
            f'{xi_hat_threshold}!\n')
    elif (    xi_hats[0] < xi_hat_threshold 
          and xi_hats[1] >= xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Right tail hat{{xi}} '
            f'({xi_hats[1]:.3f}) exceeds '
            f'{xi_hat_threshold}!\n')
    elif (    xi_hats[0] >= xi_hat_threshold 
          and xi_hats[1] < xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Left tail hat{{xi}} '
            f'({xi_hats[0]:.3f}) exceeds '
            f'{xi_hat_threshold}!\n')
  
  if no_warning:
    print('Expectand appears to be sufficiently integrable.\n')
  else:
    desc = ('  Large tail xi_hats suggest that the expectand might'
            'not be sufficiently integrable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Compute empirical mean and variance of a given sequence with a single
# pass using Welford accumulators.
# @params A one-dimensional array of expectand values.
# @return The empirical mean and variance.
def welford_summary(fs):
  """Welford accumulator for empirical mean and variance of a
     given sequence"""
  mean = 0
  var = 0
  
  for n, f in enumerate(fs):
    delta = f - mean
    mean += delta / (n + 1)
    var += delta * (f - mean)
    
  var /= (len(fs) - 1)
  
  return [mean, var]

# Check expectand output ensemble for vanishing empirical variance.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
def check_variances(samples, max_width=72):
  """Check expectand output ensemble for vanishing empirical variance"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
  
  C = samples.shape[0]
  no_warning = True
  
  for c in range(C):
    var = welford_summary(samples[c,:])[1]
    if var < 1e-10:
      no_warning = True
      print(f'  Chain {c + 1}: Expectand is constant!\n')
  
  if no_warning:
    print('Expectand is varying in all Markov chains.\n')
  else:
    desc = ('  If the expectand is not expected (haha) to be '
            'constant then the Markov transitions are misbehaving.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Split a sequence of expectand values in half to create an initial and 
# terminal Markov chains
# @params chain A sequence of expectand values derived from a single 
#               Markov chain.
# @return Two subsequences of expectand values.
def split_chain(chain):
  """Split a Markov chain into initial and terminal Markov chains"""
  N = len(chain)
  M = N // 2
  return [ chain[0:M], chain[M:N] ]

# Compute split hat{R} for the expectand values across a Markov chain 
# ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @return Split Rhat estimate.
def compute_split_rhat(samples):
  """Compute split hat{R} for an expectand output ensemble across
     a collection of Markov chains"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
  
  split_chains = [ c for chain in samples for c in split_chain(chain) ]
  N_chains = len(split_chains)
  N = sum([ len(chain) for chain in split_chains ])
  
  means = [None] * N_chains
  vars = [None] * N_chains
  
  for c, chain in enumerate(split_chains):
    summary = welford_summary(chain)
    means[c] = summary[0]
    vars[c] = summary[1]
  
  total_mean = sum(means) / N_chains
  W = sum(vars) / N_chains
  B = N * sum([ (mean - total_mean)**2 / (N_chains - 1) 
                for mean in means ])
  
  rhat = math.nan
  if abs(W) > 1e-10:
    rhat = math.sqrt( (N - 1 + B / W) / N )
  
  return rhat

# Compute split hat{R} for all input expectands
# @param expectand_samples A dictionary of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
def compute_split_rhats(expectand_samples):
  """Compute split hat{R} for all expectand output ensembles across
     a collection of Markov chains"""
  if type(expectand_samples) is not dict:
    print(('Input variable `expectand_samples` '
           'is not a standard dictionary!'))
    return
    
  rhats = []
  for name in expectand_samples:
    samples = expectand_samples[name]
    rhats.append(compute_split_rhat(samples))
  
  return rhats

# Check split hat{R} across a given expectand output ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
def check_rhat(samples, max_width=72):
  """Check split hat{R} for all expectand output ensembles across
     a collection of Markov chains"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
    
  rhat = compute_split_rhat(samples)
  
  no_warning = True
  
  if math.isnan(rhat):
    print('All Markov chains appear to be frozen!')
  elif rhat > 1.1:
    print(f'Split hat{{R}} is {rhat:.3f}!')
    no_warning = False
 
  if no_warning:
    desc = ('Markov chain behavior is consistent with equilibrium.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  else:
    desc = ('Split hat{R} larger than 1.1 suggests that at least one '
            'of the Markov chains has not reached an equilibrium.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Compute empirical integrated autocorrelation time for a sequence
# of expectand values, known here as \hat{tau}.
# @param fs A one-dimensional array of expectand values.
# @return Left and right shape estimators.
def compute_tau_hat(fs):
  """Compute empirical integrated autocorrelation time for a sequence"""
  # Compute empirical autocorrelations
  N = len(fs)
  m, v = welford_summary(fs)
  zs = [ f - m for f in fs ]
  
  if v < 1e-10:
    return math.inf
  
  B = 2**math.ceil(math.log2(N)) # Next power of 2 after N
  zs_buff = zs + [0] * (B - N)
  
  Fs = numpy.fft.fft(zs_buff)
  Ss = numpy.abs(Fs)**2
  Rs = numpy.fft.ifft(Ss)
  
  acov_buff = numpy.real(Rs)
  rhos = acov_buff[0:N] / acov_buff[0]
  
  # Drop last lag if (L + 1) is odd so that the lag pairs are complete
  L = N
  if (L + 1) % 2 == 1:
    L = L - 1
  
  # Number of lag pairs
  P = (L + 1) // 2
  
  # Construct asymptotic correlation from initial monotone sequence
  old_pair_sum = rhos[0] + rhos[1]
  for p in range(1, P):
    current_pair_sum = rhos[2 * p] + rhos[2 * p + 1]
    
    if current_pair_sum < 0:
      rho_sum = sum(rhos[1:(2 * p)])
      
      if rho_sum <= -0.25:
        rho_sum = -0.25
      
      asymp_corr = 1.0 + 2 * rho_sum
      return asymp_corr
    
    if current_pair_sum > old_pair_sum:
      current_pair_sum = old_pair_sum
      rhos[2 * p]     = 0.5 * old_pair_sum
      rhos[2 * p + 1] = 0.5 * old_pair_sum
    
    # if p == P:
      # throw some kind of error when autocorrelation
      # sequence doesn't get terminated
    
    old_pair_sum = current_pair_sum

# Compute the maximum empirical effective sample size across the 
# Markov chains for the given expectands
# @param expectand_samples A named list of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
def compute_min_eesss(expectand_samples):
  """Compute the minimimum empirical integrated autocorrelation time
     across a collection of Markov chains for all expectand output
     ensembles"""
  if type(expectand_samples) is not dict:
    print(('Input variable `expectand_samples` '
           'is not a standard dictionary!'))
    return
      
  min_eesss = []
  for name in expectand_samples:
    samples = expectand_samples[name]
    C = samples.shape[0]
    S = samples.shape[0]
    
    eesss = [None] * 4
    for c in range(C):
      tau_hat = compute_tau_hat(samples[c,:])
      eesss[c] = S / tau_hat
    
    min_eesss.append(min(eesss))
  
  return min_eesss

# Check the empirical effective sample size (EESS) for all a given 
# expectand output ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param min_eess_per_chain The minimum empirical effective sample size
#                           before a warning message is passed.
# @param max_width Maximum line width for printing
def check_eess(samples, min_eess_per_chain=100, max_width=72):
  """Check the empirical effective sample size for all expectand 
     output ensembles"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
  
  no_warning = True
  C = samples.shape[0]
  S = samples.shape[1]
  
  for c in range(C):
    tau_hat = compute_tau_hat(samples[c,:])
    eess = S / tau_hat
    if eess < min_eess_per_chain:
      print(f'Chain {c + 1}: The empirical effective sample size '
            f'{eess :.1f} is too small!')
      no_warning = False
  
  if no_warning:
    desc = ('The empirical effective sample size is large enough for '
            'Markov chain Monte Carlo estimation to be reliable '
            'assuming that a central limit theorem holds.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  else:
    desc = ('If the empirical effective sample size is too small than '
            'Markov chain Monte Carlo estimation may be unreliable '
            'even when a central limit theorem holds.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Check all expectand-specific diagnostics.
# @param expectand_samples A dictionary of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
# @param min_eess_per_chain The minimum empirical effective sample size
#                           before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @param max_width Maximum line width for printing
def check_all_expectand_diagnostics(expectand_samples,
                                    min_eess_per_chain=100,
                                    exclude_zvar=False,
                                    max_width=72):
  """Check all expectand diagnostics"""
  if type(expectand_samples) is not dict:
    print(('Input variable `expectand_samples` '
           'is not a standard dictionary!'))
    return
  
  no_xi_hat_warning = True 
  no_zvar_warning = True
  no_rhat_warning = True
  no_eess_warning = True
  
  message = ""
  
  for name in expectand_samples:
    samples = expectand_samples[name]
    C = samples.shape[0]
    S = samples.shape[1]
    
    local_warning = False
    local_message = name + ':\n'
  
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(samples[c,:])[1]
        if var < 1e-10:
          any_zvar = True
      
      if any_zvar:
        continue
    
    for c in range(C):
      fs = samples[c,:]
      
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(fs)
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) and math.isnan(xi_hats[1]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Both left and right tail '
                          'hat{{xi}}s are Nan!\n')
      elif math.isnan(xi_hats[0]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Left tail '
                          'hat{{xi}} is Nan!\n')
      elif math.isnan(xi_hats[1]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Right tail '
                          'hat{{xi}} is Nan!\n')
      if (    xi_hats[0] >= xi_hat_threshold 
          and xi_hats[1] >= xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Both left and right tail '
                          f'hat{{xi}}s ({xi_hats[0]:.3f}, '
                          f'{xi_hats[1]:.3f}) exceed '
                          f'{xi_hat_threshold}!\n')
      elif (    xi_hats[0] < xi_hat_threshold 
            and xi_hats[1] >= xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Right tail hat{{xi}} '
                          f'({xi_hats[1]:.3f}) exceeds '
                          f'{xi_hat_threshold}!\n')
      elif (    xi_hats[0] >= xi_hat_threshold 
            and xi_hats[1] < xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Left tail hat{{xi}} '
                          f'({xi_hats[0]:.3f}) exceeds '
                          f'{xi_hat_threshold}!\n')
      
      # Check empirical variance in each Markov chain
      var = welford_summary(fs)[1]
      if var < 1e-10:
        no_zvar_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Expectand exhibits '
                          'vanishing empirical variance!\n')
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(samples)
    
    if math.isnan(rhat):
      local_message += '  Split hat{R} is ill-defined!\n'
    elif rhat > 1.1:
      no_rhat_warning = False
      local_warning = True
      local_message += f'  Split hat{{R}} ({rhat:.3f}) exceeds 1.1!\n'
    
    for c in range(C):
      # Check empirical effective sample size
      fs = samples[c,:]
      
      tau_hat = compute_tau_hat(fs)
      eess = S / tau_hat
      if eess < min_eess_per_chain:
        no_eess_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: hat{{ESS}} ({eess:.1f}) '
                          'is smaller than desired '
                          f'({min_eess_per_chain:.0f})!\n')
    
    if local_warning:
      message += local_message + '\n'
  
  if (    no_xi_hat_warning and no_zvar_warning
      and no_rhat_warning   and no_eess_warning):
    desc = ('All expectands checked appear to be behaving well enough '
            'for reliable Markov chain Monte Carlo estimation.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
    return
  
  print(message)
  
  if not no_xi_hat_warning:
    desc = ('Large tail hat{xi}s suggest that the expectand '
            'might not be sufficiently integrable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  if not no_zvar_warning:
    desc = ('If the expectands are not constant then zero empirical '
            'variance suggests that the Markov transitions may be '
            'misbehaving.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  if not no_rhat_warning:
    desc = ('Split Rhat larger than 1.1 suggests that at least one of '
            'the Markov chains has not reached an equilibrium.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  if not no_eess_warning:
    desc = ('Small empirical effective sample sizes indicate strong '
            'empirical autocorrelations in the realized Markov chains. '
            'If the empirical effective sample size is too '
            'small then Markov chain Monte Carlo estimation '
            'may be unreliable even when a central limit '
            'theorem holds.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
    
  return

# Summary all expectand-specific diagnostics.
# @param expectand_samples A dictionary of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
# @param min_eess_per_chain The minimum empirical effective sample size
#                           before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @param max_width Maximum line width for printing
def summarize_expectand_diagnostics(expectand_samples,
                                    min_eess_per_chain=100,
                                    exclude_zvar=False,
                                    max_width=72):
  """Summarize expectand diagnostics"""
  if type(expectand_samples) is not dict:
    print(('Input variable `expectand_samples` '
           'is not a standard dictionary!'))
    return
  
  failed_names = []
  failed_xi_hat_names = []
  failed_zvar_names = []
  failed_rhat_names = []
  failed_eess_names = []
  
  for name in expectand_samples:
    samples = expectand_samples[name]
    C = samples.shape[0]
    S = samples.shape[1]
    
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(samples[c,:])[1]
        if var < 1e-10:
          any_zvar = True
      if any_zvar:
        continue
    
    for c in range(C):
      fs = samples[c,:]
            
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(fs)
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) or math.isnan(xi_hats[1]):
        failed_names.append(name)
        failed_xi_hat_names.append(name)
      if xi_hats[0] >= xi_hat_threshold or xi_hats[1] >= xi_hat_threshold:
        failed_names.append(name)
        failed_xi_hat_names.append(name)
    
      # Check empirical variance in each Markov chain
      var = welford_summary(fs)[1]
      if var < 1e-10:
        failed_names.append(name)
        failed_zvar_names.append(name)
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(samples)
    
    if math.isnan(rhat):
      failed_names.append(name)
      failed_rhat_names.append(name)
    elif rhat > 1.1:
      failed_names.append(name)
      failed_rhat_names.append(name)
    
    for c in range(C):
      # Check empirical effective sample size
      tau_hat = compute_tau_hat(samples[c,:])
      eess = S / tau_hat
      
      if eess < min_eess_per_chain:
        failed_names.append(name)
        failed_eess_names.append(name)
  
  failed_names = list(numpy.unique(failed_names))
  if len(failed_names):
    desc = (f'The expectands {", ".join(failed_names)} '
             'triggered diagnostic warnings.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  else:
    desc = ('All expectands checked appear to be behaving well enough '
            'for reliable Markov chain Monte Carlo estimation.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  failed_xi_hat_names = list(numpy.unique(failed_xi_hat_names))
  if len(failed_xi_hat_names):
    desc = (f'The expectands {", ".join(failed_xi_hat_names)} '
             'triggered tail hat{xi} warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))
    
    desc = ('  Large tail hat{k}s suggest that the expectand '
            'might not be sufficiently integrable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  failed_zvar_names = list(numpy.unique(failed_zvar_names))
  if len(failed_zvar_names):
    desc = (f'The expectands {", ".join(failed_zvar_names)} '
             'triggered zero variance warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))
    
    desc = ('  If the expectands are not constant then zero empirical'
            ' variance suggests that the Markov'
            ' transitions may be misbehaving.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
      
  failed_rhat_names = list(numpy.unique(failed_rhat_names))
  if len(failed_rhat_names):
    desc = (f'The expectands {", ".join(failed_rhat_names)} '
             'triggered hat{R} warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))
    
    desc = ('  Split Rhat larger than 1.1 suggests that at '
            'least one of the Markov chains has not reached '
            'an equilibrium.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
   
  failed_eess_names = list(numpy.unique(failed_eess_names))
  if len(failed_eess_names):
    desc = (f'The expectands {", ".join(failed_rhat_names)} '
             'triggered hat{ESS} warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))
    
    desc = ('Small empirical effective sample sizes indicate strong '
            'empirical autocorrelations in the realized Markov chains. '
            'If the empirical effective sample size is too '
            'small then Markov chain Monte Carlo estimation '
            'may be unreliable even when a central limit '
            'theorem holds.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Summarize Hamiltonian Monte Carlo and expectand diagnostics
# into a binary encoding
# @param expectand_samples A named list of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
# @param diagnostics A named list of two-dimensional arrays for 
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param adapt_target Target acceptance proxy statistic for step size 
#                     adaptation.
# @param max_treedepth The maximum numerical trajectory treedepth
# @param min_eess_per_chain The minimum empirical effective sample size
#                           before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @return warning_code An eight bit binary summary of the diagnostic 
#                      output.
def summarize_all_diagnostics(expectand_diagnostics,
                              diagnostics,
                              adapt_target=0.801,
                              max_treedepth=10,
                              min_eess_per_chain=100,
                              exclude_zvar=False):
  """Summarize Hamiltonian Monte Carlo and expectand diagnostics
     into a binary encoding"""
  
  warning_code = 0
  
  # Check divergences
  if sum(diagnostics['divergent__'].flatten()) > 0: 
    no_warning = False
    warning_code = warning_code | (1 << 0)
  
  # Check transitions that ended prematurely due to maximum tree depth limit
  if sum([ 1 for td in diagnostics['treedepth__'].flatten() 
           if td == max_treedepth ]) > 0:
    warning_code = warning_code | (1 << 1)
  
  C = diagnostics['energy__'].shape[0]
  no_efmi_warning = True
  no_accept_warning = True
  
  for c in range(C):
  # Checks the energy fraction of missing information (E-FMI)
    energies = diagnostics['energy__'][c,:]
    numer = sum((energies[i] - energies[i - 1])**2 for 
                i in range(1, len(energies))) / len(energies)
    denom = numpy.var(energies)
    if numer / denom < 0.2:
      no_efmi_warning = False
      
    # Check convergence of the stepsize adaptation
    ave_accept_proxy = numpy.mean(diagnostics['accept_stat__'][c,:])
    if ave_accept_proxy < 0.9 * adapt_target:
      no_accept_warning = False
  
  if not no_efmi_warning:
    warning_code = warning_code | (1 << 2)
  
  if not no_accept_warning:
    warning_code = warning_code | (1 << 3)
  
  zvar_warning = False
  xi_hat_warning = False
  rhat_warning = False
  eess_warning = False
  
  for name in expectand_samples:
    samples = expectand_samples[name]
    C = samples.shape[0]
    S = samples.shape[1]
    
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(samples[c,])[1]
        if var < 1e-10:
          any_zvar = True
      if any_zvar:
        continue
    
    for c in range(C):
      fs = samples[c,:]
       
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(fs)
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) or math.isnan(xi_hats[1]):
        xi_hat_warning = True
      elif (   xi_hats[0] >= xi_hat_threshold 
            or xi_hats[1] >= xi_hat_threshold):
        xi_hat_warning = True
    
      # Check empirical variance in each Markov chain
      var = welford_summary(fs)[1]
      if var < 1e-10:
        zvar_warning = True
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(samples)
    
    if math.isnan(rhat):
      rhat_warning = True
    elif rhat > 1.1:
      rhat_warning = True
    
    for c in range(C):
       # Check empirical effective sample size
      tau_hat = compute_tau_hat(samples[c,:])
      eess = S / tau_hat
      if neff < min_eess_per_chain:
        eess_warning = True
  
  if xi_hat_warning:
    warning_code = warning_code | (1 << 4)
  if zvar_warning:
    warning_code = warning_code | (1 << 5)
  if rhat_warning:
    warning_code = warning_code | (1 << 6)  
  if eess_warning:
    warning_code = warning_code | (1 << 7)
  
  return warning_code

# Translate binary diagnostic codes to human readable output.
# @params warning_code An eight bit binary summary of the diagnostic 
#                      output.
def parse_warning_code(warning_code):
    """Parses warning code into individual failures"""
    if warning_code & (1 << 0):
        print("  divergence warning")
    if warning_code & (1 << 1):
        print("  treedepth warning")
    if warning_code & (1 << 2):
        print("  E_FMI warning")
    if warning_code & (1 << 3):
        print("  average acceptance proxy warning")
    if warning_code & (1 << 4):
        print("  xi_hat warning")
    if warning_code & (1 << 5):
        print("  zero variance warning")
    if warning_code & (1 << 6):
        print("  Rhat warning")
    if warning_code & (1 << 7):
        print("  min empirical effective sample size warning")

# Compute empirical autocorrelations for a given Markov chain sequence
# @parmas fs A one-dimensional array of sequential expectand values.
# @return A one-dimensional array of empirical autocorrelations at each 
#         lag up to the length of the sequence.
def compute_rhos(fs):
  """Visualize empirical autocorrelations for a given sequence"""
  # Compute empirical autocorrelations
  N = len(fs)
  m, v = welford_summary(fs)
  zs = [ f - m for f in fs ]
  
  if v < 1e-10:
    return [1] * N
  
  B = 2**math.ceil(math.log2(N)) # Next power of 2 after N
  zs_buff = zs + [0] * (B - N)
  
  Fs = numpy.fft.fft(zs_buff)
  Ss = numpy.abs(Fs)**2
  Rs = numpy.fft.ifft(Ss)
  
  acov_buff = numpy.real(Rs)
  rhos = acov_buff[0:N] / acov_buff[0]
  
  # Drop last lag if (L + 1) is odd so that the lag pairs are complete
  L = N
  if (L + 1) % 2 == 1:
    L = L - 1
  
  # Number of lag pairs
  P = (L + 1) // 2
  
  # Construct asymptotic correlation from initial monotone sequence
  old_pair_sum = rhos[1] + rhos[2]
  max_L = N
  
  for p in range(1, P):
    current_pair_sum = rhos[2 * p] + rhos[2 * p + 1]
    
    if current_pair_sum < 0:
      max_L = 2 * p
      rhos[max_L:N] = [0] * (N - max_L)
      break
    
    if current_pair_sum > old_pair_sum:
      current_pair_sum = old_pair_sum
      rhos[2 * p]     = 0.5 * old_pair_sum
      rhos[2 * p + 1] = 0.5 * old_pair_sum
    
    # if p == P:
      # throw some kind of error when autocorrelation
      # sequence doesn't get terminated
    
    old_pair_sum = current_pair_sum
  
  return rhos

# Plot empirical correlograms for a given expectand across a Markov 
# chain ensemble.
# @ax Matplotlib axis object
# @param fs A two-dimensional array of scalar Markov chain states 
#           with the first dimension indexing the Markov chains and 
#           the second dimension indexing the sequential states 
#           within each Markov chain.
# @param max_L Maximum autocorrelation lag
# @param rho_lim Plotting range of autocorrelation values
# @display_name Name of expectand
def plot_empirical_correlogram(ax,
                               fs,
                               max_L,
                               rho_lim=[-0.2, 1.1],
                               name=""):
  """Plot empirical correlograms for the expectand output ensembels in a
     collection of Markov chains"""
  if len(fs.shape) != 2:
    print('Input variable `fs` is not a two-dimensional array!')
    return
  C = fs.shape[0]
  
  idxs = [ idx for idx in range(max_L) for r in range(2) ]
  xs = [ idx + delta for idx in range(max_L) for delta in [-0.5, 0.5]]
  
  colors = [dark, dark_highlight, mid, light_highlight]
  
  for c in range(C):
    rhos = compute_rhos(fs[c,:])
    pad_rhos = [ rhos[idx] for idx in idxs ]
    ax.plot(xs, pad_rhos, colors[c % 4], linewidth=2)
  
  ax.axhline(y=0, linewidth=2, color="#DDDDDD")
  
  ax.set_title(name)
  ax.set_xlabel("Lag")
  ax.set_xlim(-0.5, max_L + 0.5)
  ax.set_ylabel("Empirical\nAutocorrelation")
  ax.set_ylim(rho_lim[0], rho_lim[1])
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

# Visualize the projection of a Markov chain ensemble along two 
# expectands as a pairs plot.  Point colors darken along each Markov 
# chain to visualize the autocorrelation.
# @param f1s A two-dimensional array of expectand values with the first 
#            dimension indexing the Markov chains and the second 
#            dimension indexing the sequential states  within each 
#            Markov chain.
# @params display_name1 Name of first expectand
# @param f2s A two-dimensional array of expectand values with the first 
#            dimension indexing the Markov chains and the second 
#            dimension indexing the sequential states  within each 
#            Markov chain.
# @params display_name2 Name of second expectand
def plot_pairs_by_chain(f1s, display_name1,
                        f2s, display_name2):
  """Plot two expectand output ensembles againt each other separated by
     Markov chain """
  if len(f1s.shape) != 2:
    print('Input variable `f1s` is not a two-dimensional array!')
    return
  C1 = f1s.shape[0]
  S1 = f1s.shape[1]
  
  if len(f2s.shape) != 2:
    print('Input variable `f2s` is not a two-dimensional array!')
    return
  C2 = f2s.shape[0]
  S2 = f2s.shape[1]
    
  if C1 != C2:
    C = min(C1, C2)
    C1 = C
    C2 = C
    print(f'Plotting only {C} Markov chains.')
  
  if S1 != S2:
    S = min(S1, S2)
    S1 = S
    S2 = S
    print(f'Plotting only {S} samples per Markov chain.')
  
  colors = ["#DCBCBC", "#C79999", "#B97C7C",
            "#A25050", "#8F2727", "#7C0000"]
  cmap = LinearSegmentedColormap.from_list("reds", colors, N=S1)
  
  min_x = min(f1s.flatten())
  max_x = max(f1s.flatten())
  
  min_y = min(f2s.flatten())
  max_y = max(f2s.flatten())
  
  N_plots = C1
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0
  
  for c in range(C1):
    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1
    
    axarr[idx1, idx2].scatter(f1s.flatten(), f2s.flatten(),
                              color="#DDDDDD", s=5, zorder=3)
    axarr[idx1, idx2].scatter(f1s[c,:], f2s[c,:],
                              cmap=cmap, c=range(S1), s=5, zorder=4)
    
    axarr[idx1, idx2].set_title(f'Chain {c + 1}')
    axarr[idx1, idx2].set_xlabel(display_name1)
    axarr[idx1, idx2].set_xlim([min_x, max_x])
    axarr[idx1, idx2].set_ylabel(display_name2)
    axarr[idx1, idx2].set_ylim([min_y, max_y])
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)
  
  plot.show()

# Evaluate an expectand at the states of a Markov chain ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param expectand Scalar function to be applied to the Markov chain 
#                  states.
# @return A two-dimensional array of expectand values with the 
#         first dimension indexing the Markov chains and the 
#         second dimension indexing the sequential states within 
#         each Markov chain.
def pushforward_chains(samples, expectand):
  """Evaluate an expectand along a Markov chain"""
  return numpy.vectorize(expectand)(samples)

# Estimate expectand exectation value from a single Markov chain.
# @param fs A one-dimensional array of sequential expectand values.
# @return The Markov chain Monte Carlo estimate, its estimated standard 
#         error, and empirical effective sample size.
def mcmc_est(fs):
  """Estimate expectand expectation value from a Markov chain"""
  S = len(fs)
  if S == 1:
    return [fs[0], 0, math.nan]
  
  summary = welford_summary(fs)
  
  if summary[1] == 0:
    return [summary[0], 0, math.nan]
  
  tau_hat = compute_tau_hat(fs)
  eess = S / tau_hat
  return [summary[0], math.sqrt(summary[1] / eess), eess]

# Estimate expectand exectation value from a Markov chain ensemble.
# @param samples A two-dimensional array of expectand values with the 
#                first dimension indexing the Markov chains and the 
#                second dimension indexing the sequential states within 
#                each Markov chain.
# @return The ensemble Markov chain Monte Carlo estimate, its estimated
#         standard error, and empirical effective sample size.
def ensemble_mcmc_est(samples):
  """Estimate expectand exectation value from a collection of 
     Markov chains"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return [math.nan, math.nan, math.nan]
    
  C = samples.shape[0]
  chain_ests = [ mcmc_est(samples[c,:]) for c in range(C) ]
  
  # Total effective sample size
  total_ess = sum([ est[2] for est in chain_ests ])
  
  if math.isnan(total_ess):
    m  = numpy.mean([ est[0] for est in chain_ests ])
    se = numpy.mean([ est[1] for est in chain_ests ])
    return [m, se, math.nan]
  
  # Ensemble average weighted by effective sample size
  mean = sum([ est[0] * est[2] for est in chain_ests ]) / total_ess
  
  # Ensemble variance weighed by effective sample size
  # including correction for the fact that individual Markov chain
  # variances are defined relative to the individual mean estimators
  # and not the ensemble mean estimator
  vars = [0] * C
  
  for c in range(C):
    est = chain_ests[c]
    chain_var = est[2] * est[1]**2
    var_update = (est[0] - mean)**2
    vars[c] = est[2] * (var_update + chain_var)
  var = sum(vars) / total_ess
  
  return [mean, math.sqrt(var / total_ess), total_ess]

# Visualize pushforward distribution of a given expectand as a 
# histogram, using Markov chain Monte Carlo estimators to estimate the 
# output bin probabilities.  Bin probability estimator error is shown 
# in gray.
# @ax Matplotlib axis object
# @param samples A two-dimensional array of expectand values with the 
#                first dimension indexing the Markov chains and the 
#                second dimension indexing the sequential states within 
#                each Markov chain.
# @param B The number of histogram bins
# @param display_name Exectand name
# @param flim Optional histogram range
# @param baseline Optional baseline value for visual comparison
def plot_expectand_pushforward(ax, samples, B, display_name="f", 
                               flim=None, baseline=None):
  """Plot pushforward histogram of a given expectand using Markov chain
     Monte Carlo estimators to estimate the output bin probabilities"""
  if len(samples.shape) != 2:
    print('Input variable `samples` is not a two-dimensional array!')
    return
    
  if flim is None:
    # Automatically adjust histogram binning to range of outputs
    min_f = min(samples.flatten())
    max_f = max(samples.flatten())
    
    # Add bounding bins
    delta = (max_f - min_f) / B
    min_f = min_f - delta
    max_f = max_f + delta
    flim = [min_f, max_f]
    
    bins = numpy.arange(min_f, max_f + delta, delta)
    B = B + 2
  else:
    delta = (flim[1] - flim[0]) / B
    bins = numpy.arange(flim[0], flim[1] + delta, delta)
  
  mean_p = [0] * B
  delta_p = [0] * B
  
  for b in range(B):
    def bin_indicator(x):
      return 1.0 if bins[b] <= x and x < bins[b + 1] else 0.0
    
    indicator_samples = pushforward_chains(samples, bin_indicator)
    est = ensemble_mcmc_est(indicator_samples)
    
    # Normalize bin probabilities by bin width to allow
    # for direct comparison to probability density functions
    width = bins[b + 1] - bins[b]
    mean_p[b] = est[0] / width
    delta_p[b] = est[1] / width
  
  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[b + o] for b in range(B) for o in range(2) ]
  
  lower_inter = [ max(mean_p[idx] - 2 * delta_p[idx], 0)
                  for idx in idxs ]
  upper_inter = [ min(mean_p[idx] + 2 * delta_p[idx], 1 / width) 
                  for idx in idxs ]
  
  min_y =        min(lower_inter)
  max_y = 1.05 * max(upper_inter)
  
  ax.fill_between(xs, lower_inter, upper_inter,
                  facecolor=light, color="#DDDDDD")
  ax.plot(xs, [ mean_p[idx] for idx in idxs ], color=dark, linewidth=2)
  
  if baseline is not None:
    ax.axvline(x=baseline, linewidth=4, color="white")
    ax.axvline(x=baseline, linewidth=2, color="black")
  
  ax.set_xlim(flim)
  ax.set_xlabel(display_name)
  ax.set_ylim([min_y, max_y])
  ax.set_ylabel("Estimated Bin\nProbabilities / Bin Width")
  ax.get_yaxis().set_visible(False)
  ax.spines["top"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.spines["right"].set_visible(False)

