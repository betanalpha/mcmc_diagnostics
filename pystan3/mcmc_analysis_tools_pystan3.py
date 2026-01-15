################################################################################
#
# The code is copyright 2024 Michael Betancourt and licensed under the
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

import numpy
import math
import textwrap
import inspect
import re

import stan

light = "#DCBCBC"
light_highlight = "#C79999"
mid = "#B97C7C"
mid_highlight = "#A25050"
dark = "#8F2727"
dark_highlight = "#7C0000"

light_teal = "#6B8E8E"
mid_teal = "#487575"
dark_teal = "#1D4F4F"

# Extract unpermuted expectand values from a StanFit object and format
# them for convenient access
# @param stan_fit A StanFit object
# @return A dictionary of two-dimensional arrays for each expectand in
#         the StanFit object.  The first dimension of each element
#         indexes the Markov chains and the second dimension indexes the
#         sequential states within each Markov chain.
def extract_expectand_vals(stan_fit):
  nom_params = stan_fit._draws
  offset = len(stan_fit.sample_and_sampler_param_names)
  
  base_names = stan_fit.constrained_param_names
  formatted_names = []
  for base_name in base_names:
    name = re.sub('\.', '[', base_name, count=1)
    name = re.sub('\.', ',', name)
    if '[' in name:
      name += ']'
    formatted_names.append(name)

  params = { name: numpy.transpose(nom_params[k + offset,:,:])
             for k, name in enumerate(formatted_names) }
  return params

# Validate two-dimensional array structure of input object.
# @param obj Object to be validated
# @param name Object name
def validate_array(obj, name):
  if not isinstance(obj, numpy.ndarray):
    raise TypeError(f'Input variable {name} is not a numpy array.')
  if obj.dtype != numpy.dtype('float64') or len(obj.shape) != 2:
    raise TypeError(f'Input variable {name} is not a two-dimensional '
                     'numpy array of doubles.')

# Validate dictionary of two-dimensional array structure of input
# object.
# @param obj Object to be validated
# @param name Object name
def validate_dict_of_arrays(obj, name):
  if not isinstance(obj, dict):
    raise TypeError(f'Input variable {name} is not a dictionary.')

  if not all([ isinstance(v, numpy.ndarray)
               for k, v in obj.items() ]):
    raise TypeError(f'The elements of input variable {name} '
                      'are not all numpy arrays.')

  if not all([ v.dtype == numpy.dtype('float64') and len(v.shape) == 2
               for k, v in obj.items() ]):
    raise TypeError(f'The elements of input variable {name} '
                      'are not all two-dimensional double arrays.')

  dims = ([ v.shape for k, v in obj.items() ])
  if (   [ d[0] for d in dims ].count(dims[0][0]) != len(dims)
      or [ d[1] for d in dims ].count(dims[0][1]) != len(dims)):
    raise TypeError(f'The elements of input variable {name} '
                      'do not have consistent sizes.')

# Extract Hamiltonian Monte Carlo diagnostics values from a StanFit
# object and format them for convenient access
# @param stan_fit A StanFit object
# @return A dictionary of two-dimensional arrays for each expectand in
#         the StanFit object.  The first dimension of each element
#         indexes the Markov chains and the second dimension indexes the
#         sequential states within each Markov chain.
def extract_hmc_diagnostics(stan_fit):
  d_names = ['divergent__', 'treedepth__', 'n_leapfrog__',
             'stepsize__', 'energy__', 'accept_stat__' ]
  for dn in d_names:
    if dn not in stan_fit.sample_and_sampler_param_names:
      print(f'Diagnostic variable {dn} not found in stan_fit.')
      return

  d_idxs = [ idx for dn in d_names
             for idx, sn
             in enumerate(stan_fit.sample_and_sampler_param_names)
             if sn == dn ]
  
  params = { name: numpy.transpose(stan_fit._draws[idx,:,:])
           for idx, name in zip(d_idxs, d_names) }
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
  """Check all Hamiltonian Monte Carlo diagnostics for an
     ensemble of Markov chains"""
  validate_dict_of_arrays(diagnostics, 'diagnostics')

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
                            f'transitions ({n_tds / S:.3%})')
      local_messages.append(f'           saturated the maximum '
                            f'treedepth of {max_treedepth}.')
    
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
      local_messages.append(f'  Chain {c + 1}: Average proxy acceptance '
                            f'statistic ({ave_accept_proxy:.3f})')
      local_messages.append(f'           is smaller than 90% of the '
                            f'target ({adapt_target:.3f}).')
    
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
  
  C = len(stan_fit.stan_outputs)
  stepsize_header = b'["Adaptation terminated"]}\n'
  inv_metric_header = b'["Diagonal elements of inverse mass matrix:"]}\n'
  
  stepsizes = []
  inv_metric_elems = []
  
  for c in range(C):
    first_split = stan_fit.stan_outputs[c].partition(stepsize_header)[2]
    stepsize_info = first_split.partition(b'\n')[0]
    stepsizes.append(float(eval(stepsize_info)['values'][0].split(' = ')[1]))

    first_split = stan_fit.stan_outputs[c].partition(inv_metric_header)[2]
    inv_metric_str = first_split.partition(b'\n')[0]
    inv_metric_dict = eval(inv_metric_str)['values'][0].split(', ')
    inv_metric_elems.append([ float(x) for x in inv_metric_dict ])

  min_elem = min([ min(a) for a in inv_metric_elems ])
  max_elem = max([ max(a) for a in inv_metric_elems ])
  
  delta = (max_elem - min_elem) / B
  min_elem = min_elem - delta
  max_elem = max_elem + delta
  bins = numpy.arange(min_elem, max_elem + delta, delta)
  B = B + 2
  
  max_y = max([ max(numpy.histogram(a, bins=bins)[0])
                for a in inv_metric_elems ])
  
  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[idx + delta] for idx in range(B) for delta in [0, 1]]
  
  N_plots = C
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0
  
  sci_formatter = matplotlib.ticker.FuncFormatter(lambda x,
                                                  lim: f'{x:.1e}')
  
  for c in range(C):
    counts = numpy.histogram(inv_metric_elems[c], bins=bins)[0]
    ys = counts[idxs]
  
    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1
  
    axarr[idx1, idx2].plot(xs, ys, dark)
    axarr[idx1, idx2].set_title(f'Chain {c + 1}\n(Stepsize'
                                f' = {stepsizes[c]:.3e})')
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
  validate_dict_of_arrays(diagnostics, 'diagnostics')

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
  """Display symplectic integrator trajectory lengths"""
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  lengths = diagnostics['n_leapfrog__']
  
  C = lengths.shape[0]
  colors = [dark_highlight, dark, mid_highlight, mid, light_highlight]
  
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
  validate_dict_of_arrays(diagnostics, 'diagnostics')

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
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  lengths = diagnostics['n_leapfrog__']
  C = lengths.shape[0]
  eps = [ diagnostics['stepsize__'][c][0] for c in range(C) ]

  if tlim is None:
    # Automatically adjust histogram binning to range of outputs
    min_t = max([ eps[c] * max(lengths[c]) for c in range(C) ])
    max_t = max([ eps[c] * max(lengths[c]) for c in range(C) ])

    tlim = [min_t, max_t]
    delta = (tlim[1] - tlim[0]) / B
    bins = numpy.arange(tlim[0] - delta, tlim[1] + delta, delta)
    B = B + 2
  else:
    delta = (tlim[1] - tlim[0]) / B
    bins = numpy.arange(tlim[0], tlim[1] + delta, delta)

  colors = [dark_highlight, dark, mid_highlight, mid, light_highlight]

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
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  lengths = diagnostics['n_leapfrog__']
  C = lengths.shape[0]
  eps = [ diagnostics['stepsize__'][c][0] for c in range(C) ]
  
  if tlim is None:
    # Automatically adjust histogram binning to range of outputs
    min_t = max([ eps[c] * max(lengths[c]) for c in range(C) ])
    max_t = max([ eps[c] * max(lengths[c]) for c in range(C) ])
    
    tlim = [min_t, max_t]
    delta = (tlim[1] - tlim[0]) / B
    bins = numpy.arange(tlim[0] - delta, tlim[1] + delta, delta)
    B = B + 2
  else:
    delta = (tlim[1] - tlim[0]) / B
    bins = numpy.arange(tlim[0], tlim[1] + delta, delta)
  
  colors = [dark_highlight, dark, mid_highlight, mid, light_highlight]
  
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
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  proxy_stats = diagnostics['accept_stat__']
  C = proxy_stats.shape[0]
  
  for c in range(C):
    proxy_stat = numpy.mean(proxy_stats[c,:])
    print(  f'Chain {c + 1}: Average proxy acceptance '
          + f'statistic = {proxy_stat:.3f}')

# Apply transformation identity, log, or logit transformation to
# named values and flatten the output.  Transformation defaults to
# identity if name is not included in `transforms` dictionary.  A
# ValueError is thrown if values are not properly constrained.
# @param name Expectand name.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param transforms A dictionary with expectand names for keys and
#                   transformation flags for values.
# @return The transformed expectand name and a one-dimensional array of
#         flattened transformation outputs.
def apply_transform(name, expectand_vals_dict, transforms):
  t = transforms.get(name, 0)
  transformed_name = ""
  transformed_vals = 0
  if t == 0:
    transformed_name = name
    transformed_vals = expectand_vals_dict[name].flatten()
  elif t == 1:
    if numpy.amin(expectand_vals_dict[name]) <= 0:
      raise ValueError( 'Log transform requested for expectand '
                       f'{name} but expectand values are not strictly ' 
                        'positive.')
    transformed_name = f'log({name})'
    transformed_vals = [ math.log(x) for x in
                         expectand_vals_dict[name].flatten() ]
  elif t == 2:
    if (   numpy.amin(expectand_vals_dict[name]) <= 0
        or numpy.amax(expectand_vals_dict[name]) >= 1):
      raise ValueError( 'Logit transform requested for expectand '
                       f'{name} but expectand values are not strictly '
                        'confined to the unit interval.')
    transformed_name = f'logit({name})'
    transformed_vals = [ math.log(x / (1 - x)) for x in
                         expectand_vals_dict[name].flatten() ]
  return transformed_name, transformed_vals

# Plot pairwise scatter plots with non-divergent and divergent
# transitions separated by color
# @param x_names A list of expectand names to be plotted on the x axis.
# @param y_names A list of expectand names to be plotted on the y axis.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param diagnostics A dictionary of two-dimensional arrays for
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
def plot_div_pairs(x_names, y_names, expectand_vals_dict,
                   diagnostics, transforms={},
                   xlim=None, ylim=None, 
                   plot_mode=0, max_width=72):
  """Plot pairwise scatter plots with non-divergent and divergent 
     transitions separated by color"""
  if not isinstance(x_names, list):
    raise TypeError(('Input variable `x_names` is not a list.'))

  if not isinstance(y_names, list):
    raise TypeError(('Input variable `y_names` is not a list.'))
    
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  if not isinstance(transforms, dict):
    raise TypeError('Input variable `transforms` is not a dictionary.')
  
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
    
  # Transform expectand values
  transformed_vals = {}
  
  transformed_x_names = []
  for name in x_names:
    try: 
      t_name, t_vals = apply_transform(name,
                                       expectand_vals_dict,
                                       transforms)
    except ValueError as error:
      desc = textwrap.wrap(error, max_width)
      print('\n'.join(desc))
      return
    
    transformed_x_names.append(t_name)
    if t_name not in transformed_vals:
      transformed_vals[t_name] = t_vals
      
  transformed_y_names = []
  for name in y_names:
    try: 
      t_name, t_vals = apply_transform(name,
                                       expectand_vals_dict,
                                       transforms)
    except ValueError as error:
      desc = textwrap.wrap(error, max_width)
      print('\n'.join(desc))
    
    transformed_y_names.append(t_name)
    if t_name not in transformed_vals:
      transformed_vals[t_name] = t_vals
      
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
    if sum(divergences) > 0:
      div_nlfs = [ x for x, d in
                   zip(diagnostics['n_leapfrog__'].flatten(),
                       divergences)
                   if d == 1  ]
      max_nlf = max(div_nlfs)
      nom_colors = [light_teal, mid_teal, dark_teal]
      cmap = LinearSegmentedColormap.from_list("teals", nom_colors,
                                               N=max_nlf)
    else:
      div_nlfs = []
      nom_colors = [light_teal, mid_teal, dark_teal]
      cmap = LinearSegmentedColormap.from_list("teals", nom_colors,
                                               N=1)
  
  # Set plot layout dynamically
  N_pairs = len(pairs)
  
  if N_pairs == 1:
    N_cols = 1
    N_rows = 1
  else:
    N_cols = 3
    N_rows = math.ceil(N_pairs / N_cols)
    
  if N_rows > 3:
    N_rows = 3
    
  # Plot
  k = 0
  
  for pair in pairs:
    if k == 0:
      f, axarr = plot.subplots(N_rows, N_cols, layout="constrained",
                               squeeze=False)
      
    x_name = pair[0]
    x_nondiv_vals = [ x for x, d in
                      zip(transformed_vals[x_name], divergences)
                      if d == 0  ]
    x_div_vals    = [ x for x, d in
                      zip(transformed_vals[x_name], divergences)
                      if d == 1  ]
    
    if xlim is None:
      xmin = min(numpy.concatenate((x_nondiv_vals, x_div_vals)))
      xmax = max(numpy.concatenate((x_nondiv_vals, x_div_vals)))
      local_xlim = [xmin, xmax]
    else:
      local_xlim = xlim
    
    y_name = pair[1]
    y_nondiv_vals = [ y for y, d in
                      zip(transformed_vals[y_name], divergences)
                      if d == 0  ]
    y_div_vals    = [ y for y, d in
                      zip(transformed_vals[y_name], divergences)
                      if d == 1  ]
    
    if ylim is None:
      ymin = min(numpy.concatenate((y_nondiv_vals, y_div_vals)))
      ymax = max(numpy.concatenate((y_nondiv_vals, y_div_vals)))
      local_ylim = [ymin, ymax]
    else:
      local_ylim = ylim
     
    idx1 = k // N_cols
    idx2 = k % N_cols
    
    if plot_mode == 0:
      axarr[idx1, idx2].scatter(x_nondiv_vals, y_nondiv_vals, s=5,
                                color=dark_highlight, alpha=0.05)
      axarr[idx1, idx2].scatter(x_div_vals, y_div_vals, s=5,
                                color="#00FF00", alpha=0.25)
    elif plot_mode == 1:
      axarr[idx1, idx2].scatter(x_nondiv_vals, y_nondiv_vals,
                                s=5, color="#DDDDDD")
      if len(x_div_vals) > 0:
        axarr[idx1, idx2].scatter(x_div_vals, y_div_vals, s=5,
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
# @params vals A one-dimensional array of positive values.
# @return Shape parameter estimate.
def compute_xi_hat(vals):
  """Compute empirical Pareto shape configuration for a positive sample"""
  N = len(vals)
  sorted_vals = sorted(vals)
  
  if sorted_vals[0] == sorted_vals[-1]:
    return -2
  
  if (sorted_vals[0] < 0):
    print("Sequence values must be positive.")
    return math.nan
  
  # Estimate 25% quantile
  q = sorted_vals[math.floor(0.25 * N + 0.5)]
  if q == sorted_vals[0]:
    return -2
    
  # Heuristic Pareto configuration
  M = 20 + math.floor(math.sqrt(N))
  
  b_hat_vec = [None] * M
  log_w_vec = [None] * M
  
  for m in range(M):
    b_hat_vec[m] =   1 / sorted_vals[-1] \
                   + (1 - math.sqrt(M / (m + 0.5))) / (3 * q)
    if b_hat_vec[m] != 0:
      xi_hat = numpy.mean( [ math.log(1 - b_hat_vec[m] * f) 
                             for f in sorted_vals ] )
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

  return numpy.mean( [ math.log(1 - b_hat * f) for f in sorted_vals ] )

# Compute empirical generalized Pareto shape for upper and lower tails
# for an arbitrary sample of expectand values, ignoring any
# autocorrelation between the values.
# @param vals A one-dimensional array of expectand values.
# @return Left and right shape estimators.
def compute_tail_xi_hats(vals):
  """Compute empirical Pareto shape configuration for upper and lower tails"""
  v_center = numpy.median(vals)
  
  # Isolate lower and upper tails which can be adequately modeled by a 
  # generalized Pareto shape for sufficiently well-behaved distributions
  vals_left = [ math.fabs(v - v_center) for v in vals if v <= v_center ]
  N = len(vals_left)
  M = int(min(0.2 * N, 3 * 3 * math.sqrt(N)))
  vals_left = vals_left[M:N]
  
  vals_right = [ v - v_center for v in vals if v > v_center ]
  N = len(vals_right)
  M = int(min(0.2 * N, 3 * 3 * math.sqrt(N)))
  vals_right = vals_right[M:N]
  
  # Default to NaN if left tail is ill-defined
  xi_hat_left = math.nan
  if len(vals_left) > 40:
    xi_hat_left = compute_xi_hat(vals_left)
  
  # Default to NaN if right tail is ill-defined
  xi_hat_right = math.nan
  if len(vals_right) > 40:
    xi_hat_right = compute_xi_hat(vals_right)
    
  return [xi_hat_left, xi_hat_right]

# Check upper and lower tail behavior of a given expectand output
# ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
def check_tail_xi_hats(expectand_vals, max_width=72):
  """Check empirical Pareto shape configuration for upper and lower 
     tails of a given expectand output ensemble"""
  validate_array(expectand_vals, 'expectand_vals')
  
  C = expectand_vals.shape[0]
  no_warning = True
  
  for c in range(C):
    xi_hats = compute_tail_xi_hats(expectand_vals[c,:])
    xi_hat_threshold = 0.25
    if math.isnan(xi_hats[0]) and math.isnan(xi_hats[1]):
      no_warning = False
      print(f'  Chain {c + 1}: Both left and right tail '
            'hat{{xi}}s are Nan.\n')
    elif math.isnan(xi_hats[0]):
      no_warning = False
      print(f'  Chain {c + 1}: Left tail '
            'hat{{xi}} is Nan.\n')
    elif math.isnan(xi_hats[1]):
      no_warning = False
      print(f'  Chain {c + 1}: Right tail '
            'hat{{xi}} is Nan.\n')
    elif (    xi_hats[0] >= xi_hat_threshold 
         and xi_hats[1] >= xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Both left and right tail '
            f'hat{{xi}}s ({xi_hats[0]:.3f}, '
            f'{xi_hats[1]:.3f}) exceed '
            f'{xi_hat_threshold}.\n')
    elif (    xi_hats[0] < xi_hat_threshold 
          and xi_hats[1] >= xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Right tail hat{{xi}} '
            f'({xi_hats[1]:.3f}) exceeds '
            f'{xi_hat_threshold}.\n')
    elif (    xi_hats[0] >= xi_hat_threshold 
          and xi_hats[1] < xi_hat_threshold):
      no_warning = False
      print(f'  Chain {c + 1}: Left tail hat{{xi}} '
            f'({xi_hats[0]:.3f}) exceeds '
            f'{xi_hat_threshold}.\n')
  
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
# @params vals A one-dimensional array of sequential expectand values.
# @return The empirical mean and variance.
def welford_summary(vals):
  """Welford accumulator for empirical mean and variance of a
     given sequence"""
  mean = 0
  var = 0
  
  for n, v in enumerate(vals):
    delta = v - mean
    mean += delta / (n + 1)
    var += delta * (v - mean)
    
  var /= (len(vals) - 1)
  
  return [mean, var]

# Check expectand output ensemble for vanishing empirical variance.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
def check_variances(expectand_vals, max_width=72):
  """Check expectand output ensemble for vanishing empirical variance"""
  validate_array(expectand_vals, 'expectand_vals')
  
  C = expectand_vals.shape[0]
  no_warning = True
  
  for c in range(C):
    var = welford_summary(expectand_vals[c,:])[1]
    if var < 1e-10:
      no_warning = True
      print(f'  Chain {c + 1}: Expectand is constant.\n')

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
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @return Split Rhat estimate.
def compute_split_rhat(expectand_vals):
  """Compute split hat{R} for an expectand output ensemble across
     a collection of Markov chains"""
  validate_array(expectand_vals, 'expectand_vals')
  
  split_chain_vals = [ c for chain_vals in expectand_vals
                       for c in split_chain(chain_vals) ]
  N_chains = len(split_chain_vals)
  N = sum([ len(vals) for vals in split_chain_vals ])
  
  means = [None] * N_chains
  vars = [None] * N_chains
  
  for c, vals in enumerate(split_chain_vals):
    summary = welford_summary(vals)
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
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
def compute_split_rhats(expectand_vals_dict):
  """Compute split hat{R} for all expectand output ensembles across
     a collection of Markov chains"""
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')
    
  rhats = []
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    rhats.append(compute_split_rhat(expectand_vals))
  
  return rhats

# Check split hat{R} across a given expectand output ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
def check_rhat(expectand_vals, max_width=72):
  """Check split hat{R} for all expectand output ensembles across
     a collection of Markov chains"""
  validate_array(expectand_vals, 'expectand_vals')
    
  rhat = compute_split_rhat(expectand_vals)

  no_warning = True
  
  if math.isnan(rhat):
    print('All Markov chains appear to be frozen.')
  elif rhat > 1.1:
    print(f'Split hat{{R}} is {rhat:.3f}.')
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

# Plot a histogram of the split hat{R}s for all expectands in
# expectands_var_list.  Plot range and number of bins expands to
# include the split hat{R} threshold.
# @param ax Matplotlib axis object
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @params B The number of histogram bins
def plot_rhats(ax, expectand_vals_dict, B=25):
  validate_dict_of_arrays(expectand_vals_dict,
                          'expectand_vals_dict')

  N = len(expectand_vals_dict)
  rhats = [ numpy.nan ] * N

  n = 0
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    validate_array(expectand_vals, 'expectand_vals')

    rhat = compute_split_rhat(expectand_vals)
    if math.isfinite(rhat):
      rhats[n] = rhat
    n += 1

  val_min = numpy.nanmin(rhats)
  val_max = numpy.nanmax(rhats)

  plot_min = min(1.1, val_min)
  plot_max = max(1.1, val_max)
  B = B * int((plot_max - plot_min) / (val_max - val_min))

  delta = (plot_max - plot_min) / B
  plot_min = plot_min - delta
  plot_max = plot_max + delta
  bins = numpy.arange(plot_min, plot_max + delta, delta)
  B = B + 2

  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[idx + delta] for idx in range(B) for delta in [0, 1]]

  counts = numpy.histogram(rhats, bins=bins)[0]
  max_y = max(counts)
  ys = counts[idxs]

  ax.plot(xs, ys, dark, linewidth=1)
  ax.set_xlabel("Split hatRs")
  ax.set_xlim([plot_min, plot_max])
  ax.set_ylabel("")
  ax.get_yaxis().set_visible(False)
  ax.set_ylim([0, 1.05 * max_y])
  ax.spines["top"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.axvline(x=1.1, linewidth=2, linestyle="dashed", color="#DDDDDD")


# Compute empirical integrated autocorrelation time, \hat{tau}, for a
# sequence of expectand values.
# @param vals A one-dimensional array of expectand values.
# @return Left and right shape estimators.
def compute_tau_hat(vals):
  """Compute empirical integrated autocorrelation time for a sequence"""
  # Compute empirical autocorrelations
  N = len(vals)
  m = welford_summary(vals)[0]
  zs = [ val - m for val in vals ]
  
  B = 2**math.ceil(math.log2(N)) # Next power of 2 after N
  zs_buff = zs + [0] * (B - N)
  
  Fs = numpy.fft.fft(zs_buff)
  Ss = numpy.abs(Fs)**2
  Rs = numpy.fft.ifft(Ss)

  acov_buff = numpy.real(Rs)
  if acov_buff[0] == 0:
    return math.inf

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

    if p == P:
      return math.nan
    
    old_pair_sum = current_pair_sum


# Check the incremental empirical integrated autocorrelation time for
# all the given expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
def check_inc_tau_hat(expectand_vals, max_width=72):
  """Check that the incremental empirical integrated autocorrelation
     time of the given expectand values is sufficiently large."""
  validate_array(expectand_vals, 'expectand_vals')

  no_warning = True
  C = expectand_vals.shape[0]
  S = expectand_vals.shape[1]

  for c in range(C):
    tau_hat = compute_tau_hat(expectand_vals[c,:])

    if math.isinf(tau_hat):
      print(f'Chain {c + 1}: The calculation of hat{{tau}} is ',
             'unreliable because\nthe empirical variance ',
             'is zero.\n')
      no_warning = False
    elif math.isnan(tau_hat):
      print(f'Chain {c + 1}: The calculation of hat{{tau}} is ',
             'unreliable because\nthe empirical ',
             'autocorrelations did not decay sufficiently ',
             'quickly.\n')
      no_warning = False

    inc_tau_hat = tau_hat / S
    if inc_tau_hat > 5:
      print(f'Chain {c + 1}: Incremental hat{{tau}} '
            f'({inc_tau_hat :.3f}) is too large.\n')
      no_warning = False

  if no_warning:
    desc = ('The incremental empirical integrated ',
            'autocorrelation time is sufficiently well-behaved ',
            'for the empirical autocorrelation estimates to be ',
            'reliable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  else:
    desc = ('If the incremental empirical integrated ',
            'autocorrelation times are unreliable or too large ',
            'then the Markov chains have not explored long ',
            'enough for the autocorrelation estimates to be ',
            'reliable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Compute the minimum empirical effective sample size, or \hat{ESS},
# across the Markov chains for the given expectands.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
def compute_min_ess_hats(expectand_vals_dict):
  """Compute the minimum empirical integrated autocorrelation time
     across a collection of Markov chains for all expectand output
     ensembles"""
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')
      
  min_ess_hats = []
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    C = expectand_vals.shape[0]
    S = expectand_vals.shape[0]
    
    ess_hats = [None] * 4
    for c in range(C):
      tau_hat = compute_tau_hat(expectand_vals[c,:])
      ess_hats[c] = S / tau_hat
    
    min_ess_hats.append(min(ess_hats))
  
  return min_ess_hats

# Check the empirical effective sample size \hat{ESS} for the given
# expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param min_ess_hat_per_chain The minimum empirical effective sample
#                              size before a warning message is passed.
# @param max_width Maximum line width for printing
def check_ess_hat(expectand_vals,
                  min_ess_hat_per_chain=100,
                  max_width=72):
  """Check the empirical effective sample size for all expectand 
     output ensembles"""
  validate_array(expectand_vals, 'expectand_vals')
  
  no_warning = True
  C = expectand_vals.shape[0]
  S = expectand_vals.shape[1]
  
  for c in range(C):
    tau_hat = compute_tau_hat(expectand_vals[c,:])
    ess_hat = S / tau_hat
    if ess_hat < min_ess_hat_per_chain:
      print(f'Chain {c + 1}: The empirical effective sample size '
            f'{ess_hat :.1f} is too small.')
      no_warning = False
  
  if no_warning:
    desc = ('Assuming that a central limit theorem holds the '
            'empirical effective sample size is large enough '
            'for Markov chain Monte Carlo estimation to be'
            'reasonably precise.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  else:
    desc = ('Small empirical effective sample sizes result in '
            'imprecise Markov chain Monte Carlo estimators.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))


# Plot a histogram of the empirical effective sample sizes for all
# expectands in expectands_var_list, separated by Markov chain.
# Plot range and number of bins expands to include the min_ess_hat
# threshold.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param min_ess_hat The minimum empirical effective sample size
# @params B The number of histogram bins
def plot_ess_hats(expectand_vals_dict,
                  min_ess_hat=100, B=25):
  validate_dict_of_arrays(expectand_vals_dict,
                          'expectand_vals_dict')

  N = len(expectand_vals_dict)

  [name, ev] = next(iter(expectand_vals_dict.items()))
  validate_array(ev, f'expectand_vals_dict[{name}]')

  C = ev.shape[0]
  S = ev.shape[1]
  ess_hats = numpy.zeros((C, N))

  n = 0
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    validate_array(expectand_vals, 'expectand_vals')

    for c in range(C):
      tau_hat = compute_tau_hat(expectand_vals[c,:])
      ess_hats[c, n] = S / tau_hat
    n += 1

  val_min = numpy.nanmin(ess_hats)
  val_max = numpy.nanmax(ess_hats)

  plot_min = min(1.1, val_min)
  plot_max = max(1.1, val_max)
  B = B * int((plot_max - plot_min) / (val_max - val_min))

  delta = (plot_max - plot_min) / B
  plot_min = plot_min - delta
  plot_max = plot_max + delta
  bins = numpy.arange(plot_min, plot_max + delta, delta)
  B = B + 2

  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ bins[idx + delta] for idx in range(B) for delta in [0, 1]]

  max_y = max([ max(numpy.histogram(ess_hats[c,:], bins=bins)[0])
                for c in range(C) ])

  N_plots = C
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0

  for c in range(C):
    counts = numpy.histogram(ess_hats[c,:], bins=bins)[0]
    ys = counts[idxs]

    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1

    axarr[idx1, idx2].plot(xs, ys, dark)
    axarr[idx1, idx2].set_title(f'Chain {c + 1}')
    axarr[idx1, idx2].set_xlabel("Empirical Effective Sample Sizes")
    axarr[idx1, idx2].set_xlim([plot_min, plot_max])
    axarr[idx1, idx2].set_ylabel("")
    axarr[idx1, idx2].get_yaxis().set_visible(False)
    axarr[idx1, idx2].set_ylim([0, 1.05 * max_y])
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["left"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)

    axarr[idx1, idx2].axvline(x=min_ess_hat, linewidth=2,
                              linestyle="dashed",
                              color="#DDDDDD")

  plot.show()

# Check all expectand-specific diagnostics.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param min_ess_hat_per_chain The minimum empirical effective sample
#                              size before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @param max_width Maximum line width for printing
def check_all_expectand_diagnostics(expectand_vals_dict,
                                    min_ess_hat_per_chain=100,
                                    exclude_zvar=False,
                                    max_width=72):
  """Check all expectand diagnostics"""
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')
  
  no_xi_hat_warning = True 
  no_zvar_warning = True
  no_rhat_warning = True
  no_inc_tau_hat_warning = True
  no_ess_hat_warning = True
  
  message = ""
  
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    C = expectand_vals.shape[0]
    S = expectand_vals.shape[1]
    
    local_warning = False
    local_message = name + ':\n'
  
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(expectand_vals[c,:])[1]
        if var < 1e-10:
          any_zvar = True
      
      if any_zvar:
        continue
    
    for c in range(C):
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(expectand_vals[c,:])
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) and math.isnan(xi_hats[1]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Both left and right tail '
                          'hat{{xi}}s are Nan.\n')
      elif math.isnan(xi_hats[0]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Left tail '
                          'hat{{xi}} is Nan.\n')
      elif math.isnan(xi_hats[1]):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Right tail '
                          'hat{{xi}} is Nan.\n')
      if (    xi_hats[0] >= xi_hat_threshold 
          and xi_hats[1] >= xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Both left and right tail '
                          f'hat{{xi}}s ({xi_hats[0]:.3f}, '
                          f'{xi_hats[1]:.3f}) exceed '
                          f'{xi_hat_threshold}.\n')
      elif (    xi_hats[0] < xi_hat_threshold 
            and xi_hats[1] >= xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Right tail hat{{xi}} '
                          f'({xi_hats[1]:.3f}) exceeds '
                          f'{xi_hat_threshold}.\n')
      elif (    xi_hats[0] >= xi_hat_threshold 
            and xi_hats[1] < xi_hat_threshold):
        no_xi_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Left tail hat{{xi}} '
                          f'({xi_hats[0]:.3f}) exceeds '
                          f'{xi_hat_threshold}.\n')
      
      # Check empirical variance in each Markov chain
      var = welford_summary(expectand_vals[c,:])[1]
      if var < 1e-10:
        no_zvar_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: Expectand exhibits '
                          'vanishing empirical variance.\n')
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(expectand_vals)

    if math.isnan(rhat):
      local_message += '  Split hat{R} is ill-defined.\n'
    elif rhat > 1.1:
      no_rhat_warning = False
      local_warning = True
      local_message += f'  Split hat{{R}} ({rhat:.3f}) exceeds 1.1.\n'

    for c in range(C):
      tau_hat = compute_tau_hat(expectand_vals[c,:])

      # Check incremental empirical integrated autocorrelation time
      if math.isinf(tau_hat):
        no_inc_tau_hat_warning = False
        local_warning = True
        local_message += (f'Chain {c + 1}: The calculation of ',
                           'hat{{tau}} is unreliable because\nthe ',
                           'empirical variance is zero.\n')
      elif math.isnan(tau_hat):
        no_inc_tau_hat_warning = False
        local_warning = True
        print(f'Chain {c + 1}: The calculation of hat{{tau}} is ',
               'unreliable because\nthe empirical ',
               'autocorrelations did not decay sufficiently ',
               'quickly.\n')
      else:
        inc_tau_hat = tau_hat / S
        if inc_tau_hat > 5:
          no_inc_tau_hat_warning = False
          local_warning = True
          local_message += (f'Chain {c + 1}: Incremental hat{{tau}} '
                            f'({inc_tau_hat :.31}) is too large.\n')

      # Check empirical effective sample size
      ess_hat = S / tau_hat
      if ess_hat < min_ess_hat_per_chain:
        no_ess_hat_warning = False
        local_warning = True
        local_message += (f'  Chain {c + 1}: hat{{ESS}} ({ess_hat:.1f}) '
                          'is smaller than desired '
                          f'({min_ess_hat_per_chain:.0f}).\n')
    
    if local_warning:
      message += local_message + '\n'
  
  if (    no_xi_hat_warning and no_zvar_warning
      and no_rhat_warning   and no_inc_tau_hat_warning
      and no_ess_hat_warning):
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

  if not no_inc_tau_hat_warning:
    desc = ('If the incremental empirical integrated ',
            'autocorrelation times are unreliable or too large ',
            'then the Markov chains have not explored long ',
            'enough for the autocorrelation estimates to be ',
            'reliable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
  
  if not no_ess_hat_warning:
    desc = ('Small empirical effective sample sizes result in '
            'imprecise Markov chain Monte Carlo estimators.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))
    
  return

# Summary all expectand-specific diagnostics.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param min_ess_hat_per_chain The minimum empirical effective sample
#                              size before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @param max_width Maximum line width for printing
def summarize_expectand_diagnostics(expectand_vals_dict,
                                    min_ess_hat_per_chain=100,
                                    exclude_zvar=False,
                                    max_width=72):
  """Summarize expectand diagnostics"""
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')

  failed_names = []
  failed_xi_hat_names = []
  failed_zvar_names = []
  failed_rhat_names = []
  failed_inc_tau_hat_names = []
  failed_ess_hat_names = []

  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    C = expectand_vals.shape[0]
    S = expectand_vals.shape[1]
    
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(expectand_vals[c,:])[1]
        if var < 1e-10:
          any_zvar = True
      if any_zvar:
        continue
    
    for c in range(C):
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(expectand_vals[c,:])
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) or math.isnan(xi_hats[1]):
        failed_names.append(name)
        failed_xi_hat_names.append(name)
      if (   xi_hats[0] >= xi_hat_threshold
          or xi_hats[1] >= xi_hat_threshold):
        failed_names.append(name)
        failed_xi_hat_names.append(name)
    
      # Check empirical variance in each Markov chain
      var = welford_summary(expectand_vals[c,:])[1]
      if var < 1e-10:
        failed_names.append(name)
        failed_zvar_names.append(name)
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(expectand_vals)
    
    if math.isnan(rhat):
      failed_names.append(name)
      failed_rhat_names.append(name)
    elif rhat > 1.1:
      failed_names.append(name)
      failed_rhat_names.append(name)
    
    for c in range(C):
      tau_hat = compute_tau_hat(expectand_vals[c,:])

      # Check incremental empirical integrated autocorrelation time
      if math.isinf(tau_hat) or math.isnan(tau_hat) or (tau_hat / S) > 5:
        failed_names.append(name)
        failed_inc_tau_hat_names.append(name)

      # Check empirical effective sample size
      ess_hat = S / tau_hat
      
      if ess_hat < min_ess_hat_per_chain:
        failed_names.append(name)
        failed_ess_hat_names.append(name)
  
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
    
    desc = ('  Large tail hat{xi}s suggest that the expectand '
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

  failed_inc_tau_hat_names = list(numpy.unique(failed_inc_tau_hat_names))
  if len(failed_inc_tau_hat_names):
    desc = (f'The expectands {", ".join(failed_rhat_names)} '
             'triggered incremental hat{tau} warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))

    desc = ('If the incremental empirical integrated ',
            'autocorrelation times are unreliable or too large ',
            'then the Markov chains have not explored long ',
            'enough for the autocorrelation estimates to be ',
            'reliable.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

  failed_ess_hat_names = list(numpy.unique(failed_ess_hat_names))
  if len(failed_ess_hat_names):
    desc = (f'The expectands {", ".join(failed_ess_hat_names)} '
             'triggered hat{ESS} warnings.')
    desc = textwrap.wrap(desc, max_width)
    print('\n'.join(desc))
    
    desc = ('Small empirical effective sample sizes result in '
            'imprecise Markov chain Monte Carlo estimators.')
    desc = textwrap.wrap(desc, max_width)
    desc.append(' ')
    print('\n'.join(desc))

# Summarize Hamiltonian Monte Carlo and expectand diagnostics
# into a binary encoding
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param diagnostics A dictionary of two-dimensional arrays for
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param adapt_target Target acceptance proxy statistic for step size 
#                     adaptation.
# @param max_treedepth The maximum numerical trajectory treedepth
# @param min_ess_hat_per_chain The minimum empirical effective sample
#                              size before a warning message is passed.
# @param exclude_zvar Binary variable to exclude all expectands with
#                     vanishing empirical variance from other diagnostic
#                     checks.
# @return warning_code An eight bit binary summary of the diagnostic 
#                      output.
def encode_all_diagnostics(expectand_vals_dict,
                           diagnostics,
                           adapt_target=0.801,
                           max_treedepth=10,
                           min_ess_hat_per_chain=100,
                           exclude_zvar=False):
  """Summarize Hamiltonian Monte Carlo and expectand diagnostics
     into a binary encoding"""
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')
  validate_dict_of_arrays(diagnostics, 'diagnostics')

  warning_code = 0
  
  # Check divergences
  if sum(diagnostics['divergent__'].flatten()) > 0: 
    warning_code = warning_code | (1 << 0)
  
  # Check transitions that ended prematurely due to maximum tree depth limit
  if sum([ 1 for td in diagnostics['treedepth__'].flatten() 
           if td >= max_treedepth ]) > 0:
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
  inc_tau_hat_warning = False
  ess_hat_warning = False
  
  for name in expectand_vals_dict:
    expectand_vals = expectand_vals_dict[name]
    C = expectand_vals.shape[0]
    S = expectand_vals.shape[1]
    
    if exclude_zvar:
      # Check zero variance across all Markov chains for exclusion
      any_zvar = False
      for c in range(C):
        var = welford_summary(expectand_vals[c,])[1]
        if var < 1e-10:
          any_zvar = True
      if any_zvar:
        continue
    
    for c in range(C):
      # Check tail xi_hats in each Markov chain
      xi_hats = compute_tail_xi_hats(expectand_vals[c,])
      xi_hat_threshold = 0.25
      if math.isnan(xi_hats[0]) or math.isnan(xi_hats[1]):
        xi_hat_warning = True
      elif (   xi_hats[0] >= xi_hat_threshold 
            or xi_hats[1] >= xi_hat_threshold):
        xi_hat_warning = True
    
      # Check empirical variance in each Markov chain
      var = welford_summary(expectand_vals[c,])[1]
      if var < 1e-10:
        zvar_warning = True
    
    # Check split Rhat across Markov chains
    rhat = compute_split_rhat(expectand_vals)
    
    if math.isnan(rhat):
      rhat_warning = True
    elif rhat > 1.1:
      rhat_warning = True
    
    for c in range(C):
      tau_hat = compute_tau_hat(expectand_vals[c,:])

      # Check incremental empirical integrated autocorrelation time
      if math.isinf(tau_hat) or math.isnan(tau_hat) or (tau_hat / S) > 5:
        inc_tau_hat_warning = True

      # Check empirical effective sample size
      ess_hat = S / tau_hat
      if ess_hat < min_ess_hat_per_chain:
        ess_hat_warning = True
  
  if xi_hat_warning:
    warning_code = warning_code | (1 << 4)
  if zvar_warning:
    warning_code = warning_code | (1 << 5)
  if rhat_warning:
    warning_code = warning_code | (1 << 6)  
  if inc_tau_hat_warning:
    warning_code = warning_code | (1 << 7)
  if ess_hat_warning:
    warning_code = warning_code | (1 << 8)

  return warning_code


# Translate binary diagnostic codes to human readable output.
# @params warning_code An eight bit binary summary of the diagnostic
#                      output.
def decode_warning_code(warning_code):
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
        print("  incremental tau_hat warning")
    if warning_code & (1 << 8):
        print("  min ess_hat warning")

# Filter `expectand_vals_dict` by name.
# @param expectand_vals_dict A dictionary of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param requested_names List of expectand names to keep.
# @param check_arrays Binary variable indicating whether or not
#                     requested names should be expanded to array
#                     components.
# @param max_width Maximum line width for printing
# @return A dictionary of two-dimensional arrays for each requested
#         expectand.
def filter_expectands(expectand_vals_dict, requested_names,
                      check_arrays=False, max_width=72):
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')

  if len(requested_names) == 0:
    raise ValueError('Input variable `requested_names` '
                     'must be non-empty.')
  
  if check_arrays is True:
    good_names = []
    bad_names = []
    for name in requested_names:
      # Search for array suffix
      array_names = [ key for key in expectand_vals_dict.keys()
                      if re.match('^' + name + '\[', key) ]
      # Append array names, if found
      if len(array_names) > 0:
        good_names += array_names
      else:
        if name in expectand_vals_dict.keys():
          # Append bare name, if found
          good_names.append(name)
        else:
          # Add to list of bad names
          bad_names.append(name)
  else:
    bad_names = \
      set(requested_names).difference(expectand_vals_dict.keys())
    good_names = \
      set(requested_names).intersection(expectand_vals_dict.keys())
    
  if len(bad_names) > 0:
    message = (f'The expectands {", ".join(bad_names)} '
               'were not found in the `expectand_vals_dict` '
               'object and will be ignored.\n\n')
    message = textwrap.wrap(message, max_width)
    message.append(' ')
    print('\n'.join(message))
  
  return { name: expectand_vals_dict[name] for name in good_names }

# Compute empirical autocorrelations for a given Markov chain sequence
# @parmas vals A one-dimensional array of sequential expectand values.
# @return A one-dimensional array of empirical autocorrelations at each
#         lag up to the length of the sequence.
def compute_rhos(vals):
  """Visualize empirical autocorrelations for a given sequence"""
  # Compute empirical autocorrelations
  N = len(vals)
  m = welford_summary(vals)[0]
  zs = [ val - m for val in vals ]
  
  B = 2**math.ceil(math.log2(N)) # Next power of 2 after N
  zs_buff = zs + [0] * (B - N)
  
  Fs = numpy.fft.fft(zs_buff)
  Ss = numpy.abs(Fs)**2
  Rs = numpy.fft.ifft(Ss)
  
  acov_buff = numpy.real(Rs)
  if acov_buff[0] == 0:
    return math.inf

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
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_L Maximum autocorrelation lag
# @param rho_lim Plotting range of autocorrelation values
# @display_name Name of expectand
def plot_empirical_correlogram(ax,
                               expectand_vals,
                               max_L,
                               rho_lim=[-0.2, 1.1],
                               name=""):
  """Plot empirical correlograms for a given expectand across a Markov
     chain ensemble."""
  validate_array(expectand_vals, 'expectand_vals')

  C = expectand_vals.shape[0]
  
  idxs = [ idx for idx in range(max_L) for r in range(2) ]
  xs = [ idx + delta for idx in range(max_L) for delta in [-0.5, 0.5]]
  
  colors = [dark, dark_highlight, mid, light_highlight]
  
  for c in range(C):
    rhos = compute_rhos(expectand_vals[c,:])
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
# @param expectand1_vals A two-dimensional array of expectand values
#                        with the first dimension indexing the Markov
#                        chains and the second dimension indexing the
#                        sequential states within each Markov chain.
# @params display_name1 Name of first expectand
# @param expectand2_vals A two-dimensional array of expectand values
#                        with the first dimension indexing the Markov
#                        chains and the second dimension indexing the
#                        sequential states within each Markov chain
# @params display_name2 Name of second expectand
def plot_pairs_by_chain(expectand1_vals, display_name1,
                        expectand2_vals, display_name2):
  """Plot two expectand output ensembles against each other separated by
     Markov chain """
  validate_array(expectand1_vals, 'expectand1_vals')
  C1 = expectand1_vals.shape[0]
  S1 = expectand1_vals.shape[1]
  
  validate_array(expectand2_vals, 'expectand2_vals')
  C2 = expectand2_vals.shape[0]
  S2 = expectand2_vals.shape[1]
    
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

  min_x = min(expectand1_vals.flatten())
  max_x = max(expectand1_vals.flatten())
  
  min_y = min(expectand2_vals.flatten())
  max_y = max(expectand2_vals.flatten())
  
  N_plots = C1
  N_cols = 2
  N_rows = math.ceil(N_plots / N_cols)
  f, axarr = plot.subplots(N_rows, N_cols, layout="constrained")
  k = 0
  
  for c in range(C1):
    idx1 = k // N_cols
    idx2 = k % N_cols
    k += 1
    
    axarr[idx1, idx2].scatter(expectand1_vals.flatten(),
                              expectand2_vals.flatten(),
                              color="#DDDDDD", s=5, zorder=3)
    axarr[idx1, idx2].scatter(expectand1_vals[c,:], expectand2_vals[c,:],
                              cmap=cmap, c=range(S1), s=5, zorder=4)
    
    axarr[idx1, idx2].set_title(f'Chain {c + 1}')
    axarr[idx1, idx2].set_xlabel(display_name1)
    axarr[idx1, idx2].set_xlim([min_x, max_x])
    axarr[idx1, idx2].set_ylabel(display_name2)
    axarr[idx1, idx2].set_ylim([min_y, max_y])
    axarr[idx1, idx2].spines["top"].set_visible(False)
    axarr[idx1, idx2].spines["right"].set_visible(False)
  
  plot.show()

# Evaluate an expectand on the values of a one-dimensional input
# variable.
# @param input_vals A two-dimensional array of expectand values with
#                   the first dimension indexing the Markov chains
#                   and the second dimension indexing the sequential
#                   states within each Markov chain.
# @param expectand Scalar function to be applied to the Markov chain
#                  states.
# @return A two-dimensional array of expectand values with the
#         first dimension indexing the Markov chains and the
#         second dimension indexing the sequential states within
#         each Markov chain.
def eval_uni_expectand_pushforward(input_vals, expectand):
  """Evaluate an expectand along a Markov chain"""
  return numpy.vectorize(expectand)(input_vals)

# Create a nested list of element names, including indexing information,
# from the specified dimensions.  For example `name_array('x', [2, 3]))`
# returns the nested list
# >> [['x[1,1]', 'x[1,2]', 'x[1,3]'],
# >>  ['x[2,1]', 'x[2,2]', 'x[2,3]']]
# @ param base Base name.
# @ param dims List of array dimensions.
# @ param current_idxs Dimensions at current level of recursion.
# @ return Array of element names with dimensions given by dims.
def name_nested_list(base, dims, current_idxs=None):
  if current_idxs is None:
    current_idxs = []
  next_dim = len(current_idxs)
  if next_dim == len(dims):
    str_idxs = ','.join([ str(idx + 1) for idx in current_idxs ])
    return f'{base}[{str_idxs}]'
  else:
    return [ name_nested_list(base, dims, current_idxs + [d])
             for d in range(dims[next_dim]) ]

# Create a numpy array of element names from the specified dimensions.
# For example `name_array('x', [2, 3]))` returns the array
# >> array([['x[1,1]', 'x[1,2]', 'x[1,3]'],
# >>        ['x[2,1]', 'x[2,2]', 'x[2,3]']], dtype='<U6')
# @ param base Base name.
# @ param dims Vector of array dimensions.
# @ return Array of element names with dimensions given by dims.
def name_array(base, dims):
  # Validate inputs
  if not isinstance(base, str):
    raise TypeError(f'Input variable {base} is not a string.')

  if not isinstance(dims, list):
    raise TypeError(f'Input variable {dims} is not a list.')

  if not all([ isinstance(d, int) for d in dims ]):
    raise TypeError(f'The elements of input variable {dims} '
                      'are not all integers.')

  # Create element names and format them into desired array.
  return numpy.array(name_nested_list(base, dims))

# Evaluate a dictionary of expectands on the values of an arbitrary
# number of input variables.  Expectands must all return a float,
# integer, or logical output.
#
# By default expectand argument values are accessed by name
# in expectand_vals_dict.  If a non-null alt_arg_names is provided then
# the alternate names are used to access values in expectand_vals_dict.
# The elements of alt_arg_names can also be numpy arrays of arbitrary
# dimension in which case the individual element values are first
# accessed then formatted into matching numpy arrays before being
# passed to the expectand.
#
# @param expectand_vals_dict A dictionary of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param expectand_dict List of functions with the same arguments.
# @param alt_arg_names Optional dictionary of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return A list of two-dimensional arrays, one for each element of
#         expectand_list with the same keys as expectand_dict.
def eval_expectand_pushforwards(expectand_vals_dict,
                                expectand_dict,
                                alt_arg_names=None):
  """Evaluate a collection of expectands on the values of an arbitrary
     number of input variables."""
  # Validate inputs
  validate_dict_of_arrays(expectand_vals_dict, 'expectand_vals_dict')

  if not isinstance(expectand_dict, dict):
    raise TypeError('Input variable `expectand_dict` is '
                    'not a dictionary.')

  if not all([ callable(v) for k, v in expectand_dict.items() ]):
    raise TypeError('The elements of input variable `expectand_dict` '
                    'are not all functions.')

  if alt_arg_names is not None:
    if not isinstance(alt_arg_names, dict):
      raise TypeError('Input variable `alt_arg_names` '
                      'is not a dictionary.')

  # Check existence of all expectand arguments
  def compare_args(e):
    return (   set(inspect.getfullargspec(e).args)
            == set(nominal_arg_names)              )

  nominal_arg_names = []
  arg_consistency = True
  for e in expectand_dict.values():
    if not nominal_arg_names:
      nominal_arg_names = inspect.getfullargspec(e).args
    else:
      arg_consistency &= compare_args(e)

  if not arg_consistency:
    raise ValueError('The arguments of the functions in '
                     '`expectand_list` are not consistent '
                     'with each other.')

  # Ensure that all argument replacements are arrays
  if alt_arg_names is not None:
    alt_arg_names_array = { k: numpy.array(v)
                            for k, v in alt_arg_names.items() }

  # Check existence of all expectand arguments
  if alt_arg_names is None:
    check_arg_names = nominal_arg_names
  else:
    alt_keys = alt_arg_names.keys()
    missing_args = set(nominal_arg_names).difference(alt_keys)

    if len(missing_args) == 1:
      raise ValueError( 'The nominal expectand argument '
                       f'{", ".join(missing_args)} does not have '
                        'a replacement in `alt_arg_names`.')
    elif len(missing_args) > 1:
      raise ValueError( 'The nominal expectand arguments '
                       f'{", ".join(missing_args)} do not have '
                        'replacements in `alt_arg_names`.')

    arglistlist = [ v.flatten().tolist()
                    if isinstance(v, numpy.ndarray)
                    else [v]
                    for k, v in alt_arg_names.items() ]
    check_arg_names = [ arg for arglist in arglistlist
                        for arg in arglist ]

  missing_args = \
    set(check_arg_names).difference(expectand_vals_dict.keys())
  if len(missing_args) == 1:
    raise ValueError( 'The expectand argument '
                     f'{", ".join(missing_args)} is not in '
                      '`expectand_vals_dict`.')
  elif len(missing_args) > 1:
    raise ValueError( 'The expectand arguments '
                     f'{", ".join(missing_args)} are not in '
                      '`expectand_vals_dict`.')

  # Apply expectand to all inputs
  C = next(iter(expectand_vals_dict.values())).shape[0]
  S = next(iter(expectand_vals_dict.values())).shape[1]

  pushforward_vals = { k: numpy.zeros([C, S])
                       for k in expectand_dict.keys() }

  if alt_arg_names is not None:
    alt_names = [ alt_arg_names[name]
                  for name in nominal_arg_names ]

  def access_val(name):
    return expectand_vals_dict[name][c, s]

  for c in range(C):
    for s in range(S):
      if alt_arg_names is None:
        arg_vals = [ access_val(name)
                     for name in nominal_arg_names ]
      else:
        arg_vals = [ numpy.vectorize(access_val)(name)
                     for name in alt_names ]

      for n, e in expectand_dict.items():
        pushforward_vals[n][c, s] = e(*arg_vals)

  return pushforward_vals

# Evaluate an expectand on the values of an arbitrary number of input
# variables.  Expectands must all return a float, integer, or logical
# output.
#
# By default expectand argument values are accessed by name
# in expectand_vals_dict.  If a non-null alt_arg_names is provided then
# the alternate names are used to access values in expectand_vals_dict.
# The elements of alt_arg_names can also be numpy arrays of arbitrary
# dimension in which case the individual element values are first
# accessed then formatted into matching numpy arrays before being
# passed to the expectand.
#
# @param expectand_vals_dict A dictionary of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param expectand Functions with arbitrary number of scalar and array
#                  input arguments.
# @param alt_arg_names Optional dictionary of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return A two-dimensional array.
def eval_expectand_pushforward(expectand_vals_dict,
                               expectand,
                               alt_arg_names=None):
  pushforward_vals = eval_expectand_pushforwards(expectand_vals_dict,
                                                 { 'e': expectand },
                                                 alt_arg_names)
  return pushforward_vals['e']

# Estimate expectand expectation value from a single Markov chain.
# @param vals A one-dimensional array of sequential expectand values.
# @return The Markov chain Monte Carlo estimate, its estimated standard
#         error, and empirical effective sample size.
def mcmc_est(vals):
  """Estimate expectand expectation value from a Markov chain"""
  S = len(vals)
  if S == 1:
    return [vals[0], 0, math.nan]
  
  summary = welford_summary(vals)
  
  if summary[1] == 0:
    return [summary[0], 0, math.nan]
  
  tau_hat = compute_tau_hat(vals)
  ess_hat = S / tau_hat
  return [summary[0], math.sqrt(summary[1] / ess_hat), ess_hat]

# Estimate expectand expectation value from a Markov chain ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @return The ensemble Markov chain Monte Carlo estimate, its estimated
#         standard error, and empirical effective sample size.
def ensemble_mcmc_est(expectand_vals):
  """Estimate expectand expectation value from a collection of
     Markov chains"""
  validate_array(expectand_vals, 'expectand_vals')
    
  C = expectand_vals.shape[0]
  chain_ests = [ mcmc_est(expectand_vals[c,:]) for c in range(C) ]
  
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

# Estimate the probability allocated to a subset implicitly defined by
# an indicator function.
# @param expectand_vals_dict A dictionary of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param indicator Function with logical or 0/1 numeric outputs.
# @param alt_arg_names Optional dictionary of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return The probability estimate and its standard error.
def implicit_subset_prob(expectand_vals_dict,
                         indicator,
                         alt_arg_names):
  # Evaluate indicator function
  indicator_samples = eval_expectand_pushforward(expectand_vals_dict,
                                                 indicator,
                                                 alt_arg_names)

  # Verify outputs
  unique_vals = set(indicator_samples.flatten())

  if not (  unique_vals == set([0])
          | unique_vals == set([1])
          | unique_vals == set([0, 1])):
    raise ValueError('The function `indicator` must return only '
                     'logical or 0/1 numeric outputs.')

  # Return probability as expectation value of indicator function
  return ensemble_mcmc_est(indicator_samples)[0:2]

# Estimate expectand pushforward quantiles from a Markov chain ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param probs An array of quantile percentages in [0, 100].
# @return The ensemble Markov chain Monte Carlo quantile estimate.
def ensemble_mcmc_quantile_est(expectand_vals, probs):
  # Validate inputs
  validate_array(expectand_vals, 'expectand_vals')

  if not isinstance(probs, list):
    raise TypeError(('Input variable `probs` is not a list.'))

  # Estimate and return quantile
  q = numpy.zeros(len(probs))

  C = expectand_vals.shape[0]
  for c in range(C):
    q += numpy.percentile(expectand_vals[c,:], probs) / C

  return q

# Visualize pushforward distribution of a given expectand as a
# histogram, using Markov chain Monte Carlo estimators to estimate the
# output bin probabilities.  Bin probability estimator error is shown
# in gray.
# @param ax Matplotlib axis object
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param B The number of histogram bins
# @param display_name Expectand name
# @param flim Optional histogram range
# @param ylim Optional y-axis range; ignored if add is TRUE
# @param color Color for plotting weighted bin probabilities; defaults
#              to dark.
# @param border Color for plotting estimator error; defaults to gray
# @param border_opacity Opacity for plotting estimator error; defaults
#                       to 1.
# @param add Configure plot to overlay over existing plot; defaults to
#            FALSE
# @param title Optional plot title
# @param baseline Optional baseline value for visual comparison
# @param baseline_color Color for plotting baseline value; defaults to
#                       "black"
def plot_expectand_pushforward(ax, expectand_vals, B, display_name="f",
                               flim=None, ylim=None,
                               color=dark, border="#DDDDDD",
                               border_opacity=1,
                               add=False, title=None,
                               baseline=None, baseline_color="black"):
  """Plot pushforward histogram of a given expectand using Markov chain
     Monte Carlo estimators to estimate the output bin probabilities"""
  validate_array(expectand_vals, 'expectand_vals')
    
  if flim is None:
    # Automatically adjust histogram binning to range of outputs
    min_f = min(expectand_vals.flatten())
    max_f = max(expectand_vals.flatten())
    delta = (max_f - min_f) / B

    # Add bounding bins
    B = B + 2
    min_f = min_f - delta
    max_f = max_f + delta
    flim = [min_f, max_f]
    
    bins = numpy.arange(min_f, max_f + delta, delta)
  else:
    min_f = flim[0]
    max_f = flim[1]
    delta = (max_f - min_f) / B

    bins = numpy.arange(min_f, max_f + delta, delta)
  
  # Check sample containment
  S = expectand_vals.size

  S_low = sum(expectand_vals.flatten() < min_f)
  if S_low == 1:
    print(f'{S_low} value ({S_low / S:.2%})'
           ' fell below the histogram binning.')
  elif S_low > 1:
    print(f'{S_low} values ({S_low / S:.2%})'
           ' fell below the histogram binning.')

  S_high = sum(max_f < expectand_vals.flatten())
  if S_high == 1:
    print(f'{S_high} value ({S_high / S:.2%})'
           ' fell above the histogram binning.')
  elif S_high > 1:
    print(f'{S_high} values ({S_high / S:.2%})'
           ' fell above the histogram binning.')

  # Compute bin heights
  mean_p = [0] * B
  delta_p = [0] * B
  
  for b in range(B):
    def bin_indicator(x):
      return 1.0 if bins[b] <= x and x < bins[b + 1] else 0.0
    
    indicator_vals = eval_uni_expectand_pushforward(expectand_vals,
                                                    bin_indicator)
    est = ensemble_mcmc_est(indicator_vals)
    
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
  
  if add:
    ax.fill_between(xs, lower_inter, upper_inter,
                    color=border, facecolor=border,
                    alpha=border_opacity)
    ax.plot(xs, [ mean_p[idx] for idx in idxs ],
            color=color, linewidth=2)
  else:
    if ylim is None:
      ylim = [ 0, 1.05 * max(upper_inter) ]

    ax.fill_between(xs, lower_inter, upper_inter,
                    color=border, facecolor=border,
                    alpha=border_opacity)
    ax.plot(xs, [ mean_p[idx] for idx in idxs ],
            color=color, linewidth=2)

    if title is not None:
      ax.set_title(title)
    ax.set_xlim(flim)
    ax.set_xlabel(display_name)
    ax.set_ylim(ylim)
    ax.set_ylabel("Estimated Bin\nProbabilities / Bin Width")
    ax.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

  if baseline is not None:
    ax.axvline(x=baseline, linewidth=4, color="white")
    ax.axvline(x=baseline, linewidth=2, color=baseline_color)

