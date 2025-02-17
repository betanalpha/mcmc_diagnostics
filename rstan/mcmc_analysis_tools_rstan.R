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
library(rstan)
library(colormap)

# Graphic configuration
c_light <- c("#DCBCBC")
c_light_highlight <- c("#C79999")
c_mid <- c("#B97C7C")
c_mid_highlight <- c("#A25050")
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")

c_light_teal <- c("#6B8E8E")
c_mid_teal <- c("#487575")
c_dark_teal <- c("#1D4F4F")

# Extract unpermuted expectand values from a StanFit object and format 
# them for convenient access.  Removes the auxiliary `lp__` variable.
# @param stan_fit A StanFit object
# @return A named list of two-dimensional arrays for each expectand in 
#         the StanFit object.  The first dimension of each element 
#         indexes the Markov chains and the second dimension indexes the 
#         sequential states within each Markov chain. 
extract_expectand_vals <- function(stan_fit) {
  nom_params <- rstan:::extract(stan_fit, permuted=FALSE)
  N <- dim(nom_params)[3] - 1
  params <- lapply(1:N, function(n) t(nom_params[,,n]))
  names(params) <- names(stan_fit)[1:N]
  (params)
}

# Validate named list structure of input object.
# @param obj Object to be validated
# @param name Object name
validate_named_list <- function(obj, name) {
  if ( !is.list(obj) |
        is.null(names(obj)) ) {
    stop(paste0('Input variable ', name, ' is not a named list.'))
  }
}

# Validate two-dimensional array structure of input object.
# @param obj Object to be validated
# @param name Object name
validate_array <- function(obj, name) {
  if (length(dim(obj)) != 2 | !is.double(obj) ) {
    stop(paste0('Input variable ', name, ' is not a ',
                'two-dimensional numeric array.'))
  }
}

# Validate named list of two-dimensional array structure of input
# object.
# @param obj Object to be validated
# @param name Object name
validate_named_list_of_arrays <- function(obj, name) {
  validate_named_list(obj, name)

  if (!Reduce("&", sapply(obj,
                          function(s) is.double(s) &
                          length(dim(s)) == 2))) {
    stop(paste0('The elements of input variable ', name,
                ' are not all two-dimensional numeric arrays.'))
  }

  dims <- sapply(obj, function(s) dim(s))
  if ( !identical(unname(dims[1,][-length(dims[1,])]),
                  unname(dims[1,][-1])) |
       !identical(unname(dims[2,][-length(dims[2,])]),
                  unname(dims[2,][-1])) ) {
    stop(paste0('The elements of input variable ', name,
                ' do not have consistent dimensions.'))
  }
}

# Extract Hamiltonian Monte Carlo diagnostics values from a StanFit
# object and format them for convenient access.
# @param stan_fit A StanFit object
# @return A named list of two-dimensional arrays for each expectand in 
#         the StanFit object.  The first dimension of each element 
#         indexes the Markov chains and the second dimension indexes the 
#         sequential states within each Markov chain. 
extract_hmc_diagnostics <- function(stan_fit) {
  diagnostic_names <- c('divergent__', 'treedepth__', 'n_leapfrog__', 
                        'stepsize__', 'energy__', 'accept_stat__')

  nom_params <- get_sampler_params(stan_fit, inc_warmup=FALSE)
  C <- length(nom_params)
  params <- lapply(diagnostic_names, 
                   function(name) t(sapply(1:C, function(c) 
                                  nom_params[c][[1]][,name])))
  names(params) <- diagnostic_names
  (params)
}

# Check all Hamiltonian Monte Carlo Diagnostics 
# for an ensemble of Markov chains
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @param adapt_target Target acceptance proxy statistic for step size 
#                     adaptation.
# @param max_treedepth The maximum numerical trajectory treedepth
# @param max_width Maximum line width for printing
check_all_hmc_diagnostics <- function(diagnostics,
                                      adapt_target=0.801,
                                      max_treedepth=10,
                                      max_width=72) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')
  
  no_warning <- TRUE
  no_divergence_warning <- TRUE
  no_treedepth_warning <- TRUE
  no_efmi_warning <- TRUE
  no_accept_warning <- TRUE
  
  message <- ""
  
  C <- dim(diagnostics[['divergent__']])[1]
  S <- dim(diagnostics[['divergent__']])[2]
  
  for (c in 1:C) {
    local_message <- ""
    # Check for divergences
    n_div <- sum(diagnostics[['divergent__']][c,])
    
    if (n_div > 0) {
      no_warning <- FALSE
      no_divergence_warning <- FALSE
      local_message <- 
        paste0(local_message,
               sprintf('  Chain %s: %s of %s transitions (%.1f%%) ', 
                       c, n_div, S, 100 * n_div / S),
               'diverged.\n')
    }
    
    # Check for tree depth saturation
    n_tds <- sum(sapply(diagnostics[['treedepth__']][c,], 
                        function(s) s >= max_treedepth))
    
    if (n_tds > 0) {
      no_warning <- FALSE
      no_treedepth_warning <- FALSE
      local_message <- 
        paste0(local_message,
               sprintf('  Chain %s: %s of %s transitions (%.2f%%)\n',
                       c, n_tds, S, 100 * n_tds / S),
                       '           saturated the maximum treedepth ',
               sprintf('of %s.\n', max_treedepth))
    }
    
    # Check the energy fraction of missing information (E-FMI)
    energies = diagnostics[['energy__']][c,]
    numer = sum(diff(energies)**2) / length(energies)
    denom = var(energies)
    efmi <- numer / denom
    if (efmi < 0.2) {
      no_warning <- FALSE
      no_efmi_warning <- FALSE
      local_message <- 
        paste0(local_message, 
               sprintf('  Chain %s: E-FMI = %.3f.\n', c, efmi))
    }
    
    # Check convergence of the stepsize adaptation
    ave_accept_proxy <- mean(diagnostics[['accept_stat__']][c,])
    if (ave_accept_proxy < 0.9 * adapt_target) {
      no_warning <- FALSE
      no_accept_warning <- FALSE
      local_message <- 
        paste0(local_message,
               sprintf('  Chain %s: Average proxy acceptance ', c),
               sprintf('statistic (%.3f)\n', ave_accept_proxy),
                       '           is smaller than 90% of the target ',
               sprintf('(%.3f).\n', adapt_target))
    }
    
    if (local_message != "") {
      message <- paste0(message, local_message, '\n')
    }
  }

  if (no_warning) {
    desc <- paste0('All Hamiltonian Monte Carlo diagnostics are ',
                   'consistent with reliable Markov chain Monte Carlo.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  if (!no_divergence_warning) {
    desc <- paste0('Divergent Hamiltonian transitions result from ',
                   'unstable numerical trajectories.  These ',
                   'instabilities are often due to degenerate target ',
                   'geometry, especially "pinches".  If there are ',
                   'only a small number of divergences then running ',
                   'with adept_delta larger ',
                   sprintf('than %.3f may reduce the ', adapt_target),
                   'instabilities at the cost of more expensive ',
                   'Hamiltonian transitions.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }

  if (!no_treedepth_warning) {
    desc <- paste0('Numerical trajectories that saturate the ',
                   'maximum treedepth have terminated prematurely.  ',
                   sprintf('Increasing max_depth above %s ', max_treedepth),
                   'should result in more expensive, but more ',
                   'efficient, Hamiltonian transitions.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }

  if (!no_efmi_warning) {
    desc <- paste0('E-FMI below 0.2 arise when a funnel-like geometry ',
                   'obstructs how effectively Hamiltonian trajectories ',
                   'can explore the target distribution.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }

  if (!no_accept_warning) {
    desc <- paste0('A small average proxy acceptance statistic ',
                   'indicates that the adaptation of the numerical ',
                   'integrator step size failed to converge.  This is ',
                   'often due to discontinuous or imprecise ',
                   'gradients.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }

  cat(message)
}

# Plot outcome of inverse metric adaptation
# @params stan_fit A StanFit object
# @params B The number of bins for the inverse metric element histograms.
plot_inv_metric <- function(stan_fit, B=25) {
  adaptation_info <- rstan:::get_adaptation_info(stan_fit)
  C <- length(adaptation_info)

  inv_metric_elems <- list()
  for (c in 1:C) {
    raw_info <- adaptation_info[[c]]
    clean1 <- sub("# Adaptation terminated\n# Step size = [0-9.]*\n#",
                  "", raw_info)
    clean2 <- sub(" [a-zA-Z ]*:\n# ", "", clean1)
    clean3 <- sub("\n$", "", clean2)
    inv_metric_elems[[c]] <- as.numeric(strsplit(clean3, ',')[[1]])
  }

  min_elem <- min(unlist(inv_metric_elems))
  max_elem <- max(unlist(inv_metric_elems))

  delta <- (max_elem - min_elem) / B
  min_elem <- min_elem - delta
  max_elem <- max_elem + delta
  bins <- seq(min_elem, max_elem, delta)
  B <- B + 2  

  max_y <- max(sapply(1:C, function(c)
    max(hist(inv_metric_elems[[c]], breaks=bins, plot=FALSE)$counts)))

  idx <- rep(1:B, each=2)
  x <- sapply(1:length(idx), function(b) if(b %% 2 == 1) bins[idx[b]]
                                         else bins[idx[b] + 1])

  par(mfrow=c(2, 2), mar = c(5, 2, 2, 1))
  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)

  for (c in 1:C) {
    counts <- hist(inv_metric_elems[[c]], breaks=bins, plot=FALSE)$counts
    y <- counts[idx]

    plot(x, y, type="l", main=paste("Chain", c), col=colors[c],
         xlim=c(min_elem, max_elem), xlab="Inverse Metric Elements",
         ylim=c(0, 1.05 * max_y), ylab="", yaxt="n")
  }
}

# Display adapted symplectic integrator step sizes
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
display_stepsizes <- function(diagnostics) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')
  
  stepsizes <- diagnostics[['stepsize__']]
  C <- dim(stepsizes)[1]
  
  for (c in 1:C) {
    stepsize <- stepsizes[c, 1]
    cat(sprintf('Chain %s: Integrator Step Size = %f\n',
                c, stepsize))
  }
}

# Display symplectic integrator trajectory lengths
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
plot_num_leapfrogs <- function(diagnostics) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')
  
  lengths <- diagnostics[['n_leapfrog__']]
  C <- dim(lengths)[1]

  max_length <- max(lengths) + 1
  max_count <- max(sapply(1:C, function(c) max(table(lengths[c,]))))

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)

  idx <- rep(1:max_length, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  plot(0, type="n",
       xlab="Numerical Trajectory Length", 
       xlim=c(0.5, max_length + 0.5),
       ylab="", ylim=c(0, 1.1 * max_count), yaxt='n')

  for (c in 1:C) {
    counts <- hist(lengths[c,], 
                   seq(0.5, max_length + 0.5, 1), 
                   plot=FALSE)$counts
    pad_counts <- counts[idx]
    lines(xs, pad_counts, lwd=2, col=colors[c])
  }
}

# Display symplectic integrator trajectory lengths by Markov chain
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
plot_num_leapfrogs_by_chain <- function(diagnostics) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')
  
  lengths <- diagnostics[['n_leapfrog__']]
  C <- dim(lengths)[1]

  max_length <- max(lengths) + 1
  max_count <- max(sapply(1:C, function(c) max(table(lengths[c,]))))

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)

  idx <- rep(1:max_length, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  par(mfrow=c(2, 2), mar = c(5, 2, 2, 1))

  for (c in 1:C) {
    stepsize <- round(diagnostics[['stepsize__']][c,1], 3)
  
    counts <- hist(lengths[c,], 
                   seq(0.5, max_length + 0.5, 1), 
                   plot=FALSE)$counts
    pad_counts <- counts[idx]
  
    plot(xs, pad_counts, type="l",  lwd=2, col=colors[c],
         main=paste0("Chain ", c, " (Stepsize = ", stepsize, ")"),
         xlab="Numerical Trajectory Length", xlim=c(0.5, max_length + 0.5),
         ylab="", ylim=c(0, 1.1 * max_count), yaxt='n')
  }
}

# Display empirical average of the proxy acceptance statistic across 
# each Markov chain
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
display_ave_accept_proxy <- function(diagnostics) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')
  
  proxy_stats <- diagnostics[['accept_stat__']]
  C <- dim(proxy_stats)[1]

  for (c in 1:C) {
    ave_accept_proxy <- mean(proxy_stats[c,])
    cat(sprintf('Chain %s: Average proxy acceptance statistic = %.3f\n',
                c, ave_accept_proxy))
  }
}

# Display symplectic integrator trajectory times
# @param diagnostics A named list of two-dimensional arrays for
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the
#                    second dimension indexes the sequential
#                    states within each Markov chain.
# @param B The number of histogram bins
# @param nlim Optional histogram range
plot_int_times <- function(diagnostics, B, tlim=NULL) {
  validate_named_list_of_arrays(diagnostics, 'diagnostics')

  lengths <- diagnostics[['n_leapfrog__']]
  C <- dim(lengths)[1]
  eps <- sapply(1:C, function(c) diagnostics[['stepsize__']][c, 1])

  if (is.null(tlim)) {
    # Automatically adjust histogram binning to range of outputs
    min_t <- min(sapply(1:C, function(c) eps[c] * min(lengths[c,])))
    max_t <- max(sapply(1:C, function(c) eps[c] * max(lengths[c,])))

    tlim <- c(min_t, max_t)
    delta <- (tlim[2] - tlim[1]) / B
    bins <- seq(tlim[1] - delta, tlim[2] + delta, delta)
    B = B + 2
  } else {
    delta <- (tlim[2] - tlim[1]) / B
    bins <- seq(tlim[1], tlim[2], delta)
  }

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)

  idx <- rep(1:B, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 1) bins[idx[b]]
                                          else bins[idx[b] + 1])

  max_counts <- 0
  for (c in 1:C) {
    counts <- hist(eps[c] * lengths[c,], bins, plot=FALSE)$counts
    max_counts <- max(max_counts, max(counts))
  }

  plot(0, type="n",
       xlab="Trajectory Integration Times",
       xlim=tlim,
       ylab="", ylim=c(0, 1.1 * max_counts), yaxt='n')

  for (c in 1:C) {
    counts <- hist(eps[c] * lengths[c,], bins, plot=FALSE)$counts
    pad_counts <- counts[idx]
    lines(xs, pad_counts, lwd=2, col=colors[c])
  }
}

# Apply transformation identity, log, or logit transformation to
# named values and flatten the output.  Transformation defaults to
# identity if name is not included in `transforms` dictionary.  A 
# ValueError is thrown if values are not properly constrained.
# @param name Expectand name.
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param transforms A named list of transformation flags for each 
#                   expectand.
# @return The transformed expectand name and a one-dimensional array of
#         flattened transformation outputs.
apply_transform <- function(name, expectand_vals_list, transforms) {
  t <- transforms[[name]]
  if (is.null(t)) t <- 0
  
  transformed_name <- ""
  transformed_vals <- 0
 
  if (t == 0) {
    transformed_name <- name
    transformed_vals <- c(t(expectand_vals_list[[name]]),
                            recursive=TRUE)
  } else if (t == 1) {
    if (min(expectand_vals_list[[name]]) <= 0) {
      stop(paste0('Log transform requested for expectand ',
                  sprintf('%s ', name),
                  'but expectand values are not strictly positive.'))
    }
    transformed_name <- paste0('log(', name, ')')
    transformed_vals <- log(c(t(expectand_vals_list[[name]]),
                              recursive=TRUE))
  } else if (t == 2) {
    if (min(expectand_vals_list[[name]]) <= 0 |
        max(expectand_vals_list[[name]] >= 1)) {
      stop(paste0('Logit transform requested for expectand ',
                  sprintf('%s ' , name),
                  'but expectand values are not strictly confined ',
                  'to the unit interval.'))
    }
    transformed_name <- paste0('logit(', name, ')')
    transformed_vals <- sapply(c(t(expectand_vals_list[[name]]),
                                 recursive=TRUE),
                               function(x) log(x / (1 - x)))
  }
  return (list('t_name' = transformed_name, 
               't_vals' = transformed_vals))
}

# Plot pairwise scatter plots with non-divergent and divergent 
# transitions separated by color
# @param x_names An array of expectand names to be plotted on the x axis.
# @param y_names An array of expectand names to be plotted on the y axis.
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param diagnostics A named list of two-dimensional arrays for 
#                    each diagnostic.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
# @params transforms A named list of flags configuring which if any
#                    transformation to apply to each named expectand:
#                      0: identity
#                      1: log
#                      2: logit
# @param xlim       Optional global x-axis bounds for all pair plots.
#                   Defaults to dynamic bounds for each pair plot.
# @param ylim       Optional global y-axis bounds for all pair plots.
#                   Defaults to dynamic bounds for each pair plot.
# @params plot_mode Plotting style configuration: 
#                     0: Non-divergent transitions are plotted in 
#                        transparent red while divergent transitions are
#                        plotted in transparent green.
#                     1: Non-divergent transitions are plotted in gray 
#                        while divergent transitions are plotted in 
#                        different shades of teal depending on the 
#                        trajectory length.  Transitions from shorter
#                        trajectories should cluster somewhat closer to 
#                        the neighborhoods with problematic geometries.
# @param max_width Maximum line width for printing
plot_div_pairs <- function(x_names, y_names, 
                           expectand_vals_list, diagnostics,
                           transforms=list(), xlim=NULL, ylim=NULL,
                           plot_mode=0, max_width=72) {
  if (!is.vector(x_names)) {
    stop('Input variable `x_names` is not an array.')
  }
  
  if (!is.vector(y_names)) {
    stop('Input variable `y_names` is not an array.')
  }
  
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')

  validate_named_list_of_arrays(diagnostics, 'diagnostics')

  if (length(transforms) > 0)
    validate_named_list(transforms, 'transforms')
  
  # Check transform flags
  for (name in names(transforms)) {
    if (transforms[[name]] < 0 | transforms[[name]] > 2) {
      warning <- 
        paste0(sprintf('The transform flag %s for expectand %s ', 
                       transforms[[name]], name),
               'is invalid.  Plot will default to no transformation.')
      warning <- paste0(strwrap(warning, max_width, 0), collapse='\n')
      cat(warning)
    }
  }
  
  # Check plot mode
  if (plot_mode < 0 | plot_mode > 1) {
    stop(sprintf('Invalid `plot_mode` value %s.', plot_mode))
  }
  
  # Transform expectand values
  transformed_vals = list()
  
  transformed_x_names <- c()
  for (name in x_names) {
    r <- apply_transform(name, expectand_vals_list, transforms)
    if (is.null(r))
      stop()
    transformed_x_names <- c(transformed_x_names, r$t_name)
    if (! r$t_name %in% transformed_vals) {
      transformed_vals[[r$t_name]] <- r$t_vals
    }
  }
  
  transformed_y_names <- c()
  for (name in y_names) {
    r <- apply_transform(name, expectand_vals_list, transforms)
    if (is.null(r))
      stop()
    transformed_y_names <- c(transformed_y_names, r$t_name)
    if (! r$t_name %in% transformed_vals) {
      transformed_vals[[r$t_name]] <- r$t_vals
    }
  }
  
  # Create pairs of transformed expectands, dropping duplicates
  pairs <- list()
  for (x_name in transformed_x_names) {
    for (y_name in transformed_y_names) {
      if (x_name == y_name) next
      if (any(sapply(pairs, identical, c(x_name, y_name)))) next
      if (any(sapply(pairs, identical, c(y_name, x_name)))) next
      pairs[[length(pairs) + 1]] <- c(x_name, y_name)
    }
  }
  
  # Extract non-divergent and divergent transition indices
  divs <- diagnostics[['divergent__']]
  C <- dim(divs)[1]
  nondiv_filter <- c(sapply(1:C, function(c) divs[c,] == 0))
  div_filter    <- c(sapply(1:C, function(c) divs[c,] == 1))
  
  if (plot_mode == 1) {
    if (sum(div_filter) > 0) {
      nlfs <- c(sapply(1:C,
                       function(c) diagnostics[['n_leapfrog__']][c,]))
      div_nlfs <- nlfs[div_filter]
      max_nlf <- max(div_nlfs)
      nom_colors <- c(c_light_teal, c_mid_teal, c_dark_teal)
      cmap <- colormap(colormap=nom_colors, nshades=max_nlf)
    } else {
      div_nlfs <- c()
      nom_colors <- c(c_light_teal, c_mid_teal, c_dark_teal)
      cmap <- colormap(colormap=nom_colors, nshades=1)
    }
  }

  # Set plot layout dynamically
  N_cols <- 3
  N_plots <- length(pairs)
  if (N_plots <= 3) {
    par(mfrow=c(1, N_plots), mar = c(5, 5, 2, 1))
  } else if (N_plots == 4) {
    par(mfrow=c(2, 2), mar = c(5, 5, 2, 1))
  } else if (N_plots == 6) {
    par(mfrow=c(2, 3), mar = c(5, 5, 2, 1))
  } else {
    par(mfrow=c(3, N_cols), mar = c(5, 5, 2, 1))
  }
  
  # Plot
  c_dark_trans <- c("#8F272780")
  c_green_trans <- c("#00FF0080")
  
  for (pair in pairs) {
    x_name <- pair[1]
    x_nondiv_vals <- transformed_vals[[x_name]][nondiv_filter]
    x_div_vals    <- transformed_vals[[x_name]][div_filter]
    
    if (is.null(xlim)) {
      xmin = min(transformed_vals[[x_name]])
      xmax = max(transformed_vals[[x_name]])
      local_xlim <- c(xmin, xmax)
    } else {
      local_xlim <- xlim
    }
    
    y_name <- pair[2]
    y_nondiv_vals <- transformed_vals[[y_name]][nondiv_filter]
    y_div_vals    <- transformed_vals[[y_name]][div_filter]
    
    if (is.null(ylim)) {
      ymin = min(transformed_vals[[y_name]])
      ymax = max(transformed_vals[[y_name]])
      local_ylim <- c(ymin, ymax)
    } else {
      local_ylim <- ylim
    }
 
    if (plot_mode == 0) {
      plot(x_nondiv_vals, y_nondiv_vals,
           col=c_dark_trans, pch=16, main="",
           xlab=x_name, xlim=local_xlim, 
           ylab=y_name, ylim=local_ylim)
      points(x_div_vals, y_div_vals,
             col=c_green_trans, pch=16)
    }
    if (plot_mode == 1) {
      plot(x_nondiv_vals, y_nondiv_vals,
           col="#DDDDDD", pch=16, main="",
           xlab=x_name, xlim=local_xlim, 
           ylab=y_name, ylim=local_ylim)
      points(x_div_vals, y_div_vals,
             col=cmap[div_nlfs], pch=16)
    }
  }
}

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
compute_xi_hat <- function(vals) {
  N <- length(vals)
  sorted_vals <- sort(vals)

  # Return erroneous result if all input values are the same 
  if (sorted_vals[1] == sorted_vals[N]) {
    return (NaN)
  }

  # Return erroneous result if all input values are not positive
  if (sorted_vals[1] < 0) {
    cat("Input values must be positive.")
    return (NaN)
  }

  # Estimate 25% quantile
  q <- sorted_vals[floor(0.25 * N + 0.5)]

  if (q == sorted_vals[1]) {
    return (-2)
  }

  # Heurstic generalized Pareto shape configuration
  M <- 20 + floor(sqrt(N))

  b_hat_vec <- rep(0, M)
  log_w_vec <- rep(0, M)

  for (m in 1:M) {
    b_hat_vec[m] <- 1 / sorted_vals[N] +
                 (1 - sqrt(M / (m - 0.5))) / (3 * q)
    if (b_hat_vec[m] != 0) {
      xi_hat <- mean( log(1 - b_hat_vec[m] * sorted_vals) )
      log_w_vec[m] <- N * ( log(-b_hat_vec[m] / xi_hat) - xi_hat - 1 )
    } else {
      log_w_vec[m] <- 0
    }
  }
  
  # Remove terms that don't contribute to average to improve numerical 
  # stability
  log_w_vec <- log_w_vec[b_hat_vec != 0]
  b_hat_vec <- b_hat_vec[b_hat_vec != 0]

  max_log_w <- max(log_w_vec)
  b_hat <- sum(b_hat_vec * exp(log_w_vec - max_log_w)) /
           sum(exp(log_w_vec - max_log_w))

  mean( log (1 - b_hat * sorted_vals) )
}

# Compute empirical generalized Pareto shape for upper and lower tails
# for the given expectand values, ignoring any autocorrelation between
# the values.
# @param vals A one-dimensional array of sequential expectand values.
# @return Left and right shape estimators.
compute_tail_xi_hats <- function(vals) {
  v_center <- median(vals)
  
  # Isolate lower and upper tails which can be adequately modeled by a 
  # generalized Pareto shape for sufficiently well-behaved distributions
  vals_left <- abs(vals[vals < v_center] - v_center)
  N <- length(vals_left)
  M <- min(0.2 * N, 3 * sqrt(N))
  vals_left <- vals_left[M:N]
  
  vals_right <- vals[vals > v_center] - v_center
  N <- length(vals_right)
  M <- min(0.2 * N, 3 * sqrt(N))
  vals_right <- vals_right[M:N]
  
  # Default to NaN if left tail is ill-defined
  xi_hat_left <- NaN
  if (length(vals_left) > 40)
    xi_hat_left <- compute_xi_hat(vals_left)

  # Default to NaN if right tail is ill-defined
  xi_hat_right <- NaN
  if (length(vals_right) > 40)
    xi_hat_right <- compute_xi_hat(vals_right)

  c(xi_hat_left, xi_hat_right)
}

# Check upper and lower tail behavior of the expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
check_tail_xi_hats <- function(expectand_vals, max_width=72) {
  validate_array(expectand_vals, 'expectand_vals')
  C <- dim(expectand_vals)[1]
  
  no_warning <- TRUE
  message <- ""
  
  for (c in 1:C) {
    xi_hats <- compute_tail_xi_hats(expectand_vals[c,])
    xi_hat_threshold <- 0.25
    if ( is.nan(xi_hats[1]) & is.nan(xi_hats[2]) ) {
      no_warning <- FALSE
      body <- '  Chain %s: Both left and right hat{xi}s are NaN.\n'
      message <- paste0(message, sprintf(body, c))
    } 
    else if ( is.nan(xi_hats[1]) ) {
      no_warning <- FALSE
      body <- '  Chain %s: Left hat{xi} is NaN.\n'
      message <- paste0(message, sprintf(body, c))
    } else if ( is.nan(xi_hats[2]) ) {
      no_warning <- FALSE
      body <- '  Chain %s: Right hat{xi} is NaN.\n'
      message <- paste0(message, sprintf(body, c))
    } else if (xi_hats[1] >= xi_hat_threshold & 
      xi_hats[2] >= xi_hat_threshold) {
      no_warning <- FALSE
      body <- paste0('  Chain %s: Both left and right tail ',
                     'hat{xi}s (%.3f, %.3f) exceed %.2f.\n')
      message <- paste0(message, sprintf(body, c,
                                         xi_hats[1], xi_hats[2],
                                         xi_hat_threshold))
    } else if (xi_hats[1] <  xi_hat_threshold &
               xi_hats[2] >= xi_hat_threshold   ) {
      no_warning <- FALSE
      body <- paste0('  Chain %s: Right tail hat{xi} (%.3f) ',
                     'exceeds %.2f.\n')
      message <- paste0(message, sprintf(body, c, xi_hats[2],
                                         xi_hat_threshold))
    } else if (xi_hats[1] >= xi_hat_threshold & 
               xi_hats[2] <  xi_hat_threshold   ) {
      no_warning <- FALSE
      body <- paste0('  Chain %s: Left tail hat{xi} (%.3f) ',
                     'exceeds %.2f.\n')
      message <- paste0(message, sprintf(body, c, xi_hats[1],
                                         xi_hat_threshold))
    }
  }
  
  if (no_warning) {
    desc <- 'Expectand appears to be sufficiently integrable.\n\n'
    message <- paste0(message, desc)
  } else {
    desc <- paste0('Large tail xi_hats suggest that the expectand ',
                   'might not be sufficiently integrable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  cat(message)
}

# Compute empirical mean and variance of a given sequence with a single
# pass using Welford accumulators.
# @params vals A one-dimensional array of sequential expectand values.
# @return The empirical mean and variance.
welford_summary <- function(vals) {
  mean <- 0
  var <- 0

  N <- length(vals)
  for (n in 1:N) {
    delta <- vals[n] - mean
    mean <- mean + delta / n
    var <- var + delta * (vals[n] - mean)
  }

  var <- var/ (N - 1)

  return(c(mean, var))
}

# Check expectand values for vanishing empirical variance.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
check_variances <- function(expectand_vals, max_width=72) {
  validate_array(expectand_vals, 'expectand_vals')
  C <- dim(expectand_vals)[1]
  
  no_warning <- TRUE
  message <- ""

  for (c in 1:C) {
    var <- welford_summary(expectand_vals[c,])[2]
    if (var < 1e-10) {
      body <- '  Chain %s: Expectand values are constant.\n'
      message <- paste0(message, sprintf(body, c))
      no_warning <- FALSE
    }
  }
  
  if (no_warning) {
    desc <- 'Expectand values are varying across all Markov chains.\n\n'
    message <- paste0(message, desc)
  } else {
    desc <- paste0('If the expectand is not expected to be nearly ',
                   'constant then the Markov transition might be ',
                   'misbehaving.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  cat(message)
}

# Split a sequence of expectand values in half to create an initial and 
# terminal Markov chains
# @params chain A sequence of expectand values derived from a single 
#               Markov chain.
# @return Two subsequences of expectand values.
split_chain <- function(chain) {
  N <- length(chain)
  M <- N %/% 2
  list(chain1 <- chain[1:M], chain2 <- chain[(M + 1):N])
}

# Compute split hat{R} for the expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @return Split Rhat estimate.
compute_split_rhat <- function(expectand_vals) {
  validate_array(expectand_vals, 'expectand_vals')
  C <- dim(expectand_vals)[1]
  
  split_chain_vals <- unlist(lapply(1:C,
                                    function(c)
                                    split_chain(expectand_vals[c,])),
                             recursive=FALSE)

  N_chains <- length(split_chain_vals)
  N <- sum(sapply(1:C, function(c) length(expectand_vals[c,])))

  means <- rep(0, N_chains)
  vars <- rep(0, N_chains)

  for (c in 1:N_chains) {
    summary <- welford_summary(split_chain_vals[[c]])
    means[c] <- summary[1]
    vars[c] <- summary[2]
  }

  total_mean <- sum(means) / N_chains
  W = sum(vars) / N_chains
  B = N * sum(sapply(means, function(m)
                            (m - total_mean)**2)) / (N_chains - 1)

  rhat = NaN
  if (abs(W) > 1e-10)
    rhat = sqrt( (N - 1 + B / W) / N )

  (rhat)
}

# Compute split hat{R} for all input expectands.
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @return Array of split Rhat estimates.
compute_split_rhats <- function(expectand_vals_list) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')

  rhats <- c()
  for (name in names(expectand_vals_list)) {
    expectand_vals <- expectand_vals_list[[name]]
    rhats <- c(rhats, compute_split_rhat(expectand_vals))
  }
  return(rhats)
}

# Check split hat{R} across the given expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
check_rhat <- function(expectand_vals, max_width=72) {
  validate_array(expectand_vals, 'expectand_vals')

  rhat <- compute_split_rhat(expectand_vals)

  no_warning <- TRUE
  message <- ""

  if (is.nan(rhat)) {
    message <- paste0(message, 
                      'All Markov chains appear to be frozen.\n')
  } else if (rhat > 1.1) {
    message <- paste0(message, sprintf('Split hat{R} is %f.\n', rhat))
    no_warning <- FALSE
  }
  
  if (no_warning) {
    desc <- 'Markov chain behavior is consistent with equilibrium.\n\n'
    message <- paste0(message, desc)
  } else {
    desc <- paste0('Split hat{R} larger than 1.1 suggests that at ',
                   'least one of the Markov chains has not reached ',
                   'an equilibrium.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  cat(message)
}

# Compute empirical integrated autocorrelation time, \hat{tau}, for a
# sequence of expectand values.
# @param vals A one-dimensional array of sequential expectand values.
# @return Left and right shape estimators.
compute_tau_hat <- function(vals) {
  # Compute empirical autocorrelations
  N <- length(vals)
  zs <- vals - mean(vals)
  
  if (var(vals) < 1e-10)
    return(Inf)

  B <- 2**ceiling(log2(N)) # Next power of 2 after N
  zs_buff <- c(zs, rep(0, B - N))

  Fs <- fft(zs_buff)
  Ss <- Fs * Conj(Fs)
  Rs <- fft(Ss, inverse=TRUE)

  acov_buff <- Re(Rs)
  rhos <- head(acov_buff, N) / acov_buff[1]

  # Drop last lag if (L + 1) is odd so that the lag pairs are complete
  L <- N
  if ((L + 1) %% 2 == 1)
    L <- L - 1

  # Number of lag pairs
  P <- (L + 1) / 2

  # Construct asymptotic correlation from initial monotone sequence
  old_pair_sum <- rhos[1] + rhos[2]
  for (p in 2:P) {
    current_pair_sum <- rhos[2 * p - 1] + rhos[2 * p]
  
    if (current_pair_sum < 0) {
      rho_sum <- sum(rhos[2:(2 * p)])
    
      if (rho_sum <= -0.25)
        rho_sum <- -0.25
    
      asymp_corr <- 1.0 + 2 * rho_sum
      return (asymp_corr)
    }
  
    if (current_pair_sum > old_pair_sum) {
      current_pair_sum <- old_pair_sum
      rhos[2 * p - 1] <- 0.5 * old_pair_sum
      rhos[2 * p] <- 0.5 * old_pair_sum
    }
  
    if (p == P) {
      return (NaN)
    }
  
    old_pair_sum <- current_pair_sum
  }
}

# Check the incremental empirical integrated autocorrelation time for
# all the given expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_width Maximum line width for printing
check_inc_tau_hat <- function(expectand_vals,
                              max_width=72) {
  validate_array(expectand_vals, 'expectand_vals')

  C <- dim(expectand_vals)[1]
  S <- dim(expectand_vals)[2]

  no_warning <- TRUE
  message <- ""

  for (c in 1:C) {
    tau_hat <- compute_tau_hat(expectand_vals[c,])
    inc_tau_hat <- tau_hat / S
    if (inc_tau_hat > 5) {
      body <- print0('  Chain %s: The incremental empirical ',
                     'integrated autocorrelation time %.3f ',
                     'is too large.\n')
      message <- paste0(message, sprintf(body, c, inc_tau_hat))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    desc <- paste0('The incremental empirical integrated ',
                   'autocorrelation time is small enough for the ',
                   'empirical autocorrelation estimates to be ',
                   'reliable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  } else {
    desc <- paste0('If the incremental empirical integrated ',
                   'autocorrelation times are too large then the ',
                   'Markov chains have not explored long enough for ',
                   'the autocorrelation estimates to be reliable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }

  cat(message)
}


# Compute the minimum empirical effective sample size, or \hat{ESS},
# across the Markov chains for the given expectands.
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
compute_min_ess_hats <- function(expectand_vals_list) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')

  min_ess_hats <- c()
  for (name in names(expectand_vals_list)) {
    expectand_vals <- expectand_vals_list[[name]]
    C <- dim(expectand_vals)[1]
    S <- dim(expectand_vals)[2]
    
    ess_hats <- rep(0, C)
    for (c in 1:C) {
      tau_hat <- compute_tau_hat(expectand_vals[c,])
      ess_hats[c] <- S / tau_hat
    }
    min_ess_hats <- c(min_ess_hats, min(ess_hats))
  }
  return(min_ess_hats)
}

# Check the empirical effective sample size \hat{ESS} for the given
# expectand values.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param min_ess_hat_per_chain The minimum empirical effective sample
#                              size before a warning message is passed.
# @param max_width Maximum line width for printing
check_ess_hat <- function(expectand_vals,
                          min_ess_hat_per_chain=100,
                          max_width=72) {
  validate_array(expectand_vals, 'expectand_vals')

  C <- dim(expectand_vals)[1]
  S <- dim(expectand_vals)[2]
  
  no_warning <- TRUE
  message <- ""
  
  for (c in 1:C) {
    tau_hat <- compute_tau_hat(expectand_vals[c,])
    ess_hat <- S / tau_hat
    if (ess_hat < min_ess_hat_per_chain) {
      body <- paste0('  Chain %s: The empirical effective sample size',
                     '%.3f is too small.\n')
      message <- paste0(message, sprintf(body, c, ess_hat))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    desc <- paste0('Assuming that a central limit theorem holds the ',
                   'empirical effective sample size is large enough ',
                   'for Markov chain Monte Carlo estimation to be',
                   'reasonably precise.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  } else {
    desc <- paste0('Small empirical effective sample sizes result in ',
                   'imprecise Markov chain Monte Carlo estimators.\n\n')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  cat(message)
}

# Check all expectand-specific diagnostics.
# @param expectand_vals_list A named list of two-dimensional arrays for
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
check_all_expectand_diagnostics <- function(expectand_vals_list,
                                            min_ess_hat_per_chain=100,
                                            exclude_zvar=FALSE,
                                            max_width=72) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')
  
  no_xi_hat_warning <- TRUE
  no_zvar_warning <- TRUE
  no_rhat_warning <- TRUE
  no_inc_tau_hat_warning <- TRUE
  no_ess_hat_warning <- TRUE

  message <- ""

  for (name in names(expectand_vals_list)) {
    if (is.null(expectand_vals_list[[name]])) {
      cat(sprintf('The values for expectand `%s` are ill-formed.',
          name))
      next
    }
    
    expectand_vals <- expectand_vals_list[[name]]
    C <- dim(expectand_vals)[1]
    S <- dim(expectand_vals)[2]
    
    local_warning <- FALSE
    local_message <- paste0(name, ':\n')
  
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(expectand_vals[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }
  
    for (c in 1:C) {
      vals <- expectand_vals[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(vals)
      if ( is.nan(xi_hats[1]) & is.nan(xi_hats[2]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Both left and right hat{xi}s are NaN.\n'
        local_message <- paste0(local_message, sprintf(body, c))
      }
      else if ( is.nan(xi_hats[1]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Left hat{xi} is NaN.\n'
        local_message <- paste0(local_message, sprintf(body, c))
      } else if ( is.nan(xi_hats[2]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Right hat{xi} is NaN.\n'
        local_message <- paste0(local_message, sprintf(body, c))
      } else if (xi_hats[1] >= xi_hat_threshold & 
          xi_hats[2] >= xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- paste0('  Chain %s: Both left and right tail hat{xi}s ',
                       '(%.3f, %.3f) exceed %.2f.\n')
        local_message <- paste0(local_message,
                                sprintf(body, c,
                                        xi_hats[1], xi_hats[2],
                                        xi_hat_threshold))
      } else if (xi_hats[1] < xi_hat_threshold & 
                 xi_hats[2] >= xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Right tail hat{xi} (%.3f) exceeds %.2f.\n'
        local_message <- paste0(local_message,
                                sprintf(body, c,
                                        xi_hats[2],
                                        xi_hat_threshold))
      } else if (xi_hats[1] >= xi_hat_threshold & 
                 xi_hats[2] < xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Left tail hat{xi} (%.3f) exceeds %.2f.\n'
        local_message <- paste0(local_message,
                                sprintf(body, c,
                                        xi_hats[1],
                                        xi_hat_threshold))
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(vals)[2]
      if (var < 1e-10) {
        no_zvar_warning <- FALSE
        local_warning <- TRUE
        body <- '  Chain %s: Empirical variance is effectively zero.\n'
        local_message <- paste0(local_message, sprintf(body, c))
      }
    }
  
    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(expectand_vals)

    if (is.nan(rhat)) {
      local_message <- paste0(local_message,
                              '  Split hat{R} is ill-defined.\n')
    } else if (rhat > 1.1) {
      no_rhat_warning <- FALSE
      local_warning <- TRUE
      body <- '  Split hat{R} (%.3f) exceeds 1.1.\n'
      local_message <- paste0(local_message, sprintf(body, rhat))
    }

    for (c in 1:C) {
      vals <- expectand_vals[c,]
      tau_hat <- compute_tau_hat(vals)

      # Check incremental empirical integrated autocorrelation time
      inc_tau_hat <- tau_hat / S

      if (inc_tau_hat > 5) {
        no_inc_tau_hat_warning <- FALSE
        local_warning <- TRUE
        body <- paste0('  Chain %s: Incremental hat{tau} (%.3f) is ',
                       'too large.\n')
        local_message <- paste0(local_message,
                                sprintf(body, inc_tau_hat))
      }

      # Check empirical effective sample size
      ess_hat <- S / tau_hat
      
      if (ess_hat < min_ess_hat_per_chain) {
        no_ess_hat_warning <- FALSE
        local_warning <- TRUE
        body <- paste0('  Chain %s: hat{ESS} (%.3f) is smaller than ',
                       'desired (%s).\n')
        local_message <- paste0(local_message,
                                sprintf(body, c, ess_hat,
                                        min_ess_hat_per_chain))
      }
    }
    
    if (local_warning) {
      message <- paste0(message, local_message, '\n')
    }
  }

  if (!no_xi_hat_warning) {
    desc <- paste0('Large tail hat{xi}s suggest that the expectand ',
                   'might not be sufficiently integrable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc, '\n')
  }
  if (!no_zvar_warning) {
    desc <- paste0('If the expectands are not constant then zero ',
                   'empirical variance suggests that the Markov ',
                   'transitions may be misbehaving.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc, '\n')
  }
  if (!no_rhat_warning) {
    desc <- paste0('Split Rhat larger than 1.1 suggests that at ',
                   'least one of the Markov chains has not reached ',
                   'an equilibrium.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc, '\n')
  }
  if (!no_inc_tau_hat_warning) {
    desc <- paste0('If the incremental empirical integrated ',
                   'autocorrelation times are too large then the ',
                   'Markov chains have not explored long enough for',
                   'the autocorrelation estimates to be reliable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc)
  }
  if (!no_ess_hat_warning) {
    desc <- paste0('Small empirical effective sample sizes result in ',
                   'imprecise Markov chain Monte Carlo estimators.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc)
  }

  if(no_xi_hat_warning & no_zvar_warning & 
     no_rhat_warning & no_inc_tau_hat_warning & no_ess_hat_warning) {
    desc <- paste0('All expectands checked appear to be behaving ',
                   'well enough for reliable Markov chain Monte ',
                   'Carlo estimation.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  }

  cat(message)
}

# Summarize all expectand-specific diagnostics.
# @param expectand_vals_list A named list of two-dimensional arrays for
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
summarize_expectand_diagnostics <- function(expectand_vals_list,
                                            min_ess_hat_per_chain=100,
                                            exclude_zvar=FALSE,
                                            max_width=72) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')

  failed_names <- c()
  failed_xi_hat_names <- c()
  failed_zvar_names <- c()
  failed_rhat_names <- c()
  failed_inc_tau_hat_names <- c()
  failed_ess_hat_names <- c()

  for (name in names(expectand_vals_list)) {
    if (is.null(expectand_vals_list[[name]])) {
      cat(sprintf('The values for expectand `%s` are ill-formed.',
                  name))
      next
    }
    
    expectand_vals <- expectand_vals_list[[name]]
    C <- dim(expectand_vals)[1]
    S <- dim(expectand_vals)[2]
    
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(expectand_vals[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }

    for (c in 1:C) {
      vals <- expectand_vals[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(vals)
      if ( is.nan(xi_hats[1]) | is.nan(xi_hats[2]) ) {
        failed_names <- c(failed_names, name)
        failed_xi_hat_nameas <- c(failed_xi_hat_names, name)
      } else if (xi_hats[1] >= xi_hat_threshold | 
                 xi_hats[2] >= xi_hat_threshold) {
        failed_names <- c(failed_names, name)
        failed_xi_hat_nameas <- c(failed_xi_hat_names, name)
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(vals)[2]
      if (var < 1e-10) {
        failed_names <- c(failed_names, name)
        failed_zvar_names <- c(failed_zvar_names, name) 
      }
    }

    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(expectand_vals)

    if (is.nan(rhat)) {
      failed_names <- c(failed_names, name)
      failed_rhat_names <- c(failed_rhat_names, name)
    } else if (rhat > 1.1) {
      failed_names <- c(failed_names, name)
      failed_rhat_names <- c(failed_rhat_names, name)
    }

    for (c in 1:C) {
      tau_hat <- compute_tau_hat(expectand_vals[c,])

      # Check incremental empirical integrated autocorrelation time
      inc_tau_hat <- tau_hat / S

      if (inc_tau_hat > 5) {
        failed_names <- c(failed_names, name)
        failed_inc_tau_hat_names <- c(failed_inc_tau_hat_names, name)
      }

      # Check empirical effective sample size
      ess_hat <- S / tau_hat
      
      if (ess_hat < min_ess_hat_per_chain) {
        failed_names <- c(failed_names, name)
        failed_ess_hat_names <- c(failed_ess_hat_names, name)
      }
    }
  }
  
  message <- ""
  
  failed_names <- unique(failed_names)
  if (length(failed_names)) {
    desc <- 
      sprintf('The expectands %s triggered diagnostic warnings.\n\n',
              paste(failed_names, collapse=", "))
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  } else {
    desc <- paste0('All expectands checked appear to be behaving ',
                   'well enough for reliable Markov chain Monte ',
                   'Carlo estimation.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  }

  failed_xi_hat_names <- unique(failed_xi_hat_names)
  if (length(failed_xi_hat_names)) {
    desc <- 
      paste0(sprintf('The expectands %s triggered hat{xi} warnings.\n\n',
             paste(failed_xi_hat_names, collapse=", ")),
             '  Large tail hat{xi}s suggest that the expectand ',
             'might not be sufficiently integrable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }
        
  failed_zvar_names <- unique(failed_zvar_names)
  if (length(failed_zvar_names)) { 
    desc <- 
      paste0(sprintf('The expectands %s triggered zero variance warnings.\n\n',
             paste(failed_zvar_names, collapse=", ")),
             '  If the expectands are not constant then zero ',
             'empirical variance suggests that the Markov ',
             'transitions may be misbehaving.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }
  
  failed_rhat_names <- unique(failed_rhat_names)
  if (length(failed_rhat_names)) {
    desc <- 
      paste0(sprintf('The expectands %s triggered hat{R} warnings.\n\n',
             paste(failed_rhat_names, collapse=", ")),
             '  Split Rhat larger than 1.1 suggests that at ',
             'least one of the Markov chains has not reached ',
             'an equilibrium.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }
  
  failed_inc_tau_hat_names <- unique(failed_inc_tau_hat_names)
  if (length(failed_inc_tau_hat_names)) {
    desc <-
      body <- paste0('The expectands %s triggered incremental ',
                     'hat{tau} warnings.\n\n')
      paste0(sprintf(body,
                     paste(failed_inc_tau_hat_names, collapse=", ")),
             'If the incremental empirical integrated autocorrelation ',
             'times are too large then the Markov chains have not ',
             'explored long enough for the autocorrelation estimates ',
             'to be reliable.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }

  failed_ess_hat_names <- unique(failed_ess_hat_names)
  if (length(failed_ess_hat_names)) {
    desc <- 
      paste0(sprintf('The expectands %s triggered hat{ESS} warnings.\n\n',
             paste(failed_ess_hat_names, collapse=", ")),
             'Small empirical effective sample sizes result in ',
             'imprecise Markov chain Monte Carlo estimators.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }
  
  cat(message)
}

# Summarize Hamiltonian Monte Carlo and expectand diagnostics
# into a binary encoding
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param diagnostics A named list of two-dimensional arrays for
#                    each diagnostic.  The first dimension of each
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
encode_all_diagnostics <- function(expectand_vals_list,
                                   diagnostics,
                                   adapt_target=0.801,
                                   max_treedepth=10,
                                   min_ess_hat_per_chain=100,
                                   exclude_zvar=FALSE) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')
  validate_named_list_of_arrays(diagnostics, 'diagnostics')

  warning_code <- 0
  C <- dim(diagnostics[['divergent__']])[1]
  
  # Check divergences
  n = sum(sapply(1:C, function(c) diagnostics[['divergent__']][c,]))
  if (n > 0) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 0))
  }

  # Check transitions that ended prematurely due to maximum tree depth 
  # limit
  n = sum(sapply(1:C, function(c) 
                      diagnostics[['treedepth__']][c,] >= max_treedepth))

  if (n > 0) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 1))
  }

  # Checks the energy fraction of missing information (E-FMI)
  no_efmi_warning <- TRUE
  no_accept_warning <- TRUE
  
  for (c in 1:C) {
    # Check the energy fraction of missing information (E-FMI)
    energies = diagnostics[['energy__']][c,]
    numer = sum(diff(energies)**2) / length(energies)
    denom = var(energies)
    efmi <- numer / denom
    if (efmi < 0.2) {
      no_efmi_warning <- FALSE
    }
  
    # Check convergence of the stepsize adaptation
    ave_accept_proxy <- mean(diagnostics[['accept_stat__']][c,])
    if (ave_accept_proxy < 0.9 * adapt_target) {
      no_accept_warning <- FALSE
    }
  }

  if (!no_efmi_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 2))
  }

  if (!no_accept_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 3))
  }
  
  zvar_warning <- FALSE
  xi_hat_warning <- FALSE
  rhat_warning <- FALSE
  inc_tau_hat_warning <- FALSE
  ess_hat_warning <- FALSE
  
  for (name in names(expectand_vals_list)) {
    if (is.null(expectand_vals_lsit[[name]])) {
      cat(sprintf('The values for expectand `%s` are ill-formed.', name))
      continue
    }
    
    expectand_vals <- expectand_vals_list[[name]]
    C <- dim(expectand_vals)[1]
    S <- dim(expectand_vals)[2]
  
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(expectand_vals[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }
  
    for (c in 1:C) {
      vals <- expectand_vals[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(vals)
      if (is.nan(xi_hats[1]) | is.nan(xi_hats[2])) {
        xi_hat_warning <- TRUE
      } else if (xi_hats[1] >= xi_hat_threshold | 
          xi_hats[2] >= xi_hat_threshold) {
        xi_hat_warning <- TRUE
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(vals)[2]
      if (var < 1e-10) {
        zvar_warning <- TRUE
      }

      # Check empirical integrated autocorrelation time
      tau_hat <- compute_tau_hat(vals)
      inc_tau_hat <- tau_hat /  S

      if (int_tau_hat > 5) {
        inc_tau_hat_warning <- TRUE
      }

      # Check empirical effective sample size
      ess_hat <- S / tau_hat

      if (ess_hat < min_ess_hat_per_chain) {
        ess_hat_warning <- TRUE
      }
    }
  
    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(expectand_vals)

    if (is.nan(rhat)) {
      rhat_warning <- TRUE
    } else if (rhat > 1.1) {
      rhat_warning <- TRUE
    }
  }  
  
  if (xi_hat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 4))
  }
        
  if (zvar_warning) { 
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 5))
  }
  
  if (rhat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 6))
  }
  
  if (inc_tau_hat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 7))
  }

  if (ess_hat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 8))
  }
  
  (warning_code)
}

# Translate binary diagnostic codes to human readable output.
# @params warning_code An eight bit binary summary of the diagnostic 
#                      output.
decode_warning_code <- function(warning_code) {
  if (bitwAnd(warning_code, bitwShiftL(1, 0)))
    print("  divergence warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 1)))
    print("  treedepth warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 2)))
    print("  E-FMI warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 3)))
    print("  average acceptance proxy warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 4)))
    print("  xi_hat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 5)))
    print("  zero variance warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 6)))
    print("  Rhat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 7)))
    print("  incremental tau_hat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 7)))
    print("  min ess_hat warning")
}

# Filter `expectand_vals_list` object by name.
# @param expectand_vals_list A named list of two-dimensional arrays for
#                            each expectand.  The first dimension of
#                            each element indexes the Markov chains and
#                            the second dimension indexes the sequential
#                            states within each Markov chain.
# @param requested_names Expectand names to keep.
# @param check_arrays Binary variable indicating whether or not
#                     requested
#                     names should be expanded to array components.
# @param max_width Maximum line width for printing
# @return A named list of two-dimensional arrays for each requested
#         expectand.
filter_expectands <- function(expectand_vals_list, requested_names,
                              check_arrays=FALSE, max_width=72) {
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')
  
  if (length(requested_names) == 0) {
    stop('Input variable requested_names must be non-empty.')
  }
  
  if (check_arrays == TRUE) {
    good_names <- c()
    bad_names <- c()
    for (name in requested_names) {
      # Search for array suffix
      array_names <- grep(paste0('^', name, '\\['),
                          names(expectand_vals_list),
                          value=TRUE)
      
      # Append array names, if found
      if (length(array_names) > 0) {
        good_names <- c(good_names, array_names)
      } else {
        if (name %in% names(expectand_vals_list)) {
          # Append bare name, if found
          good_names <- c(good_names, name)
        }  else {
          # Add to list of bad names
          bad_names <- c(bad_names, name)
        }
      }
    }
  } else {
    bad_names <- setdiff(requested_names, names(expectand_vals_list))
    good_names <- intersect(requested_names, names(expectand_vals_list))
  }
    
  if (length(bad_names) == 1) {
    message <- paste0(sprintf('The expectand %s ',
                              paste(bad_names, collapse=", ")),
                      'was not found in the expectand_vals_list ',
                      'object and will be ignored.\n\n')
    message <- paste0(strwrap(message, max_width, 0), collapse='\n')
    cat(message)
  } else if (length(bad_names) > 1) {
    message <- paste0(sprintf('The expectands %s ',
                              paste(bad_names, collapse=", ")),
                      'were not found in the expectand_vals_list ',
                      'object and will be ignored.\n\n')
    message <- paste0(strwrap(message, max_width, 0), collapse='\n')
    cat(message)
  }
  
  expectand_vals_list[good_names]
}

# Compute empirical autocorrelations for a given Markov chain sequence
# @parmas vals A one-dimensional array of sequential expectand values.
# @return A one-dimensional array of empirical autocorrelations at each
#         lag up to the length of the sequence.
compute_rhos <- function(vals) {
  # Compute empirical autocorrelations
  N <- length(vals)
  zs <- vals - mean(vals)
  
  if (var(vals) < 1e-10)
    return(rep(1, N))

  B <- 2**ceiling(log2(N)) # Next power of 2 after N
  zs_buff <- c(zs, rep(0, B - N))

  Fs <- fft(zs_buff)
  Ss <- Fs * Conj(Fs)
  Rs <- fft(Ss, inverse=TRUE)

  acov_buff <- Re(Rs)
  rhos <- head(acov_buff, N) / acov_buff[1]

  # Drop last lag if (L + 1) is odd so that the
  # lag pairs are complete
  L <- N
  if ((L + 1) %% 2 == 1)
    L <- L - 1

  # Number of lag pairs
  P <- (L + 1) / 2

  # Construct asymptotic correlation from initial monotone sequence
  old_pair_sum <- rhos[1] + rhos[2]
  max_L <- N

  for (p in 2:P) {
    current_pair_sum <- rhos[2 * p - 1] + rhos[2 * p]
  
    if (current_pair_sum < 0) {
      max_L <- 2 * p
      rhos[(max_L + 1):N] <- 0
      break
    }
  
    if (current_pair_sum > old_pair_sum) {
      current_pair_sum <- old_pair_sum
      rhos[2 * p - 1] <- 0.5 * old_pair_sum
      rhos[2 * p] <- 0.5 * old_pair_sum
    }
  
    old_pair_sum <- current_pair_sum
  }
  return(rhos)
}

# Plot empirical correlograms for a given expectand across a Markov 
# chain ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param max_L Maximum autocorrelation lag
# @param rho_lim Plotting range of autocorrelation values
# @display_name Name of expectand
plot_empirical_correlogram <- function(expectand_vals,
                                       max_L,
                                       rho_lim=c(-0.2, 1.1),
                                       display_name="") {
  validate_array(expectand_vals, 'expectand_vals')
  C <- dim(expectand_vals)[1]
  
  idx <- rep(0:max_L, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  plot(0, type="n", main=display_name,
       xlab="Lag", xlim=c(-0.5, max_L + 0.5),
       ylab="Empirical Autocorrelation", ylim=rho_lim)
  abline(h=0, col="#DDDDDD", lty=2, lwd=2)

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)
  for (c in 1:C) {
    rhos <- compute_rhos(expectand_vals[c,])
    pad_rhos <- unlist(lapply(idx, function(n) rhos[n + 1]))
    lines(xs, pad_rhos, lwd=2, col=colors[c])
  }
}

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
#                        sequential states within each Markov chain.
# @params display_name2 Name of second expectand
plot_pairs_by_chain <- function(expectand1_vals, display_name1,
                                expectand2_vals, display_name2) {
  validate_array(expectand1_vals, 'expectand1_vals')
  C1 <- dim(expectand1_vals)[1]
  S1 <- dim(expectand1_vals)[2]

  validate_array(expectand2_vals, 'expectand2_vals')
  C2 <- dim(expectand2_vals)[1]
  S2 <- dim(expectand2_vals)[2]
  
  if (C1 != C2) {
    C <- min(C1, C2)
    C1 <- C
    C2 <- C
    cat(sprintf('Plotting only %s Markov chains.\n', C))
  }

  nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C",
                  "#A25050", "#8F2727", "#7C0000")
  cmap <- colormap(colormap=nom_colors, nshades=max(S1, S2))

  min_x <- min(sapply(1:C1, function(c) min(expectand1_vals[c,])))
  max_x <- max(sapply(1:C1, function(c) max(expectand1_vals[c,])))

  min_y <- min(sapply(1:C2, function(c) min(expectand2_vals[c,])))
  max_y <- max(sapply(1:C2, function(c) max(expectand2_vals[c,])))

  par(mfrow=c(2, 2), mar = c(5, 5, 3, 1))

  for (c in 1:C1) {
    plot(0, type="n", main=paste("Chain", c),
         xlab=display_name1, xlim=c(min_x, max_x),
         ylab=display_name2, ylim=c(min_y, max_y))
  
    points(unlist(lapply(1:C1, function(c) expectand1_vals[c,])),
           unlist(lapply(1:C2, function(c) expectand2_vals[c,])),
           col="#DDDDDD", pch=16, cex=1.0)
    points(expectand1_vals[c,], expectand2_vals[c,],
           col=cmap, pch=16, cex=1.0)
  }
}

# Evaluate an expectand on the values of a one-dimensional input
# variable.
# @param input_vals A two-dimensional array of expectand values with
#                   the first dimension indexing the Markov chains
#                   and the second dimension indexing the sequential
#                   states within each Markov chain.
# @param expectand Expectand with one-dimensional input space.
# @return A two-dimensional array of expectand values with the 
#         first dimension indexing the Markov chains and the 
#         second dimension indexing the sequential states within 
#         each Markov chain.
eval_uni_expectand_pushforward <- function(input_vals, expectand) {
  if (dim(input_vals)[1] == 1) {
    as.matrix(t(apply(input_vals, 2, expectand)))
  } else {
    apply(input_vals, 2, expectand)
  }
}

# Recursively create vector of element names, including indexing
# information, with column-major ordering from the specified dimensions.
# For example `elem_names('x', c(2, 3)))` returns
# >>  "x[1,1]" "x[2,1]" "x[1,2]" "x[2,2]" "x[1,3]" "x[2,3]"
#
# @ param base Base name.
# @ param dims Vector of array dimensions.
# @ param current_idxs Dimensions at current level of recursion.
# @ return Vector of element names in column-major order.
elem_names <- function(base, dims, current_idxs=c()) {
  next_dim <- length(dims) - length(current_idxs)
  if (next_dim == 0) {
    name <- do.call(paste0, lapply(current_idxs,
                                   function(idx)
                                   paste0(as.character(idx), ',')))
    name <- paste0(base, '[', gsub(',$', '', name), ']')
    return(name)
  } else {
    names <- c()
    for (d in 1:dims[next_dim]) {
      names <- c(names, Recall(base, dims, c(d, current_idxs)))
    }
    return(names)
  }
}

# Create array of element names from the specified dimensions.
# For example `name_array('x', c(2, 3)))` returns the array
# >>      [,1]     [,2]     [,3]
# >> [1,] "x[1,1]" "x[1,2]" "x[1,3]"
# >> [2,] "x[2,1]" "x[2,2]" "x[2,3]"
# @ param base Base name.
# @ param dims Vector of array dimensions.
# @ return Array of element names with dimensions given by dims.
name_array <- function(base, dims) {
  # Validate inputs
  if ( !is.character(base) ) {
    stop(paste0('Input variable ', base, ' is not a character.'))
  }

  if ( !is.vector(dims) | !is.numeric(dims) ) {
    stop(paste0('Input variable ', dims, ' is not a numeric vector.'))
  }

  # Create element names and format them into desired array.
  # Assumes that R arrays are stored in column-major order.
  names <- elem_names(base, dims)
  array(names, dims)
}

# Evaluate a list of expectands on the values of an arbitrary number of
# input variables.  Expectands must all return a scalar numeric or
# logical output.
#
# By default expectand argument values are accessed by name
# in expectand_vals_list.  If a non-null alt_arg_names is provided then
# the alternate names are used to access values in expectand_vals_list.
# The elements of alt_arg_names can also be character arrays of
# arbitrary dimension in which case the individual element values are
# first accessed then formatted into matching numeric arrays before
# being passed to the expectand.
#
# @param expectand_vals_list A named list of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param expectand_list List of functions with the same arguments.
# @param alt_arg_names Optional named list of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return A list of two-dimensional arrays, one for each element of
#         expectand_list with the same names as expectand_list.
eval_expectand_pushforwards <- function(expectand_vals_list,
                                        expectand_list,
                                        alt_arg_names=NULL) {
  # Validate inputs
  validate_named_list_of_arrays(expectand_vals_list,
                                'expectand_vals_list')

  if (!is.list(expectand_list)) {
    stop(paste0('Input variable `expectand_list` is not a list.'))
  }


  if (!Reduce("&", sapply(expectand_list, is.function))) {
    stop(paste0('The elements of input variable `expectand_list` ',
                'are not all functions.'))
  }

  if (!is.null(alt_arg_names)) {
    if ( !is.list(alt_arg_names) |
          is.null(names(alt_arg_names)) ) {
      stop(paste0('Input variable `alt_arg_names` ',
                  'is not a named list.'))
    }
  }

  # Check consistent of expectand arguments
  nominal_arg_names <- formalArgs(expectand_list[[1]])

  compare_args <- function(e) {
    Reduce("&", formalArgs(e) == nominal_arg_names)
  }
  arg_consistency <- Reduce("&", sapply(expectand_list, compare_args))
  if(!arg_consistency) {
    stop(paste0('The arguments of the functions in `expectand_list` ',
                'are not consistent with each other.'))
  }

  # Ensure that all argument replacements are arrays
  alt_arg_names_array = list()
  for (name in names(alt_arg_names)) {
    alt_arg_names_array[[name]] <- as.array(alt_arg_names[[name]])
  }

  # Check existence of all expectand arguments
  if (is.null(alt_arg_names)) {
    check_arg_names <- nominal_arg_names
  } else {
    missing_args <- setdiff(nominal_arg_names, names(alt_arg_names))
    if (length(missing_args) == 1) {
      stop(paste0('The nominal expectand argument ',
                  paste(missing_args, collapse=", "),
                  ' does not have a replacement in ',
                  '`alt_arg_names`.'))
    } else if (length(missing_args) > 1) {
      stop(paste0('The nominal expectand arguments ',
                  paste(missing_args, collapse=", "),
                  ' do not have replacements in ',
                  '`alt_arg_names`.'))
    }

    check_arg_names <- c(sapply(alt_arg_names,
                                function(alt) as.list(alt)),
                         recursive=TRUE)
  }

  missing_args <- setdiff(check_arg_names, names(expectand_vals_list))
  if (length(missing_args) == 1) {
    stop(paste0('The expectand argument ',
                paste(missing_args, collapse=", "),
                ' is not in `expectand_vals_list`.'))
  } else if (length(missing_args)) {
    stop(paste0('The expectand arguments ',
                paste(missing_args, collapse=", "),
                ' are not in `expectand_vals_list`.'))
  }

  # Apply expectand to all inputs
  C <- dim(expectand_vals_list[[1]])[1]
  S <- dim(expectand_vals_list[[1]])[2]
  pushforward_vals <- lapply(expectand_list,
                             function(e) matrix(NA, nrow=C, ncol=S))

  if (!is.null(alt_arg_names))
    alt_names <- lapply(nominal_arg_names,
                        function(name) alt_arg_names_array[[name]])

  expectand_vals_env <- as.environment(expectand_vals_list)
  access_val <- function(name) {
    expectand_vals_env[[name]][c, s]
  }

  for (c in 1:C) {
    for (s in 1:S) {
      if (is.null(alt_arg_names)) {
        arg_vals <- lapply(nominal_arg_names, access_val)
      } else {
        arg_vals <- vector("list", length=length(alt_names))
        for (n in seq_along(arg_vals)) {
          arg_vals[[n]] <- apply(alt_names[[n]],
                                 seq_along(dim(alt_names[[n]])),
                                 access_val)
        }
      }
      for (i in seq_along(pushforward_vals)) {
        pushforward_vals[[i]][c, s] <-
          as.numeric(do.call(expectand_list[[i]], arg_vals))
      }
    }
  }

  return(pushforward_vals)
}

# Evaluate an expectand on the values of an arbitrary number of input
# variables.  Expectand must return a scalar numeric or logical output.
#
# By default expectand argument values are accessed by name
# in expectand_vals_list.  If a non-null alt_arg_names is provided then
# the alternate names are used to access values in expectand_vals_list.
# The elements of alt_arg_names can also be character arrays of
# arbitrary dimension in which case the individual element values are
# first accessed then formatted into matching numeric arrays before
# being passed to the expectand.
#
# @param expectand_vals_list A named list of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param expectand Function with arbitrary number of scalar and array
#                  input arguments.
# @param alt_arg_names Optional named list of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return A two-dimensional array.
eval_expectand_pushforward <- function(expectand_vals_list,
                                       expectand,
                                       alt_arg_names=NULL) {
  pushforward_vals <- eval_expectand_pushforwards(expectand_vals_list,
                                                  list(expectand),
                                                  alt_arg_names)
  pushforward_vals[[1]]
}

# Estimate expectand expectation value from a single Markov chain.
# @param vals A one-dimensional array of sequential expectand values.
# @return The Markov chain Monte Carlo estimate, its estimated standard 
#         error, and empirical effective sample size.
mcmc_est <- function(vals) {
  S <- length(vals)
  if (S == 1) {
    return(c(vals[1], 0, NaN))
  }

  summary <- welford_summary(vals)

  if (summary[2] == 0) {
    return(c(summary[1], 0, NaN))
  }

  tau_hat <- compute_tau_hat(vals)
  ess_hat <- S / tau_hat
  return(c(summary[1], sqrt(summary[2] / ess_hat), ess_hat))
}

# Estimate expectand expectation value from a Markov chain ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @return The ensemble Markov chain Monte Carlo estimate, its estimated
#         standard error, and empirical effective sample size.
ensemble_mcmc_est <- function(expectand_vals) {
  validate_array(expectand_vals, 'expectand_vals')
  
  C <- dim(expectand_vals)[1]
  chain_ests <- lapply(1:C, function(c) mcmc_est(expectand_vals[c,]))
  
  # Total effective sample size
  total_ess <- sum(sapply(chain_ests, function(est) est[3]))
  
  if (is.nan(total_ess)) {
    m <- mean(sapply(chain_ests, function(est) est[1]))
    se <- mean(sapply(chain_ests, function(est) est[2]))
    return (c(m, se, NaN))
  }
  
  # Ensemble average weighted by effective sample size
  mean <- sum(sapply(chain_ests,
                     function(est) est[3] * est[1])) / total_ess
  
  # Ensemble variance weighed by effective sample size
  # including correction for the fact that individual Markov chain
  # variances are defined relative to the individual mean estimators
  # and not the ensemble mean estimator
  vars <- rep(0, C)
  
  for (c in 1:C) {
    est <- chain_ests[[c]]
    chain_var <- est[3] * est[2]**2
    var_update <- (est[1] - mean)**2
    vars[c] <- est[3] * (var_update + chain_var)
  }
  var <- sum(vars) / total_ess
  
  c(mean, sqrt(var / total_ess), total_ess)
}


# Estimate the probability allocated to a subset implicitly defined by
# an indicator function.
# @param expectand_vals_list A named list of two-dimensional arrays.
#                            The first dimension of each element indexes
#                            the Markov chains and the second dimension
#                            indexes the sequential states within each
#                            Markov chain.
# @param indicator Function with logical or 0/1 numeric outputs.
# @param alt_arg_names Optional named list of alternate names for the
#                      nominal expectand argument names; when used all
#                      expectand argument names must be included.
# @return The probability estimate and its standard error.
implicit_subset_prob <- function(expectand_vals_list,
                                 indicator,
                                 alt_arg_names) {
  # Evaluate indicator function
  indicator_samples <- eval_expectand_pushforward(expectand_vals_list,
                                                  indicator,
                                                  alt_arg_names)

  # Verify outputs
  unique_vals <- unique(c(indicator_samples, recursive=TRUE))

  valid_outputs = TRUE
  if (length(unique_vals) == 1) {
    if (!(unique_vals == 0 | unique_vals == 1))
      valid_outputs = FALSE
  } else if (length(unique_vals) == 2) {
    if ( !(  Reduce("&", unique_vals == c(0, 1))
           | Reduce("&", unique_vals == c(1, 0))) )
      valid_outputs = FALSE
  } else {
    valid_outputs = FALSE
  }
  if(!valid_outputs)
    stop(paste0('The function `indicator` must return only ',
                'logical or 0/1 numeric outputs.'))

  # Return probability as expectation value of indicator function
  ensemble_mcmc_est(indicator_samples)[1:2]
}

# Estimate expectand pushforward quantiles from a Markov chain ensemble.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param probs An array of quantile probabilities.
# @return The ensemble Markov chain Monte Carlo quantile estimate.
ensemble_mcmc_quantile_est <- function(expectand_vals, probs) {
  # Validate inputs
  validate_array(expectand_vals, 'expectand_vals')

  if (!is.vector(probs)) {
    stop(paste0('Input variable `probs` is not a ',
                'one-dimensional numeric array.'))
  }

  # Estimate and return quantile
  q <- 0 * probs

  C <- dim(expectand_vals)[1]
  for (c in 1:C) {
    q <- q + quantile(expectand_vals[c,], probs=probs) / C
  }

  return(q)
}

# Visualize pushforward distribution along a given expectand as a
# sequence of bin probabilities weighted by bin widths that approximates
# the pushforward probability density function.  Markov chain Monte
# Carlo estimates the output bin probabilities from the input samples,
# with the bin probability estimator errors visualized in the border
# color.
# @param expectand_vals A two-dimensional array of expectand values with
#                       the first dimension indexing the Markov chains
#                       and the second dimension indexing the sequential
#                       states within each Markov chain.
# @param B The number of histogram bins
# @param display_name Expectand name
# @param flim Optional histogram range
# @param ylim Optional y-axis range; ignored if add is TRUE
# @param col Color for plotting weighted bin probabilities; defaults to
#            c_dark.
# @param border Color for plotting estimator error; defaults to gray
# @param add Configure plot to overlay over existing plot; defaults to
#            FALSE
# @param main Optional plot title
# @param baseline Optional baseline value for visual comparison
# @param baseline_col Color for plotting baseline value; defaults to
#                     "black"
plot_expectand_pushforward <- function(expectand_vals, B,
                                       display_name="f",
                                       flim=NULL, ylim=NULL,
                                       col=c_dark, border="#DDDDDD",
                                       add=FALSE, main="",
                                       baseline=NULL,
                                       baseline_col="black") {
  validate_array(expectand_vals, 'expectand_vals')

  # Automatically adjust histogram range to range of expectand values
  # if range is not already set as an input variable
  if (is.null(flim)) {
    min_f <- min(expectand_vals)
    max_f <- max(expectand_vals)
    delta <- (max_f - min_f) / B

    # Add bounding bins
    B <- B + 2
    min_f <- min_f - delta
    max_f <- max_f + delta
    flim <- c(min_f, max_f)

    bins <- seq(min_f, max_f, delta)
  } else {
    min_f <- flim[1]
    max_f <- flim[2]

    delta <- (max_f - min_f) / B
    bins <- seq(min_f, max_f, delta)
  }

  # Check value containment
  S <- dim(expectand_vals)[1] * dim(expectand_vals)[2]

  S_low <- sum(c(expectand_vals, recursive=TRUE) < min_f)
  if (S_low == 1)
    warning(sprintf('%i value (%.1f%%) fell below the histogram binning.',
                    S_low, 100 * S_low / S))
  else if (S_low > 1)
    warning(sprintf('%i values (%.1f%%) fell below the histogram binning.',
                    S_low, 100 * S_low / S))

  S_high <- sum(max_f < c(expectand_vals, recursive=TRUE))
  if (S_low == 1)
    warning(sprintf('%i value (%.1f%%) fell above the histogram binning.',
                    S_high, 100 * S_high / S))
  else if (S_low > 1)
    warning(sprintf('%i values (%.1f%%) fell above the histogram binning.',
                    S_high, 100 * S_high / S))

  # Compute bin heights
  mean_p <- rep(0, B)
  delta_p <- rep(0, B)

  for (b in 1:B) {
    # Estimate bin probabilities
    bin_indicator <- function(x) {
      ifelse(bins[b] <= x & x < bins[b + 1], 1, 0)
    }
    indicator_vals <- eval_uni_expectand_pushforward(expectand_vals,
                                                     bin_indicator)
    est <- ensemble_mcmc_est(indicator_vals)

    # Normalize bin probabilities by bin width to allow
    # for direct comparison to probability density functions
    width = bins[b + 1] - bins[b]
    mean_p[b] = est[1] / width
    delta_p[b] = est[2] / width
  }

  # Plot histogram
  idx <- rep(1:B, each=2)
  x <- sapply(1:length(idx), function(b) if(b %% 2 == 1) bins[idx[b]]
              else bins[idx[b] + 1])
  lower_inter <- sapply(idx, function (n)
    max(mean_p[n] - 2 * delta_p[n], 0))
  upper_inter <- sapply(idx, function (n)
    min(mean_p[n] + 2 * delta_p[n], 1 / width))

  if (add) {
    polygon(c(x, rev(x)), c(lower_inter, rev(upper_inter)),
            col=border, border=NA)
    lines(x, mean_p[idx], col=col, lwd=2)
  } else {
    if (is.null(ylim)) {
      ylim=c(0, max(1.05 * upper_inter))
    }

    plot(1, type="n", main=main,
         xlim=flim, xlab=display_name,
         ylim=ylim, ylab="", yaxt="n")
    title(ylab="Estimated Bin\nProbabilities / Bin Width",
          mgp=c(1, 1, 0))

    polygon(c(x, rev(x)), c(lower_inter, rev(upper_inter)),
            col=border, border=NA)
    lines(x, mean_p[idx], col=col, lwd=2)
  }

  # Plot baseline if applicable
  if (!is.null(baseline)) {
    abline(v=baseline, col="white", lty=1, lwd=4)
    abline(v=baseline, col=baseline_col, lty=1, lwd=2)
  }
}
