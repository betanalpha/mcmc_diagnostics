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
extract_expectands <- function(stan_fit) {
  nom_params <- rstan:::extract(stan_fit, permuted=FALSE)
  N <- dim(nom_params)[3] - 1
  params <- lapply(1:N, function(n) t(nom_params[,,n]))
  names(params) <- names(stan_fit)[1:N]
  (params)
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
#                    each expectand.  The first dimension of each
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
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
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
               sprintf('  Chain %s: %s of %s transitions (%s%%) ', 
                       c, n_tds, S, 100 * n_tds / S),
               sprintf('saturated the maximum treedepth of %s.\n', 
                       max_treedepth))
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
               sprintf('  Chain %s: Averge proxy acceptance ', c),
               sprintf('statistic (%.3f) is\n', ave_accept_proxy),
                       '           smaller than 90% of the target ',
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
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
display_stepsizes <- function(diagnostics) {
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
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
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
plot_num_leapfrogs <- function(diagnostics) {
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
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
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
plot_num_leapfrogs_by_chain <- function(diagnostics) {
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
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
#                    each expectand.  The first dimension of each
#                    element indexes the Markov chains and the 
#                    second dimension indexes the sequential 
#                    states within each Markov chain.
display_ave_accept_proxy <- function(diagnostics) {
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
  proxy_stats <- diagnostics[['accept_stat__']]
  C <- dim(proxy_stats)[1]

  for (c in 1:C) {
    ave_accept_proxy <- mean(proxy_stats[c,])
    cat(sprintf('Chain %s: Average proxy acceptance statistic = %.3f\n',
                c, ave_accept_proxy))
  }
}

# Apply transformation identity, log, or logit transformation to
# named samples and flatten the output.  Transformation defaults to 
# identity if name is not included in `transforms` dictionary.  A 
# ValueError is thrown if samples are not properly constrained.
# @param name Expectand name.
# @param samples A named list of two-dimensional arrays for 
#                each expectand.  The first dimension of each element 
#                indexes the Markov chains and the second dimension 
#                indexes the sequential states within each Markov chain.
# @param transforms A named list of transformation flags for each 
#                   expectand.
# @return The transformed expectand name and a one-dimensional array of
#         flattened transformation outputs.
apply_transform <- function(name, samples, transforms) {
  t <- transforms[[name]]
  if (is.null(t)) t <- 0
  
  transformed_name <- ""
  transformed_samples <- 0
 
  if (t == 0) {
    transformed_name <- name
    transformed_samples <- c(t(samples[[name]]), recursive=TRUE)
  } else if (t == 1) {
    if (min(samples[[name]]) <= 0) {
      cat(paste0('Log transform requested for expectand ',
                 sprintf('%s ', name),
                 'but expectand values are not strictly positive.'))
      return (NULL)
    }
    transformed_name <- paste0('log(', name, ')')
    transformed_samples <- log(c(t(samples[[name]]), recursive=TRUE))
  } else if (t == 2) {
    if (min(samples[[name]]) <= 0 | max(samples[[name]] >= 1)) {
      cat(paste0('Logit transform requested for expectand ',
                 sprintf('%s ' , name),
                 'but expectand values are not strictly confined ',
                 'to the unit interval.'))
      return (NULL)
    }
    transformed_name <- paste0('logit(', name, ')')
    transformed_samples <- sapply(c(t(samples[[name]]), recursive=TRUE), 
                                  function(x) log(x / 1 - x))
  }
  return (list('t_name' = transformed_name, 
               't_samples' = transformed_samples))
}

# Plot pairwise scatter plots with non-divergent and divergent 
# transitions separated by color
# @param x_names A list of expectand names to be plotted on the x axis.
# @param y_names A list of expectand names to be plotted on the y axis.
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
# @params transforms A named list of flags configurating which if any
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
                           expectand_samples, diagnostics, 
                           transforms=list(), xlim=NULL, ylim=NULL,
                           plot_mode=0, max_width=72) {
  if (!is.vector(x_names)) {
    cat('Input variable `x_names` is not a list!')
    return
  }
  
  if (!is.vector(y_names)) {
    cat('Input variable `y_names` is not a list!')
    return
  }
  
  if (!is.vector(expectand_samples)) {
    cat('Input variable `expectand_samples` is not a named list!')
    return
  }
  
  if (!is.vector(diagnostics)) {
    cat('Input variable `diagnostics` is not a named list!')
    return
  }
  
  if (!is.vector(transforms)) {
    cat('Input variable `transforms` is not a named list!')
    return
  }
  
  # Check transform flags
  for (name in names(transforms)) {
    if (transforms[[name]] < 0 | transforms[[name]] > 2) {
      warning <- 
        paste0(sprintf('The transform flag %s for expectand %s ', 
                       transforms[[name]], name),
               'is invalid.  Plot will default to no tranformation.')
      warning <- paste0(strwrap(warning, max_width, 0), collapse='\n')
      cat(warning)
    }
  }
  
  # Check plot mode
  if (plot_mode < 0 | plot_mode > 1) {
    cat(sprintf('Invalid `plot_mode` value %s.', plot_mode))
    return
  }
  
  # Transform expectand samples
  transformed_samples = list()
  
  transformed_x_names <- c()
  for (name in x_names) {
    r <- apply_transform(name, expectand_samples, transforms)
    if (is.null(r))
      return (NULL)
    transformed_x_names <- c(transformed_x_names, r$t_name)
    if (! r$t_name %in% transformed_samples) {
      transformed_samples[[r$t_name]] <- r$t_samples
    }
  }
  
  transformed_y_names <- c()
  for (name in y_names) {
    r <- apply_transform(name, expectand_samples, transforms)
    if (is.null(r))
      return (NULL)
    transformed_y_names <- c(transformed_y_names, r$t_name)
    if (! r$t_name %in% transformed_samples) {
      transformed_samples[[r$t_name]] <- r$t_samples
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
  
  nlfs <- c(sapply(1:C, 
                   function(c) diagnostics[['n_leapfrog__']][c,]))
  div_nlfs <- nlfs[div_filter]
  max_nlf <- max(div_nlfs)
  nom_colors <- c(c_light_teal, c_mid_teal, c_dark_teal)
  cmap <- colormap(colormap=nom_colors, nshades=max_nlf)
  
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
  
  # Plot!
  c_dark_trans <- c("#8F272780")
  c_green_trans <- c("#00FF0080")
  
  for (pair in pairs) {
    x_name <- pair[1]
    x_nondiv_samples <- transformed_samples[[x_name]][nondiv_filter]
    x_div_samples    <- transformed_samples[[x_name]][div_filter]
    
    if (is.null(xlim)) {
      xmin = min(transformed_samples[[x_name]])
      xmax = max(transformed_samples[[x_name]])
      local_xlim <- c(xmin, xmax)
    } else {
      local_xlim <- xlim
    }
    
    y_name <- pair[2]
    y_nondiv_samples <- transformed_samples[[y_name]][nondiv_filter]
    y_div_samples    <- transformed_samples[[y_name]][div_filter]
    
    if (is.null(ylim)) {
      ymin = min(transformed_samples[[y_name]])
      ymax = max(transformed_samples[[y_name]])
      local_ylim <- c(ymin, ymax)
    } else {
      local_ylim <- ylim
    }
 
    if (plot_mode == 0) {
      plot(x_nondiv_samples, y_nondiv_samples,
           col=c_dark_trans, pch=16, main="",
           xlab=x_name, xlim=local_xlim, 
           ylab=y_name, ylim=local_ylim)
      points(x_div_samples, y_div_samples,
             col=c_green_trans, pch=16)
    }
    if (plot_mode == 1) {
      plot(x_nondiv_samples, y_nondiv_samples,
           col="#DDDDDD", pch=16, main="",
           xlab=x_name, xlim=local_xlim, 
           ylab=y_name, ylim=local_ylim)
      points(x_div_samples, y_div_samples,
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
# @params fs A one-dimensional array of positive values.
# @return Shape parameter estimate.
compute_xi_hat <- function(fs) {
  N <- length(fs)
  sorted_fs <- sort(fs)

  # Return erroneous result if all input values are the same 
  if (sorted_fs[1] == sorted_fs[N]) {
    return (NaN)
  }

  # Return erroneous result if all input values are not positive
  if (sorted_fs[1] < 0) {
    cat("Input values must be positive!")
    return (NaN)
  }

  # Estimate 25% quantile
  q <- sorted_fs[floor(0.25 * N + 0.5)]

  if (q == sorted_fs[1]) {
    return (-2)
  }

  # Heurstic generalized Pareto shape configuration
  M <- 20 + floor(sqrt(N))

  b_hat_vec <- rep(0, M)
  log_w_vec <- rep(0, M)

  for (m in 1:M) {
    b_hat_vec[m] <- 1 / sorted_fs[N] + 
                 (1 - sqrt(M / (m - 0.5))) / (3 * q)
    if (b_hat_vec[m] != 0) {
      xi_hat <- mean( log(1 - b_hat_vec[m] * sorted_fs) )
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

  mean( log (1 - b_hat * sorted_fs) )
}

# Compute empirical generalized Pareto shape for upper and lower tails
# for an arbitrary sample of expectand values, ignoring any 
# autocorrelation between the values.
# @param fs A one-dimensional array of expectand values.
# @return Left and right shape estimators.
compute_tail_xi_hats <- function(fs) {
  f_center <- median(fs)
  
  # Isolate lower and upper tails which can be adequately modeled by a 
  # generalized Pareto shape for sufficiently well-behaved distributions
  fs_left <- abs(fs[fs < f_center] - f_center)
  N <- length(fs_left)
  M <- min(0.2 * N, 3 * sqrt(N))
  fs_left <- fs_left[M:N]
  
  fs_right <- fs[fs > f_center] - f_center
  N <- length(fs_right)
  M <- min(0.2 * N, 3 * sqrt(N))
  fs_right <- fs_right[M:N]
  
  # Default to NaN if left tail is ill-defined
  xi_hat_left <- NaN
  if (length(fs_left) > 40)
    xi_hat_left <- compute_xi_hat(fs_left)

  # Default to NaN if right tail is ill-defined
  xi_hat_right <- NaN
  if (length(fs_right) > 40)
    xi_hat_right <- compute_xi_hat(fs_right)

  c(xi_hat_left, xi_hat_right)
}

# Check upper and lower tail behavior of a given expectand output 
# ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
check_tail_xi_hats <- function(samples, max_width=72) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }
  C <- dim(samples)[1]
  
  no_warning <- TRUE
  message <- ""
  
  for (c in 1:C) {
    xi_hats <- compute_tail_xi_hats(samples[c,])
    xi_hat_threshold <- 0.25
    if ( is.nan(xi_hats[1]) & is.nan(xi_hats[2]) ) {
      no_warning <- FALSE
      message <-
        paste0(message,
               sprintf('  Chain %s: Both left and right ', c),
               'hat{xi}s are NaN!\n')
    } 
    else if ( is.nan(xi_hats[1]) ) {
      no_warning <- FALSE
      message <-
        paste0(message,
               sprintf('  Chain %s: Left hat{xi} is NaN!\n', c))
    } else if ( is.nan(xi_hats[2]) ) {
      no_warning <- FALSE
      message <-
        paste0(message,
               sprintf('  Chain %s: Right hat{xi} is NaN!\n', c))
    } else if (xi_hats[1] >= xi_hat_threshold & 
      xi_hats[2] >= xi_hat_threshold) {
      no_warning <- FALSE
      message <-
        paste0(message,
              sprintf('  Chain %s: Both left and right tail ', c),
              sprintf('hat{xi}s (%.3f, %.3f) exceed %.2f!\n', 
                      xi_hats[1], xi_hats[2], xi_hat_threshold))
    } else if (xi_hats[1] < xi_hat_threshold & 
               xi_hats[2] >= xi_hat_threshold) {
      no_warning <- FALSE
      message <-
        paste0(message,
               sprintf('  Chain %s: Right tail hat{k} ', c),
               sprintf('(%.3f) exceeds %.2f!\n',
                       xi_hats[2], xi_hat_threshold))
    } else if (xi_hats[1] >= xi_hat_threshold & 
               xi_hats[2] < xi_hat_threshold) {
      no_warning <- FALSE
      message <-
        paste0(message,
               sprintf('  Chain %s: Left tail hat{k} ', c),
               sprintf('(%.3f) exceeds %.2f!\n',
                       xi_hats[1], xi_hat_threshold))
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
# @params A one-dimensional array of expectand values.
# @return The empirical mean and variance.
welford_summary <- function(fs) {
  mean <- 0
  var <- 0

  N <- length(fs)
  for (n in 1:N) {
    delta <- fs[n] - mean
    mean <- mean + delta / n
    var <- var + delta * (fs[n] - mean)
  }

  var <- var/ (N - 1)

  return(c(mean, var))
}

# Check expectand output ensemble for vanishing empirical variance.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
check_variances <- function(samples, max_width=72) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }
  C <- dim(samples)[1]
  
  no_warning <- TRUE
  message <- ""

  for (c in 1:C) {
    var <- welford_summary(samples[c,])[2]
    if (var < 1e-10) {
      message <- paste0(message,
                        sprintf('Chain %s: Expectand is constant!\n', c))
      no_warning <- FALSE
    }
  }
  
  if (no_warning) {
    desc <- 'Expectand is varying across all Markov chains.\n\n'
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

# Compute split hat{R} for the expectand values across a Markov chain 
# ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @return Split Rhat estimate.
compute_split_rhat <- function(samples) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }
  C <- dim(samples)[1]
  
  split_chains <- unlist(lapply(1:C, 
                                function(c) split_chain(samples[c,])),
                         recursive=FALSE)

  N_chains <- length(split_chains)
  N <- sum(sapply(1:C, function(c) length(samples[c,])))

  means <- rep(0, N_chains)
  vars <- rep(0, N_chains)

  for (c in 1:N_chains) {
    summary <- welford_summary(split_chains[[c]])
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

# Compute split hat{R} for all input expectands
# @param expectand_samples A named list of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
compute_split_rhats <- function(expectand_samples) {
  if (!is.vector(expectand_samples)) {
    cat('Input variable `expectand_samples` is not a named list!')
    return
  }

  rhats <- c()
  for (name in names(expectand_samples)) {
    samples <- expectand_samples[[name]]
    rhats <- c(rhats, compute_split_rhat(samples))
  }
  return(rhats)
}

# Check split hat{R} across a given expectand output ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param max_width Maximum line width for printing
check_rhat <- function(samples, max_width=72) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }

  rhat <- compute_split_rhat(samples)

  no_warning <- TRUE
  message <- ""

  if (is.nan(rhat)) {
    message <- paste0(message, 
                      'All Markov chains appear to be frozen!\n')
  } else if (rhat > 1.1) {
    message <- paste0(message, sprintf('Split hat{R} is %f!\n', rhat))
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

# Compute empirical integrated autocorrelation time for a sequence
# of expectand values, known here as \hat{tau}.
# @param fs A one-dimensional array of expectand values.
# @return Left and right shape estimators.
compute_tau_hat <- function(fs) {
  # Compute empirical autocorrelations
  N <- length(fs)
  zs <- fs - mean(fs)
  
  if (var(fs) < 1e-10)
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

# Compute the minimum empirical effective sample size across the 
# Markov chains for the given expectands
# @param expectand_samples A named list of two-dimensional arrays for 
#                          each expectand.  The first dimension of each
#                          element indexes the Markov chains and the 
#                          second dimension indexes the sequential 
#                          states within each Markov chain.
compute_min_eesss <- function(expectand_samples) {
  if (!is.vector(expectand_samples)) {
    cat('Input variable `expectand_samples` is not a named list!')
    return
  }

  min_eesss <- c()
  for (name in names(expectand_samples)) {
    samples <- expectand_samples[[name]]
    C <- dim(samples)[1]
    S <- dim(samples)[2]
    
    eesss <- rep(0, C)
    for (c in 1:C) {
      tau_hat <- compute_tau_hat(samples[c,])
      eesss[c] <- S / tau_hat
    }
    min_eesss <- c(min_eesss, min(eesss))
  }
  return(min_eesss)
}

# Check the empirical effective sample size (EESS) for all a given 
# expectand output ensemble.
# @param samples A two-dimensional array of scalar Markov chain states 
#                with the first dimension indexing the Markov chains and 
#                the second dimension indexing the sequential states 
#                within each Markov chain.
# @param min_eess_per_chain The minimum empirical effective sample size
#                           before a warning message is passed.
# @param max_width Maximum line width for printing
check_eess <- function(samples,
                       min_eess_per_chain=100,
                       max_width=72) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }
  
  C <- dim(samples)[1]
  N <- dim(samples)[2]
  
  no_warning <- TRUE
  message <- ""
  
  for (c in 1:C) {
    tau_hat <- compute_tau_hat(samples[c,])
    eess <- N / tau_hat
    if (eess < min_eess_per_chain) {
      message <- paste0(message,
                        sprintf('Chain %s: The empirical effective ', c),
                        sprintf('sample size %f is too small!\n', eess))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    desc <- paste0('The empirical effective sample size is large ',
                   'enough for Markov chain Monte Carlo estimation ',
                   'to be reliable assuming that a central limit ',
                   'theorem holds.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  } else {
    desc <- paste0('Small empirical effective sample sizes indicate strong ',
                   'empirical autocorrelations in the realized Markov chains. ',
                   'If the empirical effective sample size is too ',
                   'small then Markov chain Monte Carlo estimation ',
                   'may be unreliable even when a central limit ',
                   'theorem holds.\n\n.')
    desc <- paste0(strwrap(desc, max_width, 2), collapse='\n')
    message <- paste0(message, desc)
  }
  
  cat(message)
}

# Check all expectand-specific diagnostics.
# @param expectand_samples A named list of two-dimensional arrays for 
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
check_all_expectand_diagnostics <- function(expectand_samples,
                                            min_eess_per_chain=100,
                                            exclude_zvar=FALSE,
                                            max_width=72) {
  if (!is.vector(expectand_samples)) {
    cat('Input variable `expectand_samples` is not a named list!')
    return
  }
  
  no_xi_hat_warning <- TRUE
  no_zvar_warning <- TRUE
  no_rhat_warning <- TRUE
  no_eess_warning <- TRUE

  message <- ""

  for (name in names(expectand_samples)) {
    samples <- expectand_samples[[name]]
    C <- dim(samples)[1]
    S <- dim(samples)[2]
    
    local_warning <- FALSE
    local_message <- paste0(name, ':\n')
  
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(samples[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }
  
    for (c in 1:C) {
      fs <- samples[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(fs)
      if ( is.nan(xi_hats[1]) & is.nan(xi_hats[2]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Both left and right ', c),
                 'hat{xi}s are NaN!\n')
      } 
      else if ( is.nan(xi_hats[1]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Left hat{xi} is NaN!\n', c))
      } else if ( is.nan(xi_hats[2]) ) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Right hat{xi} is NaN!\n', c))
      } else if (xi_hats[1] >= xi_hat_threshold & 
          xi_hats[2] >= xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                sprintf('  Chain %s: Both left and right tail ', c),
                sprintf('hat{xi}s (%.3f, %.3f) exceed %.2f!\n', 
                        xi_hats[1], xi_hats[2], xi_hat_threshold))
      } else if (xi_hats[1] < xi_hat_threshold & 
                 xi_hats[2] >= xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Right tail hat{k} ', c),
                 sprintf('(%.3f) exceeds %.2f!\n',
                         xi_hats[2], xi_hat_threshold))
      } else if (xi_hats[1] >= xi_hat_threshold & 
                 xi_hats[2] < xi_hat_threshold) {
        no_xi_hat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Left tail hat{k} ', c),
                 sprintf('(%.3f) exceeds %.2f!\n',
                         xi_hats[1], xi_hat_threshold))
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        no_zvar_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Expectand exhibits vanishing ', c),
                         'empirical variance!\n')
      }
    }
  
    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(samples)

    if (is.nan(rhat)) {
      local_message <- paste0(local_message,
                              '  Split hat{R} is ill-defined!\n')
    } else if (rhat > 1.1) {
      no_rhat_warning <- FALSE
      local_warning <- TRUE
      local_message <-
        paste0(local_message,
               sprintf('  Split hat{R} (%.3f) exceeds 1.1!\n', rhat))
    }

    for (c in 1:C) {
      # Check empirical effective sample size
      fs <- samples[c,]
      
      tau_hat <- compute_tau_hat(fs)
      eess <- S / tau_hat
      
      if (eess < min_eess_per_chain) {
        no_eess_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: hat{ESS} (%.3f) is smaller than ',
                         c, eess),
                 sprintf('desired (%s)!\n', min_eess_per_chain))
      }
    }
    
    if (local_warning) {
      message <- paste0(message, local_message, '\n')
    }
  }

  if (!no_xi_hat_warning) {
    desc <- paste0('Large tail hat{xi}s suggest that the expectand ',
                   ' might not be sufficiently integrable.\n\n')
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
  if (!no_eess_warning) {
    desc <- paste0('Small empirical effective sample sizes indicate strong ',
                   'empirical autocorrelations in the realized Markov chains. ',
                   'If the empirical effective sample size is too ',
                   'small then Markov chain Monte Carlo estimation ',
                   'may be unreliable even when a central limit ',
                   'theorem holds.\n\n.')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, '\n', desc)
  }

  if(no_xi_hat_warning & no_zvar_warning & 
     no_rhat_warning & no_eess_warning) {
    desc <- paste0('All expectands checked appear to be behaving ',
                   'well enough for reliable Markov chain Monte ',
                   'Carlo estimation.\n\n')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc)
  }

  cat(message)
}

# Summary all expectand-specific diagnostics.
# @param expectand_samples A named list of two-dimensional arrays for 
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
summarize_expectand_diagnostics <- function(expectand_samples,
                                            min_eess_per_chain=100,
                                            exclude_zvar=FALSE,
                                            max_width=72) {
  if (!is.vector(expectand_samples)) {
    cat('Input variable `expectand_samples` is not a named list!')
    return
  }

  failed_names <- c()
  failed_xi_hat_names <- c()
  failed_zvar_names <- c()
  failed_rhat_names <- c()
  failed_eess_names <- c()

  for (name in names(expectand_samples)) {
    samples <- expectand_samples[[name]]
    C <- dim(samples)[1]
    S <- dim(samples)[2]
    
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(samples[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }

    for (c in 1:C) {
      fs <- samples[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(fs)
      if ( is.nan(xi_hats[1]) | is.nan(xi_hats[2]) ) {
        failed_names <- c(failed_names, name)
        failed_xi_hat_nameas <- c(failed_xi_hat_names, name)
      } else if (xi_hats[1] >= xi_hat_threshold | 
                 xi_hats[2] >= xi_hat_threshold) {
        failed_names <- c(failed_names, name)
        failed_xi_hat_nameas <- c(failed_xi_hat_names, name)
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        failed_names <- c(failed_names, name)
        failed_zvar_names <- c(failed_zvar_names, name) 
      }
    }

    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(samples)

    if (is.nan(rhat)) {
      failed_names <- c(failed_names, name)
      failed_rhat_names <- c(failed_rhat_names, name)
    } else if (rhat > 1.1) {
      failed_names <- c(failed_names, name)
      failed_rhat_names <- c(failed_rhat_names, name)
    }

    for (c in 1:C) {
      # Check empirical effective sample size
      tau_hat <- compute_tau_hat(samples[c,])
      eess <- S / tau_hat
      
      if (eess < min_eess_per_chain) {
        failed_names <- c(failed_names, name)
        failed_eess_names <- c(failed_eess_names, name)
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
  
  failed_eess_names <- unique(failed_eess_names)
  if (length(failed_eess_names)) {
    desc <- 
      paste0(sprintf('The expectands %s triggered hat{ESS} warnings.\n\n',
             paste(failed_eess_names, collapse=", ")),
             '  Small empirical effective sample sizes indicate strong ',
             'empirical autocorrelations in the realized Markov chains. ',
             'If the empirical effective sample size is too ',
             'small then Markov chain Monte Carlo estimation ',
             'may be unreliable even when a central limit ',
             'theorem holds.\n\n.')
    desc <- paste0(strwrap(desc, max_width, 0), collapse='\n')
    message <- paste0(message, desc, '\n\n')
  }
  
  cat(message)
}

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
encode_all_diagnostics <- function(expectand_samples,
                                   diagnostics,
                                   adapt_target=0.801,
                                   max_treedepth=10,
                                   min_eess_per_chain=100,
                                   exclude_zvar=FALSE) {
  warning_code <- 0
  
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
  C <- dim(diagnostics[['energy__']])[1]
  
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
  eess_warning <- FALSE
  
  for (name in names(expectand_samples)) {
    samples <- expectand_samples[[name]]
    C <- dim(samples)[1]
    S <- dim(samples)[2]
  
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        var <- welford_summary(samples[c,])[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }
  
    for (c in 1:C) {
      fs <- samples[c,]
      
      # Check tail behavior in each Markov chain
      xi_hat_threshold <- 0.25
      xi_hats <- compute_tail_xi_hats(fs)
      if (isnan(xi_hats[1]) | isnan(xi_hats[2])) {
        xi_hat_warning <- TRUE
      } else if (xi_hats[1] >= xi_hat_threshold | 
          xi_hats[2] >= xi_hat_threshold) {
        xi_hat_warning <- TRUE
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        zvar_warning <- TRUE
      }
    }
  
    # Check split Rhat across Markov chains
    rhat <- compute_split_rhat(samples)

    if (is.nan(rhat)) {
      rhat_warning <- TRUE
    } else if (rhat > 1.1) {
      rhat_warning <- TRUE
    }

    for (c in 1:C) {
      # Check empirical effective sample size
      fs <- samples[c,]
      
      tau_hat <- compute_tau_hat(fs)
      eess <- S / tau_hat
      
      if (eess < min_eess_per_chain) {
        eess_warning <- TRUE
      }
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
  
  if (eess_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 7))
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
    print("  min empirical effective sample size warning")
}

# Compute empirical autocorrelations for a given Markov chain sequence
# @parmas fs A one-dimensional array of sequential expectand values.
# @return A one-dimensional array of empirical autocorrelations at each 
#         lag up to the length of the sequence.
compute_rhos <- function(fs) {
  # Compute empirical autocorrelations
  N <- length(fs)
  zs <- fs - mean(fs)
  
  if (var(fs) < 1e-10)
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
# @param fs A two-dimensional array of scalar Markov chain states 
#           with the first dimension indexing the Markov chains and 
#           the second dimension indexing the sequential states 
#           within each Markov chain.
# @param max_L Maximum autocorrelation lag
# @param rho_lim Plotting range of autocorrelation values
# @display_name Name of expectand
plot_empirical_correlogram <- function(fs,
                                       max_L,
                                       rho_lim=c(-0.2, 1.1),
                                       display_name="") {
  if (length(dim(fs)) != 2) {
    cat('Input variable `fs` has the wrong dimensions!')
  }
  C <- dim(fs)[1]
  
  idx <- rep(0:max_L, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  plot(0, type="n", main=display_name,
       xlab="Lag", xlim=c(-0.5, max_L + 0.5),
       ylab="Empirical Autocorrelation", ylim=rho_lim)
  abline(h=0, col="#DDDDDD", lty=2, lwd=2)

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)
  for (c in 1:C) {
    rhos <- compute_rhos(fs[c,])
    pad_rhos <- unlist(lapply(idx, function(n) rhos[n + 1]))
    lines(xs, pad_rhos, lwd=2, col=colors[c])
  }
}

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
plot_pairs_by_chain <- function(f1s, display_name1,
                                f2s, display_name2) {
  if (length(dim(f1s)) != 2) {
    cat('Input variable `f1s` has the wrong dimensions!')
    return
  }
  C1 <- dim(f1s)[1]
  S1 <- dim(f1s)[2]

  if (length(dim(f2s)) != 2) {
    cat('Input variable `f1s` has the wrong dimensions!')
    return
  }
  C2 <- dim(f2s)[1]
  S2 <- dim(f2s)[2]
  
  if (C1 != C2) {
    C <- min(C1, C2)
    C1 <- C
    C2 <- C
    cat(sprintf('Plotting only %s Markov chains.\n', C))
  }

  nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C",
                  "#A25050", "#8F2727", "#7C0000")
  cmap <- colormap(colormap=nom_colors, nshades=max(S1, S2))

  min_x <- min(sapply(1:C1, function(c) min(f1s[c,])))
  max_x <- max(sapply(1:C1, function(c) max(f1s[c,])))

  min_y <- min(sapply(1:C2, function(c) min(f2s[c,])))
  max_y <- max(sapply(1:C2, function(c) max(f2s[c,])))

  par(mfrow=c(2, 2), mar = c(5, 5, 3, 1))

  for (c in 1:C1) {
    plot(0, type="n", main=paste("Chain", c),
         xlab=display_name1, xlim=c(min_x, max_x),
         ylab=display_name2, ylim=c(min_y, max_y))
  
    points(unlist(lapply(1:C1, function(c) f1s[c,])),
           unlist(lapply(1:C1, function(c) f2s[c,])),
           col="#DDDDDD", pch=16, cex=1.0)
    points(f1s[c,], f2s[c,], col=cmap, pch=16, cex=1.0)
  }
}

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
pushforward_chains <- function(samples, expectand) {
  apply(samples, 2, expectand)
}

# Estimate expectand exectation value from a single Markov chain.
# @param fs A one-dimensional array of sequential expectand values.
# @return The Markov chain Monte Carlo estimate, its estimated standard 
#         error, and empirical effective sample size.
mcmc_est <- function(fs) {
  S <- length(fs)
  if (S == 1) {
    return(c(fs[1], 0, NaN))
  }

  summary <- welford_summary(fs)

  if (summary[2] == 0) {
    return(c(summary[1], 0, NaN))
  }

  tau_hat <- compute_tau_hat(fs)
  eess <- S / tau_hat
  return(c(summary[1], sqrt(summary[2] / eess), eess))
}

# Estimate expectand exectation value from a Markov chain ensemble.
# @param samples A two-dimensional array of expectand values with the 
#                first dimension indexing the Markov chains and the 
#                second dimension indexing the sequential states within 
#                each Markov chain.
# @return The ensemble Markov chain Monte Carlo estimate, its estimated
#         standard error, and empirical effective sample size.
ensemble_mcmc_est <- function(samples) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return (c(NaN, NaN, NaN))
  }
  
  C <- dim(samples)[1]
  chain_ests <- lapply(1:C, function(c) mcmc_est(samples[c,]))
  
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

# Visualize pushforward distribution of a given expectand as a 
# histogram, using Markov chain Monte Carlo estimators to estimate the 
# output bin probabilities.  Bin probability estimator error is shown 
# in gray.
# @param samples A two-dimensional array of expectand values with the 
#                first dimension indexing the Markov chains and the 
#                second dimension indexing the sequential states within 
#                each Markov chain.
# @param B The number of histogram bins
# @param display_name Exectand name
# @param flim Optional histogram range
# @param baseline Optional baseline value for visual comparison
plot_expectand_pushforward <- function(samples, B, display_name="f", 
                                       flim=NULL, baseline=NULL) {
  if (length(dim(samples)) != 2) {
    cat('Input variable `samples` has the wrong dimension')
    return
  }
  
  # Automatically adjust histogram range to range of expectand values
  # if range is not already set as an input variable
  if (is.null(flim)) {
    min_f <- min(samples)
    max_f <- max(samples)
    
    # Add bounding bins
    delta <- (max_f - min_f) / B
    min_f <- min_f - delta
    max_f <- max_f + delta
    flim <- c(min_f, max_f)
    
    bins <- seq(min_f, max_f, delta)
    B <- B + 2
  } else {
    delta <- (flim[2] - flim[1]) / B
    bins <- seq(flim[1], flim[2], delta)
  }
  
  # Compute bin heights
  mean_p <- rep(0, B)
  delta_p <- rep(0, B)
  
  for (b in 1:B) {
    # Estimate bin probabilities
    bin_indicator <- function(x) {
      ifelse(bins[b] <= x & x < bins[b + 1], 1, 0)
    }
    indicator_samples <- pushforward_chains(samples, bin_indicator)
    est <- ensemble_mcmc_est(indicator_samples)
    
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
  
  min_y <- min(lower_inter)
  max_y <- max(1.05 * upper_inter)
  
  plot(1, type="n", main="",
       xlim=flim, xlab=display_name,
       ylim=c(min_y, max_y), ylab="", yaxt="n")
  title(ylab="Estimated Bin\nProbabilities / Bin Width", mgp=c(1, 1, 0))
  
  polygon(c(x, rev(x)), c(lower_inter, rev(upper_inter)),
          col = "#DDDDDD", border = NA)
  lines(x, mean_p[idx], col=c_dark, lwd=2)
  
  # Plot baseline if applicable
  if (!is.null(baseline)) {
    abline(v=baseline, col="white", lty=1, lwd=4)
    abline(v=baseline, col="black", lty=1, lwd=2)
  }
}
