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

# Check all Hamiltonian Monte Carlo Diagnostics 
# for an ensemble of Markov chains
check_all_hmc_diagnostics <- function(fit,
                                      adapt_target=0.801,
                                      max_treedepth=10) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)

  no_warning <- TRUE

  # Check divergences
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  n = sum(divergent)
  N = length(divergent)

  if (n > 0) {
    no_warning <- FALSE
    cat(sprintf('%s of %s iterations ended with a divergence (%s%%).\n',
                n, N, 100 * n / N))

    cat('  Divergences are due unstable numerical integration.  ')
    cat('These instabilities are often due to posterior degeneracies.\n')
    cat('  If there are only a small number of divergences then running ')
    cat(sprintf('with adapt_delta larger than %.3f may reduce the divergences ',
                adapt_target))
    cat('at the cost of more expensive transitions.\n\n')
  }

  # Check transitions that ended prematurely due to maximum tree depth limit
  treedepths <- do.call(rbind, sampler_params)[,'treedepth__']
  n = length(treedepths[sapply(treedepths, function(x) x >= max_treedepth)])
  N = length(treedepths)

  if (n > 0) {
    no_warning <- FALSE
    cat(sprintf('%s of %s iterations saturated the maximum tree depth of %s (%s%%).',
                n, N, max_treedepth, 100 * n / N))

    cat('  Increasing max_depth will increase the efficiency of the transitions.\n\n')
  }

  # Checks the energy fraction of missing information (E-FMI)
  no_efmi_warning <- TRUE
  for (c in 1:length(sampler_params)) {
    energies = sampler_params[c][[1]][,'energy__']
    numer = sum(diff(energies)**2) / length(energies)
    denom = var(energies)
    if (numer / denom < 0.2) {
      no_warning <- FALSE
      no_efmi_warning <- FALSE
      cat(sprintf('Chain %s: E-FMI = %s.\n', c, numer / denom))
    }
  }
  if (!no_efmi_warning) {
    cat('  E-FMI below 0.2 suggests a funnel-like geometry hiding ')
    cat('somewhere in the posterior distribution.\n\n')
  }

  # Check convergence of the stepsize adaptation
  no_accept_warning <- TRUE
  for (c in 1:length(sampler_params)) {
    ave_accept_proxy <- mean(sampler_params[[c]][,'accept_stat__'])
    if (ave_accept_proxy < 0.9 * adapt_target) {
      no_warning <- FALSE
      no_accept_warning <- FALSE
      cat(sprintf('Chain %s: Average proxy acceptance statistic (%.3f)\n',
                   c, ave_accept_proxy))
      cat(sprintf('          is smaller than 90%% of the target (%.3f).\n',
                  adapt_target))
    }
  }
  if (!no_accept_warning) {
    cat('  A small average proxy acceptance statistic indicates that the ')
    cat('integrator step size adaptation failed to converge.  This is often ')
    cat('due to discontinuous or inexact gradients.\n\n')
  }

  if (no_warning) {
    cat('All Hamiltonian Monte Carlo diagnostics are consistent with ')
    cat('accurate Markov chain Monte Carlo.\n\n')
  }
}

# Plot outcome of inverse metric adaptation
plot_inv_metric <- function(fit, B=25) {
  chain_info <- get_adaptation_info(fit)
  C <- length(chain_info)

  inv_metric_elems <- list()
  for (c in 1:C) {
    raw_info <- chain_info[[c]]
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

# Display outcome of symplectic integrator step size adaptation
display_stepsizes <- function(fit) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  for (c in 1:4) {
    stepsize <- sampler_params[[c]][,'stepsize__'][1]
    cat(sprintf('Chain %s: Integrator Step Size = %f\n',
                c, stepsize))
  }
}

# Display symplectic integrator trajectory lenghts
plot_num_leapfrog <- function(fit) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)

  max_n <- max(sapply(1:4, function(c) max(sampler_params[[c]][,'n_leapfrog__']))) + 1
  max_count <- max(sapply(1:4, function(c) max(table(sampler_params[[c]][,'n_leapfrog__']))))

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)

  idx <- rep(1:max_n, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  par(mfrow=c(2, 2), mar = c(5, 5, 3, 1))

  for (c in 1:4) {
    stepsize <- round(sampler_params[[c]][,'stepsize__'][1], 3)
  
    counts <- hist(sampler_params[[c]][,'n_leapfrog__'],
                   seq(0.5, max_n + 0.5, 1), plot=FALSE)$counts
    pad_counts <- counts[idx]
  
    plot(xs, pad_counts, type="l",  lwd=2, col=colors[c],
         main=paste0("Chain ", c, " (Stepsize = ", stepsize, ")"),
         xlab="Numerical Trajectory Length", xlim=c(0.5, max_n + 0.5),
         ylab="", ylim=c(0, 1.1 * max_count), yaxt='n')
  }
}

# Display empirical average of the proxy acceptance statistic
# across each individual Markov chains
display_ave_accept_proxy <- function(fit) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)

  for (c in 1:length(sampler_params)) {
    ave_accept_proxy <- mean(sampler_params[[c]][,'accept_stat__'])
    cat(sprintf('Chain %s: Average proxy acceptance statistic = %.3f\n',
                c, ave_accept_proxy))
  }
}

# Separate Markov chain states into those sampled from non-divergent
# numerical Hamiltonian trajectories and those sampled from divergent
# numerical Hamiltonian trajectories
partition_div <- function(fit) {
  nom_params <- rstan:::extract(fit, permuted=FALSE)
  n_chains <- dim(nom_params)[2]
  params <- as.data.frame(do.call(rbind, lapply(1:n_chains, function(n) nom_params[,n,])))

  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  params$divergent <- divergent

  div_params <- params[params$divergent == 1,]
  nondiv_params <- params[params$divergent == 0,]

  return(list(div_params, nondiv_params))
}

# Plot pairwise scatter plots with non-divergent and divergent 
# transitions separated by color
plot_div_pairs <- function(fit, names, transforms) {
  c_dark_trans <- c("#8F272780")
  c_green_trans <- c("#00FF0080")

  N <- length(names)
  N_plots <- choose(N, 2)

  if (N_plots <= 3) {
    par(mfrow=c(1, N_plots), mar = c(5, 5, 2, 1))
  } else if (N_plots == 4) {
    par(mfrow=c(2, 2), mar = c(5, 5, 2, 1))
  } else {
    par(mfrow=c(2, 3), mar = c(5, 5, 2, 1))
  }

  partition <- partition_div(fit)
  div_samples <- partition[[1]]
  nondiv_samples <- partition[[2]]

  for (n in 1:(N - 1)) {
    for (m in (n + 1):N) {
    
      name_x <- names[n]
      if (transforms[n] == 0) {
        x_nondiv_samples <- nondiv_samples[name_x][,1]
        x_div_samples <- div_samples[name_x][,1]
        x_name <- name_x
      } else if (transforms[n] == 1) {
        x_nondiv_samples <- log(nondiv_samples[name_x][,1])
        x_div_samples <- log(div_samples[name_x][,1])
        x_name <- paste("log(", name_x, ")", sep="")
      }
      xlims <- range(c(x_nondiv_samples, x_div_samples))
    
      name_y <- names[m]
      if (transforms[m] == 0) {
        y_nondiv_samples <- nondiv_samples[name_y][,1]
        y_div_samples <- div_samples[name_y][,1]
        y_name <- name_y
      } else if (transforms[m] == 1) {
        y_nondiv_samples <- log(nondiv_samples[name_y][,1])
        y_div_samples <- log(div_samples[name_y][,1])
        y_name <- paste("log(", name_y, ")", sep="")
      }
      ylims <- range(c(y_nondiv_samples, y_div_samples))
    
      plot(x_nondiv_samples, y_nondiv_samples,
           col=c_dark_trans, pch=16, main="",
           xlab=x_name, xlim=xlims, ylab=y_name, ylim=ylims)
      points(x_div_samples, y_div_samples,
             col=c_green_trans, pch=16)
    }
  }
}

# Compute empirical Pareto shape for a positive sample
compute_khat <- function(fs) {
  N <- length(fs)
  sorted_fs <- sort(fs)

  if (sorted_fs[1] == sorted_fs[N]) {
    return (-2)
  }

  if (sorted_fs[1] < 0) {
    cat("x must be positive!")
    return (NA)
  }

  # Estimate 25% quantile
  q <- sorted_fs[floor(0.25 * N + 0.5)]

  if (q == sorted_fs[1]) {
    return (-2)
  }

  # Heurstic Pareto configuration
  M <- 20 + floor(sqrt(N))

  b_hat_vec <- rep(0, M)
  log_w_vec <- rep(0, M)

  for (m in 1:M) {
    b_hat_vec[m] <- 1 / sorted_fs[N] + (1 - sqrt(M / (m - 0.5))) / (3 * q)
    if (b_hat_vec[m] != 0) {
      k_hat <- - mean( log(1 - b_hat_vec[m] * sorted_fs) )
      log_w_vec[m] <- N * ( log(b_hat_vec[m] / k_hat) + k_hat - 1)
    } else {
      log_w_vec[m] <- 0
    }
  }
  
  # Remove terms that don't contribute to improve numerical stability of average
  log_w_vec <- log_w_vec[b_hat_vec != 0]
  b_hat_vec <- b_hat_vec[b_hat_vec != 0]

  max_log_w <- max(log_w_vec)
  b_hat <- sum(b_hat_vec * exp(log_w_vec - max_log_w)) /
           sum(exp(log_w_vec - max_log_w))

  mean( log (1 - b_hat * sorted_fs) )
}

# Compute empirical Pareto shape for upper and lower tails
compute_tail_khats <- function(fs) {
  f_center <- median(fs)
  fs_left <- abs(fs[fs < f_center] - f_center)
  fs_right <- fs[fs > f_center] - f_center

  # Default to -2 if left tail is ill-defined
  khat_left <- -2
  if (length(fs_left) > 40)
    khat_left <- compute_khat(fs_left)

  # Default to -2 if right tail is ill-defined
  khat_right <- -2
  if (length(fs_right) > 40)
    khat_right <- compute_khat(fs_right)

  c(khat_left, khat_right)
}

# Check empirical Pareto shape for upper and lower tails of a
# given expectand output ensemble
check_tail_khats <- function(unpermuted_samples) {
  no_warning <- TRUE
  for (c in 1:4) {
    fs <- unpermuted_samples[,c]
    khats <- compute_tail_khats(fs)
    if (khats[1] >= 0.25 & khats[2] >= 0.25) {
      cat(sprintf('Chain %s: Both left and right tail khats exceed 0.25!\n',
                   c))
      no_warning <- FALSE
    } else if (khats[1] < 0.25 & khats[2] >= 0.25) {
      cat(sprintf('Chain %s: Right tail khat exceeds 0.25!\n', c))
      no_warning <- FALSE
    } else if (khats[1] >= 0.25 & khats[2] < 0.25) {
      cat(sprintf('Chain %s: Left tail khat exceeds 0.25!\n', c))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    cat('Expectand appears to be sufficiently integrable.\n\n')
  } else {
    cat('  Large tail khats suggest that the expectand might\n')
    cat('not be sufficiently integrable.\n\n')
  }
}

# Welford accumulator for empirical mean and variance of a given sequence
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

# Check expectand output ensemble for vanishing empirical variance
check_variances <- function(unpermuted_samples) {
  no_warning <- TRUE
  for (c in 1:4) {
    fs <- unpermuted_samples[,c]
    var <- welford_summary(fs)[2]
    if (var < 1e-10) {
      cat(sprintf('Chain %s: Expectand is constant!\n', c))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    cat('Expectand is varying in all Markov chains.')
  } else {
    cat('  If the expectand is not expected (haha) to be\n')
    cat('constant then the Markov transitions are misbehaving.\n')
  }
}

# Split a Markov chain into initial and terminal Markov chains
split_chain <- function(chain) {
  N <- length(chain)
  M <- N %/% 2
  list(chain1 <- chain[1:M], chain2 <- chain[(M + 1):N])
}

# Compute split hat{R} for an expectand output ensemble across
# a collection of Markov chains
compute_split_rhat <- function(chains) {
  split_chains <- unlist(lapply(chains, function(c) split_chain(c)),
                         recursive=FALSE)

  N_chains <- length(split_chains)
  N <- sum(sapply(chains, function(c) length(c)))

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

# Compute split hat{R} for all expectand output ensembles across
# a collection of Markov chains
compute_split_rhats <- function(fit, expectand_idxs=NULL) {
  unpermuted_samples <- rstan:::extract(fit, permute=FALSE)

  input_dims <- dim(unpermuted_samples)
  N <- input_dims[1]
  C <- input_dims[2]
  I <- input_dims[3]

  if (is.null(expectand_idxs)) {
    expectand_idxs <- 1:I
  }

  bad_idxs <- setdiff(expectand_idxs, 1:I)
  if (length(bad_idxs) > 0) {
    cat(sprintf('Excluding the invalid expectand indices: %s\n',
                bad_idxs))
    expectand_idxs <- setdiff(expectand_idxs, bad_idxs)
  }

  rhats <- c()
  for (idx in expectand_idxs) {
    chains <- lapply(1:C, function(c) unpermuted_samples[,c,idx])
    rhats <- c(rhats, compute_split_rhat(chains))
  }
  return(rhats)
}

# Check split hat{R} for all expectand output ensembles across
# a collection of Marko chains
check_rhat <- function(unpermuted_samples) {
  chains <- lapply(1:4, function(c) unpermuted_samples[,c])
  rhat <- compute_split_rhat(chains)

  no_warning <- TRUE
  if (is.nan(rhat)) {
    cat('All Markov chains are frozen!\n')
  } else if (rhat > 1.1) {
    cat(sprintf('Split rhat is %f!\n', rhat))
    no_warning <- FALSE
  }
  if (no_warning) {
    cat('Markov chains are consistent with equilibrium.\n')
  } else {
    cat('  Split rhat larger than 1.1 is inconsistent with equilibrium.\n')
  }
}

# Compute empirical integrated autocorrelation time for a sequence
compute_tauhat <- function(fs) {
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
      # throw some kind of error when autocorrelation
      # sequence doesn't get terminated
    }
  
    old_pair_sum <- current_pair_sum
  }
}

# Compute the minimimum empirical integrated autocorrelation time
# across a collection of Markov chains for all expectand output ensembles
compute_min_tauhat <- function(fit, expectand_idxs=NULL) {
  unpermuted_samples <- rstan:::extract(fit, permute=FALSE)

  input_dims <- dim(unpermuted_samples)
  N <- input_dims[1]
  C <- input_dims[2]
  I <- input_dims[3]

  expectand_names <- names(unpermuted_samples[1,1,])

  if (is.null(expectand_idxs)) {
    expectand_idxs <- 1:I
  }

  bad_idxs <- setdiff(expectand_idxs, 1:I)
  if (length(bad_idxs) > 0) {
    cat(sprintf('Excluding the invalid expectand indices: %s',
                bad_idxs))
    expectand_idxs <- setdiff(expectand_idxs, bad_idxs)
  }

  min_int_ac_times <- c()

  for (idx in expectand_idxs) {
    int_ac_times <- rep(0, C)
    for (c in 1:C) {
      fs <- unpermuted_samples[,c, idx]
      int_ac_times[c] <- compute_tauhat(fs)
    }
    min_int_ac_times <- c(min_int_ac_times, min(int_ac_times))
  }
  return(min_int_ac_times)
}

# Check the empirical integrated autocorrelation times across a 
# collection of Markov chains for all expectand output ensembles
check_min_tauhat <- function(unpermuted_samples) {
  N <- dim(unpermuted_samples)[1]
  no_warning <- TRUE
  for (c in 1:4) {
    fs <- unpermuted_samples[,c]
    int_ac_time <- compute_tauhat(fs)
    if (int_ac_time / N > 0.25) {
      cat(sprintf('Chain %s: The integrated autocorrelation time', c))
      cat('exceeds 0.25 * N!\n')
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    cat('Autocorrelations within each Markov chain appear to be reasonable.\n')
  } else {
    cat('  Autocorrelations in at least one Markov chain are large enough')
    cat('that Markov chain Monte Carlo estimates may not be reliable.\n')
  }
}

# Check the empirical effective sample size for all expectand output ensembles
check_neff <- function(unpermuted_samples,
                       min_neff_per_chain=100) {
  N <- dim(unpermuted_samples)[1]
  no_warning <- TRUE
  for (c in 1:4) {
    fs <- unpermuted_samples[,c]
    int_ac_time <- compute_tauhat(fs)
    neff <- N / int_ac_time
    if (neff < min_neff_per_chain) {
      cat(sprintf('Chain %s: The effective sample size %f is too small!\n',
                   c, neff))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    cat('All effective sample sizes are sufficiently large.\n')
  } else {
    cat('  If the effective sample size is too small then\n')
    cat('Markov chain Monte Carlo estimators will be imprecise.\n\n')
  }
}

# Check all expectand diagnostics
check_all_expectand_diagnostics <- function(fit,
                                            expectand_idxs=NULL,
                                            min_neff_per_chain=100,
                                            exclude_zvar=FALSE) {
  unpermuted_samples <- rstan:::extract(fit, permute=FALSE)

  input_dims <- dim(unpermuted_samples)
  N <- input_dims[1]
  C <- input_dims[2]
  I <- input_dims[3]

  expectand_names <- names(unpermuted_samples[1,1,])

  if (is.null(expectand_idxs)) {
    expectand_idxs <- 1:I
  }

  bad_idxs <- setdiff(expectand_idxs, 1:I)
  if (length(bad_idxs) > 0) {
    cat(sprintf('Excluding the invalid expectand indices: %s',
                bad_idxs))
    expectand_idxs <- setdiff(expectand_idxs, bad_idxs)
  }

  no_khat_warning <- TRUE
  no_zvar_warning <- TRUE
  no_rhat_warning <- TRUE
  no_tauhat_warning <- TRUE
  no_neff_warning <- TRUE

  message <- ""

  for (idx in expectand_idxs) {
    local_warning <- FALSE
    local_message <- paste0(expectand_names[idx], ':\n')
  
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        fs <- unpermuted_samples[, c, idx]
        var <- welford_summary(fs)[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }
  
    for (c in 1:C) {
      fs <- unpermuted_samples[, c, idx]
      
      # Check tail khats in each Markov chain
      khat_threshold <- 0.75
      khats <- compute_tail_khats(fs)
      if (khats[1] >= khat_threshold & khats[2] >= khat_threshold) {
        no_khat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                sprintf('  Chain %s: Both left and right tail hat{k}s ', c),
                sprintf('(%.3f, %.3f) exceed %.2f!\n', 
                        khats[1], khats[2], khat_threshold))
      } else if (khats[1] < khat_threshold & khats[2] >= khat_threshold) {
        no_khat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Right tail hat{k} (%.3f) exceeds %.2f!\n',
                         c, khats[2], khat_threshold))
      } else if (khats[1] >= khat_threshold & khats[2] < khat_threshold) {
        no_khat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Left tail hat{k} (%.3f) exceeds %.2f!\n',
                 c, khats[1], khat_threshold))
      }
      
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        no_zvar_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: Expectand has vanishing', c),
                 ' empirical variance!\n')
      }
    }
  
    # Check split Rhat across Markov chains
    chains <- lapply(1:C, function(c) unpermuted_samples[,c,idx])
    rhat <- compute_split_rhat(chains)

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
      # Check empirical integrated autocorrelation time
      fs <- unpermuted_samples[,c, idx]
      int_ac_time <- compute_tauhat(fs)
      if (int_ac_time / N > 0.25) {
        no_tauhat_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: hat{tau} per iteration (%.3f)',
                         c, int_ac_time / N),
                 ' exceeds 0.25!\n')
      }

      # Check empirical effective sample size
      neff <- N / int_ac_time
      if (neff < min_neff_per_chain) {
        no_neff_warning <- FALSE
        local_warning <- TRUE
        local_message <-
          paste0(local_message,
                 sprintf('  Chain %s: hat{ESS} (%.3f) is smaller than',
                         c, neff),
                 sprintf(' desired (%s)!\n', min_neff_per_chain))
      }
    }
    local_message <- paste0(local_message, '\n')
    if (local_warning) {
      message <- paste0(message, local_message)
    }
  }

  if (!no_khat_warning) {
    message <- paste0(message,
                      'Large tail hat{k}s suggest that the expectand',
                      ' might not be sufficiently integrable.\n\n')
  }
  if (!no_zvar_warning) {
    message <- paste0(message,
                      'If the expectands are not constant then zero empirical',
                      ' variance suggests that the Markov',
                      ' transitions are misbehaving.\n\n')
  }
  if (!no_rhat_warning) {
    message <- paste0(message,
                      'Split hat{R} larger than 1.1 is inconsisent with',
                      ' equilibrium.\n\n')
  }
  if (!no_tauhat_warning) {
    message <- paste0(message,
                      'hat{tau} larger than a quarter of the Markov chain',
                      ' length suggests that Markov chain Monte Carlo,',
                      ' estimates will be unreliable.\n\n')
  }
  if (!no_neff_warning) {
    message <- paste0(message,
                      'If hat{ESS} is too small then reliable Markov chain',
                      ' Monte Carlo estimators may still be too imprecise.\n\n')
  }

  if(no_khat_warning & no_zvar_warning & no_rhat_warning & no_tauhat_warning & no_neff_warning) {
    message <- paste0('All expectands checked appear to be behaving',
                      ' well enough for reliable Markov chain Monte Carlo estimation.\n')
  }

  cat(message)
}

# Summarize expectand diagnostics
expectand_diagnostics_summary <- function(fit,
                                          expectand_idxs=NULL,
                                          min_neff_per_chain=100,
                                          exclude_zvar=FALSE) {
  unpermuted_samples <- rstan:::extract(fit, permute=FALSE)

  input_dims <- dim(unpermuted_samples)
  N <- input_dims[1]
  C <- input_dims[2]
  I <- input_dims[3]

  if (is.null(expectand_idxs)) {
    expectand_idxs <- 1:I
  }

  bad_idxs <- setdiff(expectand_idxs, 1:I)
  if (length(bad_idxs) > 0) {
    cat(sprintf('Excluding the invalid expectand indices: %s',
                bad_idxs))
    expectand_idxs <- setdiff(expectand_idxs, bad_idxs)
  }

  failed_idx <- c()
  failed_khat_idx <- c()
  failed_zvar_idx <- c()
  failed_rhat_idx <- c()
  failed_tauhat_idx <- c()
  failed_neff_idx <- c()

  for (idx in expectand_idxs) {
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        fs <- unpermuted_samples[, c, idx]
        var <- welford_summary(fs)[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }

    for (c in 1:C) {
      # Check tail khats in each Markov chain
      fs <- unpermuted_samples[,c, idx]
      khats <- compute_tail_khats(fs)
      khat_threshold <- 0.75
      if (khats[1] >= khat_threshold | khats[2] >= khat_threshold) {
        failed_idx <- c(failed_idx, idx)
        failed_khat_idx <- c(failed_khat_idx, idx)
      }
    
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        failed_idx <- c(failed_idx, idx)
        failed_zvar_idx <- c(failed_zvar_idx, idx)
      }
    }
  
    # Check split Rhat across Markov chains
    chains <- lapply(1:C, function(c) unpermuted_samples[,c,idx])
    rhat <- compute_split_rhat(chains)

    if (is.nan(rhat)) {
      failed_idx <- c(failed_idx, idx)
      failed_rhat_idx <- c(failed_rhat_idx, idx)
    } else if (rhat > 1.1) {
      failed_idx <- c(failed_idx, idx)
      failed_rhat_idx <- c(failed_rhat_idx, idx)
    }

    for (c in 1:C) {
      # Check empirical integrated autocorrelation time
      fs <- unpermuted_samples[,c, idx]
      int_ac_time <- compute_tauhat(fs)
      if (int_ac_time / N > 0.25) {
        failed_idx <- c(failed_idx, idx)
        failed_tauhat_idx <- c(failed_tauhat_idx, idx)
      }

      # Check empirical effective sample size
      neff <- N / int_ac_time
      if (neff < min_neff_per_chain) {
        failed_idx <- c(failed_idx, idx)
        failed_neff_idx <- c(failed_neff_idx, idx)
      }
    }
  }
  
  failed_idx <- unique(failed_idx)
  if (length(failed_idx)) {
    cat(sprintf('The expectands %s triggered diagnostic warnings.\n\n',
                paste(failed_idx, collapse=", ")))
  } else {
    cat(paste0('All expectands checked appear to be behaving',
               'well enough for Markov chain Monte Carlo estimation.\n'))
  }

  failed_khat_idx <- unique(failed_khat_idx)
  if (length(failed_khat_idx)) {
    cat(sprintf('The expectands %s triggered hat{k} warnings.\n',
                paste(failed_khat_idx, collapse=", ")))
    cat(paste0('  Large tail hat{k}s suggest that the expectand',
               ' might not be sufficiently integrable.\n\n'))
  }
        
  failed_zvar_idx <- unique(failed_zvar_idx)
  if (length(failed_zvar_idx)) { 
    cat(sprintf('The expectands %s triggered zero variance warnings.\n',
                paste(failed_zvar_idx, collapse=", ")))
    cat(paste0('  If the expectands are not constant then zero empirical',
               ' variance suggests that the Markov',
               ' transitions are misbehaving.\n\n'))
  }
  
  failed_rhat_idx <- unique(failed_rhat_idx)
  if (length(failed_rhat_idx)) {
    cat(sprintf('The expectands %s triggered hat{R} warnings.\n',
                paste(failed_rhat_idx, collapse=", ")))
    cat(paste0('  Split hat{R} larger than 1.1 is inconsistent with', 
               ' equilibrium.\n\n'))
  }
  
  failed_tauhat_idx <- unique(failed_tauhat_idx)
  if (length(failed_tauhat_idx)) {
    cat(sprintf('The expectands %s triggered hat{tau} warnings.\n',
                paste(failed_tauhat_idx, collapse=", ")))
    cat(paste0('  hat{tau} larger than a quarter of the Markov chain',
               ' length suggests that Markov chain Monte Carlo',
               ' estimates may be unreliable.\n\n'))
  }
  
  failed_neff_idx <- unique(failed_neff_idx)
  if (length(failed_neff_idx)) {
    cat(sprintf('The expectands %s triggered hat{ESS} warnings.\n',
                paste(failed_neff_idx, collapse=", ")))
    cat(paste0('  If hat{ESS} is too small then even reliable Markov chain',
               ' Monte Carlo estimators may still be too imprecise.\n\n'))
  }
}

# Summarize Hamiltonian Monte Carlo and expectand diagnostics
# into a binary encoding
summarize_all_diagnostics <- function(fit,
                                      adapt_target=0.801,
                                      max_treedepth=10,
                                      expectand_idxs=NULL,
                                      min_neff_per_chain=100,
                                      exclude_zvar=FALSE) {
  warning_code <- 0
    
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)

  # Check divergences
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  n = sum(divergent)
  N = length(divergent)

  if (n > 0) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 0))
  }

  # Check transitions that ended prematurely due to maximum tree depth limit
  treedepths <- do.call(rbind, sampler_params)[,'treedepth__']
  n = length(treedepths[sapply(treedepths, function(x) x >= max_treedepth)])
  N = length(treedepths)

  if (n > 0) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 1))
  }

  # Checks the energy fraction of missing information (E-FMI)
  no_efmi_warning <- TRUE
  for (c in 1:length(sampler_params)) {
    energies = sampler_params[c][[1]][,'energy__']
    numer = sum(diff(energies)**2) / length(energies)
    denom = var(energies)
    if (numer / denom < 0.2) {
      no_efmi_warning <- FALSE
    }
  }
  if (!no_efmi_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 2))
  }

  # Check convergence of the stepsize adaptation
  no_accept_warning <- TRUE
  for (c in 1:length(sampler_params)) {
    ave_accept_proxy <- mean(sampler_params[[c]][,'accept_stat__'])
    if (ave_accept_proxy < 0.9 * adapt_target) {
      no_accept_warning <- FALSE
    }
  }
  if (!no_accept_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 3))
  }
  
  unpermuted_samples <- rstan:::extract(fit, permute=FALSE)

  input_dims <- dim(unpermuted_samples)
  N <- input_dims[1]
  C <- input_dims[2]
  I <- input_dims[3]

  if (is.null(expectand_idxs)) {
    expectand_idxs <- 1:I
  }

  bad_idxs <- setdiff(expectand_idxs, 1:I)
  if (length(bad_idxs) > 0) {
    cat(sprintf('Excluding the invalid expectand indices: %s',
                bad_idxs))
    expectand_idxs <- setdiff(expectand_idxs, bad_idxs)
  }

  khat_warning <- FALSE
  zvar_warning <- FALSE
  rhat_warning <- FALSE
  tauhat_warning <- FALSE
  neff_warning <- FALSE

  for (idx in expectand_idxs) {
    if (exclude_zvar) {
      # Check zero variance across all Markov chains for exclusion
      any_zvar <- FALSE
      for (c in 1:C) {
        fs <- unpermuted_samples[, c, idx]
        var <- welford_summary(fs)[2]
        if (var < 1e-10)
          any_zvar <- TRUE
      }
      if (any_zvar) {
        next
      }
    }

    for (c in 1:C) {
      # Check tail khats in each Markov chain
      fs <- unpermuted_samples[,c, idx]
      khats <- compute_tail_khats(fs)
      khat_threshold <- 0.75
      if (khats[1] >= khat_threshold | khats[2] >= khat_threshold) {
        khat_warning <- TRUE
      }
    
      # Check empirical variance in each Markov chain
      var <- welford_summary(fs)[2]
      if (var < 1e-10) {
        zvar_warning <- TRUE
      }
    }
  
    # Check split Rhat across Markov chains
    chains <- lapply(1:C, function(c) unpermuted_samples[,c,idx])
    rhat <- compute_split_rhat(chains)

    if (is.nan(rhat)) {
      rhat_warning <- TRUE
    } else if (rhat > 1.1) {
      rhat_warning <- TRUE
    }

    for (c in 1:C) {
      # Check empirical integrated autocorrelation time
      fs <- unpermuted_samples[,c, idx]
      int_ac_time <- compute_tauhat(fs)
      if (int_ac_time / N > 0.25) {
        tauhat_warning <- TRUE
      }

      # Check empirical effective sample size
      neff <- N / int_ac_time
      if (neff < min_neff_per_chain) {
        neff_warning <- TRUE
      }
    }
  }
  
  if (khat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 4))
  }
        
  if (zvar_warning) { 
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 5))
  }
  
  if (rhat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 6))
  }
  
  if (tauhat_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 7))
  }
  
  if (neff_warning) {
    warning_code <- bitwOr(warning_code, bitwShiftL(1, 8))
  }
  
  (warning_code)
}

parse_warning_code <- function(warning_code) {
  if (bitwAnd(warning_code, bitwShiftL(1, 0)))
    print("  divergence warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 1)))
    print("  treedepth warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 2)))
    print("  E-FMI warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 3)))
    print("  average acceptance proxy warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 4)))
    print("  khat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 5)))
    print("  zero variance warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 6)))
    print("  Rhat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 7)))
    print("  tauhat warning")
  if (bitwAnd(warning_code, bitwShiftL(1, 8)))
    print("  min effective sample size warning")
}

# Visualize empirical autocorrelations for a given sequence
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

# Plot empirical correlograms for the expectand output ensembels in a 
# collection of Markov chains
plot_empirical_correlogram <- function(unpermuted_fs,
                                       max_L,
                                       rholim=c(-0.2, 1.1),
                                       name="") {
  idx <- rep(0:max_L, each=2)
  xs <- sapply(1:length(idx), function(b) if(b %% 2 == 0) idx[b] + 0.5
                                          else idx[b] - 0.5)

  plot(0, type="n", main=name,
       xlab="Lag", xlim=c(-0.5, max_L + 0.5),
       ylab="Empirical Autocorrelation", ylim=rholim)
  abline(h=0, col="#DDDDDD", lty=2, lwd=2)

  colors <- c(c_dark, c_mid_highlight, c_mid, c_light_highlight)
  for (c in 1:4) {
    fs <- unpermuted_fs[,c]
    rhos <- compute_rhos(fs)
    pad_rhos <- unlist(lapply(idx, function(n) rhos[n + 1]))
    lines(xs, pad_rhos, lwd=2, col=colors[c])
  }
}

# Plot two expectand output ensembles againt each other separated by
# Markov chain 
plot_chain_sep_pairs <- function(unpermuted_f1s, name_x,
                                 unpermuted_f2s, name_y) {
  N <- dim(unpermuted_f1s)[1]

  nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C",
                  "#A25050", "#8F2727", "#7C0000")
  cmap <- colormap(colormap=nom_colors, nshades=N)

  min_x <- min(sapply(1:4, function(c) min(unpermuted_f1s[,c])))
  max_x <- max(sapply(1:4, function(c) max(unpermuted_f1s[,c])))

  min_y <- min(sapply(1:4, function(c) min(unpermuted_f2s[,c])))
  max_y <- max(sapply(1:4, function(c) max(unpermuted_f2s[,c])))

  par(mfrow=c(2, 2), mar = c(5, 5, 3, 1))

  for (c in 1:4) {
    plot(0, type="n", main=paste("Chain", c),
         xlab=name_x, xlim=c(min_x, max_x),
         ylab=name_y, ylim=c(min_y, max_y))
  
    points(unlist(lapply(1:4, function(c) unpermuted_f1s[,c])),
           unlist(lapply(1:4, function(c) unpermuted_f2s[,c])),
           col="#DDDDDD", pch=16, cex=1.0)
    points(unpermuted_f1s[,c], unpermuted_f2s[,c],
         col=cmap, pch=16, cex=1.0)
  }
}

# Evaluate an expectand along a Markov chain
pushforward_chains <- function(chains, expectand) {
  lapply(chains, function(c) sapply(c, function(x) expectand(x)))
}

# Estimate expectand expectation value from a Markov chain
mcmc_est <- function(fs) {
  N <- length(fs)
  if (N == 1) {
    return(c(fs[1], 0, NaN))
  }

  summary <- welford_summary(fs)

  if (summary[2] == 0) {
    return(c(summary[1], 0, NaN))
  }

  int_ac_time <- compute_tauhat(fs)
  neff <- N / int_ac_time
  return(c(summary[1], sqrt(summary[2] / neff), neff))
}

# Estimate expectand exectation value from a collection of Markov chains
ensemble_mcmc_est <- function(chains) {
  C <- length(chains)
  chain_ests <- lapply(chains, function(c) mcmc_est(c))

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

# Plot pushforward histogram of a given expectand using Markov chain
# Monte Carlo estimators to estimate the output bin probabilities
plot_pushforward_hist <- function(unpermuted_samples, B, flim=NULL, name="f") {
  if (is.null(flim)) {
    # Automatically adjust histogram binning to range of outputs
    min_f <- min(unpermuted_samples)
    max_f <- max(unpermuted_samples)
    
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
  
  mean_p <- rep(0, B)
  delta_p <- rep(0, B)

  C <- dim(unpermuted_samples)[2]
  chains <- lapply(1:C, function(c) unpermuted_samples[,c])

  for (b in 1:B) {
    bin_indicator <- function(x) {
      ifelse(bins[b] <= x & x < bins[b + 1], 1, 0)
    }
    indicator_chains <- pushforward_chains(chains, bin_indicator)
    est <- ensemble_mcmc_est(indicator_chains)
  
    # Normalize bin probabilities by bin width to allow
    # for direct comparison to probability density functions
    width = bins[b + 1] - bins[b]
    mean_p[b] = est[1] / width
    delta_p[b] = est[2] / width
  }

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
       xlim=flim, xlab=name,
       ylim=c(min_y, max_y), ylab="", yaxt="n")
  title(ylab="Estimated Bin\nProbabilities", mgp=c(1, 1, 0))

  polygon(c(x, rev(x)), c(lower_inter, rev(upper_inter)),
          col = c_light, border = NA)
  lines(x, mean_p[idx], col=c_dark, lwd=2)
}
