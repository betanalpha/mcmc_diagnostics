General Markov chain Monte Carlo, and specific Hamiltonian Monte Carlo,
diagnostic code compatible with RStan, PyStan2, and PyStan3.

The diagnostics can also be fully adapted to any other Hamiltonian Monte Carlo 
implementation by re-implementing the `extract_expectands`,
`extract_hmc_diagnostics`, and `plot_inv_metric` functions.

Recommendations for code optimization are welcomed and appreciated.

### Acknowledgements {-}

I thank Sean Talts and Dan Waxman for Python code improvements.  Raoul Kima
originally suggested separating divergent transitions by numerical trajectory
length.
