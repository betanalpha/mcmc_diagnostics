General Markov chain Monte Carlo, and specific Hamiltonian Monte Carlo, diagnostic code
compatible with RStan and PyStan2.

The diagnostics can also be fully adapted to any other Hamiltonain Monte Carlo 
implementation by reimplementing the `extract_expectands`,  
`extract_hmc_diagnostics`, and `plot_inv_metric` functions.

Recommendations for code optimization are welcomed and appreciated.

### Acknowledgements {-}

I thank Sean Talts for Python code improvements.  Raoul Kima originally suggested separating 
divergent transitions by numerical trajectory length.
