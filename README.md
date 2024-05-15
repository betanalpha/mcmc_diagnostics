A suite of analysis and diagnostics tools in `R` and `python` for working with
Markov chain Monte Carlo generally and Hamiltonian Monte Carlo specifically.
The suite includes functions for interfacing with `RStan`, `PyStan2`, and
`PyStan3` and notebooks demonstrating their use.

These tools can also be interfaced with any Hamiltonian Monte Carlo code by
implementing appropriate `extract_expectands`, `extract_hmc_diagnostics`, and
`plot_inv_metric` functions.

Recommendations for code optimization are welcomed and appreciated.

### Acknowledgements {-}

I thank Sean Talts and Dan Waxman for Python code improvements.  Raoul Kima
originally suggested separating divergent transitions by numerical trajectory
length.
