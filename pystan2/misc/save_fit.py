import pystan

import multiprocessing
multiprocessing.set_start_method("fork")

import numpy

import stan_utility_pystan2 as util

model = util.compile_model('stan_programs/simu_logistic_reg.stan')
simu = model.sampling(iter=1, warmup=0, chains=1, chain_id=[1],
                      refresh=1000, seed=4838282,
                      algorithm="Fixed_param")

X = simu.extract()['X'][0]
y = simu.extract()['y'][0].astype(numpy.int64)

data = dict(M = 3, N = 1000, x0 = [-1, 0, 1], X = X, y = y)

interval_inits = [None] * 4

for c in range(4):
  beta = [0, 0, 0]
  alpha = 0.5
  interval_inits[c] = dict(alpha = alpha, beta = beta)

model = util.compile_model('stan_programs/bernoulli_linear.stan')
fit = model.sampling(data=data, seed=8438338, warmup=1000, iter=2024,
                     chain_id=[1, 2, 3, 4], refresh=0,
                     init=interval_inits)

import pickle
with open('model.obj','wb') as f:
  pickle.dump(model, f)
with open('fit.obj','wb') as f:
  pickle.dump(fit, f)
