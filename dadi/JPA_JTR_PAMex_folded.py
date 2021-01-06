# Numpy is the numerical library dadi is built upon
from numpy import array

import dadi

# In demographic_models.py, we've defined a custom model for this problem
import demographic_models

# Load the data
# Parse the data file to generate the data dictionary
dd = dadi.Misc.make_data_dict('dadi_input_JPA_JTR.tsv')

# Extract the spectrum for ['JPA','JTR'] from that dictionary, with both
# projected down to x samples per population.

# If we didn't have outgroup information, we could use the folded  version 
# of the fs.
data = dadi.Spectrum.from_data_dict(dd, ['JPA','JTR'], [49,41],
                                      polarized=False)


ns = data.sample_sizes

# These are the grid point settings will use for extrapolation.
pts_l = [60,50]

# The Demographics1D and Demographics2D modules contain a few simple models,
# mostly as examples. We could use one of those.
# func = dadi.Demographics2D.split_mig
# Instead, we'll work with our custom model
func = demographic_models.PAMex


# Now let's optimize parameters for this model.

# The upper_bound and lower_bound lists are for use in optimization.
# Occasionally the optimizer will try wacky parameter values. We in particular
# want to exclude values with very long times, very small population sizes, or
# very high migration rates, as they will take a long time to evaluate.
# Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
#     nu1, nu2, Ts, O = params SI
#  nu1, nu2, m12, m21, Ts, O = params IM
upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12]
lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0]

# This is our initial guess for the parameters, which is somewhat arbitrary.
params = ('nu1a', 'nu2a', 'nu1', 'nu2', 'm12', 'm21', 'Tam', 'Ts', 'Te')
p0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# Make the extrapolating version of our demographic model function.
func_ex = dadi.Numerics.make_extrap_log_func(func)

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
                              lower_bound=lower_bound)
                              
# Do the optimization. By default we assume that theta is a free parameter,
# since it's trivial to find given the other parameters. If you want to fix
# theta, add a multinom=False to the call.
# The maxiter argument restricts how long the optimizer will run. For real 
# runs, you will want to set this value higher (at least 10), to encourage
# better convergence. You will also want to run optimization several times
# using multiple sets of intial parameters, to be confident you've actually
# found the true maximum likelihood parameters.
print('Beginning optimization ************************************************')
popt = dadi.Inference.optimize_log(p0, data, func_ex, pts_l, 
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound,
                                   verbose=len(p0), maxiter=10)
# The verbose argument controls how often progress of the optimizer should be
# printed. It's useful to keep track of optimization process.
print('Finished optimization **************************************************')


# These are the actual best-fit model parameters, which we found through
# longer optimizations and confirmed by running multiple optimizations.


# Calculate the best-fit model AFS.
model = func_ex(popt, ns, pts_l)


# Computation of statistics
ll_opt = dadi.Inference.ll_multinom(model, data)
theta = dadi.Inference.optimal_sfs_scaling(model, data)
AIC = 2*len(params)-2*ll_opt


# Print results
print 'Optimized parameters', repr(popt)
print 'Optimized log-likelihood:', ll_opt
print 'theta:', theta
print 'AIC:', AIC
	
# Plot a comparison of the resulting fs with the data.
import pylab
pylab.figure(1)

dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=1, resid_range=3,
			    pop_ids =('JPA','JTR'))

pylab.show()



