#########################################################
# acquisitionFunctions.py
# Available acquisition functions:
# - SBO (surrogate based optimization): directly using the prediction of the surrogate model ($\mu$)
# - LCB (Lower Confidence bound): using the confidence interval : $\mu -3 \times \sigma$
# - EI for expected Improvement (EGO)
# Note: be sure to add any user defined aquisition function to get_acq_func
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize, brute
#########################################################
# expected improvement: 
def EI(GP,points,f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    if var.size == 1 and var == 0.0:  
        raise Exception('Must evaluate EI for more than one point.') #return 0.0
    ei = args1 + args2
    return ei
#########################################################
# surrogate Based optimization: min the Surrogate model by using the mean mu
def SBO(GP,points):
    res = GP.predict_values(points)
    return res
#########################################################
# lower confidence bound optimization: minimize by using mu - 3*sigma
def LCB(GP,points):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    res = pred-3.*np.sqrt(var)
    return res
#########################################################
# maximal standard deviation: tries to minimize the max standard deviation
def MSD(GP,points):
    var = GP.predict_variances(points)
    res = -np.sqrt(var)
    return res
#########################################################
def get_acq_func(IC,gpr,f_min_k):
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k)[:,0]
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x))
    elif IC == 'LCB':
        obj_k = lambda x: LCB(gpr,np.atleast_2d(x))
    elif IC == 'MSD':
        obj_k = lambda x: MSD(gpr,np.atleast_2d(x))
    else:
        raise Exception('Unrecognized acq_func specified.')
    return obj_k

#########################################################
# Perform a constrained optimization of the continous arguments for a specific choice
# of the integer arguments
# Input: xi is a list of integer arguments (specifies all of the ordered and categorical types)
# Output #2: the optimal choice of continous arguments (for the particular choice of input integer arguments)
# that minimizes the objective function obj_k
# Output #1: the minimum achieved for this choice of continuous and integer arguments.
def min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,xi):
    opt_all = np.array([])
    for i_s in range(bo_ops.n_opt_pts):
        opt_all = np.append(opt_all,minimize(lambda xf: float(obj_k(np.append(xf,xi))), xc_start[i_s], method=bo_ops.minimization_method, bounds=xclimits_num))
    opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the entries of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to opt_all[0].success
    obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
    ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
    opt = opt_success[ind_min] # the full output for the best initial guess
    xf_opt = opt['x'] # the x value at which the min occurs
    return np.min(obj_success),xf_opt

#########################################################
# Return the argmin of the objective function obj_k
# Minimization of continuous arguments uses scipy minimize
# Minimization of integer arguments uses scipy brute
def minimize_acq_func(model,obj_k,x_start,bo_ops):
    # # naive random sampling:
    # #x_start = np.zeros([bo_ops.n_opt_pts,n_dim])
    # #for i_r in range(n_dim):
    # #    x_start[:,i_r] = np.random.rand(bo_ops.n_opt_pts)*(xlimits[i_r][1]-xlimits[i_r][0])+xlimits[i_r][0]
    # opt_all = np.array([])
    # # The initial points for the minimization are random, but scipy minimize will use the numpy random
    # # seed in performing the minimization algorithm. Set this deterministically.
    # np.random.seed(7) # XXX not sure if this is needed
    # for i_s in range(bo_ops.n_opt_pts):
    #     # minimization_method values that are sometimes appropriate:
    #     # Powell: slow for continuous. Works for virtualEngineering. Doesn't work with enumerated types (it doesn't understand if a list of points is integer points).Warns initial guess not in specified bounds for mixed types
    #     # SLSQP: fast for continuous. works for mixed types. `x0` violates bound constraints for virtualEngineering
    #     # L-BFGS-B: fast for continuous. very slow for mixed types
    #     # TNC: bit slower than SLSQP for continuous. mixed types raises: `x0` violates bound constraints.
    #     # minimization_method values that should not be used:
    #     # CG, BFGS, Newton-CG, COBYLA: can not handle bounds. Nelder-Mead: version on Eagle can not handle bounds
    #     # trust-constr: warnings from approximate Hessian
    #     # dogleg, trust-ncg, trust-exact, trust-krylov: Jacobian required
    #     #opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s,:], method=bo_ops.minimization_method, bounds=xlimits_num))
    #     opt_all = np.append(opt_all,minimize(lambda x0: float(obj_k(np.append(x0,2,'a'))), x_start[i_s,:], method=bo_ops.minimization_method, bounds=xlimits_num))
    # opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the entries of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to opt_all[0].success
    # obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
    # ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
    # opt = opt_success[ind_min] # the full output for the best initial guess
    # x_et_k = opt['x'] # the x value at which the min occurs
    # return x_et_k # note that y_et_k will be the objective function rather than the acquisition function value at this point

    xc_start = x_start[:,0:model.n_cont_vars]
    xclimits_num = model.xlimits_num[0:model.n_cont_vars]
    
    ranges = ()
    for i in range(model.n_cont_vars,model.n_dim):
        ranges = ranges+(slice(model.xlimits_num[i][0], model.xlimits_num[i][-1]+1, 1),)
    if model.n_cont_vars == 0: # if all data types are discrete
        x_et_k = brute(lambda x: float(obj_k(x)), ranges, finish=None)
    elif model.n_cont_vars == model.n_dim: # if all data types are continuous
        x_et_k = min_for_cont_vars([])[1]
    else: # if there are a mixture of continuous and discrete data types
        xi_opt = brute(lambda x: min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,x)[0], ranges, finish=None)
        xf_opt = min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,xi_opt)[1]
        x_et_k = np.append(xf_opt,xi_opt)
    return x_et_k


#########################################################