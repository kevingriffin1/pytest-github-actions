import numpy as np
import sys
sys.path.insert(0, '../') # add the path to the AdaptiveComputing directory
from ac_common import *

def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]

def test_lhs():
    n_lhs_samp = np.atleast_1d(5)
    from smt.sampling_methods import LHS
    deterministic = True
    n_fl = 1
    n_samp = 0
    n_dim = 2
    x_data = np.empty([0,n_dim])
    xlimits = np.array([[0, 1], [4,4.2]])
    for i in range(n_fl):
        if n_lhs_samp[i] > 0: 
            if n_lhs_samp[i] <= 1:
                raise Exception('LatinHypercubeSampler requires n_lhs_samp ==0 or >=2')
            rand_state = np.random.RandomState()
            if deterministic:
                # Ensurses the fidelity levels all have unique seeds.
                # Note: that the sampler will increment the rand_state for each sample
                rand_state = int(n_samp+1.0 + sum(n_lhs_samp[:i]))
            
            sampling = LHS(xlimits=xlimits,criterion='maximin',random_state=rand_state)
            x_data_lhs = sampling(n_lhs_samp[i])
            x_data = np.append(x_data,x_data_lhs,axis=0)
            n_samp = n_samp + n_lhs_samp[i]
    print(x_data)
    computed_values = x_data.flatten()
    expected_samples = np.array([[0.48283585,4.06777601],[0.13995167,4.08199814]
                                 ,[0.2828112,4.19778379],[0.90297782,4.00409338]
                                 ,[0.70717928,4.14655179]])
    expected_values = expected_samples.flatten()
    tolerances = [1e-7]*n_dim*n_lhs_samp[0]
    for i in range(len(expected_values)):
        assert abs(expected_values[i] - computed_values[i]) < tolerances[i]
    print('Test lhs passed!') 

def func_mt(x):
    # evaluate the categorical variable
    if x[2] == 'a':
        s = 10.
    elif x[2] == 'b':
        s = 5.
    elif x[2] == 'c':
        s = 7.5
    elif x[2] == 'd':
        s = 6.
    else:
        raise Exception('Unrecognized value for categorical variable x[2]')
    return pow((x[0]-5.0),2.0) + pow((x[1]-4.0),2.0) + s - 5.0

def test_bo():
    x0 = Param()
    x0.type = 'continuous'
    x0.min_val = 0
    x0.max_val = 8

    # Use an ordered integer when the order of the discrete values has significance,
    # that is, we expect neigboring values to have objective function values that are correlated
    x1 = Param()
    x1.type = 'ordered'
    x1.min_val = 2
    x1.max_val = 6 # domain: 2,3,4,5,6.

    # Use categorical type if the order of the categories is arbitrary.
    x2 = Param()
    x2.type = 'categorical'
    x2.categories = ['a','b','c','d']

    params = [x0, x1, x2]
    
    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Perform the optimization
    my_model = Model(func_mt, params, mod_ops)
    my_model.add_lhs_samples(8)
    viz_ops = VizOptions()
    # viz_ops.plot_nd=True
    my_model.add_bo_samples(25,viz_ops=viz_ops)
    [x_opt, y_opt] = my_model.find_min()
    print('The minimum should be y = 0 at the location [x0_opt, x1_opt, x2_opt] = [5, 4, b]')
    print('The minimum found is y = ', y_opt, ' at the location [', x_opt[0],', ',x_opt[1],', ',x2.categories[int(x_opt[2])],']')
    computed_values = [x_opt[0], x_opt[1], x_opt[2], y_opt[0]]
    expected_values = [5.0, 4.0, 1.0, 0.0] # Note: 1 maps to 'b' for x2
    assert(x2.categories[int(x_opt[2])]=='b')
    tolerances = [0.2, 1e-12, 1e-12, 0.2]
   
    for i in range(len(expected_values)):
        assert abs(expected_values[i] - computed_values[i]) < tolerances[i]
    print('Test bo passed!') 

if __name__ == '__main__':
#    test_lhs()
   test_bo()
