import numpy as np

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
            
            sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=rand_state)
            x_data_lhs = sampling(n_lhs_samp[i])
            x_data = np.append(x_data,x_data_lhs,axis=0)
            n_samp = n_samp + n_lhs_samp[i]
    print(x_data)
    computed_values = x_data.flatten()
    expected_samples = np.array([[0.48283585,4.06777601],[0.13995167,4.08199814],[0.2828112,4.19778379],[0.90297782,4.00409338],[0.70717928,4.14655179]])
    expected_values = expected_samples.flatten()
    tolerances = [1e-7]*n_dim*n_lhs_samp[0]
    for i in range(len(expected_values)):
        assert abs(expected_values[i] - computed_values[i]) < tolerances[i]
    print('Test lhs passed!') 

if __name__ == '__main__':
   test_lhs()