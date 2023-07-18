"""<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (EGO method) for optimal parameter selection for a simple 1D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm.</ol>
</div>

```python"""
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
"""```


Import the objective function defined in func_1d.py


```python"""
from func_1d import func_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_1d():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 25
    params = [x0]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Perform the optimization
    import time
    t = time.time()
    my_model2 = Model(func_1d, params, mod_ops)
    my_model2.add_lhs_samples(2)
    viz_ops = VizOptions()
    # viz_ops.animation_1d=True
    # my_model2.add_bo_samples(7,viz_ops=viz_ops)
    my_model2.add_bo_samples(1,viz_ops=viz_ops)
    [x_opt, y_opt] = my_model2.find_min()
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be approximately [x,y] = [18.9352,-15.1251]')
    print('The minimum found is [', x_opt[0], ',', y_opt,']')
    computed_values = [my_model2.x_data[0][-1,0],my_model2.y_data[0][-1,0]]
    # expected_values = [10.931257746200416, 5.205871653595352]
    expected_values = [4.430890164405155, 0.27181479523752633]
    tolerances = [1e-5, 1e-5]

    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_1d()
"""```"""
