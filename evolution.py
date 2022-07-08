import cma
import numpy as np
optimizer=cma.CMAEvolutionStrategy(x0=[100,100],sigma0=1)
def function(x):
    return x[0]**2+x[1]**2

optimizer.optimize(objective_fct=function,verb_disp=1)
print(optimizer.result.xbest)