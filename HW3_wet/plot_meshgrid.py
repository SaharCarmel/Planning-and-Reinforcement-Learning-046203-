import matplotlib.pyplot as plt
import numpy as np
from radial_basis_function_extractor import RadialBasisFunctionExtractor

RBF = RadialBasisFunctionExtractor([12,10])
x = np.linspace(-0.12, 0.6, 1000)
y = np.linspace(-0.07,0.07, 1000)
xx, yy = np.meshgrid(x, y)

z = xx+yy
h = plt.contourf(x,y,z)
plt.show()
np.array(list(zip(x,y)))

res = RBF.encode_states_with_radial_basis_functions(np.array(list(zip(x,y))))

first = np.expand_dims([z[0] for z in res],0)
second = np.expand_dims([z[1] for z in res],0)
h = plt.contourf(x,y,first)
plt.show()

h = plt.contourf(x,y,second)
plt.show()