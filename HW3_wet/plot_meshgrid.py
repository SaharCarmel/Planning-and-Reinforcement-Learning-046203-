import matplotlib.pyplot as plt
import numpy as np
from radial_basis_function_extractor import RadialBasisFunctionExtractor


x = np.linspace(-0.12, 0.6, 1000)
y = np.linspace(-0.07,0.07, 1000)
xx, yy = np.meshgrid(x, y)

all_points = []
for i in range(1000):
    for j in range(1000):
        all_points.append([x[i],y[j]])

RBF = RadialBasisFunctionExtractor([12,10])
res = RBF.encode_states_with_radial_basis_functions(np.array(all_points))[:,:2]

mat_feature_1 = np.zeros([1000,1000])
mat_feature_2 = np.zeros([1000,1000])
for i in range(1000):
    for j in range(1000):
        mat_feature_1[i,j] = res[i*1000 + j][0]
        mat_feature_2[i,j] = res[i*1000 + j][1]


h1 = plt.contourf(x,y,mat_feature_1)
plt.title('Feature 1')
plt.ylabel('Velocity')
plt.xlabel('Position')
plt.show()

h2 = plt.contourf(x,y,mat_feature_2)
plt.title('Feature 2')
plt.ylabel('Velocity')
plt.xlabel('Position')
plt.show()