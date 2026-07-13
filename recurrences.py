import numpy as np
import math

from distances import dist_m, dist_p

theta = np.array([0, 3/5*np.pi, np.pi/4, 3/8 *np.pi])

n = len(theta)

#init matrices full of nothing, indexing is i,j, alpha
u_p = np.full((n, n, n), '#', dtype=object)
u_m = np.full((n, n, n), '#', dtype=object)

v_p = np.full((n, n, n), '#', dtype=object)
v_m = np.full((n, n, n), '#', dtype=object)

s_p = np.full((n, n), '#', dtype=object)
s_m = np.full((n, n), '#', dtype=object)

phi_p = np.full((n, n), '#', dtype=object)
phi_m = np.full((n, n), '#', dtype=object)

delta_p = np.full((n, n), '#', dtype=object)
delta_m = np.full((n, n), '#', dtype=object)

lambda_p = np.full((n), '#', dtype=object)
lambda_m = np.full((n), '#', dtype=object)


#seeds s and phi
for i in range(1,n):
    s_m[i][0] = 0
    phi_m[i][0] = 0

    s_p[i][0] = 0
    phi_p[i][0] = 2*np.pi


for j in range(1,n):
    for i in range(n):
        if (phi_m[i][j-1] == '#' or 
            (dist_m(theta[i], theta[j]) < dist_m(theta[i], phi_m[i][j-1])
            and dist_m(theta[i], theta[j]) != 0
        )): #argmax update
            s_m[i][j] = j
            phi_m[i][j] = theta[j]
        else: #no argmax update
            s_m[i][j] = s_m[i][j-1]
            phi_m[i][j] = phi_m[i][j-1]

        if (phi_p[i][j-1] == '#' or 
            (dist_p(theta[i], theta[j]) < dist_p(theta[i], phi_p[i][j-1])
            and dist_p(theta[i], theta[j]) != 0
        )): #argmin update
            s_p[i][j] = j
            phi_p[i][j] = theta[j]
        else: #no argmin update
            s_p[i][j] = s_p[i][j-1]
            phi_p[i][j] = phi_p[i][j-1]    


delta_m[0][0] = 1
delta_p[0][0] = 1

for i in range(1, n):
    if dist_m(theta[i], phi_m[i][0]) > 2/3 * np.pi:
        delta_m[i][0] = 1
    else:
        delta_m[i][0] = 1 / (2 * math.cos(dist_m(theta[i], phi_m[i][0]) / 2))
    
    if dist_p(theta[i], phi_p[i][0]) > 2/3 * np.pi:
        delta_p[i][0] = 1
    else:
        delta_p[i][0] = 1 / (2 * math.cos(dist_p(theta[i], phi_p[i][0]) / 2))

for j in range(1, n):
    for i in range(n):
        if dist_m(theta[i], phi_m[i][j]) > 2/3 * np.pi:
            delta_m[i][j] = 1
        else:
            delta_m[i][j] = 1 / (2 * math.cos(dist_m(theta[i], phi_m[i][j]) / 2))
        
        if dist_p(theta[i], phi_p[i][j]) > 2/3 * np.pi:
            delta_p[i][j] = 1
        else:
            delta_p[i][j] = 1 / (2 * math.cos(dist_p(theta[i], phi_p[i][j]) / 2))

print(delta_m)
print(delta_p)

lambda_m[0] = 0
lambda_p[0] = 0
for i in range(1,n):
    if dist_m(theta[i], phi_m[i][i]) >= 2/3 * np.pi:
        lambda_m[i] = 0
    if dist_m(theta[i], phi_m[i][i]) < 2/3 * np.pi:
        lambda_m[i] = lambda_p[s_m[i][i]] + 1

    if dist_p(theta[i], phi_p[i][i]) >= 2/3 * np.pi:
        lambda_p[i] = 0
    if dist_p(theta[i], phi_p[i][i]) < 2/3 * np.pi:
        lambda_p[i] = lambda_m[s_p[i][i]] + 1

print(lambda_m)
print(lambda_p)