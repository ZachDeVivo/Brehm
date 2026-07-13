import numpy as np
import math

from distances import dist_m, dist_p, get_new_magnitude

theta = np.array([0, np.pi, np.pi/2, np.pi/4, 3/8 * np.pi, 7/16 * np.pi])

n = len(theta)

#init matrices full of nothing, indexing is i,j, alpha
u_p = np.full((n, n, n), '#', dtype=object)
u_m = np.full((n, n, n), '#', dtype=object)

s_p = np.full((n, n), '#', dtype=object)
s_m = np.full((n, n), '#', dtype=object)

phi_p = np.full((n, n), '#', dtype=object)
phi_m = np.full((n, n), '#', dtype=object)

delta_p = np.full((n, n), '#', dtype=object)
delta_m = np.full((n, n), '#', dtype=object)

lambda_p = np.full((n), '#', dtype=object)
lambda_m = np.full((n), '#', dtype=object)


#seeds s and phi
for j in range(n):
    for i in range(j + 1):
        candidates = [k for k in range(j+1)]
        s_p[i][j] = min(candidates, key=lambda k: dist_p(theta[i], theta[k]))
        phi_p[i][j] = theta[s_p[i][j]]

        s_m[i][j] = min(candidates, key=lambda k: dist_m(theta[i], theta[k]))
        phi_m[i][j] = theta[s_m[i][j]]


for j in range(n):
    for i in range(j + 1):
        if dist_m(theta[i], phi_m[i][j]) > 2/3 * np.pi:
            delta_m[i][j] = 1
        else:
            delta_m[i][j] = 1 / (2 * math.cos(dist_m(theta[i], phi_m[i][j]) / 2))
        
        if dist_p(theta[i], phi_p[i][j]) > 2/3 * np.pi:
            delta_p[i][j] = 1
        else:
            delta_p[i][j] = 1 / (2 * math.cos(dist_p(theta[i], phi_p[i][j]) / 2))


for i in range(n):
    if dist_m(theta[i], phi_m[i][i]) >= 2/3 * np.pi:
        lambda_m[i] = 0
    if dist_m(theta[i], phi_m[i][i]) < 2/3 * np.pi:
        lambda_m[i] = int(lambda_p[s_m[i][i]]) + 1
        u_m[i][i][lambda_m[i] - 1] = delta_m[i][i]

    if dist_p(theta[i], phi_p[i][i]) >= 2/3 * np.pi:
        lambda_p[i] = 0
    if dist_p(theta[i], phi_p[i][i]) < 2/3 * np.pi:
        lambda_p[i] = int(lambda_m[s_p[i][i]]) + 1
        u_p[i][i][lambda_p[i] - 1] = delta_p[i][i]

print(lambda_m)
print(u_m)

for j in range(n):
    for i in range(j + 1):
        for alpha in range(lambda_m[i]):
            if (phi_m[i][j] != phi_m[i][j-1] and phi_m[i][j-1] != '#'):
                old_angle = (theta[i] + phi_m[i][j-1])/2
                old_magnitude = u_m[i][j-1][alpha]

                new_angle = (theta[i] + phi_m[i][j])/2
                new_magnitude = get_new_magnitude(theta[i], old_angle, new_angle, old_magnitude)
                u_m[i][j][alpha] = new_magnitude
            else: 
                u_m[i][j][alpha] = u_m[i][j-1][alpha]
