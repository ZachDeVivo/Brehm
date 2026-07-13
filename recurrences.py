import numpy as np
import math

from distances import dist_m, dist_p, get_new_magnitude

theta = np.array([0, np.pi/2, np.pi/4, np.pi, 5/8*np.pi])

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


TWO_PI_3 = 2 / 3 * np.pi

for j in range(n):
    for i in range(j + 1):
        K = range(j + 1)

        # nearest neighbors
        s_p[i, j] = min(K, key=lambda k: dist_p(theta[i], theta[k]))
        s_m[i, j] = min(K, key=lambda k: dist_m(theta[i], theta[k]))
        phi_p[i, j] = theta[s_p[i, j]]
        phi_m[i, j] = theta[s_m[i, j]]

        # δ
        d_p = dist_p(theta[i], phi_p[i, j])
        d_m = dist_m(theta[i], phi_m[i, j])
        delta_p[i, j] = 1 if d_p >= TWO_PI_3 else 1 / (2 * math.cos(d_p / 2))
        delta_m[i, j] = 1 if d_m >= TWO_PI_3 else 1 / (2 * math.cos(d_m / 2))

        if i == j:
            # λ, u birth (minus then plus)
            if d_m >= TWO_PI_3:
                lambda_m[i] = 0
            else:
                p = s_m[i, i]
                lambda_m[i] = lambda_p[p] + 1
                u_m[i, i, lambda_m[i] - 1] = delta_m[i, i]
                for alpha in range(lambda_m[i] - 1):
                    u_m[i, i, alpha] = u_p[p, i, alpha] = get_new_magnitude(
                        theta[p],
                        (theta[p] + phi_p[p, i - 1]) / 2,
                        (theta[p] + phi_p[p, i]) / 2,
                        u_p[p, i - 1, alpha],
                    )

            if d_p >= TWO_PI_3:
                lambda_p[i] = 0
            else:
                p = s_p[i, i]
                lambda_p[i] = lambda_m[p] + 1
                u_p[i, i, lambda_p[i] - 1] = delta_p[i, i]
                for alpha in range(lambda_p[i] - 1):
                    u_p[i, i, alpha] = u_m[p, i, alpha] = get_new_magnitude(
                        theta[p],
                        (theta[p] + phi_m[p, i - 1]) / 2,
                        (theta[p] + phi_m[p, i]) / 2,
                        u_m[p, i - 1, alpha],
                    )
        else:
            # recurrence in j
            for alpha in range(lambda_m[i]):
                if phi_m[i, j] != phi_m[i, j - 1]:
                    u_m[i, j, alpha] = get_new_magnitude(
                        theta[i],
                        (theta[i] + phi_m[i, j - 1]) / 2,
                        (theta[i] + phi_m[i, j]) / 2,
                        u_m[i, j - 1, alpha],
                    )
                else:
                    u_m[i, j, alpha] = u_m[i, j - 1, alpha]

            for alpha in range(lambda_p[i]):
                if phi_p[i, j] != phi_p[i, j - 1]:
                    u_p[i, j, alpha] = get_new_magnitude(
                        theta[i],
                        (theta[i] + phi_p[i, j - 1]) / 2,
                        (theta[i] + phi_p[i, j]) / 2,
                        u_p[i, j - 1, alpha],
                    )
                else:
                    u_p[i, j, alpha] = u_p[i, j - 1, alpha]

print(s_p)
print(s_m)
print(phi_p)
print(phi_m)
print(lambda_m)
print(lambda_p)
print(u_m)
print(u_p)