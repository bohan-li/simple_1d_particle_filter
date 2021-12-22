import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def run_pf(pos, velo, measurement, num_particles, epsilon, nt, dt):
    a = []
    b = []
    a.append(pos(0) + epsilon * np.random.normal(size=[num_particles]))

    for t in range(1, nt):
        b.append(measurement(a[t - 1]) + epsilon * np.random.normal())
        p_b_given_a = norm.pdf(b[t - 1], loc=measurement(a[t - 1]), scale=1 / epsilon)
        p_a_given_b = norm.pdf(a[t - 1], loc=pos(t - 1), scale=1/epsilon) * p_b_given_a
        p_a_given_b /= np.sum(p_a_given_b)
        a.append(np.random.choice(a[t - 1], size=[num_particles], p=p_a_given_b))
        a[t] += dt * velo(t - 1)
    return a, b

if __name__ == '__main__':
    num_particles = 200
    epsilon = 0.01

    nt = 1000
    dt = 0.1

    alpha = 1
    beta = 1
    pos = lambda t: alpha * np.sin(beta * t)
    velo = lambda t: alpha * beta * np.cos(beta * t)

    pole_loc = -0.25
    dist = lambda x: np.abs(pole_loc - x)
    a, b = run_pf(pos, velo, dist, num_particles, epsilon, nt, dt)

    a_avg = [np.sum(elem) / num_particles for elem in a]
    plt.plot(range(nt), a_avg)
    plt.show()
