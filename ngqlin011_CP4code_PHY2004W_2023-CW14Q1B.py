import numpy as np
import matplotlib.pyplot as plt


def main():
    """Functions correspond to different parts for submission
    """
    euler(dt=0.01, T=10)
    period_dependance(n=50)
    

def euler(dt:float, T:int) -> None:
    """Runs a symplectic-euler simulation of the non-linear simple pendulum
    and plots angle, angular velocity, and total energy against time plus a phase plot.
    """
    # Initial Condition
    dt = dt
    omega0 = 5
    g = 9.8
    L = g / omega0**2
    m = 10
    N = int(T / dt + 1.5)

    t = np.zeros(N)
    p = np.zeros(N)
    q = np.zeros(N)
    E = np.zeros(N)

    p[0] = 0
    q[0] = np.pi / 2
    E[0] = 0.5 * m * L**2 * p[0]**2 + m * g * L * (1 - np.cos(q[0]))    

    for i in range(N - 1):
        t[i + 1] = (i + 1) * dt  
        p[i + 1] = p[i] - omega0**2 * np.sin(q[i]) * dt
        q[i + 1] = q[i] + p[i + 1] * dt
        E[i + 1] = m * L * (0.5 * L * p[i]**2 + g * (1 - np.cos(q[i]))) #0.5 * m * L**2 * p[i]**2 + m * g * L * (1 - np.cos(q[i + 1]))
                                      
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    ax[0, 0].plot(t, q, 'b-')
    ax[0, 0].set(xlabel='time [s]', ylabel='angle [rad]')
    ax[0, 1].plot(t, p, 'b-')
    ax[0, 1].set(xlabel='time [s]', ylabel='velocity [rad/s]')
    ax[1, 0].plot(q, p, 'b-')
    ax[1, 0].set(xlabel='angle [rad]', ylabel='velocity [rad/s]')
    ax[1, 1].plot(t, E, 'r-')
    ax[1, 1].set(xlabel='time [s]', ylabel='Energy [J]')

    fig.tight_layout()
    fig.show()
    

def period_dependance(n:int) -> None:
    """Plots period of a simple pendulum experiment
    against different choices of initial angle: theta from equilibrium
    between pi/16 and pi/2
    """
    theta = np.linspace(np.pi/ 16, np.pi / 2, n)
    data = np.stack((theta, np.zeros(n), np.zeros(n)), axis=1)

    # Generate theta(t) vs time data
    for j, theta_i in enumerate(theta):
        
        # Initial Condition
        dt = 0.01
        duration = 10
        omega0 = 5
        g = 9.8
        L = g / omega0**2
        m = 10
        N = int(duration / dt + 1.5)

        t = np.zeros(N)
        p = np.zeros(N)
        q = np.zeros(N)
        E = np.zeros(N)

        p[0] = 0
        q[0] = theta_i

        # Data Generating Loop
        for i in range(N - 1):
            t[i + 1] = (i + 1) * dt  
            p[i + 1] = p[i] - omega0**2 * np.sin(q[i]) * dt
            q[i + 1] = q[i] + p[i + 1] * dt

        # Append t and q data to data array
        data[j, 1], data[j, 2] = get_period(t, q)

    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.errorbar(np.degrees(data[:, 0]), data[:, 1], data[:, 2], fmt='ko', capsize=3, label=f'simulation data\nN = {n}')
    ax.set(title='Period T vs Initial Angle $(\\theta)$',
           xlabel='$\\theta$ [deg]',
           ylabel='T [sec]')
    ax.legend()
    fig.show()


def get_period(time:np.ndarray, theta:np.ndarray) -> np.ndarray:
    """Estimates the period of 0-equilibrium periodic data
    using zero crossings.
    returns: tuple of best estimate and standard uncertainity: 'std.dev / sqrt(N)'
    """
    indices = np.where(np.diff(np.signbit(theta)))[0] # np.where returns: tuple ([indices], dtype)
    half_periods = np.diff(time[indices])
    avg_period = np.mean(2 * half_periods)
    u_avg_period = np.std(half_periods) / np.sqrt(len(half_periods))

    return avg_period, u_avg_period


if __name__ == '__main__':
    main()




























    
