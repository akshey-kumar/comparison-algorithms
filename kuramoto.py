# Re-run the simulation after kernel reset
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # number of oscillators
K = 4.0   # coupling strength
Delta = 1.0  # width of Lorentzian frequency distribution
T = 10.0  # total time
dt = 0.001  # time step
timesteps = int(T / dt)

# Natural frequencies drawn from Lorentzian distribution
def lorentzian(omega0, Delta, size):
    return omega0 + Delta * np.tan(np.pi * (np.random.rand(size) - 0.5))

omega = lorentzian(0, Delta, N)

# Initial phases
theta = 2 * np.pi * np.random.rand(N)

# To record order parameter over time
r_micro = np.zeros(timesteps, dtype=complex)

# Simulate theta evolution
for t in range(timesteps):
    z = np.mean(np.exp(1j * theta))  # order parameter
    r_micro[t] = z
    theta += dt * (omega + K * np.abs(z) * np.sin(np.angle(z) - theta))

# Simulate mean-field ODE: dr/dt = -Δr + (K/2)r(1 - r^2)
r_macro = np.zeros(timesteps)
r_macro[0] = np.abs(r_micro)[0] # 0.01  # small initial coherence
for t in range(1, timesteps):
    r = r_macro[t - 1]
    drdt = -Delta * r + (K / 2) * r * (1 - r**2)
    r_macro[t] = r + dt * drdt

# Plot
time = np.linspace(0, T, timesteps)
plt.figure(figsize=(8, 5))
plt.plot(time, np.abs(r_micro), label='Microscopic $r(t)$ from $\Theta_i$', lw=2)
plt.plot(time, r_macro, label='Macroscopic $r(t)$ from ODE', lw=2, linestyle='--')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Order Parameter $r(t)$", fontsize=12)
plt.title("Kuramoto Synchronization: Micro vs Macro", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
