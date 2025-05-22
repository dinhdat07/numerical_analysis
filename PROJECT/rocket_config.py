# rocket_config.py

import numpy as np

# Hằng số vật lý
G = 6.67430e-11       # gravitational constant
M = 5.972e24          # mass of Earth (kg)
R = 6.371e6           # radius of Earth (m)
rho0 = 1.225          # air density at sea level (kg/m^3)
H = 8000              # scale height (m)
Cd = 0.5              # drag coefficient
A = 10.5              # cross-sectional area (m^2)

# Tên lửa - giai đoạn 1
class RocketStage:
    def __init__(self, m0, m_dry, T, burn_time):
        self.m0 = m0
        self.m_dry = m_dry
        self.T = T
        self.burn_time = burn_time
        self.mdot = (m0 - m_dry) / burn_time

STAGE1 = RocketStage(
    m0=300000,
    m_dry=150000,
    T=2.55e7,
    burn_time=100
)
