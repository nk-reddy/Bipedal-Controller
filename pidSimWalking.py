import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns  
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

class BipedalRobot:
    def __init__(self):
        self.hip = 0.0
        self.knee = 0.0
        self.ankle = 0.0

        self.hip_dot = 0.0
        self.knee_dot = 0.0
        self.ankle_dot = 0.0

        self.hip_tau = 0.0
        self.knee_tau = 0.0
        self.ankle_tau = 0.0

        self.I_hip = 1.0
        self.I_knee = 1.0
        self.I_ankle = 1.0

robot = BipedalRobot()

dt = 0.01  
t = np.arange(0, 10, dt) 

desired_hip = 0.5 * np.sin(2 * np.pi * 0.5 * t)
desired_knee = 1.0 * np.sin(2 * np.pi * 0.5 * t + np.pi/4)
desired_ankle = 0.3 * np.sin(2 * np.pi * 0.5 * t + np.pi/2)

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp  
        self.Ki = Ki  
        self.Kd = Kd  
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

hip_pid = PIDController(Kp=100, Ki=0.0, Kd=20)
knee_pid = PIDController(Kp=100, Ki=0.0, Kd=20)
ankle_pid = PIDController(Kp=100, Ki=0.0, Kd=20)

hip_actual = []
knee_actual = []
ankle_actual = []

hip_control = []
knee_control = []
ankle_control = []

for i in range(len(t)):
    hip_pid.setpoint = desired_hip[i]
    knee_pid.setpoint = desired_knee[i]
    ankle_pid.setpoint = desired_ankle[i]

    tau_hip = hip_pid.compute(robot.hip, dt)
    tau_knee = knee_pid.compute(robot.knee, dt)
    tau_ankle = ankle_pid.compute(robot.ankle, dt)

    hip_acc = tau_hip / robot.I_hip
    knee_acc = tau_knee / robot.I_knee
    ankle_acc = tau_ankle / robot.I_ankle

    robot.hip_dot += hip_acc * dt
    robot.knee_dot += knee_acc * dt
    robot.ankle_dot += ankle_acc * dt

    robot.hip += robot.hip_dot * dt
    robot.knee += robot.knee_dot * dt
    robot.ankle += robot.ankle_dot * dt

    hip_actual.append(robot.hip)
    knee_actual.append(robot.knee)
    ankle_actual.append(robot.ankle)

    hip_control.append(tau_hip)
    knee_control.append(tau_knee)
    ankle_control.append(tau_ankle)

desired_hip_deg = np.degrees(desired_hip)
desired_knee_deg = np.degrees(desired_knee)
desired_ankle_deg = np.degrees(desired_ankle)

actual_hip_deg = np.degrees(hip_actual)
actual_knee_deg = np.degrees(knee_actual)
actual_ankle_deg = np.degrees(ankle_actual)

plt.figure(figsize=(14, 8))

plt.subplot(3,1,1)
plt.plot(t, desired_hip_deg, label='Desired Hip Angle', linestyle='--')
plt.plot(t, actual_hip_deg, label='Actual Hip Angle')
plt.ylabel('Angle (degrees)')
plt.title('Hip Joint Angle Tracking')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, desired_knee_deg, label='Desired Knee Angle', linestyle='--')
plt.plot(t, actual_knee_deg, label='Actual Knee Angle')
plt.ylabel('Angle (degrees)')
plt.title('Knee Joint Angle Tracking')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, desired_ankle_deg, label='Desired Ankle Angle', linestyle='--')
plt.plot(t, actual_ankle_deg, label='Actual Ankle Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Ankle Joint Angle Tracking')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

plt.subplot(3,1,1)
plt.plot(t, hip_control, label='Hip Control Torque')
plt.ylabel('Torque (N·m)')
plt.title('Hip Joint Control Torque')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, knee_control, label='Knee Control Torque')
plt.ylabel('Torque (N·m)')
plt.title('Knee Joint Control Torque')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, ankle_control, label='Ankle Control Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N·m)')
plt.title('Ankle Joint Control Torque')
plt.legend()

plt.tight_layout()
plt.show()

rmse_hip = np.sqrt(mean_squared_error(desired_hip_deg, actual_hip_deg))
rmse_knee = np.sqrt(mean_squared_error(desired_knee_deg, actual_knee_deg))
rmse_ankle = np.sqrt(mean_squared_error(desired_ankle_deg, actual_ankle_deg))

print(f"RMSE Hip Joint: {rmse_hip:.2f} degrees")
print(f"RMSE Knee Joint: {rmse_knee:.2f} degrees")
print(f"RMSE Ankle Joint: {rmse_ankle:.2f} degrees")
