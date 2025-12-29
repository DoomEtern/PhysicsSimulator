import numpy as np
from scipy.linalg import expm, inv, pinv
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

# --- CONFIGURATION SETTINGS ---
# Consistency bounds for a 15-degree-of-freedom system
CHI_SQUARE_LOWER = 5.629  # 95% Confidence interval lower bound
CHI_SQUARE_UPPER = 26.119 # 95% Confidence interval upper bound
CONTROL_TIMESTEP = 0.01

class AdvancedEKF:
    """
    Error-State Extended Kalman Filter (ES-EKF).
    State vector: [position(3), velocity(3), orientation_error(3), gyro_bias(3), accel_bias(3)]
    Total: 15 Degrees of Freedom.
    """
    def __init__(self):
        # Initial State Estimate
        self.pos_est = np.zeros(3)
        self.vel_est = np.zeros(3)
        self.rot_est = np.eye(3)
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)
        
        # Error Covariance Matrix (P) - Represents uncertainty
        self.P_cov = np.eye(15) * 1e-4
        
        # Process Noise (Continuous Time) - Tuning parameters
        # Values represent variance for: pos, vel, rot, gyro_bias, accel_bias
        self.Q_noise = np.diag([1e-4]*3 + [1e-3]*3 + [1e-5]*3 + [1e-6]*3 + [1e-6]*3)
        
        # GPS Measurement Noise
        self.R_meas = np.diag([0.5**2]*3) 

    def skew_symmetric(self, v):
        """Helper function to create skew-symmetric matrix from vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def predict_step(self, accel_reading, gyro_reading, dt):
        """
        Prediction step using IMU data.
        Propagates state and covariance forward in time.
        """
        # 1. Integrate Nominal State
        # Correct IMU readings with estimated biases
        accel_corrected = accel_reading - self.bias_accel
        gyro_corrected = gyro_reading - self.bias_gyro
        gravity = np.array([0, 0, 9.81])
        
        # Standard kinematic equations
        self.pos_est = self.pos_est + self.vel_est * dt + 0.5 * (self.rot_est @ accel_corrected - gravity) * dt**2
        self.vel_est = self.vel_est + (self.rot_est @ accel_corrected - gravity) * dt
        # Update rotation using exponential map for stability
        self.rot_est = self.rot_est @ expm(self.skew_symmetric(gyro_corrected * dt))
        
        # 2. Build Error-State Jacobian (F_matrix)
        F_matrix = np.zeros((15, 15))
        # Position error depends on velocity
        F_matrix[0:3, 3:6] = np.eye(3)
        # Velocity error dynamics
        F_matrix[3:6, 6:9] = -self.rot_est @ self.skew_symmetric(accel_corrected)
        F_matrix[3:6, 12:15] = -self.rot_est
        # Orientation error dynamics
        F_matrix[6:9, 6:9] = -self.skew_symmetric(gyro_corrected)
        F_matrix[6:9, 9:12] = -np.eye(3)
        
        # 3. Covariance Propagation
        # Discrete time transition matrix approx
        Transition_matrix = np.eye(15) + F_matrix * dt
        Noise_discrete = self.Q_noise * dt 
        
        # Safety check: prevent covariance explosion
        if np.max(np.abs(self.P_cov)) > 1e8:
            self.P_cov = np.eye(15) * 1e8 
            
        self.P_cov = Transition_matrix @ self.P_cov @ Transition_matrix.T + Noise_discrete
        
        # Ensure covariance remains symmetric and positive definite
        self.P_cov = 0.5 * (self.P_cov + self.P_cov.T) + np.eye(15) * 1e-9

    def update_gps(self, gps_measurement):
        """
        Measurement update step using GPS data.
        Corrects the state estimate based on observation.
        """
        # Observation Matrix (H) - We only observe position directly
        H_matrix = np.zeros((3, 15))
        H_matrix[0:3, 0:3] = np.eye(3)
        
        # Innovation Covariance (S)
        S_matrix = H_matrix @ self.P_cov @ H_matrix.T + self.R_meas
        
        try:
            # Kalman Gain (K) calculation
            # Using pseudoinverse for numerical stability in edge cases
            K_gain = self.P_cov @ H_matrix.T @ inv(S_matrix)
        except:
            # Fallback if matrix inversion fails
            K_gain = np.zeros((15, 3))
        
        # Innovation (Residual)
        innovation = gps_measurement - self.pos_est
        
        # Error State Correction
        error_correction = K_gain @ innovation         
        
        if np.any(np.isnan(error_correction)): error_correction = np.zeros(15)

        # Inject Error Correction into Nominal State
        self.pos_est += error_correction[0:3]
        self.vel_est += error_correction[3:6]
        # Orientation correction (small angle approx)
        self.rot_est = self.rot_est @ expm(self.skew_symmetric(error_correction[6:9]))
        self.bias_gyro += error_correction[9:12]
        self.bias_accel += error_correction[12:15]
        
        # Update Covariance (Joseph Form for stability)
        I_minus_KH = np.eye(15) - K_gain @ H_matrix
        self.P_cov = I_minus_KH @ self.P_cov @ I_minus_KH.T + K_gain @ self.R_meas @ K_gain.T

    def calculate_consistency_metric(self, ground_truth):
        """
        Computes the Normalized Estimation Error Squared (NEES).
        This tells us if the filter is "consistent" (i.e., does its uncertainty match reality?)
        """
        # Calculate errors between estimate and truth
        pos_err = ground_truth['pos'] - self.pos_est
        vel_err = ground_truth['vel'] - self.vel_est
        
        # Orientation error is tricky - needs to be mapped to tangent space
        rot_err_matrix = ground_truth['rot'] @ self.rot_est.T
        # Calculate rotation angle
        trace_val = np.trace(rot_err_matrix)
        angle = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
        
        if np.abs(angle) < 1e-7:
            theta_err = np.zeros(3)
        else:
            # Log map for SO(3)
            theta_err = (angle / (2 * np.sin(angle))) * np.array([
                rot_err_matrix[2, 1] - rot_err_matrix[1, 2],
                rot_err_matrix[0, 2] - rot_err_matrix[2, 0],
                rot_err_matrix[1, 0] - rot_err_matrix[0, 1]
            ])
            
        accel_bias_err = ground_truth['bias_acc'] - self.bias_accel
        gyro_bias_err = ground_truth['bias_gyro'] - self.bias_gyro
        
        # Full error vector
        total_error_vec = np.hstack([pos_err, vel_err, theta_err, gyro_bias_err, accel_bias_err])
        
        try:
            # Mahalanobis Distance: e^T * P^-1 * e
            nees_score = total_error_vec.T @ inv(self.P_cov) @ total_error_vec
        except:
            nees_score = 1e6 # Flag singularity
            
        return nees_score

class SimulationEngine:
    def __init__(self, seed_val):
        np.random.seed(seed_val)
        
        # --- Parameter Randomization (Monte Carlo) ---
        # Real world parameters are never exactly known
        self.mass = 1.5 * np.random.uniform(0.95, 1.05) # +/- 5% error
        inertia_scaling = np.random.uniform(0.9, 1.1)
        self.inertia = np.diag([0.015, 0.015, 0.025]) * inertia_scaling
        self.inv_inertia = inv(self.inertia)
        
        # Actuator parameters (Truth vs Model)
        self.k_thrust_true = 3e-6 * np.random.uniform(0.9, 1.1)
        self.k_torque_true = 7e-8 * np.random.uniform(0.9, 1.1)
        
        # Nominal parameters (What the controller assumes)
        self.arm_len = 0.25
        self.k_thrust_nom = 3e-6 
        self.k_torque_nom = 7e-8
        
        # Allocation Matrix (Based on nominal model)
        self.mixer_matrix = np.array([
            [self.k_thrust_nom, self.k_thrust_nom, self.k_thrust_nom, self.k_thrust_nom],
            [0, self.k_thrust_nom*self.arm_len, 0, -self.k_thrust_nom*self.arm_len],
            [-self.k_thrust_nom*self.arm_len, 0, self.k_thrust_nom*self.arm_len, 0],
            [self.k_torque_nom, -self.k_torque_nom, self.k_torque_nom, -self.k_torque_nom]
        ])
        self.inv_mixer = pinv(self.mixer_matrix)
        
        # State Variables
        self.pos = np.zeros(3)
        self.vel_body = np.zeros(3)
        self.rot = np.eye(3)
        self.ang_vel = np.zeros(3)
        self.motor_rpm = np.zeros(4)
        
        # Sensor Biases (Truth)
        self.true_bias_acc = np.random.uniform(-0.1, 0.1, 3)
        self.true_bias_gyro = np.random.uniform(-0.01, 0.01, 3)
        
        # Initialize Estimator
        self.filter = AdvancedEKF()
        
        # Failure Scenario Setup
        # Randomly choose if a motor fails, which one, and when
        self.broken_motor_idx = np.random.choice([None, 0, 1]) 
        self.fail_time = np.random.uniform(2.0, 5.0) 
        
    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def step_physics(self, cmd_rpm_sq, dt, current_time):
        # --- Fault Injection ---
        if self.broken_motor_idx is not None and current_time > self.fail_time:
            cmd_rpm_sq[self.broken_motor_idx] = 0.0 # Motor dies
        
        # Motor Dynamics (First-order lag)
        target_rpm = np.sqrt(np.clip(cmd_rpm_sq, 0, 9500**2))
        self.motor_rpm += (target_rpm - self.motor_rpm) * (dt / 0.03)
        
        # Calculate Forces/Torques using TRUTH parameters
        thrust_forces = self.k_thrust_true * self.motor_rpm**2
        
        # Torque calculation
        tau_x = self.arm_len * (thrust_forces[1] - thrust_forces[3])
        tau_y = self.arm_len * (thrust_forces[2] - thrust_forces[0])
        tau_z = self.k_torque_true * (self.motor_rpm[0]**2 - self.motor_rpm[1]**2 + self.motor_rpm[2]**2 - self.motor_rpm[3]**2)
        
        total_thrust = np.sum(thrust_forces)
        
        # Linear Dynamics (including Drag)
        vel_mag = np.linalg.norm(self.vel_body)
        drag_force = -0.5 * 1.225 * 0.05 * vel_mag * self.vel_body # Quadratic Drag
        
        forces_body = np.array([0,0,-total_thrust]) + drag_force + self.rot.T @ np.array([0,0,self.mass*9.81])
        accel_body = forces_body/self.mass - np.cross(self.ang_vel, self.vel_body)
        
        # Angular Dynamics
        torques_body = np.array([tau_x, tau_y, tau_z]) 
        ang_accel = self.inv_inertia @ (torques_body - np.cross(self.ang_vel, self.inertia @ self.ang_vel))
        
        # Integration (Euler)
        self.vel_body += accel_body * dt
        self.ang_vel += ang_accel * dt
        self.pos += (self.rot @ self.vel_body) * dt
        self.rot = self.rot @ expm(self.skew_symmetric(self.ang_vel * dt))
        
        # IMU Readings (Truth + Bias + Physics)
        # Note: Accelerometers measure "Proper Acceleration" (Force/Mass), not kinematic acceleration
        proper_acc = accel_body + np.cross(self.ang_vel, self.vel_body) + self.rot.T @ np.array([0,0,-9.81])
        return proper_acc + self.true_bias_acc, self.ang_vel + self.true_bias_gyro

    def compute_control(self, target_pos):
        # Uses ESTIMATED state, not truth
        R_est, p_est, v_est = self.filter.rot_est, self.filter.pos_est, self.filter.vel_est
        
        # Position Control (PD)
        pos_err = p_est - target_pos
        vel_err = v_est
        
        # Desired force vector (feedforward gravity)
        f_des = -6.0*pos_err - 4.5*vel_err + np.array([0,0,self.mass*9.81]) 
        
        # Attitude Control (Geometric)
        z_body_des = f_des / np.linalg.norm(f_des)
        x_world = np.array([1,0,0])
        y_body_des = np.cross(z_body_des, x_world); y_body_des /= np.linalg.norm(y_body_des)
        x_body_des = np.cross(y_body_des, z_body_des)
        R_des = np.stack([x_body_des, y_body_des, z_body_des], axis=1)
        
        # Rotation Error
        R_err_matrix = R_des.T @ R_est - R_est.T @ R_des
        e_rot = 0.5 * np.array([R_err_matrix[2,1], R_err_matrix[0,2], R_err_matrix[1,0]])
        
        # Mixing
        thrust_cmd = np.dot(f_des, R_est[:,2])
        torque_cmd = -12.0*e_rot - 2.0*self.filter.bias_gyro # Use bias estimate for damping
        
        motor_cmds = self.inv_mixer @ np.array([thrust_cmd, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
        return motor_cmds

def execute_trial(trial_num):
    """Runs a single simulation trial and logs data."""
    sim_instance = SimulationEngine(trial_num)
    # Initialize estimator slightly off to test convergence
    sim_instance.filter.pos_est = sim_instance.pos + np.random.normal(0, 0.5, 3)
    
    log_data = {'consistency_score': [], 'timestamp': [], 'has_fault': False, 'crashed': False}
    
    for step in range(800): # 8 seconds at 100Hz
        time_now = step * CONTROL_TIMESTEP
        
        # Safety Check: Stop simulation if physics explode
        if np.any(np.isnan(sim_instance.pos)) or np.linalg.norm(sim_instance.vel_body) > 300:
            log_data['crashed'] = True
            break

        # 1. Controller Update
        motor_inputs = sim_instance.compute_control(np.array([0,0,-5])) # Hover at 5m
        
        # 2. Physics Update
        imu_acc, imu_gyro = sim_instance.step_physics(motor_inputs, CONTROL_TIMESTEP, time_now)
        
        # 3. EKF Prediction
        sim_instance.filter.predict_step(imu_acc, imu_gyro, CONTROL_TIMESTEP)
        
        # 4. GPS Update (10Hz) with simulated dropout
        # Dropout between 3s and 4s to test dead-reckoning
        if step % 10 == 0 and not (3.0 < time_now < 4.0): 
            gps_reading = sim_instance.pos + np.random.normal(0, 0.5, 3)
            sim_instance.filter.update_gps(gps_reading)
            
        # 5. Logging
        ground_truth_state = {
            'pos': sim_instance.pos, 
            'vel': sim_instance.rot @ sim_instance.vel_body, 
            'rot': sim_instance.rot, 
            'bias_acc': sim_instance.true_bias_acc, 
            'bias_gyro': sim_instance.true_bias_gyro
        }
        nees_val = sim_instance.filter.calculate_consistency_metric(ground_truth_state)
        
        # Clamp large values for cleaner plotting
        log_data['consistency_score'].append(np.clip(nees_val, 0, 1e5))
        log_data['timestamp'].append(time_now)
        
        if sim_instance.broken_motor_idx is not None and time_now > sim_instance.fail_time:
            log_data['has_fault'] = True
            
    return log_data

if __name__ == "__main__":
    print("Running Monte Carlo Analysis (50 Trials)...")
    
    # Parallel processing for speed
    with ProcessPoolExecutor() as executor:
        batch_results = list(executor.map(execute_trial, range(50)))
        
    # --- Visualization ---
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Filter Consistency (NEES)
    ax1 = fig.add_subplot(211)
    
    # Plot theoretical bounds
    ax1.axhline(CHI_SQUARE_UPPER, color='darkred', linestyle='--', label='95% Confidence Upper')
    ax1.axhline(CHI_SQUARE_LOWER, color='darkred', linestyle='--', label='95% Confidence Lower')
    
    # Aggregate data for mean calculation
    all_nees_values = np.full((800, len(batch_results)), np.nan)
    
    for idx, res in enumerate(batch_results):
        nees_trace = np.array(res['consistency_score'])
        len_trace = len(nees_trace)
        all_nees_values[:len_trace, idx] = nees_trace
        
        # Color code faults
        line_color = 'red' if res['has_fault'] else 'blue'
        ax1.plot(res['timestamp'], nees_trace, color=line_color, alpha=0.1)

    # Plot average NEES
    mean_nees = np.nanmean(all_nees_values, axis=1)
    time_vector = np.linspace(0, 8.0, 800)
    ax1.plot(time_vector, mean_nees, color='green', linewidth=2, label='Average Consistency')
    
    ax1.set_yscale('log')
    ax1.set_title("Filter Consistency Analysis (NEES Metric)")
    ax1.set_ylabel("NEES Score")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Subplot 2: Reliability Histogram
    ax2 = fig.add_subplot(212)
    # Calculate flight time for each trial
    flight_times = [len(r['consistency_score']) * CONTROL_TIMESTEP for r in batch_results]
    ax2.hist(flight_times, bins=20, color='gray', edgecolor='black', alpha=0.7)
    ax2.set_title("Reliability Distribution: Time Until Failure")
    ax2.set_xlabel("Flight Duration (s)")
    ax2.set_ylabel("Number of Trials")

    plt.tight_layout()
    plt.show()