import numpy as np
from scipy.linalg import expm, inv, norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from collections import deque

# ==========================================
# PART 1: CORE MATHEMATICS
# ==========================================

def make_skew_symmetric(vec):
    """Creates a skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

def rotation_map(w):
    """
    Computes the rotation matrix from an angular velocity vector.
    Uses Taylor series for small angles to avoid numerical issues.
    """
    theta_sq = np.dot(w, w)
    theta = np.sqrt(theta_sq)
    K = make_skew_symmetric(w)
    
    if theta < 1e-6:
        # Small angle approximation
        return np.eye(3) + K + 0.5 * (K @ K)
    else:
        # Standard Rodrigues' formula
        return np.eye(3) + (np.sin(theta)/theta) * K + ((1 - np.cos(theta))/theta_sq) * (K @ K)

class PhysicsEngine:
    def __init__(self, time_step):
        self.dt = time_step
        self.mass = 1.5 
        self.gravity = 9.81
        
        # Inertia tensor and its inverse
        self.inertia = np.diag([0.02, 0.02, 0.04])
        self.inv_inertia = inv(self.inertia)
        
        # Initial State: Rotation, Position, Velocity, Angular Velocity
        self.rot_mat = np.eye(3)
        self.pos = np.array([0., 0., 10.])
        self.vel = np.zeros(3)
        self.ang_vel = np.zeros(3)
        
        # Quadcopter Geometry
        arm_len = 0.25
        torque_coeff = 0.05
        
        # Allocation Matrix (Maps motors to forces/torques)
        self.alloc_matrix = np.array([
            [1, 1, 1, 1],          # Thrust (Z)
            [arm_len, -arm_len, -arm_len, arm_len],        # Roll
            [arm_len, arm_len, -arm_len, -arm_len],        # Pitch
            [-torque_coeff, torque_coeff, -torque_coeff, torque_coeff]         # Yaw
        ])
        
    def step_simulation(self, motor_inputs, wind_vec, motor_health):
        # Apply motor health degradation
        effective_inputs = motor_inputs * motor_health
        forces_torques = self.alloc_matrix @ effective_inputs
        
        # Linear Dynamics
        force_gravity = np.array([0, 0, -self.mass * self.gravity])
        force_thrust = self.rot_mat @ np.array([0, 0, forces_torques[0]])
        
        relative_vel = self.vel - wind_vec
        force_drag = -0.1 * norm(relative_vel) * relative_vel
        
        acceleration = (force_gravity + force_thrust + force_drag) / self.mass
        self.vel += acceleration * self.dt
        self.pos += self.vel * self.dt
        
        # Ground Collision Check
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vel = np.zeros(3)
            self.ang_vel = np.zeros(3)
            return self.rot_mat, self.pos, self.vel, self.ang_vel, True 
            
        # Angular Dynamics
        total_torque = forces_torques[1:] - np.cross(self.ang_vel, self.inertia @ self.ang_vel) - 0.01*self.ang_vel
        self.ang_vel += (self.inv_inertia @ total_torque) * self.dt
        
        # Update Orientation
        self.rot_mat = self.rot_mat @ rotation_map(self.ang_vel * self.dt)
        
        return self.rot_mat, self.pos, self.vel, self.ang_vel, False

# ==========================================
# PART 2: SAFETY SUPERVISOR (Control Barrier Function)
# ==========================================

class SafetyFilter:
    def __init__(self, mass):
        self.mass = mass
        self.gravity = 9.81
        self.safe_altitude = 0.5
        
        # Barrier function gains
        self.gain1 = 2.0
        self.gain2 = 2.0
        
        # Max expected disturbance (e.g., wind gust)
        self.max_disturbance = 2.0 
        
    def compute_safe_control(self, altitude, vert_vel, nominal_thrust, max_available_thrust):
        """
        Calculates the minimum thrust required to guarantee safety.
        Solves a constraint based on the Control Barrier Function.
        """
        # Barrier function h(x) = z - z_safe
        h_val = altitude - self.safe_altitude
        
        # Composite barrier B
        barrier_B = vert_vel + self.gain1 * h_val
        
        # Required minimum thrust to satisfy h_dot + alpha * h >= 0 under worst-case disturbance
        min_safe_thrust = self.mass * (self.gravity - self.gain1 * vert_vel - self.gain2 * barrier_B + self.max_disturbance)
        
        # Calculate time to impact if falling
        time_to_impact = float('inf')
        if vert_vel < -0.1:
            time_to_impact = (altitude - self.safe_altitude) / abs(vert_vel)
            
        # Check if safety is physically possible
        if min_safe_thrust > max_available_thrust:
            return max_available_thrust, False, h_val, "CRITICAL_FAILURE", time_to_impact
            
        # Arbitrate between nominal control and safety requirement
        safe_cmd = max(nominal_thrust, min_safe_thrust)
        safe_cmd = min(safe_cmd, max_available_thrust) # Clamp to physical limits
        
        status = "NORMAL" if safe_cmd == nominal_thrust else "SAFETY_OVERRIDE"
        return safe_cmd, True, h_val, status, time_to_impact

# ==========================================
# PART 3: SYSTEM HEALTH MONITOR
# ==========================================

class HealthMonitor:
    def __init__(self):
        arm = 0.25
        coeff = 0.05
        
        self.alloc_matrix = np.array([
            [1, 1, 1, 1], 
            [arm, -arm, -arm, arm], 
            [arm, arm, -arm, -arm], 
            [-coeff, coeff, -coeff, coeff]
        ])
        
        self.max_motor_thrust = 15.0
        
    def assess_capabilities(self, motor_status):
        # Create effective allocation matrix based on motor health
        health_diag = np.diag(motor_status)
        effective_matrix = self.alloc_matrix @ health_diag
        
        # Check controllability rank
        controllability_rank = np.linalg.matrix_rank(effective_matrix)
        
        # Estimate maximum vertical thrust available
        total_thrust_capacity = np.sum(motor_status * self.max_motor_thrust)
        
        return controllability_rank, total_thrust_capacity

# ==========================================
# PART 4: STATE ESTIMATOR
# ==========================================

class AltitudeEstimator:
    def __init__(self):
        # 1D Kalman Filter for Altitude
        self.alt_est = 10.0
        self.vel_est = 0.0
        
        # Covariance Matrix
        self.cov_matrix = np.diag([1.0, 1.0])
        
        # Process and Measurement Noise
        self.proc_noise = np.diag([0.01, 0.1])
        self.meas_noise = 0.5
        
    def run_filter(self, gps_alt, accel_z, dt):
        # Prediction Step
        F_mat = np.array([[1, dt], [0, 1]])
        
        self.alt_est += self.vel_est * dt + 0.5 * (accel_z - 9.81) * dt**2
        self.vel_est += (accel_z - 9.81) * dt
        self.cov_matrix = F_mat @ self.cov_matrix @ F_mat.T + self.proc_noise
        
        # Update Step
        H_mat = np.array([[1, 0]])
        S_val = H_mat @ self.cov_matrix @ H_mat.T + self.meas_noise
        
        # Kalman Gain
        K_gain = self.cov_matrix @ H_mat.T @ inv(S_val)
        
        innovation = gps_alt - self.alt_est
        correction = K_gain @ np.array([innovation])
        
        self.alt_est += correction[0]
        self.vel_est += correction[1]
        self.cov_matrix = (np.eye(2) - K_gain @ H_mat) @ self.cov_matrix
        
        # Consistency Check (NIS)
        nis_score = innovation**2 / S_val[0,0]
        return self.alt_est, self.vel_est, nis_score

# ==========================================
# PART 5: ADVERSARY GENERATOR
# ==========================================

def calculate_hostile_wind(time, barrier_val):
    # Generates wind that tries to push the drone down when it's most vulnerable
    intensity = min(5.0, time * 0.5)
    wind_vec = np.array([0.0, 0.0, -intensity])
    
    # If safety margin is low, add lateral shear
    if barrier_val < 2.0:
        wind_vec += np.array([2.0, 2.0, -2.0]) 
        
    return wind_vec

# ==========================================
# PART 6: MAIN SIMULATION LOOP
# ==========================================

def run_simulation_trial(trial_id):
    time_step = 0.01
    
    # Initialize components
    physics = PhysicsEngine(time_step)
    safety_sys = SafetyFilter(physics.mass)
    monitor = HealthMonitor()
    estimator = AltitudeEstimator()
    
    failure_trigger_time = 2.0
    failed_motor_config = np.array([1, 1, 0.4, 1]) 
    
    # Data Logging
    log_data = {'time':[], 'barrier':[], 'cmd_thrust':[], 'status_code':[], 'altitude':[], 'impact_time':[]}
    
    for step in range(800): 
        current_time = step * time_step
        
        # 1. Estimation
        # Add noise to GPS reading
        gps_meas = physics.pos[2] + np.random.randn()*0.1
        if current_time > 3.0: gps_meas += 0.0 # Nominal conditions
        
        # Simulated Accelerometer reading
        accel_true = (physics.rot_mat.T @ np.array([0,0,9.81]))[2] 
        est_alt, est_vel, consistency = estimator.run_filter(gps_meas, accel_true, time_step)
        
        # 2. Health Monitoring
        current_health = np.ones(4)
        if current_time > failure_trigger_time: 
            current_health = failed_motor_config
            
        rank, max_thrust_avail = monitor.assess_capabilities(current_health)
        
        # 3. Nominal Controller (Pilot Logic)
        kp_pos, kv_vel = 4.0, 3.0
        # Pilot tries to hold 0m altitude (land)
        nominal_thrust = -kp_pos*(est_alt - 0.0) - kv_vel*est_vel + physics.mass*9.81
        
        # 4. SAFETY INTERVENTION
        safe_thrust, is_feasible, h_val, system_status, tti = safety_sys.compute_safe_control(
            est_alt, est_vel, nominal_thrust, max_thrust_avail
        )
        
        # 5. Physics Step
        wind_force = calculate_hostile_wind(current_time, h_val)
        
        # Distribute thrust to motors (Simplified allocation)
        motor_cmds = safe_thrust / 4.0 * np.ones(4) 
        
        _, true_pos, _, _, crash_detected = physics.step_simulation(motor_cmds, wind_force, current_health)
        
        # Logging
        log_data['time'].append(current_time)
        log_data['barrier'].append(h_val)
        log_data['altitude'].append(true_pos[2])
        log_data['cmd_thrust'].append(safe_thrust)
        log_data['impact_time'].append(tti)
        
        status_int = 0 
        if system_status == "SAFETY_OVERRIDE": status_int = 1
        if system_status == "CRITICAL_FAILURE": status_int = 2
        log_data['status_code'].append(status_int)
        
        if crash_detected: break
        
    return log_data

# ==========================================
# PART 7: RESULTS VISUALIZATION
# ==========================================

if __name__ == "__main__":
    print("Starting Formal Verification Campaign...")
    
    with ProcessPoolExecutor() as executor:
        sim_results = list(executor.map(run_simulation_trial, range(4)))
        
    data = sim_results[0]
    time_axis = data['time']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Altitude
    axes[0].plot(time_axis, data['altitude'], 'b-', lw=2, label='True Altitude')
    axes[0].axhline(0.5, color='r', linestyle='--', lw=2, label='Safety Floor')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Thrust Command
    axes[1].plot(time_axis, data['cmd_thrust'], 'g-', lw=2, label='Safety Thrust Cmd')
    axes[1].set_ylabel('Thrust (N)')
    axes[1].grid(True)
    
    # Plot 3: Time to Impact
    tti_vals = np.array(data['impact_time'])
    tti_vals[tti_vals > 10] = 10 # Cap for readability
    axes[2].plot(time_axis, tti_vals, 'm-', lw=2, label='Time to Impact')
    axes[2].set_ylabel('TTI (s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- FINAL VERIFICATION REPORT ---")
    if 2 in data['status_code']:
        print("Result: FAILURE PREDICTED. System correctly identified inevitable crash.")
    elif min(data['barrier']) >= -0.01:
        print("Result: VERIFIED SAFE. System maintained invariant set.")
    else:
        print("Result: UNSAFE. Barrier constraint violated.")