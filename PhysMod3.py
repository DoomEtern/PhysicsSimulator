import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as ScipyRot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor
import time

# ==============================================================================
# PART 1: EXTENDED KALMAN FILTER (EKF) FOR QUADCOPTER STATE ESTIMATION
# ==============================================================================
class QuadStateEstimator:
    def __init__(self, time_step):
        self.dt = time_step
        self.state_vector = np.zeros(10)
        self.state_vector[6] = 1.0 # Initialize quaternion to identity (no rotation)
        self.P = np.eye(10) * 0.1
        
        # Process Noise Covariance (Q)
        self.Q = np.diag([
            0.5, 0.5, 0.5,       # Position
            1.0, 1.0, 1.0,       # Velocity
            0.05, 0.05, 0.05, 0.05 # Orientation
        ]) * time_step
        
        # Measurement Noise Covariance (R) - Represents uncertainty in GPS readings
        # Matches the simulated GPS noise standard deviation
        self.R_gps = np.diag([0.5, 0.5, 0.8]) ** 2 

    def predict_step(self, accel_reading, gyro_reading):
        """ 
        Prediction step using high-frequency IMU data.
        Propagates the state forward in time.
        """
        pos, vel, quat = self.state_vector[0:3], self.state_vector[3:6], self.state_vector[6:10]
        
        # 1. Update Orientation using Gyroscope Data
        current_rot = ScipyRot.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Scipy uses x,y,z,w format
        rot_matrix = current_rot.as_matrix()
        
        angular_speed = np.linalg.norm(gyro_reading)
        if angular_speed > 1e-6:
            rotation_axis = gyro_reading / angular_speed
            angle = angular_speed * self.dt
            # Quaternion representing the incremental rotation
            delta_quat = np.array([np.cos(angle/2), *list(rotation_axis * np.sin(angle/2))])
            rot_delta = ScipyRot.from_quat([delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0]])
            new_rot = current_rot * rot_delta
            new_quat_scipy = new_rot.as_quat() 
            # Convert back to w, x, y, z format
            new_quat = np.array([new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]])
        else:
            new_quat = quat

        # 2. Update Velocity using Accelerometer Data
        gravity_vector = np.array([0, 0, 9.81])
        # Transform acceleration from body frame to world frame and subtract gravity
        accel_world = rot_matrix @ accel_reading - gravity_vector
        new_vel = vel + accel_world * self.dt
        
        # 3. Update Position
        new_pos = pos + vel * self.dt + 0.5 * accel_world * self.dt**2
        
        # Update the state vector
        self.state_vector = np.concatenate([new_pos, new_vel, new_quat])
        
        # Update Covariance Matrix (Simplified Jacobian approximation)
        F_matrix = np.eye(10)
        F_matrix[0:3, 3:6] = np.eye(3) * self.dt # Position dependence on velocity
        self.P = F_matrix @ self.P @ F_matrix.T + self.Q

    def correction_step(self, gps_reading):
        """ 
        Correction step using low-frequency GPS data.
        Adjusts the state estimate based on position measurements.
        """
        if np.any(np.isnan(gps_reading)): return
        
        # Measurement Matrix (H) - Maps state to measurement space (position only)
        H_matrix = np.zeros((3, 10))
        H_matrix[0:3, 0:3] = np.eye(3)
        
        # Predicted measurement
        predicted_measurement = H_matrix @ self.state_vector
        # Innovation (Residual)
        measurement_residual = gps_reading - predicted_measurement
        
        # Innovation Covariance (S)
        S_matrix = H_matrix @ self.P @ H_matrix.T + self.R_gps
        # Kalman Gain (K)
        K_gain = self.P @ H_matrix.T @ np.linalg.inv(S_matrix)
        
        # Update State Estimate
        self.state_vector = self.state_vector + K_gain @ measurement_residual
        # Update Covariance Estimate
        self.P = (np.eye(10) - K_gain @ H_matrix) @ self.P
        
        # Re-normalize Quaternion to ensure it represents a valid rotation
        self.state_vector[6:10] /= np.linalg.norm(self.state_vector[6:10])

    def output_state(self, raw_gyro):
        """ Extracts and formats the relevant state variables for the controller. """
        q = self.state_vector[6:10]
        rot_obj = ScipyRot.from_quat([q[1], q[2], q[3], q[0]])
        rot_mat = rot_obj.as_matrix()
        
        vel_world = self.state_vector[3:6]
        # Transform velocity to body frame
        vel_body = rot_mat.T @ vel_world
        
        return {
            'position': self.state_vector[0:3],
            'velocity_body': vel_body,
            'rotation_matrix': rot_mat,
            'angular_velocity_body': raw_gyro 
        }

# ==============================================================================
# PART 2: HIGH-FIDELITY PHYSICS SIMULATOR
# ==============================================================================
class QuadcopterPhysics:
    def __init__(self, random_seed=None):
        if random_seed is not None: np.random.seed(random_seed)
        
        # Vehicle Parameters
        self.mass = 1.5 # kg
        self.inertia_matrix = np.diag([0.015, 0.015, 0.03]) # kg*m^2
        self.inv_inertia = np.linalg.inv(self.inertia_matrix)
        self.gravity = 9.81 # m/s^2
        self.arm_length = 0.25 # m
        
        # Aerodynamics and Environment
        self.drag_coeff_linear = 0.1
        self.ground_level = 0.0
        self.propeller_radius = 0.12 # m
        self.wind_vector = np.zeros(3)
        
        # Propulsion System
        self.thrust_coeff = 3e-6
        self.torque_coeff = 1e-7
        self.max_motor_rpm = 9000
        self.motor_time_constant = 0.03 # s
        self.current_motor_rpms = np.zeros(4)
        self.motor_health_status = np.ones(4) # 1.0 = healthy, 0.0 = failed
        
        # Battery Model
        self.battery_capacity = 2200 * 3.6 # Coulombs (mAh -> C)
        self.current_charge = self.battery_capacity
        self.internal_resistance = 0.02 # Ohms
        
    def skew_symmetric(self, v):
        """ Creates a skew-symmetric matrix from a vector (for cross products). """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def calculate_ground_effect(self, altitude):
        """ Simulates increased lift when close to the ground. """
        height_above_ground = max(altitude - self.ground_level, 0.1)
        if height_above_ground < 2 * self.propeller_radius:
            return 1.0 / (1 - (self.propeller_radius/(4*height_above_ground))**2)
        return 1.0

    def update_wind(self, dt):
        """ Updates wind vector using a simplified Dryden wind turbulence model. """
        self.wind_vector += (-self.wind_vector * 0.5 * dt) + \
                           1.5 * np.random.normal(0, np.sqrt(dt), 3)
        return self.wind_vector

    def run_physics_step(self, current_state, target_motor_rpms, dt):
        """ 
        Advances the simulation by one time step using RK4 integration.
        Includes motor dynamics, aerodynamics, and battery discharge.
        """
        # Simulate motor lag
        self.current_motor_rpms += (target_motor_rpms - self.current_motor_rpms) * dt / self.motor_time_constant
        # Apply failure mask (simulate broken motors)
        effective_rpms = self.current_motor_rpms * self.motor_health_status
        
        pos, vel_body, rot_mat, ang_vel_body = current_state['position'], current_state['velocity_body'], current_state['rotation_matrix'], current_state['angular_velocity_body']
        vel_world = rot_mat @ vel_body
        
        # Dynamics function for integration
        def derivatives(v_w, R, w_b):
            ground_multiplier = self.calculate_ground_effect(pos[2])
            thrust_forces = self.thrust_coeff * effective_rpms**2 * ground_multiplier
            total_thrust_body = np.array([0, 0, np.sum(thrust_forces)])
            
            # Drag force in body frame
            drag_force_body = -self.drag_coeff_linear * (R.T @ v_w)
            
            # Total force in world frame
            total_force_world = (R @ (total_thrust_body + drag_force_body)) + self.update_wind(dt) + np.array([0, 0, -self.mass*self.gravity])
            accel_world = total_force_world / self.mass
            
            # Torques in body frame
            torques = np.array([
                self.arm_length * (thrust_forces[0] - thrust_forces[1] - thrust_forces[2] + thrust_forces[3]),
                self.arm_length * (-thrust_forces[0] - thrust_forces[1] + thrust_forces[2] + thrust_forces[3]),
                self.torque_coeff * (thrust_forces[0] - thrust_forces[1] + thrust_forces[2] - thrust_forces[3])
            ])
            
            # Angular acceleration (Euler's equation for rigid body dynamics)
            ang_accel = self.inv_inertia @ (torques - np.cross(w_b, self.inertia_matrix @ w_b))
            return accel_world, ang_accel

        # RK4 Integration Steps
        k1_acc, k1_ang_acc = derivatives(vel_world, rot_mat, ang_vel_body)
        
        v2 = vel_world + 0.5*dt*k1_acc
        w2 = ang_vel_body + 0.5*dt*k1_ang_acc
        R2 = rot_mat @ expm(self.skew_symmetric(ang_vel_body * 0.5 * dt))
        k2_acc, k2_ang_acc = derivatives(v2, R2, w2)
        
        v3 = vel_world + 0.5*dt*k2_acc
        w3 = ang_vel_body + 0.5*dt*k2_ang_acc
        R3 = rot_mat @ expm(self.skew_symmetric(w2 * 0.5 * dt))
        k3_acc, k3_ang_acc = derivatives(v3, R3, w3)
        
        v4 = vel_world + dt*k3_acc
        w4 = ang_vel_body + dt*k3_ang_acc
        R4 = rot_mat @ expm(self.skew_symmetric(w3 * dt))
        k4_acc, k4_ang_acc = derivatives(v4, R4, w4)
        
        # Combine RK4 steps
        vel_world_next = vel_world + (dt/6.0)*(k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)
        ang_vel_body_next = ang_vel_body + (dt/6.0)*(k1_ang_acc + 2*k2_ang_acc + 2*k3_ang_acc + k4_ang_acc)
        pos_next = pos + vel_world * dt 
        
        # Update Rotation Matrix
        avg_ang_vel = (dt/6.0)*(k1_ang_acc + 2*k2_ang_acc + 2*k3_ang_acc + k4_ang_acc)
        rot_mat_next = rot_mat @ expm(self.skew_symmetric(ang_vel_body + 0.5*avg_ang_vel*dt))
        # Orthogonalize matrix to prevent drift
        U, _, Vt = np.linalg.svd(rot_mat_next)
        rot_mat_next = U @ Vt
        
        # Battery Simulation
        current_draw = np.sum(effective_rpms**2) * 1e-7
        self.current_charge -= current_draw * dt
        state_of_charge = max(0, self.current_charge / self.battery_capacity)
        voltage = (10.0 + 1.2*state_of_charge) - current_draw*self.internal_resistance
        
        # Calculate "Proper Acceleration" (what an IMU would measure, excluding gravity)
        ground_mult = self.calculate_ground_effect(pos[2])
        thrusts = self.thrust_coeff * effective_rpms**2 * ground_mult
        force_thrust_body = np.array([0,0,np.sum(thrusts)])
        force_drag_body = -self.drag_coeff_linear * vel_body
        proper_accel_body = (force_thrust_body + force_drag_body) / self.mass

        return {
            'position': pos_next, 
            'velocity_body': rot_mat_next.T @ vel_world_next, 
            'rotation_matrix': rot_mat_next, 
            'angular_velocity_body': ang_vel_body_next,
            'proper_acceleration': proper_accel_body, 
            'voltage': voltage
        }

    def generate_sensor_readings(self, true_state):
        """ Adds noise to true state values to simulate sensor imperfections. """
        accel_noise = np.random.normal(0, 0.2, 3)
        gyro_noise = np.random.normal(0, 0.01, 3)
        
        measured_accel = true_state['proper_acceleration'] + accel_noise
        measured_gyro = true_state['angular_velocity_body'] + gyro_noise
        
        gps_noise = np.random.normal(0, 0.5, 3)
        measured_pos = true_state['position'] + gps_noise
        
        return measured_accel, measured_gyro, measured_pos

# ==============================================================================
# PART 3: ROBUST FLIGHT CONTROLLER
# ==============================================================================
class GeometricController:
    def __init__(self, mass, inertia, k_thrust, k_torque, arm_len):
        self.m, self.J, self.kf, self.km, self.L = mass, inertia, k_thrust, k_torque, arm_len
        
        # Control Gains
        self.kp_pos = np.array([5.0, 5.0, 6.0])
        self.kv_vel = np.array([3.0, 3.0, 4.0])
        self.k_rot = 10.0
        self.k_ang_vel = 2.0
        
        self.integral_error = np.zeros(3)
        self.allocation_matrix = self._create_allocation_matrix()
        
    def _create_allocation_matrix(self):
        l, c = self.L, self.km/self.kf
        return np.array([
            [1, 1, 1, 1],
            [l, -l, -l, l],
            [-l, -l, l, l],
            [c, -c, c, -c]
        ])

    def calculate_motor_commands(self, estimated_state, target_position, failure_mask):
        """ Calculates required motor RPMs to track the target position. """
        R = estimated_state['rotation_matrix']
        p = estimated_state['position']
        v = R @ estimated_state['velocity_body'] 
        w = estimated_state['angular_velocity_body']
        
        # Position Error
        pos_error = p - target_position
        self.integral_error += pos_error * 0.01
        self.integral_error = np.clip(self.integral_error, -2, 2) # Anti-windup
        
        # Desired Acceleration
        accel_desired = -self.kp_pos*pos_error - self.kv_vel*v - 0.2*self.integral_error + np.array([0,0,9.81])
        force_desired = self.m * accel_desired
        
        # Orientation Control (Geometric method)
        z_axis_body_des = force_desired / (np.linalg.norm(force_desired) + 1e-6)
        x_axis_world = np.array([1, 0, 0])
        y_axis_body_des = np.cross(z_axis_body_des, x_axis_world)
        y_axis_body_des /= np.linalg.norm(y_axis_body_des)
        x_axis_body_des = np.cross(y_axis_body_des, z_axis_body_des)
        R_desired = np.stack([x_axis_body_des, y_axis_body_des, z_axis_body_des], axis=1)
        
        # Orientation Error
        def vee_map(M): return np.array([M[2,1], M[0,2], M[1,0]])
        rot_error = 0.5 * vee_map(R_desired.T @ R - R.T @ R_desired)
        
        # Compute desired Torques and Thrust
        total_thrust = np.dot(force_desired, R[:,2])
        torques = -self.k_rot*rot_error - self.k_ang_vel*w
        
        wrench_vector = np.array([total_thrust, torques[0], torques[1], torques[2]])
        
        # Control Allocation (solving for motor forces)
        effective_alloc_matrix = self.allocation_matrix @ np.diag(failure_mask)
        try:
            # Pseudo-inverse to handle potential matrix singularity from failures
            motor_forces_sq = np.linalg.pinv(effective_alloc_matrix) @ wrench_vector
        except:
            motor_forces_sq = np.zeros(4)
            
        # Convert forces to RPMs
        motor_forces_sq = np.clip(motor_forces_sq, 0, 9000**2 * self.kf)
        return np.sqrt(motor_forces_sq / self.kf)

# ==============================================================================
# PART 4: SIMULATION EXECUTION
# ==============================================================================
def run_simulation_trial(trial_number):
    """ Runs a single simulation trial with a unique seed. """
    sim = QuadcopterPhysics(random_seed=trial_number)
    estimator = QuadStateEstimator(time_step=0.005) # 200Hz
    controller = GeometricController(sim.mass, sim.inertia_matrix, sim.thrust_coeff, sim.torque_coeff, sim.arm_length)
    
    # Initial State
    current_state = {
        'position': np.array([0., 0., 0.1]),
        'velocity_body': np.zeros(3),
        'rotation_matrix': np.eye(3),
        'angular_velocity_body': np.zeros(3),
        'proper_acceleration': np.array([0,0,9.81]),
        'voltage': 12.6
    }
    
    setpoint = np.array([5.0, 5.0, 8.0])
    
    history_pos_true, history_pos_est = [], []
    history_voltage, history_error = [], []
    
    elapsed_time = 0
    gps_tick_counter = 0
    sim_failure_desc = "None"
    
    gyro_reading = np.zeros(3) # Initial value
    
    # Run loop for 1600 steps (8 seconds at 200Hz)
    for _ in range(1600):
        # 1. Physics Update
        motor_health = np.ones(4)
        
        # Inject failure in trial 99
        if trial_number == 99 and elapsed_time > 4.0:
            motor_health[0] = 0.0
            sim_failure_desc = "Motor 1 Failed at t=4.0s"
        sim.motor_health_status = motor_health
        
        if gps_tick_counter > 0:
            estimated_state = estimator.output_state(gyro_reading)
            motor_commands = controller.calculate_motor_commands(estimated_state, setpoint, motor_health)
        else:
            motor_commands = np.zeros(4)

        current_state = sim.run_physics_step(current_state, motor_commands, dt=0.005)
        
        # 2. Sensor Simulation & State Estimation
        accel_reading, gyro_reading, gps_reading = sim.generate_sensor_readings(current_state)
        
        estimator.predict_step(accel_reading, gyro_reading)
        
        # GPS Update (10Hz, every 20th step)
        if gps_tick_counter % 20 == 0:
            estimator.correction_step(gps_reading)
        gps_tick_counter += 1
        
        # 3. Data Logging
        history_pos_true.append(current_state['position'].copy())
        history_pos_est.append(estimator.state_vector[0:3].copy())
        history_voltage.append(current_state['voltage'])
        history_error.append(np.linalg.norm(current_state['position'] - estimator.state_vector[0:3]))
        elapsed_time += 0.005

    return {
        'id': trial_number,
        'failure_mode': sim_failure_desc,
        'path_true': np.array(history_pos_true),
        'path_est': np.array(history_pos_est),
        'voltage_log': np.array(history_voltage),
        'error_log': np.array(history_error)
    }

def visualize_data(sim_results):
    """ Generates plots for analysis. """
    fig = plt.figure(figsize=(18, 9))
    
    # Plot 1: 3D Flight Path
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for res in sim_results:
        path = res['path_true']
        est_path = res['path_est']
        transparency = 1.0 if res['id'] == 99 else 0.3
        label_text = f"Actual Path (Trial {res['id']})" if res['id'] == 99 else None
        
        ax1.plot(path[:,0], path[:,1], path[:,2], 'b-', alpha=transparency, label=label_text)
        if res['id'] == 99:
            ax1.plot(est_path[:,0], est_path[:,1], est_path[:,2], 'r--', label="EKF Estimate")
            
    ax1.set_title("3D Flight Trajectory: Ground Truth vs. Estimation")
    ax1.set_xlabel("X Position (m)"); ax1.set_ylabel("Y Position (m)"); ax1.set_zlabel("Altitude (m)")
    ax1.legend()
    
    # Plot 2: Battery Voltage
    ax2 = fig.add_subplot(2, 2, 2)
    time_axis = np.arange(len(sim_results[0]['voltage_log'])) * 0.005
    
    for res in sim_results:
        ax2.plot(time_axis, res['voltage_log'], alpha=0.5)
    ax2.set_title("Battery Voltage Drop During Flight")
    ax2.set_ylabel("Voltage (V)")
    ax2.axhline(9.0, color='r', linestyle='--', label="Critical Low Voltage")
    ax2.legend()
    
    # Plot 3: Position Error
    ax3 = fig.add_subplot(2, 2, 4)
    for res in sim_results:
        ax3.plot(time_axis, res['error_log'], alpha=0.6)
    ax3.set_title("State Estimation Error over Time")
    ax3.set_ylabel("Error Magnitude (m)")
    ax3.set_xlabel("Simulation Time (s)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting Advanced Multicopter Simulation Framework...")
    simulation_ids = [0, 1, 2, 3, 4, 99]
    
    # Run simulations in parallel
    with ProcessPoolExecutor() as executor:
        batch_results = list(executor.map(run_simulation_trial, simulation_ids))
        
    failed_trials = [r for r in batch_results if r['id'] == 99]
    print(f"\nSimulation batch complete. Total trials: {len(batch_results)}")
    
    if failed_trials:
        failure_case = failed_trials[0]
        final_pos_error = failure_case['error_log'][-1]
        print(f"--- Failure Scenario Report (Trial 99) ---")
        print(f"  Condition: {failure_case['failure_mode']}")
        print(f"  Final Position Error: {final_pos_error:.3f} meters")
        print(f"  Ending Voltage: {failure_case['voltage_log'][-1]:.2f} V")
        
    visualize_data(batch_results)
