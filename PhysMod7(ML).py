import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, block_diag, inv
from collections import deque
import warnings

# ==============================================================================
# SECTION 0: DEPENDENCY HANDLING (FIXED)
# ==============================================================================
# Try importing Stable Baselines 3 (RL Library)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("Stable Baselines3 not installed. RL Training will be skipped, running PID baseline instead.")
    
    # FIX: Define a dummy BaseCallback so the class definition below doesn't crash
    class BaseCallback:
        def __init__(self, verbose=0): pass

# ==============================================================================
# SECTION 1: MANIFOLD MATHEMATICS UTILITIES
# ==============================================================================
class LieGroupMath:
    @staticmethod
    def wedge(v):
        """Maps vector in R^3 to skew-symmetric matrix in so(3)."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def vee(M):
        """Inverse map of wedge."""
        return np.array([M[2,1], M[0,2], M[1,0]])

    @staticmethod
    def Exp_SO3(w):
        """Exponential map from so(3) to SO(3)."""
        theta_sq = np.dot(w, w)
        theta = np.sqrt(theta_sq)
        K = LieGroupMath.wedge(w)
        if theta < 1e-6:
            return np.eye(3) + K + 0.5 * (K @ K)
        else:
            return np.eye(3) + (np.sin(theta)/theta) * K + ((1 - np.cos(theta))/theta_sq) * (K @ K)

# ==============================================================================
# SECTION 2: PHYSICS ENGINE (LGVI + ENV + FAULTS)
# ==============================================================================
class DrydenWindModel:
    """
    Simulates wind gusts using a simplified Dryden model (filtered noise).
    """
    def __init__(self, dt):
        self.dt = dt
        self.state = np.zeros(3)
        self.turbulence_intensity = 0.0
    
    def step(self, mean_wind):
        # First order Gauss-Markov process to simulate turbulence
        noise = np.random.normal(0, 1, 3)
        correlation_time = 2.0  # seconds
        alpha = np.exp(-self.dt / correlation_time)
        
        sigma = self.turbulence_intensity
        self.state = alpha * self.state + (1 - alpha) * sigma * noise
        return mean_wind + self.state

class QuadrotorPhysics:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.mass = 1.5
        self.J = np.diag([0.02, 0.02, 0.04])
        self.invJ = np.linalg.inv(self.J)
        self.g = 9.81
        self.arm_length = 0.25
        
        # Environment Objects
        self.wind_model = DrydenWindModel(dt)
        self.obstacles = [{'center': np.array([2.0, 2.0, 3.0]), 'radius': 1.0}] # Static Sphere
        
        # Actuator Mixing
        self.k_thrust = 1.0 
        self.k_torque = 0.05
        
        self.reset()

    def reset(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.wind_model.state = np.zeros(3)
        return self.get_state()

    def check_collision(self):
        # Ground check
        if self.p[2] < 0: return True
        
        # Obstacle check
        for obs in self.obstacles:
            if np.linalg.norm(self.p - obs['center']) < obs['radius'] + 0.2: # +0.2 drone radius
                return True
        return False

    def step(self, action_wrench, motor_health, mean_wind_vector, turbulence_level):
        """
        Evolves state using LGVI.
        """
        self.wind_model.turbulence_intensity = turbulence_level
        current_wind = self.wind_model.step(mean_wind_vector)

        # 1. Actuator Dynamics & Fault Injection
        avg_health = np.mean(motor_health)
        # Add slight actuator lag/noise
        effective_wrench = action_wrench * avg_health * np.random.uniform(0.98, 1.02, 4)
        
        f_thrust_scalar = effective_wrench[0]
        tau_body = effective_wrench[1:]
        
        # 2. Translational Dynamics
        F_gravity = np.array([0, 0, -self.mass * self.g])
        F_thrust = self.R @ np.array([0, 0, max(0, f_thrust_scalar)]) 
        
        # Drag depends on airspeed (velocity relative to wind)
        v_air = self.v - current_wind
        F_drag = -0.1 * np.linalg.norm(v_air) * v_air
        
        acc = (F_gravity + F_thrust + F_drag) / self.mass
        self.v += acc * self.dt
        self.p += self.v * self.dt
        
        # 3. Rotational Dynamics (SO(3))
        tau_gyro = -np.cross(self.w, self.J @ self.w)
        dw = self.invJ @ (tau_body + tau_gyro)
        self.w += dw * self.dt
        self.R = self.R @ LieGroupMath.Exp_SO3(self.w * self.dt)
        
        crashed = self.check_collision()
        if crashed and self.p[2] < 0:
            self.p[2] = 0
            self.v = np.zeros(3)
            self.w = np.zeros(3)
            
        return self.get_state(), crashed

    def get_state(self):
        return {'p': self.p, 'v': self.v, 'R': self.R, 'w': self.w}

# ==============================================================================
# SECTION 3: ADAPTIVE RIGHT-INVARIANT EKF
# ==============================================================================
class InvariantEKF:
    def __init__(self, dt):
        self.dt = dt
        # State: p, v, R, b_g, b_a
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        
        self.P = np.eye(15) * 0.1
        self.Q_nominal = np.eye(15) * 0.001
        self.R_noise = np.eye(3) * 0.5 

        # Diagnostics
        self.history_nis = deque(maxlen=100)
        self.history_cond = deque(maxlen=100)

    def predict(self, u_gyro, u_acc):
        w_hat = u_gyro - self.bg
        a_hat = u_acc - self.ba
        
        self.R = self.R @ LieGroupMath.Exp_SO3(w_hat * self.dt)
        self.v = self.v + (self.R @ a_hat + np.array([0,0,-9.81])) * self.dt
        self.p = self.p + self.v * self.dt
        
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Adaptive Process Noise: If high dynamic maneuvers, inflate Q
        dynamic_factor = 1.0 + np.linalg.norm(w_hat)
        self.P = F @ self.P @ F.T + self.Q_nominal * dynamic_factor

    def update(self, y_gps):
        # Adaptive: If GPS is NaN (dropout), skip update but inflate covariance
        if np.any(np.isnan(y_gps)): 
            self.P *= 1.01 # Uncertainty grows without measurement
            return 0.0
        
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3) 
        
        resid = y_gps - self.p
        S = H @ self.P @ H.T + self.R_noise
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            correction = K @ resid
            
            self.p += correction[0:3]
            self.v += correction[3:6]
            self.R = LieGroupMath.Exp_SO3(correction[6:9]) @ self.R
            self.bg += correction[9:12]
            self.ba += correction[12:15]
            
            self.P = (np.eye(15) - K @ H) @ self.P
            
            nis = resid.T @ np.linalg.inv(S) @ resid
            self.history_nis.append(nis)
            self.history_cond.append(np.linalg.cond(self.P))
            return nis
        except:
            return 100.0

    def get_state(self):
        return {'p': self.p, 'v': self.v, 'R': self.R, 'w': np.zeros(3)}

# ==============================================================================
# SECTION 4: ENHANCED SAFETY SUPERVISOR (CBF)
# ==============================================================================
class SafetySupervisor:
    def __init__(self):
        self.z_safe = 0.2
        self.v_max_xy = 3.0  # Max lateral velocity
        self.radius_max = 8.0 # Geofence radius
        
        self.gamma_z = 2.0
        self.gamma_v = 1.0
        self.interventions = 0
        
    def filter_action(self, state, action, mass):
        """
        Extended CBF: Handles altitude, lateral geofencing, and max velocity.
        """
        p = state['p']
        v = state['v']
        safe_action = action.copy()
        intervention_type = "None"
        
        # 1. Altitude Barrier (h_z = z - z_safe)
        h_z = p[2] - self.z_safe
        barrier_z = v[2] + self.gamma_z * h_z
        
        if barrier_z < 0 and v[2] < -0.5:
            min_thrust = mass * 9.81 * 1.5
            if safe_action[0] < min_thrust:
                safe_action[0] = min_thrust
                intervention_type = "Altitude"

        # 2. Lateral Geofence Barrier (h_lat = R^2 - (x^2 + y^2))
        r_curr = np.linalg.norm(p[0:2])
        v_radial = np.dot(p[0:2], v[0:2]) / (r_curr + 1e-6)
        
        # If outside radius and moving outwards
        if r_curr > self.radius_max and v_radial > 0:
             # Reduce thrust to stop divergence? No, lean back. 
             # For RL action space, we dampen lateral inputs.
             safe_action[1:] *= 0.5 # Dampen torques
             intervention_type = "Geofence"

        # 3. Velocity Limit (Braking if too fast)
        v_mag = np.linalg.norm(v[0:2])
        if v_mag > self.v_max_xy:
            # Simple soft cap: Reduce thrust (to reduce acceleration capability) or dampen rotation
            safe_action[0] *= 0.9 
            intervention_type = "SpeedLimit"
        
        triggered = (intervention_type != "None")
        if triggered: self.interventions += 1
        
        return safe_action, triggered, intervention_type

# ==============================================================================
# SECTION 5: CURRICULUM RL ENVIRONMENT
# ==============================================================================
class QuadLearnEnv(gym.Env):
    def __init__(self):
        super(QuadLearnEnv, self).__init__()
        
        self.physics = QuadrotorPhysics()
        self.estimator = InvariantEKF(dt=0.01)
        self.supervisor = SafetySupervisor()
        
        # Action: [Thrust, TorqueX, TorqueY, TorqueZ] (Normalized -1..1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Observation: [p(3), v(3), R(9), w(3), dist(1), difficulty(1)] = 20
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.target_pos = np.array([0, 0, 5.0])
        self.max_steps = 600
        
        # Curriculum Learning
        self.difficulty = 0 # 0: Easy, 1: Medium, 2: Hard, 3: Adversarial
        self.prev_action = np.zeros(4)

    def set_difficulty(self, level):
        self.difficulty = np.clip(level, 0, 3)
        print(f"Environment Difficulty Set to Level {self.difficulty}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Physics Reset
        true_state = self.physics.reset()
        
        # 2. Estimator Reset
        self.estimator.p = true_state['p'] + np.random.normal(0, 0.5, 3)
        self.estimator.R = np.eye(3)
        self.estimator.P = np.eye(15) * 0.1
        self.estimator.history_nis.clear()
        
        # 3. Curriculum Config
        if self.difficulty == 0:
            self.mean_wind = np.zeros(3)
            self.turbulence = 0.0
            self.prob_fault = 0.0
            self.gps_fail_prob = 0.0
        elif self.difficulty == 1:
            self.mean_wind = np.random.uniform(-1, 1, 3)
            self.turbulence = 0.5
            self.prob_fault = 0.0
            self.gps_fail_prob = 0.0
        elif self.difficulty == 2:
            self.mean_wind = np.random.uniform(-3, 3, 3)
            self.turbulence = 1.0
            self.prob_fault = 0.2
            self.gps_fail_prob = 0.05
        else: # Adversarial
            self.mean_wind = np.random.uniform(-5, 5, 3)
            self.turbulence = 2.0
            self.prob_fault = 0.4
            self.gps_fail_prob = 0.2

        self.motor_health = np.ones(4)
        if np.random.rand() < self.prob_fault:
            bad_motor = np.random.randint(0, 4)
            self.motor_health[bad_motor] = np.random.uniform(0.4, 0.8) # Degradation
            
        self.step_count = 0
        self.prev_action = np.zeros(4)
        return self._get_obs(), {}
        
    def step(self, action):
        self.step_count += 1
        
        # Scale action: Thrust [0, 20N], Torques [-1, 1 Nm]
        scaled_action = np.zeros(4)
        scaled_action[0] = (action[0] + 1) * 10.0 
        scaled_action[1:] = action[1:] * 1.0
        
        # 1. Safety Filter
        true_state_prev = self.physics.get_state()
        safe_action, safety_trig, safety_type = self.supervisor.filter_action(true_state_prev, scaled_action, self.physics.mass)
        
        # 2. Physics Step
        true_state, crashed = self.physics.step(safe_action, self.motor_health, self.mean_wind, self.turbulence)
        
        # 3. Sensors (with Failures)
        acc_meas = true_state['R'].T @ (np.array([0,0,9.81]) + (true_state['v']-true_state_prev['v'])/0.01) + np.random.normal(0, 0.2, 3)
        gyro_meas = true_state['w'] + np.random.normal(0, 0.01, 3)
        self.estimator.predict(gyro_meas, acc_meas)
        
        nis = 0
        if self.step_count % 10 == 0: # 10Hz GPS
            # Simulated GPS Dropout/Jamming
            if np.random.rand() < self.gps_fail_prob:
                gps_meas = np.array([np.nan, np.nan, np.nan])
            else:
                gps_meas = true_state['p'] + np.random.normal(0, 0.5, 3)
            nis = self.estimator.update(gps_meas)
            
        # 4. Reward Shaping
        est_state = self.estimator.get_state()
        pos_error = np.linalg.norm(est_state['p'] - self.target_pos)
        
        # Smoothness (Jerk penalty)
        smoothness_penalty = np.linalg.norm(action - self.prev_action)
        self.prev_action = action
        
        r_pos = -1.0 * pos_error
        r_smooth = -0.1 * smoothness_penalty
        r_energy = -0.001 * np.sum(np.abs(safe_action))
        r_safety = -5.0 if safety_trig else 0.0
        r_crash = -100.0 if crashed else 0.0
        r_survival = 0.2 # Increased incentive to just stay alive
        
        reward = r_pos + r_smooth + r_energy + r_safety + r_crash + r_survival
        
        # 5. Termination
        truncated = False
        terminated = crashed or (self.step_count >= self.max_steps)
        if pos_error > 15.0: truncated = True # Fly away
        
        info = {
            "nis": nis, 
            "safety": safety_trig, 
            "safety_type": safety_type,
            "true_pos": true_state['p'],
            "est_pos": est_state['p'],
            "action": safe_action
        }
        
        return self._get_obs(), reward, terminated, truncated, info
        
    def _get_obs(self):
        est = self.estimator.get_state()
        dist = np.linalg.norm(est['p'] - self.target_pos)
        
        obs = np.concatenate([
            est['p'],
            est['v'],
            est['R'].flatten(),
            self.physics.w,
            [dist],
            [float(self.difficulty)]
        ])
        return obs.astype(np.float32)

# ==============================================================================
# SECTION 6: EXECUTION, BASELINES, AND VISUALIZATION
# ==============================================================================

def run_geometric_baseline(env):
    """Enhanced Geometric Controller Baseline."""
    print("Running Enhanced Geometric Control Baseline...")
    obs, _ = env.reset()
    logs = {'x':[], 'y':[], 'z':[], 'x_est':[], 'y_est':[], 'z_est':[], 'thrust':[], 'safety':[]}
    
    for _ in range(500):
        # Extract state
        p = obs[0:3]
        v = obs[3:6]
        R = obs[6:15].reshape(3,3)
        
        # PID with Feedforward
        target = env.target_pos
        err_p = p - target
        err_v = v
        
        # Feedforward gravity compensation + PID
        kp = 3.0
        kv = 2.5
        F_des = -kp * err_p - kv * err_v + np.array([0,0, env.physics.mass * 9.81])
        
        # Geometric Attitude Control
        z_b = R[:,2]
        thrust = np.dot(F_des, z_b)
        
        # Simplistic orientation logic (assume internal attitude controller for RL space mapping)
        # Here we just map thrust roughly to demonstrate physics
        thrust_norm = (thrust / 10.0) - 1.0
        action = np.array([thrust_norm, 0, 0, 0])
        
        obs, r, term, trunc, info = env.step(action)
        
        logs['x'].append(info['true_pos'][0])
        logs['y'].append(info['true_pos'][1])
        logs['z'].append(info['true_pos'][2])
        logs['x_est'].append(info['est_pos'][0])
        logs['y_est'].append(info['est_pos'][1])
        logs['z_est'].append(info['est_pos'][2])
        logs['thrust'].append(info['action'][0])
        logs['safety'].append(1 if info['safety'] else 0)
        
        if term or trunc: break
    return logs

def plot_comprehensive_results(logs):
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(logs['x'], logs['y'], logs['z'], 'b-', label='Ground Truth')
    ax1.plot(logs['x_est'], logs['y_est'], logs['z_est'], 'r--', label='EKF Est')
    # Draw Obstacle
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 2.0 + 1.0 * np.outer(np.cos(u), np.sin(v))
    y = 2.0 + 1.0 * np.outer(np.sin(u), np.sin(v))
    z = 3.0 + 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='gray', alpha=0.3)
    ax1.set_title("3D Flight Path & Obstacle")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()
    
    # 2. Altitude Tracking
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(logs['z'], label='Altitude')
    ax2.axhline(5.0, color='g', linestyle='--', label='Target')
    ax2.set_title("Altitude Tracking")
    ax2.grid()
    ax2.legend()
    
    # 3. Control Inputs
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(logs['thrust'], color='orange', label='Thrust (N)')
    ax3.set_title("Control Effort (Thrust)")
    ax3.set_xlabel("Step")
    ax3.grid()
    
    # 4. Safety Interventions
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(logs['safety'], 'k-', label='Supervisor Active')
    ax4.fill_between(range(len(logs['safety'])), logs['safety'], color='red', alpha=0.3)
    ax4.set_title("Safety Supervisor Interventions")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Nominal', 'Intervention'])
    
    plt.tight_layout()
    plt.show()

# FIX: Class definition for CurriculumCallback must inherit from BaseCallback
# If SB3 is absent, BaseCallback is defined as a dummy in Section 0
class CurriculumCallback(BaseCallback):
    """
    Custom callback for updating curriculum difficulty during training.
    """
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.eval_freq = 1000
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Simple curriculum logic: Time-based
            if self.n_calls > 4000:
                self.env.set_difficulty(3)
            elif self.n_calls > 2000:
                self.env.set_difficulty(2)
            elif self.n_calls > 1000:
                self.env.set_difficulty(1)
        return True

if __name__ == "__main__":
    env = QuadLearnEnv()
    
    if SB3_AVAILABLE:
        print("Starting RL Training with Curriculum...")
        # Start Easy
        env.set_difficulty(0)
        
        callback = CurriculumCallback(env)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, ent_coef=0.01)
        
        # Train
        model.learn(total_timesteps=6000, callback=callback)
        print("Training Complete.")
        
        # Evaluate on Hard
        env.set_difficulty(3)
        obs, _ = env.reset()
        logs = {'x':[], 'y':[], 'z':[], 'x_est':[], 'y_est':[], 'z_est':[], 'thrust':[], 'safety':[]}
        
        print("Evaluating Policy on Level 3...")
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            logs['x'].append(info['true_pos'][0])
            logs['y'].append(info['true_pos'][1])
            logs['z'].append(info['true_pos'][2])
            logs['x_est'].append(info['est_pos'][0])
            logs['y_est'].append(info['est_pos'][1])
            logs['z_est'].append(info['est_pos'][2])
            logs['thrust'].append(info['action'][0])
            logs['safety'].append(1 if info['safety'] else 0)
            
            if done: break
            
        plot_comprehensive_results(logs)
        
    else:
        # Run Baseline if no RL
        env.set_difficulty(2) # Test baseline on medium difficulty
        logs = run_geometric_baseline(env)
        plot_comprehensive_results(logs)

    print("Stage 7 Pipeline Executed Successfully.")
