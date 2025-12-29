import numpy as np
from scipy.linalg import expm, block_diag
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from collections import deque

# --- ROBUST MANIFOLD MATH (PRESERVED) ---
def wedge(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def Exp_SO3(w):
    theta_sq = np.dot(w, w)
    theta = np.sqrt(theta_sq)
    K = wedge(w)
    if theta < 1e-6:
        return np.eye(3) + K + 0.5 * (K @ K)
    else:
        return np.eye(3) + (np.sin(theta)/theta) * K + ((1 - np.cos(theta))/theta_sq) * (K @ K)

# --- 1. ACTUATOR DYNAMICS & ALLOCATION ---
class PropulsionSystem:
    def __init__(self):
        d = 0.25 
        c = 0.05 
        self.A = np.array([
            [1, 1, 1, 1],          
            [d, -d, -d, d],        
            [d, d, -d, -d],        
            [-c, c, -c, c]         
        ])
        self.A_inv = np.linalg.pinv(self.A)
        self.f_min = 0.0
        self.f_max = 15.0 # Increased to 15.0 to allow survivability
        self.ramp_rate = 100.0 
        self.u_actual = np.zeros(4) 
        
    def step(self, wrench_cmd, dt, faults):
        u_des = self.A_inv @ wrench_cmd
        delta = u_des - self.u_actual
        delta = np.clip(delta, -self.ramp_rate*dt, self.ramp_rate*dt)
        self.u_actual += delta
        self.u_actual = np.clip(self.u_actual, self.f_min, self.f_max)
        u_faulted = self.u_actual * faults
        wrench_applied = self.A @ u_faulted
        # Return F_thrust (scalar along z), Tau (vector), and Total Effort Sum
        return wrench_applied[0], wrench_applied[1:], np.sum(self.u_actual)

# --- 2. ENERGY SAFETY ENVELOPE ---
class SafetyEnvelope:
    def __init__(self, m, g):
        self.m = m
        self.g = g
        self.max_thrust = 4 * 15.0 
        
    def check_viability(self, z, vz, v_total):
        PE = self.m * self.g * max(0, z)
        a_breaking = (self.max_thrust / self.m) - self.g
        min_stop_dist = (vz**2) / (2 * a_breaking) if vz < 0 else 0
        margin = z - min_stop_dist
        return margin, PE + 0.5*self.m*v_total**2

# --- 3. PHYSICS CORE (LGVI) ---
class LGVI_TruthModel:
    def __init__(self, dt, start_z=0.0):
        self.dt = dt
        self.m = 1.5
        self.J = np.diag([0.02, 0.02, 0.04])
        self.invJ = np.linalg.inv(self.J)
        self.g = 9.81
        self.R = np.eye(3); self.p = np.array([0.0, 0.0, start_z]); self.v = np.zeros(3); self.w = np.zeros(3)
        self.propulsion = PropulsionSystem()
        
    def step(self, wrench_des, wind_field, motor_health):
        f_thrust, tau_b, total_effort = self.propulsion.step(wrench_des, self.dt, motor_health)
        F_grav = np.array([0, 0, -self.m * self.g])
        F_thrust_vec = self.R @ np.array([0, 0, f_thrust])
        v_rel = self.v - wind_field
        F_drag = -0.1 * np.linalg.norm(v_rel) * v_rel
        acc = (F_grav + F_thrust_vec + F_drag) / self.m
        self.v += acc * self.dt
        self.p += self.v * self.dt
        tau_gyro = -np.cross(self.w, self.J @ self.w)
        tau_drag = -0.01 * self.w
        dw = self.invJ @ (tau_b + tau_gyro + tau_drag)
        self.w += dw * self.dt
        self.R = self.R @ Exp_SO3(self.w * self.dt)
        u, _, vt = np.linalg.svd(self.R)
        self.R = u @ vt
        return self.R, self.p, self.v, self.w, total_effort

# --- 4. ESTIMATOR CREDIBILITY METRICS ---
class InvariantEKF:
    def __init__(self, start_z=0.0):
        self.R = np.eye(3); self.v = np.zeros(3); self.p = np.array([0.0, 0.0, start_z])
        self.bg = np.zeros(3); self.ba = np.zeros(3)
        self.P = np.eye(15) * 0.1
        self.Qc = block_diag(np.eye(3)*0.01, np.eye(3)*0.1, np.eye(3)*0.0, np.eye(3)*1e-4, np.eye(3)*1e-4)
        self.N = np.eye(3) * 0.5
        self.innov_history = deque(maxlen=20)
        
    def predict(self, u_gyro, u_acc, dt):
        w_hat = u_gyro - self.bg
        a_hat = u_acc - self.ba
        R_pred = self.R @ Exp_SO3(w_hat * dt)
        v_pred = self.v + (self.R @ a_hat + np.array([0,0,-9.81])) * dt
        p_pred = self.p + self.v * dt
        A = np.zeros((15, 15))
        A[0:3, 0:3] = -wedge(w_hat); A[0:3, 9:12] = -np.eye(3)
        A[3:6, 0:3] = -wedge(self.R @ a_hat); A[3:6, 12:15] = -self.R
        A[6:9, 3:6] = np.eye(3)
        Phi = expm(A * dt)
        self.P = Phi @ self.P @ Phi.T + Phi @ self.Qc @ Phi.T * dt
        self.P = 0.5 * (self.P + self.P.T)
        self.R = R_pred; self.v = v_pred; self.p = p_pred

    def update_gps(self, y_gps):
        H = np.zeros((3, 15)); H[0:3, 6:9] = np.eye(3)
        z = y_gps - self.p
        S = H @ self.P @ H.T + self.N
        try:
            nis = z.T @ np.linalg.inv(S) @ z
            K = self.P @ H.T @ np.linalg.inv(S)
        except: return 100.0, 0.0, 1e6
        
        dx = K @ z
        self.R = Exp_SO3(dx[0:3]) @ self.R
        self.v += dx[3:6]; self.p += dx[6:9]; self.bg += dx[9:12]; self.ba += dx[12:15]
        self.P = (np.eye(15) - K @ H) @ self.P
        self.innov_history.append(z)
        whiteness = 0.0
        if len(self.innov_history) > 2:
            current = self.innov_history[-1]
            prev = self.innov_history[-2]
            whiteness = np.abs(np.dot(current, prev) / (np.linalg.norm(current)*np.linalg.norm(prev) + 1e-9))
        cond_num = np.linalg.cond(self.P)
        return nis, whiteness, cond_num

# --- 5. FAILURE-AWARE MISSION MANAGER ---
class MissionManager:
    def __init__(self):
        self.mode = "NOMINAL"
        
    def update(self, t, nis, whiteness, impact_margin):
        if self.mode == "NOMINAL":
            # Tuned thresholds for clearer switching
            if nis > 20.0 or whiteness > 0.85:
                self.mode = "DEGRADED"
            if impact_margin < 2.0:
                self.mode = "EMERGENCY"
        elif self.mode == "DEGRADED":
            if nis > 100.0 or impact_margin < 1.0:
                self.mode = "EMERGENCY"
        return self.mode

    def get_control_gains(self):
        if self.mode == "NOMINAL": return 6.0, 3.5
        elif self.mode == "DEGRADED": return 2.0, 1.5
        else: return 0.0, 5.0

# --- 6. SIMULATION RUNNER ---
def run_scenario(scenario_config):
    dt = 0.005
    start_z = 15.0 # Start high for survivability demonstration
    truth = LGVI_TruthModel(dt, start_z=start_z)
    ekf = InvariantEKF(start_z=start_z)
    manager = MissionManager()
    safety = SafetyEnvelope(truth.m, truth.g)
    
    wind_type = scenario_config['wind']
    motor_fault_t = scenario_config['motor_fault_t'] 
    motor_efficiency = scenario_config['motor_efficiency']
    
    # Initialize State
    ekf.p = truth.p + np.array([0.5, -0.5, 0.0])
    
    # Expanded Logging
    log = {
        't': [], 'pos_err': [], 'nis': [], 'margin': [], 'mode': [], 
        'whiteness': [], 'vz': [], 'thrust': []
    }
    
    for k in range(1000): # 5 seconds
        t = k * dt
        
        # Sensing
        acc_meas = truth.R.T @ (np.array([0,0,9.81]) - truth.v*0.1) + np.random.normal(0, 0.1, 3)
        gyro_meas = truth.w + np.random.normal(0, 0.01, 3)
        ekf.predict(gyro_meas, acc_meas, dt)
        
        nis, whiteness, cond = np.nan, 0.0, 0.0
        if k % 10 == 0:
            gps = truth.p + np.random.normal(0, 0.3, 3)
            # Jamming Logic: Drift GPS slowly
            if 'jamming' in scenario_config and t > 2.0:
                gps += np.array([5.0 * (t-2.0), 5.0 * (t-2.0), 0.0])
            nis, whiteness, cond = ekf.update_gps(gps)
            
        # Safety & Control
        impact_margin, energy = safety.check_viability(ekf.p[2], ekf.v[2], np.linalg.norm(ekf.v))
        mode = manager.update(t, nis if not np.isnan(nis) else 0, whiteness, impact_margin)
        kp, kv = manager.get_control_gains()
        
        wrench_cmd = np.zeros(4)
        if mode == "EMERGENCY":
            wrench_cmd = np.array([0.0, 0, 0, 0])
            # Less conservative emergency thrust: Try to hover partially
            if ekf.p[2] > 0.5: wrench_cmd[0] = truth.m * 9.0 
        else:
            target = np.array([2.0, 2.0, 15.0]) # Hover high
            ep = ekf.p - target; ev = ekf.v
            F_des = -kp*ep - kv*ev + np.array([0,0,truth.m*9.81])
            u_thrust = np.dot(F_des, ekf.R[:,2])
            z_b = ekf.R[:,2]
            z_des = F_des / (np.linalg.norm(F_des) + 1e-6)
            axis_err = np.cross(z_b, z_des)
            tau_cmd = 3.0 * axis_err - 0.5 * ekf.bg
            wrench_cmd = np.array([u_thrust, tau_cmd[0], tau_cmd[1], tau_cmd[2]])

        # Physics
        health = np.ones(4)
        if t > motor_fault_t:
            health = motor_efficiency
        wind = np.zeros(3)
        if wind_type == "shear" and t > 2.0:
            wind = np.array([5.0, 5.0, 0.0]) # Stronger wind to force failure
            
        R, p, v, w, total_effort = truth.step(wrench_cmd, wind, health)
        
        log['t'].append(t)
        log['pos_err'].append(np.linalg.norm(truth.p - ekf.p))
        log['nis'].append(nis)
        log['margin'].append(impact_margin)
        log['whiteness'].append(whiteness)
        log['vz'].append(v[2])
        log['thrust'].append(total_effort)
        
        mode_int = 0 if mode=="NOMINAL" else (1 if mode=="DEGRADED" else 2)
        log['mode'].append(mode_int)
        
    return log

# --- 7. VISUALIZATION ---
if __name__ == "__main__":
    print("Initiating Simulating (fake) Campaign...")
    
    scenarios = []
    # 1. Nominal
    scenarios.append({'wind': 'none', 'motor_fault_t': 99.0, 'motor_efficiency': np.ones(4)})
    # 2. Rotor Loss (Recoverable from high altitude)
    scenarios.append({'wind': 'none', 'motor_fault_t': 2.0, 'motor_efficiency': np.array([1, 1, 0.6, 1])})
    # 3. GPS Jamming (Survives by degrading)
    scenarios.append({'wind': 'none', 'motor_fault_t': 99.0, 'motor_efficiency': np.ones(4), 'jamming': True})
    # 4. Wind + Motor Fault (Unrecoverable)
    scenarios.append({'wind': 'shear', 'motor_fault_t': 2.0, 'motor_efficiency': np.array([1, 1, 0.5, 1])})

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_scenario, scenarios))
        
    titles = ["Baseline", "Partial Rotor Loss (Survivable)", "GPS Jamming (Degraded)", "Wind + Motor Fault (Crash)"]
    
    # Complex 4x3 Grid for detailed analysis
    fig, axs = plt.subplots(4, 3, figsize=(18, 14))
    
    for i, res in enumerate(results):
        row = i
        t = np.array(res['t'])
        modes = np.array(res['mode'])
        
        # Helper to color background
        def shade_background(ax):
            ax.fill_between(t, -100, 100, where=modes==0, color='green', alpha=0.1)
            ax.fill_between(t, -100, 100, where=modes==1, color='yellow', alpha=0.1)
            ax.fill_between(t, -100, 100, where=modes==2, color='red', alpha=0.1)
            
        # Col 1: Vertical Velocity (Crash indicator)
        ax = axs[row, 0]
        ax.plot(t, res['vz'], 'b-', lw=2)
        shade_background(ax)
        ax.set_ylabel("Vert Vel (m/s)")
        ax.set_ylim(-10, 5)
        ax.axhline(0, color='k', linestyle=':')
        ax.set_title(f"{titles[i]} - Descent Profile", fontweight='bold')
        
        # Col 2: Thrust Applied (Effort)
        ax = axs[row, 1]
        ax.plot(t, res['thrust'], 'k-', lw=1.5)
        shade_background(ax)
        ax.set_ylabel("Total Thrust (N)")
        ax.set_ylim(0, 65)
        ax.set_title("Control Effort")
        
        # Col 3: Estimator Health (NIS)
        ax = axs[row, 2]
        # Filter NaNs for plotting
        valid_idxs = ~np.isnan(res['nis'])
        ax.plot(t[valid_idxs], np.array(res['nis'])[valid_idxs], 'm.', markersize=4)
        shade_background(ax)
        ax.set_yscale('log')
        ax.set_ylim(0.1, 1000)
        ax.axhline(20.0, color='r', linestyle='--')
        ax.set_ylabel("NIS (Log)")
        ax.set_title("Estimator Health")

    plt.tight_layout()
    plt.show()