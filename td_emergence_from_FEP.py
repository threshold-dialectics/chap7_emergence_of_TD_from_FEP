#td_emergence_from_FEP.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Double all default font sizes for better readability in exported figures
BASE_FONT_SIZE = plt.rcParams.get("font.size", 10)
plt.rcParams.update({"font.size": BASE_FONT_SIZE * 2, "savefig.dpi": 350})
import os
import time as pytime # To avoid conflict with simulation time variable
from collections import deque
from scipy.signal import savgol_filter 
from scipy.stats import pearsonr, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu, f_oneway, chi2_contingency
from itertools import combinations
import warnings

# Suppress specific RuntimeWarnings for cleaner output during batch runs
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.")
baseline_s1_metrics = {}



# --- Configuration ---
SIM_CONFIG_G7 = {
    "dt": 0.1,
    "total_time": 1000, # Increased for longer observation
    "logging_interval": 1,
    "n_monte_carlo": 30, # << INCREASED for robust results (start with 1-3 for tuning)
    "results_base_dir": "results_G7_fep_td_agent"
}

ENV_CONFIG_G7_BASE = { # Base settings for environment
    "signal_type": "sine_wave_shifting", 
    "initial_amplitude": 1.0,
    "initial_frequency": 0.1, 
    "base_noise_std": 0.02, # << Reduced for S1 to be more predictable initially
    "regime_shift_time": SIM_CONFIG_G7["total_time"] * 2, # Effectively no shift for S1,S2,S3 by default
    "shift_amplitude_factor": 1.0,
    "shift_frequency_factor": 1.0,
    "shift_noise_factor": 1.0
}

AGENT_TD_CONFIG_G7_BASE = {
    "g_initial": 2.0, # Start a bit higher to learn fast
    "beta_initial": 0.2, # Start with high exploration
    "fcrit_initial": 200.0, 
    "fcrit_floor": 10.0,
    "fcrit_replenish_rate": 0.9, # Decent replenishment for ample resource scenarios
    "w_coeffs": {"w1": 0.3, "w2": 0.3, "w3": 0.4}, 
    # "C_const": 0.05, # Placeholder - USER MUST CALIBRATE THIS FIRST FROM S1!
    "C_const": 0.021, # << STARTING GUESS - CALIBRATE THIS! (Adjusted from 0.15)
    
    "g_cost_kappa": 0.05,    # Kappa (κ) for g_lever energetic cost: P(g) = κ * g^φ1
    "g_cost_phi1": 0.75,     # Phi1 (φ1) for g_lever energetic cost
    "g_adaptation_rate": 0.1, # Slower, more deliberate adaptation
    "g_pe_thresh_improve": 0.05, 
    "g_pe_thresh_degrade": 0.2, 
    "g_min": 0.1, "g_max": 3.0,

    "beta_model_update_cost": 0.15, 
    "beta_adaptation_rate": 0.1, # Slower, more deliberate adaptation
    "beta_pe_thresh_exploit": 0.08, 
    "beta_pe_thresh_explore": 0.25,
    "beta_min": 0.05, "beta_max": 5.0,

    "beta_maintenance_cost_kappa": 0.001, # Kappa (κ) for beta_lever continuous maintenance cost
    "beta_maintenance_cost_phi": 1.0,     # Phi (φ) for beta_lever maintenance cost P(β) = κ * β^φ

    "fcrit_decay_rate_intrinsic": 0.0001, # Very small intrinsic decay

    "strain_avg_window_G7": 50, 
    "deriv_window_G7": 31,      # Adjusted for potentially smoother dynamics
    "deriv_polyorder_sg": 2,    
    "couple_window_G7": 100,
    "strain_fatigue_threshold_steps": 75 # For optional "giving up" logic
}

# --- Analysis Window Parameters ---
PRE_COLLAPSE_WINDOW_SIZE = 100  # number of time steps before collapse for detailed analysis
SEA_MAX_TIME_BEFORE_COLLAPSE = 100  # steps used in SEA plots
BASELINE_WINDOW_S2_S3_START_TIME = 1.0
BASELINE_WINDOW_S2_S3_END_TIME = 6.0
BASELINE_WINDOW_S4_PRE_SHIFT_START_TIME = 300.0
BASELINE_WINDOW_S4_PRE_SHIFT_END_TIME = 450.0

# Path for consolidated summary log
SUMMARY_LOG_PATH = os.path.join(SIM_CONFIG_G7["results_base_dir"], "summary.txt")


def log_summary(line):
    """Print line and append to summary file."""
    print(line)
    os.makedirs(os.path.dirname(SUMMARY_LOG_PATH), exist_ok=True)
    with open(SUMMARY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_simulation_parameters():
    """Write key configuration parameters to the summary."""
    log_summary("=== Simulation Configuration ===")
    log_summary(str(SIM_CONFIG_G7))
    log_summary("=== Environment Configuration ===")
    log_summary(str(ENV_CONFIG_G7_BASE))
    log_summary("=== Agent TD Configuration ===")
    log_summary(str(AGENT_TD_CONFIG_G7_BASE))

# --- TD Mechanisms within the Agent ---
class AgentTDMechanismsInternal:
    def __init__(self, config):
        self.g_lever = float(config["g_initial"])
        self.beta_lever = float(config["beta_initial"]) 
        self.fcrit_lever = float(config["fcrit_initial"])
        self.w1, self.w2, self.w3 = config["w_coeffs"]["w1"], config["w_coeffs"]["w2"], config["w_coeffs"]["w3"]
        self.C_const = float(config["C_const"])
        self.g_cost_kappa = config["g_cost_kappa"]
        self.g_cost_phi1 = config["g_cost_phi1"]
        self.g_min, self.g_max = config["g_min"], config["g_max"]
        self.beta_model_update_cost = config["beta_model_update_cost"]
        self.beta_min, self.beta_max = config["beta_min"], config["beta_max"]
        self.beta_maintenance_cost_kappa = float(config["beta_maintenance_cost_kappa"])
        self.beta_maintenance_cost_phi = float(config["beta_maintenance_cost_phi"])
        self.fcrit_replenish_rate = config["fcrit_replenish_rate"]
        self.fcrit_decay_rate_intrinsic = config["fcrit_decay_rate_intrinsic"]
        self.fcrit_floor = float(config["fcrit_floor"])

    @property
    def theta_t(self):
        g = max(1e-6, self.g_lever)
        b = max(1e-6, self.beta_lever)
        f = max(1e-6, self.fcrit_lever)
        return self.C_const * (g**self.w1) * (b**self.w2) * (f**self.w3)

    def apply_costs_and_replenishment(self, dt, current_g_lever_value, model_was_updated_this_step):
        cost_g = self.g_cost_kappa * (current_g_lever_value**self.g_cost_phi1) * dt
        cost_beta_model_update = self.beta_model_update_cost if model_was_updated_this_step else 0.0

        # Continuous energetic cost for maintaining beta_lever
        cost_beta_maintenance = self.beta_maintenance_cost_kappa * (max(1e-6, self.beta_lever)**self.beta_maintenance_cost_phi) * dt

        intrinsic_decay = self.fcrit_lever * self.fcrit_decay_rate_intrinsic * dt
        self.fcrit_lever -= (cost_g + cost_beta_model_update + cost_beta_maintenance + intrinsic_decay)
        self.fcrit_lever += self.fcrit_replenish_rate * dt
        self.fcrit_lever = max(0.0, self.fcrit_lever)
        return cost_g, cost_beta_model_update, cost_beta_maintenance

    def check_collapse(self, current_strain_avg_pe_sq):
        if current_strain_avg_pe_sq > (self.theta_t + 1e-9) and self.theta_t > 1e-9 :
            return "ThetaT_Breach_PE"
        if self.fcrit_lever < self.fcrit_floor:
            return "Fcrit_Depletion_Agent"
        return None

# --- Simple Environment ---
class SimpleEnvironmentG7:
    def __init__(self, config, sim_dt, seed=None): # Added sim_dt
        self.config = config.copy() 
        self.time = 0.0
        self.dt = sim_dt # Use passed dt
        self.amplitude = self.config["initial_amplitude"]
        self.frequency = self.config["initial_frequency"]
        self.noise_std = self.config["base_noise_std"]
        self.regime_shifted = False
        if seed is not None: np.random.seed(seed)

    def step(self):
        self.time += self.dt
        if not self.regime_shifted and self.time >= self.config["regime_shift_time"]:
            # print(f"  ENV REGIME SHIFT at t={self.time:.2f}") # Keep for debugging
            self.amplitude *= self.config["shift_amplitude_factor"]
            self.frequency *= self.config["shift_frequency_factor"]
            self.noise_std *= self.config["shift_noise_factor"]
            self.regime_shifted = True
        # ... (rest of signal generation logic remains same) ...
        if self.config["signal_type"] == "sine_wave_fixed" or self.config["signal_type"] == "sine_wave_shifting":
            signal_val = self.amplitude * np.sin(self.frequency * self.time)
        elif self.config["signal_type"] == "ar1_noisy":
            phi = 0.8
            self.last_signal = getattr(self, 'last_signal', np.random.normal(0,self.amplitude)) # Initialize if not present
            signal_val = phi * self.last_signal + np.random.normal(0, self.amplitude * (1-phi**2)**0.5) 
            self.last_signal = signal_val
        else: 
            signal_val = 0.0
        observation = signal_val + np.random.normal(0, self.noise_std)
        return observation


# --- FEP Agent ---
class FEPAgentG7:
    def __init__(self, agent_td_config, env_config, sim_dt, seed=None): # Added sim_dt
        self.dt = sim_dt
        self.config = agent_td_config.copy() # Use a copy
        self.td_mechanisms = AgentTDMechanismsInternal(self.config) # Pass the agent's config
        # TODO (Future Expansion): If agent takes actions, beta_lever should also determine
        # the precision (inverse temperature) of its action selection policy, as per FEP.
        self.env = SimpleEnvironmentG7(env_config, sim_dt, seed=seed+100 if seed is not None else None)

        self.learned_model_parameter = 0.0 
        self.model_param_learning_rate_base = 0.1 # Adjusted from 0.05 for potentially faster learning

        self.pe_history = deque(maxlen=self.config["strain_avg_window_G7"])
        self.pe_squared_avg_strain = 0.0
        self.steps = 0
        self.persistent_high_strain_counter_g = 0 # For "giving up" logic
        self.persistent_low_r_equivalent_counter_beta = 0 # For beta "giving up"

        if seed is not None: np.random.seed(seed)
    
    def predict_observation(self):
        return self.learned_model_parameter

    def update_internal_model(self, observation, prediction_error_raw):
        # g_lever modulates the perceived precision of prediction errors
        effective_pe_for_learning = self.td_mechanisms.g_lever * prediction_error_raw

        # Learning rate scaled only by beta_lever (policy precision)
        model_learning_rate_at_current_beta = (self.model_param_learning_rate_base /
                                               (self.td_mechanisms.beta_lever + 1e-3))

        self.learned_model_parameter += model_learning_rate_at_current_beta * effective_pe_for_learning * self.dt
        return True

    def adapt_td_levers(self, current_pe_abs):
        # --- Adapt g_lever ---
        delta_g = 0
        if current_pe_abs > self.config["g_pe_thresh_degrade"]:
            self.persistent_high_strain_counter_g += 1
            if self.td_mechanisms.fcrit_lever > self.td_mechanisms.fcrit_floor * 1.2:
                delta_g = self.config["g_adaptation_rate"] * self.dt
            if self.persistent_high_strain_counter_g > self.config["strain_fatigue_threshold_steps"]:
                delta_g = -self.config["g_adaptation_rate"] * self.dt * 1.5 
        elif current_pe_abs < self.config["g_pe_thresh_improve"]:
            self.persistent_high_strain_counter_g = 0 
            delta_g = -self.config["g_adaptation_rate"] * self.dt
        else: 
            self.persistent_high_strain_counter_g = 0

        self.td_mechanisms.g_lever = np.clip(self.td_mechanisms.g_lever + delta_g,
                                             self.config["g_min"], self.config["g_max"])

        # --- Adapt beta_lever ---
        delta_beta = 0
        if current_pe_abs > self.config["beta_pe_thresh_explore"]:
            self.persistent_low_r_equivalent_counter_beta +=1
            delta_beta = -self.config["beta_adaptation_rate"] * self.dt
        elif current_pe_abs < self.config["beta_pe_thresh_exploit"]:
            self.persistent_low_r_equivalent_counter_beta = 0
            if self.td_mechanisms.fcrit_lever > self.td_mechanisms.fcrit_floor * 1.1: 
                delta_beta = self.config["beta_adaptation_rate"] * self.dt
        else: 
            self.persistent_low_r_equivalent_counter_beta = 0
        
        if self.persistent_low_r_equivalent_counter_beta > self.config["strain_fatigue_threshold_steps"]:
             delta_beta -= self.config["beta_adaptation_rate"] * self.dt * 1.5 

        if self.td_mechanisms.fcrit_lever < self.td_mechanisms.fcrit_floor * 1.05:
            delta_beta -= self.config["beta_adaptation_rate"] * self.dt * 0.5 
        self.td_mechanisms.beta_lever = np.clip(self.td_mechanisms.beta_lever + delta_beta,
                                                self.config["beta_min"], self.config["beta_max"])

    def agent_step(self):
        current_observation = self.env.step()
        predicted_observation = self.predict_observation()
        raw_pe = current_observation - predicted_observation
        abs_pe = abs(raw_pe)
        self.pe_history.append(raw_pe**2)
        if self.pe_history: self.pe_squared_avg_strain = np.mean(list(self.pe_history))
        model_updated_this_step = False
        if abs_pe > self.config["beta_pe_thresh_explore"] * 0.7: 
            if self.td_mechanisms.fcrit_lever > self.td_mechanisms.fcrit_floor + self.td_mechanisms.beta_model_update_cost:
                 model_updated_this_step = self.update_internal_model(current_observation, raw_pe)
        self.adapt_td_levers(abs_pe)
        g_cost_val = self.td_mechanisms.g_lever
        cost_g_step, cost_beta_model_update_step, cost_beta_maintenance_step = self.td_mechanisms.apply_costs_and_replenishment(
            self.dt, g_cost_val, model_updated_this_step
        )
        collapse_status = self.td_mechanisms.check_collapse(self.pe_squared_avg_strain)
        log_entry = {
            "time": self.env.time, "observation": current_observation, "prediction": predicted_observation,
            "PE_raw": raw_pe, "PE_abs": abs_pe, "strain_avg_PEsq": self.pe_squared_avg_strain,
            "g_lever": self.td_mechanisms.g_lever, "beta_lever": self.td_mechanisms.beta_lever,
            "fcrit_lever": self.td_mechanisms.fcrit_lever, "theta_t": self.td_mechanisms.theta_t,
            "G_td": self.td_mechanisms.theta_t - self.pe_squared_avg_strain,
            "learned_model_param": self.learned_model_parameter, "cost_g_step": cost_g_step,
            "cost_beta_model_update_step": cost_beta_model_update_step,
            "cost_beta_maintenance_step": cost_beta_maintenance_step,
            "model_updated_flag": int(model_updated_this_step)
        }
        self.steps +=1
        return log_entry, collapse_status

# --- Simulation Loop Function --- 
def run_fep_simulation(sim_cfg_dict, td_config_agent_dict, env_config_sim_dict, scenario_name_for_log, seed=None):
    agent = FEPAgentG7(td_config_agent_dict, env_config_sim_dict, sim_cfg_dict["dt"], seed=seed)
    log_list = []
    collapse_type = None
    max_steps = int(sim_cfg_dict["total_time"] / sim_cfg_dict["dt"])

    for step_num in range(max_steps):
        log_entry, collapse_status = agent.agent_step() 
        if step_num % sim_cfg_dict["logging_interval"] == 0:
            log_list.append(log_entry)
        if collapse_status:
            collapse_type = collapse_status
            log_summary(f"  Collapse at t={agent.env.time:.2f} (step {step_num}) due to {collapse_type}")
            break
    
    if not collapse_type:
        collapse_type = "MaxTimeReached"
        log_summary(f"  Simulation {scenario_name_for_log} completed MaxTimeReached at t={agent.env.time:.2f}")
    
    if log_list: 
        log_list[-1]["collapse_type"] = collapse_type
    elif collapse_type != "MaxTimeReached": 
         log_list.append({
             "time": agent.env.time, "PE_raw": np.nan, "PE_abs": np.nan, 
             "strain_avg_PEsq": np.nan, "g_lever": agent.td_mechanisms.g_lever,
             "beta_lever": agent.td_mechanisms.beta_lever, "fcrit_lever": agent.td_mechanisms.fcrit_lever,
             "theta_t": agent.td_mechanisms.theta_t, "G_td": np.nan,
             "learned_model_param": agent.learned_model_parameter, "cost_g_step":np.nan,
             "cost_beta_model_update_step":np.nan, "cost_beta_maintenance_step": np.nan, "model_updated_flag":0,
             "collapse_type": collapse_type
        })
    return pd.DataFrame(log_list)


# --- Simplified Diagnostics Calculation for this script ---
def calculate_G7_diagnostics(df, dt, deriv_window, polyorder, couple_window):
    df = df.copy()
    for col in ['dot_beta', 'dot_fcrit', 'speed_td', 'couple_td']:
        if col not in df.columns: 
            df[col] = np.nan
            
    if len(df) < max(deriv_window, 2) : return df 

    if deriv_window % 2 == 0: deriv_window = max(3, deriv_window + 1) # Ensure deriv_window is odd
    if deriv_window <= polyorder: deriv_window = polyorder + (2 - polyorder % 2) 

    if len(df) >= deriv_window:
        try:
            df['dot_beta'] = savgol_filter(df['beta_lever'], deriv_window, polyorder, deriv=1, delta=dt, mode='interp')
            df['dot_fcrit'] = savgol_filter(df['fcrit_lever'], deriv_window, polyorder, deriv=1, delta=dt, mode='interp')
            df['speed_td'] = np.sqrt(df['dot_beta']**2 + df['dot_fcrit']**2)

            if len(df) >= couple_window:
                min_periods_corr = max(2, int(couple_window * 0.5)) 
                df['couple_td'] = df['dot_beta'].rolling(
                    window=couple_window, min_periods=min_periods_corr
                ).corr(df['dot_fcrit']).fillna(0) 
            else:
                df['couple_td'] = np.nan
        except ValueError as e:
            # print(f"SGolay/Corr filter error: {e}. Len(df): {len(df)}, deriv_window: {deriv_window}")
            df['dot_beta'] = np.nan; df['dot_fcrit'] = np.nan; df['speed_td'] = np.nan; df['couple_td'] = np.nan
            
    return df


def extract_analysis_windows(df_run, collapse_time, scenario_name):
    """Extract pre-collapse and baseline windows for analysis."""
    dt = SIM_CONFIG_G7["dt"]
    df_run = calculate_G7_diagnostics(
        df_run.copy(),
        dt,
        AGENT_TD_CONFIG_G7_BASE["deriv_window_G7"],
        AGENT_TD_CONFIG_G7_BASE["deriv_polyorder_sg"],
        AGENT_TD_CONFIG_G7_BASE["couple_window_G7"],
    )

    pre_start_time = collapse_time - PRE_COLLAPSE_WINDOW_SIZE * dt
    df_pre = df_run[(df_run["time"] < collapse_time) & (df_run["time"] >= pre_start_time)]

    df_baseline = pd.DataFrame(columns=df_run.columns)

    if scenario_name in ["S2_UnpredEnv_AmpleRes", "S3_PredEnv_ScarceRes"]:
        if df_run["time"].max() >= BASELINE_WINDOW_S2_S3_END_TIME and collapse_time > BASELINE_WINDOW_S2_S3_END_TIME:
            df_baseline = df_run[(df_run["time"] >= BASELINE_WINDOW_S2_S3_START_TIME) &
                                 (df_run["time"] <= BASELINE_WINDOW_S2_S3_END_TIME)]
    elif scenario_name == "S4_EnvShift_MidPred":
        if df_run["time"].max() >= BASELINE_WINDOW_S4_PRE_SHIFT_END_TIME and collapse_time > BASELINE_WINDOW_S4_PRE_SHIFT_END_TIME:
            df_baseline = df_run[(df_run["time"] >= BASELINE_WINDOW_S4_PRE_SHIFT_START_TIME) &
                                 (df_run["time"] <= BASELINE_WINDOW_S4_PRE_SHIFT_END_TIME)]

    return df_pre, df_baseline

# --- Plotting ---
def plot_G7_results(df, scenario_name, results_dir, agent_td_config, run_id=""):
    if df.empty or len(df) < 2:
        log_summary(f"Plotting skipped for G7 {scenario_name} run {run_id}: No data or insufficient data.")
        return

    sim_dt = SIM_CONFIG_G7['dt'] # Use global dt for plotting consistency as well

    df_diag = calculate_G7_diagnostics(df.copy(), sim_dt, # Pass a copy to avoid modifying original in list
                                 agent_td_config['deriv_window_G7'],
                                 agent_td_config['deriv_polyorder_sg'],
                                 agent_td_config['couple_window_G7'])


    fig, axs = plt.subplots(7, 1, figsize=(14, 22), sharex=True) 
    title_suffix = f" - Run {run_id}" if run_id !="" else "" # Corrected run_id check
    fig.suptitle(f"FEP-TD Agent Dynamics - {scenario_name}{title_suffix}", fontsize=32)

    axs[0].plot(df_diag["time"], df_diag["observation"], label="Observation (Env Signal)", alpha=0.6, color='gray')
    axs[0].plot(df_diag["time"], df_diag["prediction"], label="Agent Prediction", linestyle='-', color='blue')
    axs[0].plot(df_diag["time"], df_diag["learned_model_param"], label="Agent Model Param (Learned Mean)", linestyle=':', color='green')
    axs[0].set_ylabel("Signal Value"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(df_diag["time"], df_diag["PE_abs"], label="|Prediction Error|", color='red', alpha=0.7)
    axs[1].plot(df_diag["time"], df_diag["strain_avg_PEsq"], label="Strain (Avg PE^2)", linestyle='--', color='darkred')
    axs[1].set_ylabel("Error / Strain", color='darkred'); axs[1].tick_params(axis='y', labelcolor='darkred'); axs[1].legend(loc='upper left')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(df_diag["time"], df_diag["theta_t"], label="Theta_T (Tolerance)", color='purple', linestyle=':')
    ax1_twin.plot(df_diag["time"], df_diag["G_td"], label="Safety Margin (G_td)", color='cyan', linestyle='-.')
    ax1_twin.axhline(0, color='gray', linestyle='--', linewidth=0.8, label="G_td = 0")
    ax1_twin.set_ylabel("Theta_T / G_td", color='purple'); ax1_twin.tick_params(axis='y', labelcolor='purple'); ax1_twin.legend(loc='upper right')
    axs[1].grid(True)

    axs[2].plot(df_diag["time"], df_diag["g_lever"], label="g_lever (Perception/Attention)")
    axs[2].plot(df_diag["time"], df_diag["beta_lever"], label="beta_lever (1/ExplorationRate)")
    axs[2].set_ylabel("g_lever / beta_lever"); axs[2].legend(); axs[2].grid(True)

    axs[3].plot(df_diag["time"], df_diag["fcrit_lever"], label="fcrit_lever (Energetic Slack)")
    axs[3].axhline(agent_td_config["fcrit_floor"], color='r', linestyle='--', label="Fcrit Floor")
    axs[3].set_ylabel("fcrit_lever"); axs[3].legend(); axs[3].grid(True)
    
    axs[4].plot(df_diag["time"], df_diag["cost_g_step"], label="Cost of g_lever (per step)", alpha=0.7)
    axs[4].plot(df_diag["time"], df_diag["cost_beta_model_update_step"], label="Cost of Beta (Model Update)", alpha=0.7, linestyle='--')
    if "cost_beta_maintenance_step" in df_diag.columns:
        axs[4].plot(df_diag["time"], df_diag["cost_beta_maintenance_step"], label="Cost of Beta (Maintenance)", alpha=0.7, linestyle=':')
    axs[4].set_ylabel("Step Costs"); axs[4].legend(); axs[4].grid(True)

    axs[5].plot(df_diag["time"], df_diag["speed_td"], label="Speed Index (beta, fcrit)")
    axs[5].set_ylabel("Speed Index"); axs[5].legend(); axs[5].grid(True)

    axs[6].plot(df_diag["time"], df_diag["couple_td"], label="Couple Index (beta, fcrit)")
    axs[6].set_ylabel("Couple Index"); axs[6].set_xlabel("Time"); axs[6].grid(True); axs[6].set_ylim(-1.1, 1.1)
    axs[6].legend()
    
    # Corrected collapse line plotting
    if not df_diag.empty and "collapse_type" in df_diag.columns:
        # Find the first actual collapse event for marking the line
        collapse_events_df = df_diag[df_diag['collapse_type'].notna() & 
                                   ~df_diag['collapse_type'].isin(["MaxTimeReached", "None", np.nan])]
        if not collapse_events_df.empty:
            collapse_time_val = collapse_events_df['time'].iloc[0]
            collapse_reason = collapse_events_df['collapse_type'].iloc[0]
            # Add axvline to all subplots
            for ax_p_idx, ax_p in enumerate(axs):
                # Add label only to the first subplot's axvline for a cleaner composite legend
                vline_label = f"Collapse: {collapse_reason}" if ax_p_idx == 0 else None
                ax_p.axvline(collapse_time_val, color='k', linestyle=':', linewidth=2, label=vline_label)
            
            # Rebuild legend for the first subplot to include the collapse line
            handles, labels = axs[0].get_legend_handles_labels()
            if len(handles) > 0:
                # Filter out duplicate "Collapse:..." labels
                unique_labels_map = {}
                for handle, label_item in zip(handles, labels):
                    if label_item not in unique_labels_map:
                        unique_labels_map[label_item] = handle
                axs[0].legend(unique_labels_map.values(), unique_labels_map.keys(), loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    file_name_suffix = f"_run{run_id}" if str(run_id) != "" else "_mc_avg"
    fig_path = os.path.join(results_dir, f"G7_fep_agent_{scenario_name.lower().replace(' ','_')}{file_name_suffix}.png")
    plt.savefig(fig_path, dpi=350)
    log_summary(f"[FIGURE] {fig_path}")
    plt.close(fig)


def plot_G7_mc_summary_results(all_dfs_list_scenario, scenario_name, results_dir_scenario, agent_td_config_scenario):
    if not all_dfs_list_scenario:
        log_summary(f"No dataframes to average for MC summary: {scenario_name}")
        return

    valid_dfs_for_len = [df for df in all_dfs_list_scenario if len(df) > agent_td_config_scenario['deriv_window_G7']]
    if not valid_dfs_for_len:
        log_summary(f"No sufficiently long runs for MC summary: {scenario_name}")
        return
    min_len = min(len(df) for df in valid_dfs_for_len)
    
    # Use time from a valid df
    time_vector = next((df['time'].iloc[:min_len].values for df in valid_dfs_for_len if len(df) >= min_len), None)
    if time_vector is None:
        log_summary(f"Could not establish time vector for MC summary: {scenario_name}")
        return


    cols_to_avg = ["observation", "prediction", "learned_model_param", "PE_abs", "strain_avg_PEsq",
                   "g_lever", "beta_lever", "fcrit_lever", "theta_t", "G_td",
                   "cost_g_step", "cost_beta_model_update_step", "cost_beta_maintenance_step",
                   "speed_td", "couple_td"]

    for col in cols_to_avg:
        data_stacked_list = []
        for df_orig in all_dfs_list_scenario:
            if len(df_orig) >= min_len:
                # Process a copy to avoid modifying original df in list
                df_processed = calculate_G7_diagnostics(df_orig.iloc[:min_len].copy(), 
                                                        SIM_CONFIG_G7['dt'],
                                                        agent_td_config_scenario['deriv_window_G7'],
                                                        agent_td_config_scenario['deriv_polyorder_sg'],
                                                        agent_td_config_scenario['couple_window_G7'])
                if col in df_processed.columns:
                     data_stacked_list.append(df_processed[col].values)
                else: 
                    data_stacked_list.append(np.full(min_len, np.nan))
            else: 
                data_stacked_list.append(np.full(min_len, np.nan))

        if not data_stacked_list: continue
        data_stacked_np = np.array(data_stacked_list)
        
        # Check if all elements are NaN after stacking for a column
        if np.all(np.isnan(data_stacked_np)):
            log_summary(f"All data for column {col} is NaN in {scenario_name}. Skipping MC plot.")
            continue

        mean_trajectory = np.nanmean(data_stacked_np, axis=0)
        std_trajectory = np.nanstd(data_stacked_np, axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(time_vector, mean_trajectory, label=f"{col} (Mean)")
        plt.fill_between(time_vector, 
                         mean_trajectory - std_trajectory, 
                         mean_trajectory + std_trajectory, 
                         alpha=0.2, label=f"{col} (+/- 1 STD)")
        plt.xlabel("Time"); plt.ylabel(col)
        plt.title(f"MC Summary: {col} - {scenario_name} (N={SIM_CONFIG_G7['n_monte_carlo']})") # Use global MC
        plt.legend(); plt.grid(True); plt.tight_layout()
        fig_path = os.path.join(results_dir_scenario, f"G7_mc_avg_{col}.png")
        plt.savefig(fig_path, dpi=350)
        log_summary(f"[FIGURE] {fig_path}")
        plt.close()


def perform_and_plot_sea(all_collapsing_runs_data, diagnostics_for_sea, scenario_name, results_dir):
    """Perform Superposed Epoch Analysis on collapsing runs and save plots."""
    if not all_collapsing_runs_data:
        return

    min_len = min(len(df) for df in all_collapsing_runs_data)
    time_vec = None
    stacked = {d: [] for d in diagnostics_for_sea}

    for df in all_collapsing_runs_data:
        df_cur = df.tail(min_len)
        if time_vec is None:
            time_vec = df_cur["time_to_collapse"].values
        for d in diagnostics_for_sea:
            stacked[d].append(df_cur.get(d, pd.Series([np.nan]*min_len)).values)

    for diag in diagnostics_for_sea:
        data = np.vstack(stacked[diag])
        mean_traj = np.nanmean(data, axis=0)

        # >>> NEW: skip diagnostics that are entirely NaN
        if np.all(np.isnan(mean_traj)):
            log_summary(f"SEA {scenario_name} {diag}: skipped (all values are NaN)")
            continue
        std_traj = np.nanstd(data, axis=0)

        # Summary stats for text log
        slope, r2 = _linear_fit_stats(time_vec[-50:], mean_traj[-50:])
        idx_max = np.nanargmax(mean_traj)
        idx_min = np.nanargmin(mean_traj)
        t_peak = time_vec[idx_max]
        t_trough = time_vec[idx_min]
        log_summary(
            f"SEA {scenario_name} {diag}: slope_last50={slope:.4f}, R2={r2:.2f}, "
            f"peak at t={t_peak:.1f}, trough at t={t_trough:.1f}"
        )

        plt.figure(figsize=(8, 4))
        plt.plot(time_vec, mean_traj, label=f"{diag} (Mean)")
        plt.fill_between(time_vec, mean_traj - std_traj, mean_traj + std_traj, alpha=0.3, label="+/-1 STD")
        plt.axvline(0, color="k", linestyle=":")
        plt.xlabel("Time to Collapse")
        plt.ylabel(diag)
        plt.title(f"SEA {diag} - {scenario_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(results_dir, f"sea_{diag}_{scenario_name}.png")
        plt.savefig(fig_path, dpi=350)
        log_summary(f"[FIGURE] {fig_path}")
        plt.close()


def perform_G7_scenario_analysis(scenario_name, scenario_dfs_list, agent_td_config, results_dir, sim_dt):
    """Perform Monte Carlo statistical analysis for a scenario."""
    global baseline_s1_metrics
    if not scenario_dfs_list:
        return

    metrics = ["PE_abs", "g_lever", "beta_lever", "fcrit_lever", "speed_td", "couple_td"]

    # Pre-compute diagnostics for each run so derived columns exist for all analyses
    diagnostic_dfs = [
        calculate_G7_diagnostics(
            df.copy(),
            sim_dt,
            agent_td_config["deriv_window_G7"],
            agent_td_config["deriv_polyorder_sg"],
            agent_td_config["couple_window_G7"],
        )
        for df in scenario_dfs_list
    ]
    early_vals = {m: [] for m in metrics}   # early/ pre-shift
    late_vals = {m: [] for m in metrics}    # late/ post-shift/ characteristic
    pre_vals = {m: [] for m in metrics}
    base_vals = {m: [] for m in metrics}
    pre_quadrants = []
    baseline_quadrants = []
    pre_windows_all = []
    baseline_windows_all = []
    collapse_times = []
    collapse_types = []

    for df_diag in diagnostic_dfs:

        run_end = df_diag["time"].iloc[-1]
        collapse_type = df_diag.get("collapse_type", pd.Series(["MaxTimeReached"] * len(df_diag))).iloc[-1]
        collapse_types.append(collapse_type)
        if collapse_type != "MaxTimeReached":
            collapse_times.append(run_end)
            collapse_time_val = df_diag[df_diag["collapse_type"].notna() &
                                       ~df_diag["collapse_type"].isin(["MaxTimeReached", "None", np.nan])]["time"].iloc[0]
            df_pre, df_base = extract_analysis_windows(df_diag, collapse_time_val, scenario_name)
            for m in metrics:
                pre_vals[m].append(df_pre[m].mean())
                base_vals[m].append(df_base[m].mean())
            pre_quadrants.append(calculate_lever_velocity_quadrants(df_pre))
            baseline_quadrants.append(calculate_lever_velocity_quadrants(df_base))
            pre_windows_all.append(df_pre)
            baseline_windows_all.append(df_base)
        else:
            collapse_times.append(np.nan)

        if scenario_name == "S1_PredEnv_AmpleRes":
            early_window = df_diag[(df_diag["time"] >= sim_dt) & (df_diag["time"] <= 50.0)]
            late_start = SIM_CONFIG_G7["total_time"] * 0.8
            late_end = SIM_CONFIG_G7["total_time"] * 0.95
            late_window = df_diag[(df_diag["time"] >= late_start) & (df_diag["time"] <= late_end)]
        elif scenario_name in ["S2_UnpredEnv_AmpleRes", "S3_PredEnv_ScarceRes"]:
            if collapse_type != "MaxTimeReached":
                if len(df_diag) <= 100:
                    start_idx = len(df_diag) // 2
                else:
                    start_idx = len(df_diag) - 100
                late_window = df_diag.iloc[start_idx:]
            else:
                late_window = df_diag[(df_diag["time"] >= SIM_CONFIG_G7["total_time"] * 0.8) &
                                      (df_diag["time"] <= SIM_CONFIG_G7["total_time"] * 0.95)]
            early_window = pd.DataFrame(columns=df_diag.columns)
        elif scenario_name == "S4_EnvShift_MidPred":
            shift_time = SIM_CONFIG_G7["total_time"] / 2.0
            pre_window = df_diag[(df_diag["time"] >= shift_time - 100 * sim_dt) &
                                 (df_diag["time"] < shift_time)]
            post_window = df_diag[(df_diag["time"] >= shift_time) &
                                  (df_diag["time"] <= min(run_end, shift_time + 100 * sim_dt))]
            early_window, late_window = pre_window, post_window
        else:
            early_window = pd.DataFrame(columns=df_diag.columns)
            late_window = pd.DataFrame(columns=df_diag.columns)

        for m in metrics:
            early_vals[m].append(early_window[m].mean())
            late_vals[m].append(late_window[m].mean())

    lines = [f"--- Scenario {scenario_name} ---"]

    if scenario_name == "S1_PredEnv_AmpleRes":
        for m in metrics:
            t_res = ttest_rel(early_vals[m], late_vals[m], nan_policy="omit")
            lines.append(
                f"S1 (Late vs Early) {m}: {np.nanmean(early_vals[m]):.3f} -> {np.nanmean(late_vals[m]):.3f} (p={t_res.pvalue:.3e})"
            )
        baseline_s1_metrics = {f"{m}_S1_stable": late_vals[m] for m in metrics}

    elif scenario_name == "S2_UnpredEnv_AmpleRes" and baseline_s1_metrics:
        for m in metrics:
            base = baseline_s1_metrics.get(f"{m}_S1_stable", [])
            if len(base) >= 2 and len(late_vals[m]) >= 1:
                t_res = ttest_ind(late_vals[m], base, nan_policy="omit")
                lines.append(
                    f"S2 vs S1 (stable) {m}: {np.nanmean(late_vals[m]):.3f} vs {np.nanmean(base):.3f} (p={t_res.pvalue:.3e})"
                )
            else:
                lines.append(f"S2 vs S1 (stable) {m}: insufficient data")

    elif scenario_name == "S3_PredEnv_ScarceRes" and baseline_s1_metrics:
        mean_collapse = np.nanmean(collapse_times)
        counts = pd.Series(collapse_types).value_counts().to_dict()
        lines.append(f"S3 collapse mean time {mean_collapse:.2f}")
        lines.append("    Collapse types: " + ", ".join([f"{k}:{v}" for k, v in counts.items()]))
        for m in metrics:
            base = baseline_s1_metrics.get(f"{m}_S1_stable", [])
            if len(base) >= 2 and len(late_vals[m]) >= 1:
                t_res = ttest_ind(late_vals[m], base, nan_policy="omit")
                lines.append(
                    f"S3 vs S1 (stable) {m}: {np.nanmean(late_vals[m]):.3f} vs {np.nanmean(base):.3f} (p={t_res.pvalue:.3e})"
                )
            else:
                lines.append(f"S3 vs S1 (stable) {m}: insufficient data")

    elif scenario_name == "S4_EnvShift_MidPred":
        for m in metrics:
            t_res = ttest_rel(early_vals[m], late_vals[m], nan_policy="omit")
            lines.append(
                f"S4 (Post vs Pre) {m}: {np.nanmean(early_vals[m]):.3f} -> {np.nanmean(late_vals[m]):.3f} (p={t_res.pvalue:.3e})"
            )

    # Pre-collapse vs baseline comparisons
    for m in metrics:
        pre = [v for v in pre_vals[m] if not np.isnan(v)]
        base = [v for v in base_vals[m] if not np.isnan(v)]
        if pre and base:
            stat = mannwhitneyu(pre, base, alternative="two-sided")
            lines.append(
                f"{scenario_name} PreCollapse vs Baseline {m}: {np.nanmean(pre):.3f} vs {np.nanmean(base):.3f} (p={stat.pvalue:.3e})"
            )
        else:
            lines.append(f"{scenario_name} PreCollapse vs Baseline {m}: insufficient data")

    def _aggregate_counts(list_dicts):
        agg = {"Q1":0, "Q2":0, "Q3":0, "Q4":0}
        for d in list_dicts:
            for k in agg:
                agg[k] += d.get(k,0)
        return agg

    pre_count_tot = _aggregate_counts(pre_quadrants)
    base_count_tot = _aggregate_counts(baseline_quadrants)
    pre_total = sum(pre_count_tot.values())
    base_total = sum(base_count_tot.values())
    pre_props = {k: (pre_count_tot[k]/pre_total) if pre_total>0 else 0 for k in pre_count_tot}
    base_props = {k: (base_count_tot[k]/base_total) if base_total>0 else 0 for k in base_count_tot}

    obs = np.asarray([list(pre_count_tot.values()),
                    list(base_count_tot.values())], dtype=int)

    # Keep only columns where at least one observation exists
    non_zero_cols = obs.sum(axis=0) > 0
    obs = obs[:, non_zero_cols]

    if obs.shape[1] >= 2:                               # need ≥2 columns for χ²
        chi2, p, dof, exp = chi2_contingency(obs)
        lines.append(f"{scenario_name} Quadrant distribution χ² p={p:.3e}")
    else:
        lines.append(f"{scenario_name} Quadrant distribution: "
                    "not enough non-zero quadrants for χ² test")

    summary_df = pd.DataFrame({
        "Metric": metrics,
        "PreCollapse": [np.nanmean(pre_vals[m]) for m in metrics],
        "Baseline": [np.nanmean(base_vals[m]) for m in metrics]
    })
    lines.append("Summary Table (Stable Metrics):")
    lines.append(summary_df.to_string(index=False))

    # Time-series derived summaries
    ts_summary = summarize_time_series_across_runs(diagnostic_dfs, metrics)
    for m, stats in ts_summary.items():
        if stats:
            lines.append(
                f"{scenario_name}: {m} slope={stats['slope']:.4f} (R2={stats['r2']:.2f}), "
                f"max={stats['max']:.2f} at t={stats['t_max']:.1f}, "
                f"min={stats['min']:.2f} at t={stats['t_min']:.1f}"
            )

    # Run-to-run divergence summaries
    for m in metrics:
        std_t20, spread = compute_run_to_run_spread(diagnostic_dfs, m, ref_time=20.0)
        lines.append(
            f"{scenario_name}: {m} STD@t20={std_t20:.2f}; max spread={spread:.2f}"
        )

    # Correlation summaries
    pre_concat = pd.concat(pre_windows_all, ignore_index=True) if pre_windows_all else pd.DataFrame()
    base_concat = pd.concat(baseline_windows_all, ignore_index=True) if baseline_windows_all else pd.DataFrame()
    corr_metrics = ["g_lever", "beta_lever", "fcrit_lever", "PE_abs"]
    if not base_concat.empty:
        corr_base = compute_correlation_matrix(base_concat, corr_metrics)
        lines.append("Baseline correlations:\n" + corr_base.to_string())
    if not pre_concat.empty:
        corr_pre = compute_correlation_matrix(pre_concat, corr_metrics)
        lines.append("Pre-collapse correlations:\n" + corr_pre.to_string())
        

    summary_path = os.path.join(results_dir, "text_summary_report.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
            log_summary(l)
    
    return pre_props, base_props


def calculate_lever_velocity_quadrants(df_window):
    """Return counts of lever velocity quadrant occupancy."""
    if df_window.empty:
        return {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    q1 = ((df_window["dot_beta"] >= 0) & (df_window["dot_fcrit"] >= 0)).sum()
    q2 = ((df_window["dot_beta"] < 0) & (df_window["dot_fcrit"] >= 0)).sum()
    q3 = ((df_window["dot_beta"] < 0) & (df_window["dot_fcrit"] < 0)).sum()
    q4 = ((df_window["dot_beta"] >= 0) & (df_window["dot_fcrit"] < 0)).sum()
    return {"Q1": int(q1), "Q2": int(q2), "Q3": int(q3), "Q4": int(q4)}


def _linear_fit_stats(time_arr, value_arr):
    """Return slope and R^2 of a linear fit."""
    if len(time_arr) < 2 or np.all(np.isnan(value_arr)):
        return np.nan, np.nan
    mask = ~np.isnan(time_arr) & ~np.isnan(value_arr)
    if mask.sum() < 2:
        return np.nan, np.nan
    t = time_arr[mask]
    v = value_arr[mask]
    slope, intercept = np.polyfit(t, v, 1)
    pred = slope * t + intercept
    ss_res = np.sum((v - pred) ** 2)
    ss_tot = np.sum((v - np.mean(v)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def compute_time_series_stats(df, metric):
    """Compute slope and extrema information for a metric."""
    if metric not in df.columns or df.empty:
        return None

    time_vals   = df["time"].values
    # treat ±inf like NaN
    metric_vals = np.where(np.isfinite(df[metric].values), df[metric].values, np.nan)

    # >>> NEW EARLY-EXIT GUARD
    if metric_vals.size == 0 or np.all(np.isnan(metric_vals)):
        # nothing usable – skip this metric
        return None
    # <<<

    # safe linear trend (already handles NaNs)
    slope, r2 = _linear_fit_stats(time_vals, metric_vals)

    # extremal indices – now guaranteed to have a finite value
    idx_max = np.nanargmax(metric_vals)
    idx_min = np.nanargmin(metric_vals)

    return {
        "slope": slope,
        "r2":    r2,
        "max":   metric_vals[idx_max],
        "t_max": time_vals[idx_max],
        "min":   metric_vals[idx_min],
        "t_min": time_vals[idx_min],
    }



def summarize_time_series_across_runs(df_list, metrics):
    """Aggregate slope/extrema stats across multiple runs."""
    summary = {m: [] for m in metrics}
    for df in df_list:
        for m in metrics:
            stats = compute_time_series_stats(df, m)
            if stats:
                summary[m].append(stats)
    aggregated = {}
    for m, stat_list in summary.items():
        if not stat_list:
            aggregated[m] = None
            continue
        slope_mean = np.nanmean([s["slope"] for s in stat_list])
        r2_mean = np.nanmean([s["r2"] for s in stat_list])
        max_mean = np.nanmean([s["max"] for s in stat_list])
        t_max_mean = np.nanmean([s["t_max"] for s in stat_list])
        min_mean = np.nanmean([s["min"] for s in stat_list])
        t_min_mean = np.nanmean([s["t_min"] for s in stat_list])
        aggregated[m] = {
            "slope": slope_mean,
            "r2": r2_mean,
            "max": max_mean,
            "t_max": t_max_mean,
            "min": min_mean,
            "t_min": t_min_mean,
        }
    return aggregated


def compute_run_to_run_spread(df_list, metric, ref_time=20.0):
    """Return STD across runs at ref_time and max range across time."""
    if not df_list:
        return np.nan, np.nan
    min_len = min(len(df) for df in df_list)
    base_time = df_list[0]["time"].iloc[:min_len].values
    stacked = []
    for df in df_list:
        vals = np.interp(base_time, df["time"].values, df[metric].values)
        stacked.append(vals)
    data = np.vstack(stacked)
    idx = np.argmin(np.abs(base_time - ref_time))
    std_at_ref = np.nanstd(data[:, idx])
    spread = np.nanmax(np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    return std_at_ref, spread


def compute_correlation_matrix(df_concat, metrics):
    """Return correlation matrix dataframe for given metrics."""
    if df_concat.empty:
        return pd.DataFrame()
    corr_df = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for m1 in metrics:
        for m2 in metrics:
            if m1 in df_concat.columns and m2 in df_concat.columns:
                try:
                    corr, _ = pearsonr(df_concat[m1], df_concat[m2])
                except Exception:
                    corr = np.nan
            else:
                corr = np.nan
            corr_df.loc[m1, m2] = corr
    return corr_df


def plot_quadrant_proportions(pre_props, baseline_props, scenario_name, results_dir):
    quads = ["Q1", "Q2", "Q3", "Q4"]
    pre_vals = [pre_props.get(q, 0) for q in quads]
    base_vals = [baseline_props.get(q, 0) for q in quads]
    x = np.arange(len(quads))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, pre_vals, width, label="Pre-collapse")
    plt.bar(x + width / 2, base_vals, width, label="Baseline")
    plt.xticks(x, quads)
    plt.ylabel("Proportion")
    plt.title(f"Lever Velocity Quadrants - {scenario_name}")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"quadrant_props_{scenario_name}.png")
    plt.savefig(fig_path, dpi=350)
    log_summary(f"[FIGURE] {fig_path}")
    plt.close()


def analyze_late_stable_phase(df_list, metrics_to_analyze, start_time=700, end_time=900, return_all=False):
    """Calculate mean metric values in the specified late-stable window.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of dataframes from Monte Carlo runs.
    metrics_to_analyze : list of str
        Column names to evaluate.
    start_time : float, optional
        Window start time, by default 700.
    end_time : float, optional
        Window end time, by default 900.
    return_all : bool, optional
        If True return list of per-run means instead of the overall mean.

    Returns
    -------
    dict
        Mapping metric -> mean over MC runs (or lists if ``return_all``).
    """
    metrics_runs = {m: [] for m in metrics_to_analyze}
    for df in df_list:
        window_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        for m in metrics_to_analyze:
            metrics_runs[m].append(window_df[m].mean())

    if return_all:
        return metrics_runs
    return {m: np.nanmean(metrics_runs[m]) if metrics_runs[m] else np.nan for m in metrics_to_analyze}

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(SIM_CONFIG_G7["results_base_dir"], exist_ok=True)
    if os.path.exists(SUMMARY_LOG_PATH):
        os.remove(SUMMARY_LOG_PATH)
    # Start summary file with configuration
    write_simulation_parameters()
    
    # --- CALIBRATION STEP ---
    # print("--- USER ACTION REQUIRED: CALIBRATE AGENT_TD_CONFIG_G7_BASE['C_const'] ---")
    # print("1. Set SIM_CONFIG_G7['n_monte_carlo'] = 1.")
    # print("2. Run ONLY the 'S1_PredEnv_AmpleRes' scenario block below.")
    # print("3. Observe 'strain_avg_PEsq' and typical g, beta, fcrit in stable phase from the plot.")
    # print("4. Adjust 'C_const' in AGENT_TD_CONFIG_G7_BASE so Theta_T is slightly above strain (G_td ~ 0.1-0.3).")
    # print("   C_const ~ (Target_Strain + Desired_G_td) / (g^w1 * beta^w2 * fcrit^w3)")
    # print("5. Once calibrated, set n_monte_carlo to desired value (e.g., 10) and run all scenarios.")
    # print("--- Current C_const is a guess. Calibration is CRUCIAL. ---")
    # For this run, assume C_const = 0.035 is pre-calibrated
    AGENT_TD_CONFIG_G7_BASE["C_const"] = 0.021


    scenarios_G7_definitions = {
        "S1_PredEnv_AmpleRes": {"env_config": ENV_CONFIG_G7_BASE.copy(), 
                                "agent_config_mods": {}},
        "S2_UnpredEnv_AmpleRes": {"env_config": {**ENV_CONFIG_G7_BASE.copy(), 
                                                "base_noise_std": 0.30, # Higher noise
                                                "regime_shift_time": float('inf')}, # No shift
                                  "agent_config_mods": {}},
        "S3_PredEnv_ScarceRes": {"env_config": {**ENV_CONFIG_G7_BASE.copy(),
                                                "regime_shift_time": float('inf')}, # No shift
                                 "agent_config_mods": {
                                     "fcrit_initial_factor": 0.15, # Significantly less fcrit
                                     "fcrit_replenish_rate_factor": 0.05} # Very little replenishment
                                 },
        "S4_EnvShift_MidPred": {"env_config": {**ENV_CONFIG_G7_BASE.copy(), 
                                               "base_noise_std": 0.03, # Start more predictable
                                               "regime_shift_time": SIM_CONFIG_G7["total_time"] / 2.0, 
                                               "shift_amplitude_factor": 1.0, 
                                               "shift_frequency_factor": 1.0, 
                                               "shift_noise_factor": 15.0}, # Becomes much noisier
                                "agent_config_mods": {
                                    "fcrit_initial_factor": 1.5 # Give more fcrit to handle shift
                                }}
    }

    master_results_log_all_scenarios = []

    for sc_name, sc_configs_data in scenarios_G7_definitions.items(): # renamed sc_configs to sc_configs_data
        log_summary(f"\n--- Running G7 Scenario: {sc_name} ---")
        
        current_agent_td_config = AGENT_TD_CONFIG_G7_BASE.copy() 
        current_env_config = sc_configs_data["env_config"] # Use sc_configs_data

        if "fcrit_initial_factor" in sc_configs_data["agent_config_mods"]:
            current_agent_td_config["fcrit_initial"] *= sc_configs_data["agent_config_mods"]["fcrit_initial_factor"]
        if "fcrit_replenish_rate_factor" in sc_configs_data["agent_config_mods"]:
            current_agent_td_config["fcrit_replenish_rate"] *= sc_configs_data["agent_config_mods"]["fcrit_replenish_rate_factor"]
        
        scenario_runs_data_list_current_sc = [] # Corrected variable name
        
        scenario_output_dir = os.path.join(SIM_CONFIG_G7["results_base_dir"], sc_name)
        os.makedirs(scenario_output_dir, exist_ok=True)

        for i in range(SIM_CONFIG_G7["n_monte_carlo"]):
            run_seed = i + sum(ord(c) for c in sc_name) 
            log_summary(f"  MC Run {i+1}/{SIM_CONFIG_G7['n_monte_carlo']} for {sc_name}")
            
            df_run = run_fep_simulation(SIM_CONFIG_G7, current_agent_td_config, 
                                        current_env_config, sc_name, seed=run_seed)
            
            if not df_run.empty:
                df_run["scenario"] = sc_name
                df_run["run_id"] = i
                master_results_log_all_scenarios.append(df_run.copy()) 
                scenario_runs_data_list_current_sc.append(df_run) # Add to current scenario list

                if i == 0: 
                    plot_G7_results(df_run, sc_name, scenario_output_dir, current_agent_td_config, run_id=str(i))
            else:
                log_summary(f"    Run {i+1} for {sc_name} produced empty dataframe.")
        
        if scenario_runs_data_list_current_sc:
            plot_G7_mc_summary_results(scenario_runs_data_list_current_sc, sc_name, scenario_output_dir, current_agent_td_config)

        pre_props = {"Q1":0,"Q2":0,"Q3":0,"Q4":0}; base_props = {"Q1":0,"Q2":0,"Q3":0,"Q4":0}

        if sc_name in ["S2_UnpredEnv_AmpleRes", "S3_PredEnv_ScarceRes", "S4_EnvShift_MidPred"]:
            collapsing_runs_dfs = [df for df in scenario_runs_data_list_current_sc if not df.empty and df["collapse_type"].iloc[-1] != "MaxTimeReached"]
            sea_prepped_data = []
            for df_run_orig in collapsing_runs_dfs:
                df_run = calculate_G7_diagnostics(df_run_orig.copy(), SIM_CONFIG_G7["dt"], current_agent_td_config["deriv_window_G7"], current_agent_td_config["deriv_polyorder_sg"], current_agent_td_config["couple_window_G7"])
                collapse_time_val = df_run[df_run['collapse_type'].notna() & ~df_run['collapse_type'].isin(["MaxTimeReached","None", np.nan])]['time'].iloc[0]
                sea_window_df = df_run[(df_run['time'] > collapse_time_val - SEA_MAX_TIME_BEFORE_COLLAPSE * SIM_CONFIG_G7['dt']) & (df_run['time'] <= collapse_time_val)].copy()
                if not sea_window_df.empty:
                    sea_window_df['time_to_collapse'] = sea_window_df['time'] - collapse_time_val
                    sea_prepped_data.append(sea_window_df)

            if sea_prepped_data:
                diagnostics_list = ["G_td", "speed_td", "couple_td", "g_lever", "beta_lever", "fcrit_lever", "PE_abs", "strain_avg_PEsq"]
                perform_and_plot_sea(sea_prepped_data, diagnostics_list, sc_name, scenario_output_dir)

        pre_props, base_props = perform_G7_scenario_analysis(sc_name, scenario_runs_data_list_current_sc, current_agent_td_config, scenario_output_dir, SIM_CONFIG_G7["dt"])

        plot_quadrant_proportions(pre_props, base_props, sc_name, scenario_output_dir)


    if master_results_log_all_scenarios:
        all_runs_combined_df = pd.concat(master_results_log_all_scenarios, ignore_index=True)
        all_runs_combined_df.to_csv(os.path.join(SIM_CONFIG_G7["results_base_dir"], "G7_all_scenarios_combined_runs.csv"), index=False)
        log_summary(f"\nSaved combined results of all runs to {os.path.join(SIM_CONFIG_G7['results_base_dir'], 'G7_all_scenarios_combined_runs.csv')}")
    log_summary("\n--- Completed main scenario runs (S1-S4) ---")

    # -------- Energetic Cost Parameter Exploration Experiments --------

    # Experiment 1: Vary g_cost_phi1
    g_cost_phi1_values_to_test = [0.25, 0.5, 0.75, 1.0, 1.25]
    exp1_dir = os.path.join(SIM_CONFIG_G7["results_base_dir"], "Experiment1_Vary_g_cost_phi1")
    os.makedirs(exp1_dir, exist_ok=True)
    exp1_metrics_runs = {}

    for phi1_val in g_cost_phi1_values_to_test:
        scenario_name_exp1 = f"S1_g_phi1_{phi1_val}"
        agent_cfg = AGENT_TD_CONFIG_G7_BASE.copy()
        agent_cfg["g_cost_phi1"] = phi1_val
        env_cfg = ENV_CONFIG_G7_BASE.copy()
        results_dir_exp1 = os.path.join(exp1_dir, scenario_name_exp1)
        os.makedirs(results_dir_exp1, exist_ok=True)

        scenario_runs_exp1 = []
        for i in range(SIM_CONFIG_G7["n_monte_carlo"]):
            run_seed = i + int(phi1_val * 100)
            df_run = run_fep_simulation(SIM_CONFIG_G7, agent_cfg, env_cfg, scenario_name_exp1, seed=run_seed)
            if not df_run.empty:
                scenario_runs_exp1.append(df_run)
                if i == 0:
                    plot_G7_results(df_run, scenario_name_exp1, results_dir_exp1, agent_cfg, run_id=str(i))

        if scenario_runs_exp1:
            plot_G7_mc_summary_results(scenario_runs_exp1, scenario_name_exp1, results_dir_exp1, agent_cfg)
            metrics_list_exp1 = ["g_lever", "fcrit_lever", "PE_abs", "G_td"]
            exp1_metrics_runs[phi1_val] = analyze_late_stable_phase(scenario_runs_exp1, metrics_list_exp1, return_all=True)
        else:
            exp1_metrics_runs[phi1_val] = {m: [np.nan] for m in ["g_lever", "fcrit_lever", "PE_abs", "G_td"]}

    # Cross-condition summary plots and statistical tests for Experiment 1
    phi1_sorted = sorted(exp1_metrics_runs.keys())
    metrics_list_exp1 = ["g_lever", "fcrit_lever", "PE_abs", "G_td"]
    lines_exp1 = ["--- Experiment 1: g_cost_phi1 variations ---"]
    for m in metrics_list_exp1:
        means = [np.nanmean(exp1_metrics_runs[p][m]) for p in phi1_sorted]
        plt.figure()
        plt.plot(phi1_sorted, means, marker='o')
        plt.xlabel("g_cost_phi1")
        plt.ylabel(f"Mean {m} (t=700-900)")
        plt.title(f"Stable {m} vs g_cost_phi1")
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(exp1_dir, f"summary_comparison_g_phi1_{m}.png")
        plt.savefig(fig_path, dpi=350)
        log_summary(f"[FIGURE] {fig_path}")
        plt.close()

        data_groups = [exp1_metrics_runs[p][m] for p in phi1_sorted]
        if all(len(g) > 1 for g in data_groups):
            anova_res = f_oneway(*data_groups)
            lines_exp1.append(f"ANOVA {m}: p={anova_res.pvalue:.3e}")
            bonf = len(list(combinations(phi1_sorted, 2)))
            for (i, j) in combinations(range(len(phi1_sorted)), 2):
                t_res = ttest_ind(data_groups[i], data_groups[j], nan_policy='omit')
                p_adj = min(1.0, t_res.pvalue * bonf)
                lines_exp1.append(f"  {m} {phi1_sorted[i]} vs {phi1_sorted[j]}: p={t_res.pvalue:.3e} (adj={p_adj:.3e})")
        else:
            lines_exp1.append(f"{m}: insufficient data for stats")

    with open(os.path.join(exp1_dir, "statistical_summary_g_phi1.txt"), "w") as f:
        f.write("\n".join(lines_exp1))
    for l in lines_exp1:
        log_summary(l)

    # Experiment 2: Vary beta_maintenance_cost_phi
    beta_cost_phi_values_to_test = [0.5, 1.0, 1.5, 2.0]
    exp2_dir = os.path.join(SIM_CONFIG_G7["results_base_dir"], "Experiment2_Vary_beta_maint_cost_phi")
    os.makedirs(exp2_dir, exist_ok=True)
    exp2_metrics_runs = {}

    for phi_beta_val in beta_cost_phi_values_to_test:
        scenario_name_exp2 = f"S1_beta_phi_{phi_beta_val}"
        agent_cfg = AGENT_TD_CONFIG_G7_BASE.copy()
        agent_cfg["beta_maintenance_cost_phi"] = phi_beta_val
        env_cfg = ENV_CONFIG_G7_BASE.copy()
        results_dir_exp2 = os.path.join(exp2_dir, scenario_name_exp2)
        os.makedirs(results_dir_exp2, exist_ok=True)

        scenario_runs_exp2 = []
        for i in range(SIM_CONFIG_G7["n_monte_carlo"]):
            run_seed = i + int(phi_beta_val * 100)
            df_run = run_fep_simulation(SIM_CONFIG_G7, agent_cfg, env_cfg, scenario_name_exp2, seed=run_seed)
            if not df_run.empty:
                scenario_runs_exp2.append(df_run)
                if i == 0:
                    plot_G7_results(df_run, scenario_name_exp2, results_dir_exp2, agent_cfg, run_id=str(i))

        if scenario_runs_exp2:
            plot_G7_mc_summary_results(scenario_runs_exp2, scenario_name_exp2, results_dir_exp2, agent_cfg)
            metrics_list_exp2 = ["beta_lever", "fcrit_lever", "PE_abs", "G_td"]
            exp2_metrics_runs[phi_beta_val] = analyze_late_stable_phase(scenario_runs_exp2, metrics_list_exp2, return_all=True)
        else:
            exp2_metrics_runs[phi_beta_val] = {m: [np.nan] for m in ["beta_lever", "fcrit_lever", "PE_abs", "G_td"]}

    beta_sorted = sorted(exp2_metrics_runs.keys())
    metrics_list_exp2 = ["beta_lever", "fcrit_lever", "PE_abs", "G_td"]
    lines_exp2 = ["--- Experiment 2: beta_maintenance_cost_phi variations ---"]
    for m in metrics_list_exp2:
        means = [np.nanmean(exp2_metrics_runs[p][m]) for p in beta_sorted]
        plt.figure()
        plt.plot(beta_sorted, means, marker='o')
        plt.xlabel("beta_maintenance_cost_phi")
        plt.ylabel(f"Mean {m} (t=700-900)")
        plt.title(f"Stable {m} vs beta_cost_phi")
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(exp2_dir, f"summary_comparison_beta_phi_{m}.png")
        plt.savefig(fig_path, dpi=350)
        log_summary(f"[FIGURE] {fig_path}")
        plt.close()

        data_groups = [exp2_metrics_runs[p][m] for p in beta_sorted]
        if all(len(g) > 1 for g in data_groups):
            anova_res = f_oneway(*data_groups)
            lines_exp2.append(f"ANOVA {m}: p={anova_res.pvalue:.3e}")
            bonf = len(list(combinations(beta_sorted, 2)))
            for (i, j) in combinations(range(len(beta_sorted)), 2):
                t_res = ttest_ind(data_groups[i], data_groups[j], nan_policy='omit')
                p_adj = min(1.0, t_res.pvalue * bonf)
                lines_exp2.append(f"  {m} {beta_sorted[i]} vs {beta_sorted[j]}: p={t_res.pvalue:.3e} (adj={p_adj:.3e})")
        else:
            lines_exp2.append(f"{m}: insufficient data for stats")

    with open(os.path.join(exp2_dir, "statistical_summary_beta_phi.txt"), "w") as f:
        f.write("\n".join(lines_exp2))
    for l in lines_exp2:
        log_summary(l)

    log_summary("\n--- G7 FEP-TD Agent Simulation Study Complete ---")
    log_summary("\n=== Final Summary ===")
    log_summary(f"Total Scenarios Run: {len(scenarios_G7_definitions)}")
    log_summary(f"Total MC Runs: {SIM_CONFIG_G7['n_monte_carlo']}")
    log_summary(f"Summary file generated at: {SUMMARY_LOG_PATH}")

