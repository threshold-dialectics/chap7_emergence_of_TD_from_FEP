## Overview

This repository contains the Python simulation code for **Chapter 7: *Instantiating Threshold Dialectics: Validating Emergent Dynamics in an FEP-Driven Agent*** from the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness*.

The script, "td_emergence_from_FEP.py", implements a minimalist agent ("Agent G7") whose behavior is guided by the **Free Energy Principle (FEP)**. The agent's goal is to minimize prediction error about its environment under explicit resource constraints. It is designed with internal mechanisms that are direct instantiations of the core adaptive levers from **Threshold Dialectics (TD)**.

The primary purpose of this simulation is to demonstrate that an agent driven by simple, local, FEP-aligned rules will spontaneously exhibit the complex adaptive dynamics and pre-collapse diagnostic signatures predicted by the TD framework when its viability is challenged. The simulation runs several scenarios to test the agent under different conditions of environmental predictability and resource availability, generating a rich set of analyses and visualizations.

## Theoretical Context: Threshold Dialectics in a Nutshell

The code in this repository is deeply rooted in the theoretical framework of Threshold Dialectics (TD). A brief understanding of its core concepts is essential to interpret the simulation.

### The Three Adaptive Levers

TD posits that any complex adaptive system manages its viability through the dynamic interplay of three core capacities, or "levers":

- **Perception Gain ($g$):** The system's sensitivity or responsiveness to new information and prediction errors. In the code, this is "g_lever".
- **Policy Precision ($\beta$):** The system's confidence or rigidity in its current model or course of action. High precision favors exploitation; low precision allows for exploration. In the code, this is "beta_lever".
- **Energetic Slack ($\mathcal{F}_{crit}$):** The system's buffer of available resources (energy, capital, time, etc.) used to fuel adaptation and withstand shocks. In the code, this is "fcrit_lever".

### The Tolerance Sheet ($\Theta_T$)

The combined state of the three levers defines the system's **Tolerance Sheet ($\Theta_T$)**, which represents its maximum capacity to withstand sustained stress or **Systemic Strain** (time-averaged prediction error, $\langle\Delta_P\rangle_{\tau}$). The core equation is:

$$ \Theta_T = C \cdot g^{w_1} \cdot \beta^{w_2} \cdot \mathcal{F}_{crit}^{w_3} $$

Collapse occurs when systemic strain exceeds this tolerance ($\langle\Delta_P\rangle_{\tau} > \Theta_T$). In the code, "theta_t" represents $\Theta_T$ and "strain_avg_PEsq" represents the systemic strain.

### Core Diagnostics: Speed and Couple

TD argues that the best early warnings of collapse come from the *velocities* of the levers, not just their levels. Two key diagnostics are:

- **Speed Index ($\mathcal{S}$):** The joint rate of change of the levers, primarily $\beta$ and $\mathcal{F}_{crit}$. High speed indicates rapid, often destabilizing, structural change. In the code, this is "speed_td".
- **Couple Index ($\mathcal{C}$):** The correlation between the lever velocities. A strong, detrimental coupling (e.g., rigidity increasing while resources deplete) is a potent signature of escalating risk. In the code, this is "couple_td".

## About the Simulation (Agent G7)

The "FEPAgentG7" class implements these principles:
- **FEP-Driven Goal:** The agent's primary goal is to minimize prediction error by learning a single parameter ("learned_model_parameter") that predicts the environmental signal.
- **TD Instantiation:** The agent's internal state includes "g_lever", "beta_lever", and "fcrit_lever". It adjusts these levers based on simple, FEP-aligned heuristics (e.g., if prediction error is high, increase vigilance "g_lever" and exploration by reducing "beta_lever").
- **Resource Constraints:** All adaptive actions (adjusting levers, updating the model) have an energetic cost that depletes "fcrit_lever". The agent's viability is constrained by both a "ThetaT_Breach" (strain exceeds tolerance) and "Fcrit_Depletion" (running out of energy).

The simulation does **not** program the agent to manage its "Speed" or "Couple" indices. These diagnostic signatures are **emergent properties** of its FEP-driven behavior, which this simulation suite aims to validate.

## Scenarios and Experiments

The "main" block of the script executes a series of validation scenarios and experiments, running 30 Monte Carlo simulations for each to ensure statistical robustness.

### Main Scenarios (S1-S4)

1.  **S1: Predictable Environment, Ample Resources:** A baseline scenario to observe the agent's self-organization and stable operation.
2.  **S2: Unpredictable Environment, Ample Resources:** Tests the agent's resilience to high environmental noise and its "capacity overwhelm" failure mode.
3.  **S3: Predictable Environment, Scarce Resources:** Tests the agent's behavior under severe resource constraints, leading to a "resource exhaustion" failure mode.
4.  **S4: Environmental Regime Shift:** The environment abruptly becomes much noisier, testing the agent's ability to adapt to sudden, drastic changes.

### Parameter Sweep Experiments

1.  **Experiment 1: Varying $g$ Cost Exponent ("g_cost_phi1"):** Explores how the energetic cost structure of perception gain affects the agent's stable-state behavior.
2.  **Experiment 2: Varying $\beta$ Maintenance Cost Exponent ("beta_maintenance_cost_phi"):** Investigates how the ongoing cost of maintaining policy precision influences the system's dynamics.

## How to Run

### Prerequisites

You will need Python 3.9+ and the following libraries:
- "numpy"
- "pandas"
- "matplotlib"
- "scipy"

You can install them using pip:
"""bash
pip install numpy pandas matplotlib scipy
"""

### Execution

To run the full simulation suite, simply execute the script from your terminal:
"""bash
python td_emergence_from_FEP.py
"""

The script will create a main results directory named "results_G7_fep_td_agent". Inside this directory, it will create subdirectories for each scenario and experiment.

## Output and Results

The simulation generates a rich set of outputs for each scenario:

-   **Console Log & Summary File:** A "summary.txt" file in the main results directory logs the configuration, progress of the Monte Carlo runs, and all statistical analysis results.
-   **Per-Run Plots:** For the first run of each scenario, a detailed multi-panel plot is generated (e.g., "G7_fep_agent_s1_predenv_ampleres_run0.png"), showing the time series of all key levers, diagnostics, and system states.
-   **Monte Carlo Summary Plots:** For each key variable, a plot showing the mean trajectory and +/- 1 standard deviation across all Monte Carlo runs is generated (e.g., "G7_mc_avg_g_lever.png").
-   **Superposed Epoch Analysis (SEA) Plots:** For scenarios with collapses, SEA plots show the averaged diagnostic trends leading up to the collapse event (e.g., "sea_speed_td_S3_PredEnv_ScarceRes.png").
-   **Statistical Summaries:** Text files containing detailed statistical comparisons between scenarios and conditions (e.g., t-tests, ANOVA, Chi-squared tests).
-   **Combined Data:** A final CSV file, "G7_all_scenarios_combined_runs.csv", containing the raw data from all simulation runs across all main scenarios.

## Key Findings from the Simulation

The script's output validates several core hypotheses of Threshold Dialectics:

1.  **Lever Self-Organization (S1):** In a stable environment, the agent's levers self-organize into a stable, cyclical equilibrium, demonstrating FEP-rational resource management where it "spends" energy ("fcrit_lever") to maintain a state of low prediction error.
2.  **Emergent Diagnostic Signatures:** Under stress (S2, S3, S4), the agent's FEP-driven struggle generates TD-consistent diagnostic signatures. For example, the SEA plots show a characteristic sharp rise in the **Speed Index** before resource-exhaustion collapse (S3) and complex patterns in the **Couple Index** during a regime shift (S4). These are emergent phenomena, not programmed behaviors.
3.  **Distinct Failure Modes:** The simulations successfully produce distinct, TD-consistent collapse modes: "capacity overwhelm" from high strain in S2 and "resource exhaustion" from high costs in S3.
4.  **Statistical Validation:** The automated analysis reveals statistically significant differences in lever dynamics and diagnostic signatures across the different scenarios, confirming that the TD framework can distinguish between different modes of systemic stress. For example, S2 (unpredictable) shows significantly lower policy precision ("beta_lever") and higher strain than the S1 baseline.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.