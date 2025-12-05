# Quantum-Coherence-Neural-Microtubules-Gamma-Oscillation-Generation
Data Code For Simulation
# Quantum Coherence in Neural Microtubules - Simulation Code

This repository contains the source code for the computational validation presented in the paper **"Quantum Coherence in Neural Microtubules: A Conditional Framework for Testing Quantum Modulation of Gamma Oscillation Precision"** (Perry, 2025).

## Overview
The simulation implements a hybrid Quantum-Classical neural network model:
1.  **Quantum Dynamics:** Models the effective coherence factor $\rho(t)$ of a microtubule bundle using a stochastic process derived from Lindblad master equation rates (thermal, EM, mechanical) and collective protection factors.
2.  **Neural Dynamics:** Simulates a PING (Pyramidal-Interneuron Network Gamma) architecture using 500 Leaky Integrate-and-Fire (LIF) neurons (400 Exc / 100 Inh).
3.  **Scale-Bridging Interface:** Modulates synaptic time constants based on the instantaneous quantum coherence: $\tau_{syn}(t) = \tau_0 [1 - \alpha \rho(t)]$.

## Key Parameters
* **Neurons:** $N_E=400$, $N_I=100$.
* **Physics:** $C_m=0.5$ nF, $V_{thresh}=-50$ mV.
* **Quantum Coupling:** $\alpha$ (Coherence Modulation Factor) = 0.05.
* **Decoherence:** Temperature-dependent scaling $T_c \approx 12$ K.

## Reproduction of Results
To generate the data for **Figure 3** (Coherence-Precision Correlation) and **Table 2** (Temperature Scaling):

```bash
python quantum_gamma_simulation.py
