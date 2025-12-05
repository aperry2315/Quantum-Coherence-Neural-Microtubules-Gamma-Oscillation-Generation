import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

"""
Quantum Coherence in Neural Microtubules: Simulation Engine
Author: Anthony L. Perry (2025)

This script simulates a PING (Pyramidal-Interneuron Network Gamma) circuit 
modulated by a stochastic variable representing Microtubule Quantum Coherence.
"""

# ==========================================
# 1. QUANTUM DYNAMICS (Effective Mean Field)
# ==========================================
class MicrotubuleBundle:
    def __init__(self, temp_k=310.0, protection_factor=1e4, alpha_coupling=0.05):
        self.temp_k = temp_k
        self.protection = protection_factor
        self.alpha = alpha_coupling  # Coherence Modulation Factor
        
        # Physics Constants
        self.kb = 1.38e-23
        self.hbar = 1.05e-34
        self.Tc = 12.0  # Characteristic quantum temp scale (from paper)
        self.T0 = 310.0 # Ref temp
        
        # Calculate Base Coherence (Mean Field)
        # Based on Eq 6 in paper: rho(T) = rho0 * exp(-(T-T0)/Tc)
        # We model effective coherence as a stochastic process around this mean.
        if self.temp_k > (self.T0 + 10):
            self.mean_rho = 0.01 # Effectively decoherent at high T
        else:
            # Phenomenological scaling derived from protection mechanics
            delta_T = max(0, self.temp_k - self.T0)
            self.mean_rho = 0.85 * np.exp(-delta_T / self.Tc)
            
    def get_coherence_trace(self, duration_ms, dt):
        """
        Generates a stochastic coherence trajectory rho(t).
        Modeled as an Ornstein-Uhlenbeck process to represent 
        fluctuations in the protected subspace.
        """
        steps = int(duration_ms / dt)
        rho = np.zeros(steps)
        rho[0] = self.mean_rho
        
        # Fluctuation parameters (1/f noise proxy)
        tau_corr = 5.0 # Correlation time (ms)
        sigma = 0.05   # Noise magnitude
        
        noise = np.random.normal(0, 1, steps)
        
        for i in range(1, steps):
            # d_rho = -theta * (rho - mean) * dt + sigma * dW
            drift = -(rho[i-1] - self.mean_rho) / tau_corr
            diffusion = sigma * np.sqrt(dt) * noise[i]
            rho[i] = rho[i-1] + drift * dt + diffusion
            
        # Clip to physical bounds [0, 1]
        return np.clip(rho, 0.0, 1.0)

# ==========================================
# 2. NEURAL NETWORK (LIF - PING Architecture)
# ==========================================
class PINGNetwork:
    def __init__(self, n_exc=400, n_inh=100, dt=0.1):
        self.Ne = n_exc
        self.Ni = n_inh
        self.dt = dt
        
        # Neuron Parameters (LIF) - From Supplemental Table 1
        self.Cm = 0.5    # nF
        self.gl = 0.025  # uS (Leak conductance)
        self.El = -70.0  # mV (Leak potential)
        self.Vt = -50.0  # mV (Threshold)
        self.Vr = -60.0  # mV (Reset)
        self.Ref = 2.0   # ms (Refractory)
        
        # Synaptic Parameters (Base)
        self.tau_ampa = 2.0
        self.tau_gaba = 5.0  # Key determinant of Gamma freq
        
        # Connectivity (Sparse random)
        self.p_conn = 0.1
        self.Wee = 0.02 * (np.random.rand(self.Ne, self.Ne) < self.p_conn)
        self.Wei = 0.05 * (np.random.rand(self.Ni, self.Ne) < self.p_conn)
        self.Wie = 0.08 * (np.random.rand(self.Ne, self.Ni) < self.p_conn) # Strong I->E is crucial for PING
        self.Wii = 0.05 * (np.random.rand(self.Ni, self.Ni) < self.p_conn)
        
    def run(self, duration_ms, mt_bundle):
        steps = int(duration_ms / self.dt)
        time = np.arange(0, duration_ms, self.dt)
        
        # Get Quantum Coherence Trace for this run
        rho_t = mt_bundle.get_coherence_trace(duration_ms, self.dt)
        
        # State Variables
        Ve = np.ones(self.Ne) * self.El + np.random.rand(self.Ne) * 10
        Vi = np.ones(self.Ni) * self.El + np.random.rand(self.Ni) * 10
        
        # Refractory counters
        ref_e = np.zeros(self.Ne)
        ref_i = np.zeros(self.Ni)
        
        # Synaptic Gating Variables (s)
        s_ampa = np.zeros(self.Ne) # E -> others
        s_gaba = np.zeros(self.Ni) # I -> others
        
        # Recording
        lfp_trace = [] # Proxy: Mean absolute voltage
        spike_times_e = []
        
        print(f"Running Simulation: T={mt_bundle.temp_k}K, Mean Rho={mt_bundle.mean_rho:.2f}...")
        
        for t_idx in range(steps):
            # 1. QUANTUM MODULATION STEP
            # Eq 4 in paper: tau_syn = tau_0 * (1 - beta * rho)
            # We modulate GABA decay primarily as it controls the Gamma period precision
            current_rho = rho_t[t_idx]
            mod_factor = 1.0 - (mt_bundle.alpha * current_rho)
            
            eff_tau_gaba = self.tau_gaba * mod_factor
            
            # 2. CURRENT CALCULATION
            # I_syn = g * s * (V - E_rev)
            # E_rev_AMPA = 0, E_rev_GABA = -75
            
            # Input to Excitatory
            I_ampa_e = np.dot(self.Wee, s_ampa) * (Ve - 0)
            I_gaba_e = np.dot(self.Wie, s_gaba) * (Ve + 75)
            I_ext_e = 1.5 + np.random.normal(0, 0.5, self.Ne) # Poisson-like drive
            
            # Input to Inhibitory
            I_ampa_i = np.dot(self.Wei, s_ampa) * (Vi - 0)
            I_gaba_i = np.dot(self.Wii, s_gaba) * (Vi + 75)
            I_ext_i = 0.8 + np.random.normal(0, 0.2, self.Ni)
            
            # 3. UPDATE VOLTAGES (Euler integration)
            # dV/dt = (1/Cm) * (gl(El-V) - Isyn + Iext)
            
            # Excitatory Update
            dVe = (1/self.Cm) * (self.gl*(self.El - Ve) - I_ampa_e - I_gaba_e + I_ext_e)
            Ve_new = Ve + dVe * self.dt
            # Clamp refractory
            Ve_new[ref_e > 0] = self.Vr
            ref_e[ref_e > 0] -= self.dt
            
            # Inhibitory Update
            dVi = (1/self.Cm) * (self.gl*(self.El - Vi) - I_ampa_i - I_gaba_i + I_ext_i)
            Vi_new = Vi + dVi * self.dt
            Vi_new[ref_i > 0] = self.Vr
            ref_i[ref_i > 0] -= self.dt
            
            # 4. SPIKE DETECTION
            spikes_e = Ve_new > self.Vt
            spikes_i = Vi_new > self.Vt
            
            # Reset
            Ve_new[spikes_e] = self.Vr
            Vi_new[spikes_i] = self.Vr
            ref_e[spikes_e] = self.Ref
            ref_i[spikes_i] = self.Ref
            
            # Record spikes for raster
            if np.any(spikes_e):
                spike_indices = np.where(spikes_e)[0]
                current_time = t_idx * self.dt
                for idx in spike_indices:
                    spike_times_e.append((current_time, idx))
            
            # 5. SYNAPTIC DYNAMICS
            # ds/dt = -s/tau + sum(spikes)
            # Note modulation of tau_gaba here
            s_ampa += (-s_ampa / self.tau_ampa) * self.dt + spikes_e.astype(float)
            s_gaba += (-s_gaba / eff_tau_gaba) * self.dt + spikes_i.astype(float)
            
            # Update State
            Ve = Ve_new
            Vi = Vi_new
            
            # LFP proxy: Average synaptic currents into Excitatory population
            lfp_val = np.mean(I_ampa_e + I_gaba_e)
            lfp_trace.append(lfp_val)
            
        return np.array(lfp_trace), spike_times_e, rho_t

# ==========================================
# 3. ANALYSIS UTILS
# ==========================================
def analyze_precision(lfp, dt):
    """
    Calculates Phase Locking Value (PLV) and Jitter from LFP.
    """
    # Filter for Gamma (30-80 Hz)
    from scipy.signal import butter, filtfilt
    fs = 1000 / dt
    nyq = 0.5 * fs
    b, a = butter(4, [30/nyq, 80/nyq], btype='band')
    gamma_lfp = filtfilt(b, a, lfp)
    
    # Hilbert Transform for Phase
    analytic = hilbert(gamma_lfp)
    phase = np.angle(analytic)
    envelope = np.abs(analytic)
    
    # Find peaks (bursts) to measure jitter
    # Simple metric: Std Dev of period between peaks
    # Zero crossings or Peak detection
    peaks = []
    for i in range(1, len(gamma_lfp)-1):
        if gamma_lfp[i] > gamma_lfp[i-1] and gamma_lfp[i] > gamma_lfp[i+1]:
            if gamma_lfp[i] > np.std(gamma_lfp): # threshold
                peaks.append(i * dt)
    
    if len(peaks) > 2:
        isis = np.diff(peaks)
        jitter = np.std(isis) # ms
        freq = 1000 / np.mean(isis)
        precision = 1.0 / jitter if jitter > 0 else 0
    else:
        jitter = 100
        precision = 0
        
    return precision, jitter, gamma_lfp

# ==========================================
# 4. MAIN EXPERIMENT RUNNER
# ==========================================
if __name__ == "__main__":
    
    # --- Experiment A: Correlation Study ---
    temps = np.linspace(308, 320, 10) # Temperature Sweep
    precisions = []
    coherences = []
    
    # Run multiple trials
    print("Starting Temperature Sweep Experiment...")
    for T in temps:
        # Create Microtubule Physics
        mt = MicrotubuleBundle(temp_k=T)
        
        # Create Network
        net = PINGNetwork()
        
        # Run
        lfp, spikes, rho_trace = net.run(duration_ms=500, mt_bundle=mt)
        
        # Analyze
        prec, jitter, _ = analyze_precision(lfp, 0.1)
        mean_rho = np.mean(rho_trace)
        
        precisions.append(prec)
        coherences.append(mean_rho)
        print(f" -> T={T:.1f}K | Rho={mean_rho:.3f} | Precision={prec:.3f}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.scatter(coherences, precisions, c=temps, cmap='coolwarm', s=100, edgecolor='k')
    
    # Linear Fit
    if len(coherences) > 1:
        z = np.polyfit(coherences, precisions, 1)
        p = np.poly1d(z)
        plt.plot(coherences, p(coherences), "r--", label=f"Fit (Slope={z[0]:.2f})")
    
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('Microtubule Coherence Factor (rho)')
    plt.ylabel('Gamma Precision (1/std_ISI)')
    plt.title('Simulation Results: Coherence vs Gamma Precision')
    plt.grid(True)
    plt.legend()
    
    output_file = 'gamma_correlation_plot.png'
    plt.savefig(output_file)
    print(f"\nSimulation Complete. Results saved to {output_file}")
