import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

"""
Quantum Coherence in Neural Microtubules: Simulation Engine
Author: Anthony L. Perry (2025)
UPDATED VERSION: Increased drive and duration for robust Gamma generation.
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
        self.Tc = 12.0  # Characteristic quantum temp scale
        self.T0 = 310.0 # Ref temp
        
        # Calculate Base Coherence (Mean Field)
        if self.temp_k > (self.T0 + 10):
            self.mean_rho = 0.01 
        else:
            delta_T = max(0, self.temp_k - self.T0)
            self.mean_rho = 0.90 * np.exp(-delta_T / self.Tc) # Slightly higher base
            
    def get_coherence_trace(self, duration_ms, dt):
        steps = int(duration_ms / dt)
        rho = np.zeros(steps)
        rho[0] = self.mean_rho
        
        # Fluctuation parameters
        tau_corr = 5.0 
        sigma = 0.02   # Reduced noise for clearer signal
        
        noise = np.random.normal(0, 1, steps)
        for i in range(1, steps):
            drift = -(rho[i-1] - self.mean_rho) / tau_corr
            diffusion = sigma * np.sqrt(dt) * noise[i]
            rho[i] = rho[i-1] + drift * dt + diffusion
            
        return np.clip(rho, 0.0, 1.0)

# ==========================================
# 2. NEURAL NETWORK (LIF - PING Architecture)
# ==========================================
class PINGNetwork:
    def __init__(self, n_exc=400, n_inh=100, dt=0.1):
        self.Ne = n_exc
        self.Ni = n_inh
        self.dt = dt
        
        # Neuron Parameters
        self.Cm = 0.5    
        self.gl = 0.025  
        self.El = -70.0  
        self.Vt = -50.0  
        self.Vr = -60.0  
        self.Ref = 2.0   
        
        # Synaptic Parameters
        self.tau_ampa = 2.0
        self.tau_gaba = 5.0  
        
        # Connectivity
        self.p_conn = 0.1
        self.Wee = 0.02 * (np.random.rand(self.Ne, self.Ne) < self.p_conn)
        self.Wei = 0.05 * (np.random.rand(self.Ni, self.Ne) < self.p_conn)
        self.Wie = 0.10 * (np.random.rand(self.Ne, self.Ni) < self.p_conn) # Boosted I->E
        self.Wii = 0.05 * (np.random.rand(self.Ni, self.Ni) < self.p_conn)
        
    def run(self, duration_ms, mt_bundle):
        steps = int(duration_ms / self.dt)
        
        rho_t = mt_bundle.get_coherence_trace(duration_ms, self.dt)
        
        Ve = np.ones(self.Ne) * self.El + np.random.rand(self.Ne) * 10
        Vi = np.ones(self.Ni) * self.El + np.random.rand(self.Ni) * 10
        
        s_ampa = np.zeros(self.Ne) 
        s_gaba = np.zeros(self.Ni) 
        
        ref_e = np.zeros(self.Ne)
        ref_i = np.zeros(self.Ni)
        
        lfp_trace = [] 
        
        print(f"Running: T={mt_bundle.temp_k:.1f}K, Rho={mt_bundle.mean_rho:.2f}")
        
        for t_idx in range(steps):
            # QUANTUM MODULATION
            current_rho = rho_t[t_idx]
            mod_factor = 1.0 - (mt_bundle.alpha * current_rho)
            eff_tau_gaba = self.tau_gaba * mod_factor
            
            # CURRENT CALCULATION
            # INCREASED DRIVE HERE (2.5 instead of 1.5)
            I_ampa_e = np.dot(self.Wee, s_ampa) * (Ve - 0)
            I_gaba_e = np.dot(self.Wie, s_gaba) * (Ve + 75)
            I_ext_e = 2.5 + np.random.normal(0, 0.5, self.Ne) 
            
            I_ampa_i = np.dot(self.Wei, s_ampa) * (Vi - 0)
            I_gaba_i = np.dot(self.Wii, s_gaba) * (Vi + 75)
            I_ext_i = 1.5 + np.random.normal(0, 0.2, self.Ni)
            
            # UPDATE VOLTAGES
            dVe = (1/self.Cm) * (self.gl*(self.El - Ve) - I_ampa_e - I_gaba_e + I_ext_e)
            Ve_new = Ve + dVe * self.dt
            Ve_new[ref_e > 0] = self.Vr
            ref_e[ref_e > 0] -= self.dt
            
            dVi = (1/self.Cm) * (self.gl*(self.El - Vi) - I_ampa_i - I_gaba_i + I_ext_i)
            Vi_new = Vi + dVi * self.dt
            Vi_new[ref_i > 0] = self.Vr
            ref_i[ref_i > 0] -= self.dt
            
            # SPIKES
            spikes_e = Ve_new > self.Vt
            spikes_i = Vi_new > self.Vt
            
            Ve_new[spikes_e] = self.Vr
            Vi_new[spikes_i] = self.Vr
            ref_e[spikes_e] = self.Ref
            ref_i[spikes_i] = self.Ref
            
            # SYNAPSES
            s_ampa += (-s_ampa / self.tau_ampa) * self.dt + spikes_e.astype(float)
            s_gaba += (-s_gaba / eff_tau_gaba) * self.dt + spikes_i.astype(float)
            
            Ve = Ve_new
            Vi = Vi_new
            
            # LFP: Sum of abs currents
            lfp_val = np.mean(np.abs(I_ampa_e) + np.abs(I_gaba_e))
            lfp_trace.append(lfp_val)
            
        return np.array(lfp_trace), rho_t

# ==========================================
# 3. ANALYSIS UTILS
# ==========================================
def analyze_precision(lfp, dt):
    # Bandpass Filter (30-100 Hz)
    fs = 1000 / dt
    nyq = 0.5 * fs
    b, a = butter(4, [30/nyq, 100/nyq], btype='band')
    
    # Remove startup transient (first 200ms)
    startup_idx = int(200/dt)
    if len(lfp) > startup_idx:
        lfp_clean = lfp[startup_idx:]
    else:
        lfp_clean = lfp
        
    gamma_lfp = filtfilt(b, a, lfp_clean)
    
    # Peak Detection
    peaks = []
    threshold = np.mean(gamma_lfp) + 0.5 * np.std(gamma_lfp) # Lowered threshold
    
    for i in range(1, len(gamma_lfp)-1):
        if gamma_lfp[i] > gamma_lfp[i-1] and gamma_lfp[i] > gamma_lfp[i+1]:
            if gamma_lfp[i] > threshold: 
                peaks.append(i * dt)
    
    if len(peaks) > 3:
        isis = np.diff(peaks)
        jitter = np.std(isis) # ms
        precision = 1.0 / jitter if jitter > 0.01 else 0
    else:
        precision = 0 # Failed to oscillate
        
    return precision

# ==========================================
# 4. MAIN EXPERIMENT RUNNER
# ==========================================
if __name__ == "__main__":
    
    # Sweep Parameters
    temps = np.linspace(308, 320, 12) 
    precisions = []
    coherences = []
    
    print("Starting Optimized Simulation...")
    
    for T in temps:
        mt = MicrotubuleBundle(temp_k=T)
        net = PINGNetwork()
        
        # INCREASED DURATION TO 1000ms
        lfp, rho_trace = net.run(duration_ms=1000, mt_bundle=mt)
        
        prec = analyze_precision(lfp, 0.1)
        mean_rho = np.mean(rho_trace)
        
        precisions.append(prec)
        coherences.append(mean_rho)
        print(f" -> Precision={prec:.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(coherences, precisions, c=temps, cmap='coolwarm', s=100, edgecolor='k', zorder=2)
    
    # Linear Fit
    if len(coherences) > 1 and np.max(precisions) > 0:
        z = np.polyfit(coherences, precisions, 1)
        p = np.poly1d(z)
        plt.plot(coherences, p(coherences), "r--", linewidth=2, zorder=1, label=f"Fit (Slope={z[0]:.2f})")
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Temperature (K)')
    plt.xlabel('Microtubule Coherence Factor (rho)')
    plt.ylabel('Gamma Precision (1/std_ISI)')
    plt.title('Simulation Results: Coherence vs Gamma Precision')
    plt.grid(True)
    plt.legend()
    
    output_file = 'gamma_correlation_plot.png'
    plt.savefig(output_file)
    print(f"\nSaved to {output_file}")
