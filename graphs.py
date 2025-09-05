import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from numpy.random import default_rng

# -------- Load dataset -------- #
rx = np.load("rx.npy")
with open("meta.json", "r") as f:
    meta = json.load(f)

sps = meta["sps"]
timing_offset = meta["timing_offset"]
fs = meta["sample_rate"]

# crude matched filter + downsample
rrc = np.ones(sps)
rx_filtered = np.convolve(rx, rrc, mode="same")
rx_aligned = rx_filtered[timing_offset:]
symbols = rx_aligned[::sps]
symbols_norm = symbols / np.sqrt(np.mean(np.abs(symbols)**2))

#BERvsSNR for coded vs uncoded systems

rng = default_rng()
clean_bits = np.array(meta["clean_bits"], dtype=np.int32)

snr_range_db = list(range(0, 15, 2))
ber_results = []

for snr_db in snr_range_db:
    noise_var = 10**(-snr_db/10)
    noise = np.sqrt(noise_var/2) * (rng.standard_normal(symbols_norm.shape) +
                                    1j*rng.standard_normal(symbols_norm.shape))
    rx_noisy = symbols_norm + noise

    bits = (np.real(rx_noisy) > 0).astype(np.int32)
    L = min(len(bits), len(clean_bits))
    ber = np.mean(bits[:L] != clean_bits[:L])
    ber_results.append(ber)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=snr_range_db, y=ber_results,
                          mode="lines+markers", name="Uncoded BPSK"))
fig3.update_yaxes(type="log")
fig3.update_layout(title="BER vs SNR",
                   xaxis_title="SNR (dB)",
                   yaxis_title="Bit Error Rate")
fig3.show()

#Constellation diagrams at representative SNRs

fig1 = px.scatter(x=np.real(symbols_norm), y=np.imag(symbols_norm),
                  title=f"Constellation (SNR={meta['snr_db']} dB)",
                  labels={"x": "In-phase", "y": "Quadrature"})
fig1.show()


#Doppler effect compensation plots (before/after correction)

N = 4096
freqs = np.fft.fftfreq(N, 1/fs)
spectrum = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx[:N]))))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.fft.fftshift(freqs), y=spectrum, mode="lines"))
fig2.update_layout(title="Received Spectrum",
                   xaxis_title="Frequency (Hz)",
                   yaxis_title="Magnitude (dB)")
fig2.show()
