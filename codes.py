#Code for Phase-1
import numpy as np
import json
from scipy.signal import fir_filter_design as design
from scipy.signal import lfilter

def phase1_decode(rx_path="rx.npy", meta_path="meta.json", out_path="decoded_bits.npy"):
    # Load data
    rx = np.load(rx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    sps = meta["sps"]
    timing_offset = meta["timing_offset"]

    # ----- Step 1: Matched filter (rectangular since modulation is BPSK baseband) -----
    # Root Raised Cosine (RRC) is common; here, simple rectangular filter for sps samples
    rrc = np.ones(sps)  # crude filter
    rx_filtered = lfilter(rrc, 1.0, rx)

    # ----- Step 2: Timing recovery -----
    # Compensate by shifting by timing_offset
    rx_aligned = rx_filtered[timing_offset:]

    # ----- Step 3: Symbol downsampling -----
    symbols = rx_aligned[::sps]

    # ----- Step 4: BPSK decision -----
    bits = (np.real(symbols) > 0).astype(np.int32)

    # Save results
    np.save(out_path, bits)
    print(f"Saved decoded bits to {out_path}")

if __name__ == "__main__":
    phase1_decode("rx.npy", "meta.json", "decoded_bits.npy")




   #Code for Phase-2

import numpy as np
import json
from scipy.signal import lfilter

def phase2_decode(rx_path="rx.npy", meta_path="meta.json", out_path="decoded_bits.npy"):
    # Load received samples
    rx = np.load(rx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    sps = meta["sps"]
    timing_offset = meta["timing_offset"]

    # ----- Step 1: Matched filter (RRC or simple rectangular here) -----
    rrc = np.ones(sps)   # crude rectangular filter
    rx_filtered = lfilter(rrc, 1.0, rx)

    # ----- Step 2: Timing alignment -----
    rx_aligned = rx_filtered[timing_offset:]

    # ----- Step 3: Symbol downsampling -----
    symbols = rx_aligned[::sps]

    # ----- Step 4: Power calibration -----
    # Estimate signal power and noise power
    signal_power = np.mean(np.abs(symbols)**2)
    # For BPSK, ideal symbol amplitude = ±1, so normalize
    symbols_norm = symbols / np.sqrt(signal_power)

    # ----- Step 5: BPSK demodulation -----
    bits = (np.real(symbols_norm) > 0).astype(np.int32)

    # Save decoded bits
    np.save(out_path, bits)
    print(f"Saved decoded bits to {out_path}")

    # ----- Optional: BER calculation if clean bits available -----
    if "clean_bits" in meta:
        clean_bits = np.array(meta["clean_bits"], dtype=np.int32)
        min_len = min(len(bits), len(clean_bits))
        ber = np.mean(bits[:min_len] != clean_bits[:min_len])
        print(f"BER vs clean_bits: {ber:.4e}")

if __name__ == "__main__":
    phase2_decode("rx.npy", "meta.json", "decoded_bits.npy")


    #Code for Phase-3(Reed-Solomon)

import numpy as np
import json
from scipy.signal import lfilter

# ---------------- Reed-Solomon (15,11) ---------------- #
class ReedSolomon1511:
    def __init__(self):
        import reedsolo
        self.rs = reedsolo.RSCodec(4)  # (15,11) -> 4 parity symbols

    def decode(self, bits):
        # Group into 15-bit symbols
        n, k = 15, 11
        # Pack bits into bytes
        symbols = np.packbits(bits)
        decoded = []
        for i in range(0, len(symbols), n):
            block = symbols[i:i+n]
            if len(block) < n:
                break
            try:
                msg = self.rs.decode(bytearray(block))[0]
                decoded.extend(msg)
            except:
                # decoding failure
                continue
        return np.unpackbits(np.array(decoded, dtype=np.uint8))
    

#Code for Phase-3(Conv.)

def phase3_decode(rx_path="rx.npy", meta_path="meta.json", out_path="decoded_bits.npy"):
    # Load data
    rx = np.load(rx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    sps = meta["sps"]
    timing_offset = meta["timing_offset"]

    # ----- Step 1: Matched filter -----
    rrc = np.ones(sps)
    rx_filtered = lfilter(rrc, 1.0, rx)

    # ----- Step 2: Timing alignment -----
    rx_aligned = rx_filtered[timing_offset:]

    # ----- Step 3: Downsample -----
    symbols = rx_aligned[::sps]

    # ----- Step 4: Normalize -----
    signal_power = np.mean(np.abs(symbols)**2)
    symbols_norm = symbols / np.sqrt(signal_power)

    # ----- Step 5: Hard-decision BPSK -----
    bits = (np.real(symbols_norm) > 0).astype(np.int32)

    # ----- Step 6: Decode depending on coding -----
    coding = meta.get("coding", "none")
    if coding == "rs":
        rs = ReedSolomon1511()
        decoded_bits = rs.decode(bits)
    elif coding == "conv":
        decoded_bits = viterbi_decode(bits)
    else:
        decoded_bits = bits

    # Save decoded bits
    np.save(out_path, decoded_bits)
    print(f"Saved decoded bits to {out_path}")

    # BER/FER if ground truth available
    if "clean_bits" in meta:
        clean_bits = np.array(meta["clean_bits"], dtype=np.int32)
        min_len = min(len(decoded_bits), len(clean_bits))
        ber = np.mean(decoded_bits[:min_len] != clean_bits[:min_len])
        print(f"BER vs clean_bits: {ber:.4e}")

if __name__ == "__main__":
    phase3_decode("rx.npy", "meta.json", "decoded_bits.npy")



#Code for Phase-4

import numpy as np
import json
from scipy.signal import lfilter

# -------- Frequency offset estimation (Kay's method) -------- #
def estimate_freq_offset(samples, sps, sample_rate):
    # Phase difference between adjacent samples
    phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
    mean_phase = np.mean(phase_diff)
    # Convert to Hz
    freq_offset = mean_phase * sample_rate / (2 * np.pi)
    return freq_offset

def phase4_decode(rx_path="rx.npy", meta_path="meta.json", out_path="decoded_bits.npy"):
    # Load dataset
    rx = np.load(rx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    sps = meta["sps"]
    timing_offset = meta["timing_offset"]
    fs = meta["sample_rate"]

    # ----- Step 1: Frequency offset estimation -----
    est_offset = estimate_freq_offset(rx, sps, fs)
    print(f"Estimated Doppler offset: {est_offset:.2f} Hz")

    # ----- Step 2: Frequency correction -----
    n = np.arange(len(rx))
    rx_corrected = rx * np.exp(-1j * 2 * np.pi * est_offset * n / fs)

    # ----- Step 3: Matched filter -----
    rrc = np.ones(sps)
    rx_filtered = lfilter(rrc, 1.0, rx_corrected)

    # ----- Step 4: Timing recovery -----
    rx_aligned = rx_filtered[timing_offset:]

    # ----- Step 5: Downsample -----
    symbols = rx_aligned[::sps]

    # ----- Step 6: Normalize -----
    signal_power = np.mean(np.abs(symbols)**2)
    symbols_norm = symbols / np.sqrt(signal_power)

    # ----- Step 7: BPSK hard decision -----
    bits = (np.real(symbols_norm) > 0).astype(np.int32)

    # ----- Step 8: Save results -----
    np.save(out_path, bits)
    print(f"Saved decoded bits to {out_path}")

    # BER check if ground truth available
    if "clean_bits" in meta:
        clean_bits = np.array(meta["clean_bits"], dtype=np.int32)
        min_len = min(len(bits), len(clean_bits))
        ber = np.mean(bits[:min_len] != clean_bits[:min_len])
        print(f"BER vs clean_bits: {ber:.4e}")

if __name__ == "__main__":
    phase4_decode("rx.npy", "meta.json", "decoded_bits.npy")