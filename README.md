# ICARUS Phase-2 Project --- Communication Challenege
Reliable communication is the lifeline of any CubeSat mission. With limited power, bandwidth, and  hardware resources, small errors in modem design or coding can mean the difference between a  successful downlink and complete data loss.  This challenge simulates those realities through a staged dataset of received signals. Each phase  introduces controlled impairments—timing errors, noise calibration traps, coding requirements, and  Doppler shifts—that mirror conditions encountered in low Earth orbit.  Candidates must design and refine receiver algorithms to recover the original bitstreams and  demonstrate robust performance against defined thresholds. The process is intentionally iterative:  naive solutions will fail, requiring careful analysis, algorithmic insight, and systematic debugging. 

Approach: Every phase has a different algorithm to be applied. 

Starting with Phase-1, 
                      The problem that can occur during the initial stage of signal transmission is named Symbol Offset Timing. Which basically means that the receiver needs to sample the received signal sequence every (say) Tm seconds. However, the peak sample of the end of the matched filter output is not known in advance at the receiver.
                      This is where symbols come in, which are a distinct waveform or state of the communication channel that represents a unit of information, persisting for a fixed period of time. What essentially happens is that the symbol boundaries get misaligned during transmission. There is also a lot of noise(unrequired frequencies) that get added to our signal. 
                      The solution for that as mentioned, are utilising the Matched filtering and Timing recovery techniques. 
                      What Matched filtering essentially does is, it attenuates the external frequencies that are considered noise and it tries to match the shape of the transmitted signal to the signal that was initially sent. This results in a higher SNR(signal-to-noise ratio) giving us reduced symbol errors.
                      Timing recovery is used to obtain symbol synchronization. 
                      Both of these methods are applied together, continually to adjust sampling phase and frequency based on detected timing errors.

In Phase-2, 
                    SNR refers to signal-to-noise ratio. As the name suggests, it suggests the strength of the signal. If the SNR is higher, it means that the signal strength is good and clear, the amount of noise is lesser. During transmission again, there can be some inconsistencies while scaling the SNR. The signal may require long pulse durations and can have pronounced sensitivity to relaxation effects, magnetic field inhomogeneities and chemical shifts. 
                    To fix this Correct signal power calibration is implemented. What it essentially does is, it lowers the chance of inconsistencies by ensuring that both the signal and noise levels are measured accurately and consistently.

Looking into Phase-3, 
                    We are utilising error coding schemes to fix bit errors during transmission.
                    The first error correction scheme utilised is the Reed-Solomon scheme. 
                    The Convolution scheme is also used. The method is mentioned in the Document.

And lastly for Phase-4,
                    revising Doppler frequnecy shift, it refers to the change in frequency of an incident wave that occurs when it is reflected of a moving interface measured as the difference between the transmitted and observed frequency. 
                    This directly affects digital communication. Let's take an example. Say, a mobile device is moving towards/away from the base station. The radio waves transmitted by it get compressed/stretched out, leading in the change of its frequency. If this disturbance is significant, it can cause errors in data transmission and impact the quality of wireless connection. 
                    To aid with this, Frequency offset estimation and correction techniques are utilised. 


(THE PLOTS ARE IN THE DOCUMENTATION)
