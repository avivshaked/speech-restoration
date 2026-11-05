# Findings

The numbers behind the README, with what each one measured. Every figure here was read from a saved run output (scores, curves or checkpoints). Runs used 4 to 16 clips and a single seed, so small differences between them mean little.

WER throughout is the word error rate between Whisper large-v3's transcript of the processed audio and its transcript of the clean reference. It is not measured against human transcripts.

## 1. Suppressing noise did not restore speech

Zero-shot [MP-SENet](https://github.com/yxlu-0102/MP-SENet), a masking-based noise suppressor, on 11 Creative Commons podcast clips (20 s, 16 kHz) with white noise added:

| Noise level | DNSMOS background | WER, noisy input | WER, MP-SENet |
|---|---|---|---|
| 5 dB SNR | 4.04 | 0.191 | 0.229 |
| 0 dB SNR | 3.96 | 0.245 | 0.287 |

A suppressor works on what is already in the signal, so speech lost under the noise does not come back. The project moved to generative restoration after this run. (The published test-set manifest lists 10 clips; one was excluded after these runs.)

## 2. A generative restorer can change the words

Zero-shot Resemble Enhance on clean EARS speech with synthetic damage (12 s, 48 kHz). With light damage (4 clips) the restored WER was 0.022. With moderate damage (6 clips) it was 0.773, against 0.059 for the untreated input, while NISQA scored the restored audio 4.01 out of 5.

On real DAPS recordings (3 speakers, one 12 s clip each per condition):

| Condition | WER, untreated | WER, restored | NISQA, restored | Speaker similarity |
|---|---|---|---|---|
| iPad, office | 0.00 | 0.05 | 4.96 | 0.91 |
| iPad, living room | 0.00 | 0.02 | 4.99 | 0.93 |
| iPad, conference room | 0.00 | 0.00 | 5.02 | 0.94 |
| iPhone, balcony | 0.03 | 0.79 | 4.84 | 0.73 |

A synthetic "podcast" recipe (reverberation plus saturation) drove the restored WER to 0.58 on 6 EARS clips, against an average of 0.024 on the three real indoor conditions. My synthetic degradations were harsher than real ones, so from then on synthetic results were used for development only and DAPS recordings were the reference.

## 3. Fine-tuning the restorer on DAPS

Resemble Enhance fine-tuned on DAPS device recordings, with speakers f8, f10, m8 and m10 held out. Evaluation used 12 clips: those four speakers in three conditions (iPad office, iPhone living room, iPhone balcony).

| Checkpoint | Mean WER | Balcony WER |
|---|---|---|
| off-the-shelf | 0.273 | 0.769 |
| step 3,000 (about half an epoch) | 0.102 | 0.274 |
| steps 3,000 to 37,458 (six epochs) | 0.067 to 0.219, no trend | |

Caveats:

- These clips were the first 12 s of each recording, which includes lead-in silence. The evaluation code was later changed to pick a speech-dense window cut at sentence boundaries (`load_aligned_speech_window` in `src/voice_cleanup/data/daps.py`).
- The best checkpoint was chosen on the same 12 clips.
- The untreated input's WER was not recorded in this run.

On four 30 s balcony clips with the later windowing, sweeping the denoiser-strength setting (lambda 0 to 1) on the fine-tuned model gave WER between 0.157 and 0.179. The setting made little difference.

## 4. The producer

A spectral residual-mask U-Net with 2,789,249 parameters, trained from scratch on DAPS to map the unprocessed studio take to the engineer's processed version. The output mask starts at zero, so the untrained model passes audio through unchanged. Best checkpoint at step 10,325, on four held-out speakers with 30 s clips:

| Measure | Value |
|---|---|
| log-magnitude L1 to the engineer's version | 0.399 |
| WER against the engineer's version | 0.013 |
| speaker similarity | 0.998 |
| loudness change | +4.1 LU |
| crest factor change | -2.5 dB |

I preferred the model's output to the engineer's version, even though the L1 loss puts the two 0.399 apart.

A second variant was trained from the de-breathed studio take instead, so nothing in its training data teaches breath removal, and is fed the unprocessed take at inference. A measurement at the time showed it keeping the breaths; that measurement was not saved. Best checkpoint at step 5,500: L1 0.347, WER 0.010, loudness +4.6 LU. These are not comparable with the first variant's numbers because its target is easier.

## 5. The full-band rebuild

The DAPS conditions never lack high frequencies, so the restorer had no reason to learn bandwidth extension. The rebuild started again from the off-the-shelf Resemble Enhance weights and trained on VCTK and Hi-Fi TTS, which are recorded at 44.1 kHz or above, with synthetic EQ damage and bandwidth limiting as the main degradations. A quality gate scored 368,433 clips and dropped 14,520 (3.9%). The best checkpoint was at step 6,000, with validation loss down from 0.386 to 0.234.

On 16 held-out clips (four speakers, each under EQ damage, "radio voice", bandwidth limiting and a broadband mix, 30 s):

| Measure | Off-the-shelf | Fine-tuned |
|---|---|---|
| WER | 0.135 | 0.056 |
| high-frequency recovery, 8-16 kHz | 11.1 dB | 15.1 dB |
| speaker similarity | | 0.986 |

One held-out speaker sat at WER 0.144 before and after and accounts for most of the remaining mean.

When I listened to the held-out outputs they sounded unnatural and unlike their clean targets. The panel's measures cover transcripts, voice identity and band energy but not naturalness. My working explanation, which I did not test directly: the model is conditioned on the damaged input's spectrum, so colouration in the input survives into the output.

## 6. Comparing other restorers

The same reference clips went through several models, loudness-matched for listening. On three synthetic VCTK speakers with bandwidth limiting, the 8-16 kHz energy relative to the speech band:

| Speaker | Untreated | Off-the-shelf Resemble Enhance | VoiceFixer | Clean original |
|---|---|---|---|---|
| p225 | -38.4 dB | -20.1 dB | -7.5 dB | -7.5 dB |
| p226 | -41.5 dB | -24.9 dB | -15.8 dB | -15.1 dB |
| p227 | -53.9 dB | -30.8 dB | -24.6 dB | -25.9 dB |

VoiceFixer also raised the top octave of the clean originals, by 4.4 dB (p225), 0.9 dB (p226) and 1.5 dB (p227). Of the models compared, it regenerated the most voice-like content from damaged input, with too much top end. I stopped before choosing between the high-shelf settings in `src/voice_cleanup/eval/tame.py`, which include one that scales the cut with the input's brightness.

The models I would try next are AnyEnhance, and AudioSR as a stage after VoiceFixer. Miipher and FINALLY had no open weights when I checked in May 2026.

## A note on the code comments

Some comments and docstrings refer to a decision log, a learning log and handoff notes. Those were working files and are not part of this repository.
