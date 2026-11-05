# Findings

The numbers behind the README, with what each one measured. The held-out evaluation in sections 3 and 4 (`scripts/eval_heldout.py`, with paired statistics from `scripts/heldout_stats.py`) used 48 restorer clips and 20 producer clips. Most earlier runs used 4 to 16 clips and a single seed, so small differences between them mean little.

WER throughout is the word error rate between Whisper large-v3's transcript of the processed audio and its transcript of the clean reference. It is not measured against human transcripts.

## 1. Noise-suppression baseline

Zero-shot [MP-SENet](https://github.com/yxlu-0102/MP-SENet), a noise suppressor, on 11 podcast clips from archive.org (20 s, 16 kHz) with white noise added:

| Noise level | DNSMOS background, noisy input | DNSMOS background, MP-SENet | WER, noisy input | WER, MP-SENet |
|---|---|---|---|---|
| 5 dB SNR | 1.50 | 4.04 | 0.191 | 0.229 |
| 0 dB SNR | 1.28 | 3.96 | 0.245 | 0.287 |

PESQ rose too, from 1.14 to 1.71 at 5 dB and from 1.08 to 1.52 at 0 dB. By both scores the audio got cleaner, and transcription got worse at both noise levels. A suppressor can only remove what it judges to be noise; it cannot bring back speech the noise covered. The project moved to generative restoration after this run.

Ten of the clips are Creative Commons. The eleventh was tagged public domain by its uploader and was later dropped from the published test-set manifest because that tag is not reliable. It was part of these runs.

## 2. Off-the-shelf generative restoration

Zero-shot Resemble Enhance on clean EARS speech with synthetic damage (12 s, 48 kHz). With light damage (4 clips) the restored WER was 0.022. With moderate damage (6 clips) it was 0.773, where the untreated input had 0.059, while NISQA scored the restored audio 4.01. NISQA's scale is nominally 1 to 5, but its predictions are not capped, which is why the table below has a 5.02.

On real DAPS recordings (3 speakers, one 12 s clip each per condition):

| Condition | WER, untreated | WER, restored | NISQA, restored | Speaker similarity |
|---|---|---|---|---|
| iPad, office | 0.00 | 0.05 | 4.96 | 0.91 |
| iPad, living room | 0.00 | 0.02 | 4.99 | 0.93 |
| iPad, conference room | 0.00 | 0.00 | 5.02 | 0.94 |
| iPhone, balcony | 0.03 | 0.79 | 4.84 | 0.73 |

A synthetic "podcast" recipe (reverberation plus saturation) drove the restored WER to 0.58 on 6 EARS clips, against an average of 0.024 on the three real indoor conditions. My synthetic degradations were harsher than real ones. From then on DAPS recordings were the reference wherever DAPS had the damage in question. DAPS has no bandwidth loss, so the full-band work in sections 5 and 6 still relies on synthetic damage.

## 3. Fine-tuning the restorer on DAPS

Resemble Enhance fine-tuned on DAPS device recordings for six epochs (37,458 steps), with speakers f8, f10, m8 and m10 held out of training.

### Held-out evaluation

Those four speakers in all 12 DAPS device and room conditions, script 1, cut to about 30 s at sentence boundaries: 48 clips. "Last" is the final checkpoint; "best" is the checkpoint the training run selected on 12 of these recordings, so its numbers are not independent.

| System | Mean WER | Median WER | WER, balcony (8 clips) | WER, other rooms (40 clips) | Speaker similarity |
|---|---|---|---|---|---|
| untreated | 0.028 | 0.012 | 0.054 | 0.023 | 0.970 |
| off-the-shelf | 0.089 | 0.031 | 0.351 | 0.036 | 0.968 |
| fine-tuned, best (step 27,000) | 0.039 | 0.021 | 0.137 | 0.020 | 0.974 |
| fine-tuned, last | 0.038 | 0.014 | 0.126 | 0.020 | 0.976 |

Paired differences for the last checkpoint, with 95% bootstrap intervals over clips (`scripts/heldout_stats.py` prints these from the saved results):

| Comparison | Mean WER difference | 95% interval | Clips lower / equal / higher |
|---|---|---|---|
| against off-the-shelf, all clips | -0.051 | -0.084 to -0.023 | 28 / 8 / 12 |
| against off-the-shelf, balcony | -0.225 | -0.343 to -0.115 | 7 / 1 / 0 |
| against untreated, all clips | +0.010 | -0.004 to +0.022 | 9 / 17 / 22 |
| against untreated, other rooms | -0.003 | -0.016 to +0.007 | 9 / 17 / 14 |
| against untreated, balcony | +0.072 | +0.051 to +0.094 | 0 / 0 / 8 |
| against best | -0.001 | -0.009 to +0.005 | 19 / 14 / 15 |

Speaker similarity to the studio take was higher than the untreated input's on 39 of 48 clips (mean +0.006, interval +0.002 to +0.010).

### Training curve

During training the run scored 12 held-out clips (the four speakers in iPad office, iPhone living room and iPhone balcony) on the first 12 s of each recording, which includes lead-in silence:

| Checkpoint | Mean WER | Balcony WER |
|---|---|---|
| off-the-shelf | 0.273 | 0.769 |
| step 3,000 (about half an epoch) | 0.102 | 0.274 |
| steps 3,000 to 37,458 | 0.067 to 0.219, no trend | |

Those windows were the reason for the evaluation above, which picks speech-dense windows cut at sentence boundaries (`load_aligned_speech_window` in `src/voice_cleanup/data/daps.py`).

Resemble Enhance has a denoiser-strength setting (`lambd`, 0 to 1), and the held-out evaluation used the default, 0.5. An earlier sweep from 0 to 1 on four 30 s balcony clips, with the best checkpoint, moved WER only between 0.157 and 0.179.

## 4. The producer

A spectral residual-mask U-Net with 2,788,225 parameters, trained from scratch on DAPS to map the unprocessed studio take to the engineer's post-produced version. The output layer starts at zero, so the untrained model passes audio through unchanged. Held-out evaluation on the same four speakers, all five scripts, about 30 s each (20 clips):

| System | Log-magnitude L1 to target | WER | Speaker similarity | Loudness change | Crest factor change |
|---|---|---|---|---|---|
| pass-through | 0.792 | 0.024 | 0.996 | +0.0 LU | +0.0 dB |
| gain only, matched to each target's loudness | 0.660 | 0.032 | 0.996 | +4.3 LU | -0.0 dB |
| static EQ plus gain, fitted on training speakers | 0.602 | 0.033 | 0.996 | +3.5 LU | +0.8 dB |
| producer (cleanup) | 0.424 | 0.014 | 0.998 | +4.3 LU | -2.6 dB |
| producer (natural) | 0.583 | 0.023 | 0.997 | +4.6 LU | -1.5 dB |
| engineer's version | | | | +4.3 LU | -3.2 dB |

The producer was closer to the target than each of the three baselines on all 20 clips: by 0.368 on average against pass-through, 0.236 against gain only and 0.179 against the static EQ. The gain-only baseline uses each clip's target to set the gain, which favours it. None of the baselines compresses, so this does not show the producer beating a fixed chain that includes a compressor. Log-magnitude L1 is also the producer's main training loss, alongside a multi-resolution STFT loss.

The training script chose each producer's best checkpoint by this distance on the script-1 clips of the held-out speakers, which are 4 of these 20. On the other 16 (`heldout_stats.py --producer-skip _script1_`) the cleanup model scored 0.430, against 0.793 for pass-through, 0.661 for gain only and 0.605 for the static EQ, and was closer than all three on every clip.

The "natural" variant takes the de-breathed studio take as its training input, with the same target. Input and target both lack breaths, so nothing in its training teaches breath removal. At inference it is fed the unprocessed take, breaths included. Its distance to a target with the breaths removed is larger by design. `scripts/validate_breath_preservation.py` measures whether the breaths survive; its result at the time was recorded in working notes, not saved as a run output.

## 5. The full-band restorer

Nothing in the DAPS conditions removes high frequencies, and a restorer fine-tuned on them never has to learn bandwidth extension. A second fine-tune started again from the off-the-shelf Resemble Enhance weights and trained on VCTK and Hi-Fi TTS, which are recorded at 44.1 kHz or above. The main synthetic degradations were EQ damage and bandwidth limiting. An automatic quality filter scored 368,433 candidate training clips and dropped 14,520 (3.9%). The best checkpoint was at step 6,000, with validation loss down from 0.386 to 0.234.

On 16 held-out clips (four speakers, each under EQ damage, "radio voice", bandwidth limiting and a broadband mix, 30 s):

| Measure | Off-the-shelf | Fine-tuned |
|---|---|---|
| WER | 0.135 | 0.056 |
| 8-16 kHz energy added over the damaged input | 11.1 dB | 15.1 dB |
| speaker similarity | | 0.986 |

The untreated input's WER was not recorded in this run, nor was the off-the-shelf model's speaker similarity. One held-out speaker scored 0.144 with both models and accounts for most of the remaining mean.

When I listened to the held-out outputs they sounded unnatural and unlike their clean targets. This evaluation measured transcripts, voice identity and band energy, not naturalness. My working explanation, which I did not test directly: the model is conditioned on the damaged input's spectrum, so colouration in the input survives into the output.

## 6. Comparing other restorers

The same reference clips went through off-the-shelf Resemble Enhance and VoiceFixer, loudness-matched for listening. On three synthetic VCTK speakers with bandwidth limiting, the energy in the 8-16 kHz band relative to the 100 Hz-8 kHz speech band:

| Speaker | Untreated | Off-the-shelf Resemble Enhance | VoiceFixer | Clean original |
|---|---|---|---|---|
| p225 | -38.4 dB | -20.1 dB | -7.5 dB | -7.5 dB |
| p226 | -41.5 dB | -24.9 dB | -15.8 dB | -15.1 dB |
| p227 | -53.9 dB | -30.8 dB | -24.6 dB | -25.9 dB |

VoiceFixer also raised the top octave of the clean originals, by 4.4 dB (p225), 0.9 dB (p226) and 1.5 dB (p227). To my ear it regenerated more voice-like content from damaged input than Resemble Enhance did, with too much top end. I stopped before choosing between the high-shelf settings in `src/voice_cleanup/eval/tame.py`, which include one that scales the cut with the input's brightness.

The models I would try next are AnyEnhance, and AudioSR as a stage after VoiceFixer. Miipher and FINALLY had no open weights when I checked in May 2026.

## A note on the code comments

Some code comments refer to working notes (a decision log, a learning log and handoff notes) that are not included in this repository.
