# speech-restoration

Experiments in making speech recorded on phones and tablets in ordinary rooms sound like a studio recording. There are two models, trained and evaluated separately. The restorer is Resemble Enhance, an open generative speech-restoration model from Resemble AI, fine-tuned to clean up device recordings. The producer is a small model I trained from scratch to reproduce a studio engineer's post-production on clean speech. An evaluation harness scores both. Neither restorer I trained handles severe bandwidth loss (the "old radio voice").

The project's working name was Voice Cleanup, and the Python package is still `voice_cleanup`.

## Where it stands

### What works

- The producer came closer to a studio engineer's post-production than any of three simple baselines, on every one of 20 clips from speakers it never trained on. It is a 2.79M-parameter spectral U-Net trained on [DAPS](https://zenodo.org/records/4660670), a public dataset of the same scripts recorded in a studio and on phones and tablets in real rooms, with an engineer's finished version of each studio take (EQ, compression, loudness, breath removal). Its log-magnitude distance to that finished version was 0.424; the unprocessed input scored 0.792, a loudness match 0.660 and a fixed EQ plus gain fitted on the training speakers 0.602. It matched the target's loudness (+4.3 LU) and most of its compression: crest factor, the peak-to-average ratio that compression lowers, fell 2.6 dB, against 3.2 dB in the engineer's version.
  - None of those baselines compresses, and the distance measure is also the producer's training loss. The checkpoint was chosen on 4 of the 20 clips; on the other 16 the figures barely move ([findings](docs/findings.md#4-the-producer)).
- Fine-tuning fixed most of the words off-the-shelf [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) got wrong. The restorer was fine-tuned on the DAPS device recordings, with the unprocessed studio take as target. On 48 clips from the same four held-out speakers, across all 12 device and room conditions, mean Whisper word error rate (WER) fell from 0.089 to 0.038. Those figures come from the last checkpoint, which no evaluation clip helped choose.
- The evaluation harness runs intrusive scores (PESQ, ESTOI, SI-SDR), MOS predictors (DNSMOS, SQUIM, NISQA), Whisper WER, speaker similarity and spectral balance. Restoration models plug in through a registry, and every run writes a loudness-matched A/B listening pack. The held-out evaluation above uses WER, speaker similarity (a WavLM speaker-verification model) and, for the producer, distance to the target; [findings](docs/findings.md) sections 1 and 2 report DNSMOS, PESQ and NISQA from earlier runs.

### What doesn't

- Neither the off-the-shelf nor the fine-tuned model made the DAPS speech easier to transcribe than the untreated recordings, which already had a mean WER of 0.028. In the other ten conditions, all indoors, the fine-tuned model was level with them (0.020 and 0.023). On the two balcony conditions it was worse on all 8 clips (0.126 and 0.054).
- Beyond voice identity, nothing in this evaluation measures how close the restored audio comes to the studio take. The intrusive scores assume the output keeps the reference's waveform, which a generative model that resynthesises the audio does not, so this evaluation leaves them out.
- The restorer was fine-tuned only on DAPS conditions: background noise, room reverberation and device colouration. It was not trained on bandwidth loss, clipping or codec damage.
- A second restorer, trained on full-band speech (VCTK and Hi-Fi TTS, public studio-quality speech datasets) with synthetic bandwidth and EQ damage, improved on the off-the-shelf model's WER and high-frequency recovery but failed my listening check (below).

### Unfinished

Comparing other open restorers on bandwidth-limited speech. Three synthetic bandwidth-limited VCTK clips started 26 to 31 dB short of their clean originals' 8-16 kHz energy. [VoiceFixer](https://github.com/haoheliu/voicefixer) brought all three to within 1.3 dB of the originals; off-the-shelf Resemble Enhance stayed 5 to 13 dB short. VoiceFixer also brightens speech that was already clean, by up to 4.4 dB, and sounds too bright. I wrote a high-shelf filter to tame it and stopped before settling its settings.

## Metrics and listening

On moderately damaged synthetic speech (6 clips), off-the-shelf Resemble Enhance had a WER of 0.773, where the untreated input had 0.059. NISQA, which predicts a listener's score on a scale of 1 to 5, gave the same output 4.01.

The full-band restorer was Resemble Enhance fine-tuned on VCTK and Hi-Fi TTS. On 16 held-out synthetic clips its WER was 0.056, and the off-the-shelf model's was 0.135. It added 15.1 dB of 8-16 kHz energy over the damaged input, where the off-the-shelf model added 11.1 dB. The untreated input's WER was not recorded in that run. The same held-out samples did not sound natural or like the clean targets.

That evaluation measured transcripts, voice identity and band energy, and none of those checks naturalness. The MOS predictors are meant to, but they were trained on other kinds of damage, and on this material they scored hallucinated speech highly. My best explanation for the full-band restorer is that Resemble Enhance conditions on the damaged input's spectrum and rebuilds the colouration it is given. Fixing severe damage probably needs a model with a stronger prior over speech, and the strongest published restorers I found (Miipher, FINALLY) had no open weights when I checked in May 2026.

A listening pass decided whether a model moved forward. All the listening was mine (seventeen years as a sound designer), unblinded and with no formal protocol. The held-out evaluation above used 48 restorer clips and 20 producer clips; most earlier runs used 4 to 16. The numbers and their caveats are in [docs/findings.md](docs/findings.md).

## Trained models

The fine-tuned DAPS restorer and two producers are on Hugging Face at [shakeda/speech-restoration-models](https://huggingface.co/shakeda/speech-restoration-models), as safetensors files. `producer-cleanup` is the producer described above. `producer-natural` was trained with the de-breathed take as input, so it should leave breaths in. They are licensed CC BY-NC 4.0 because they were trained on DAPS. `src/voice_cleanup/infer/pretrained.py` loads them.

## Running the evaluation

Needs Linux or WSL2, an NVIDIA GPU (developed on a 24 GB RTX 4090), Python 3.11, [uv](https://docs.astral.sh/uv/) and the system `libaio` library, which DeepSpeed compiles against (on Ubuntu, `sudo apt install libaio-dev`).

```bash
git clone https://github.com/avivshaked/speech-restoration.git
cd speech-restoration
uv sync
uv run pytest -m "not slow"

# Third-party models, fetched from their sources
uv run python scripts/fetch_mpsenet.py
uv run python scripts/fetch_resemble_enhance.py
uv run python scripts/fetch_nisqa.py

# Held-out evaluation of the trained models: needs DAPS (about 16 GB) and the weights
uv run python scripts/fetch_clean_corpora.py --daps-only
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('shakeda/speech-restoration-models', local_dir='models')"
uv run python scripts/eval_heldout.py \
    --restorer last=models/restorer-daps/model.safetensors \
    --producer cleanup=models/producer-cleanup/model.safetensors
uv run python scripts/heldout_stats.py runs/heldout-eval/<timestamp>/results.json

# Synthetic-damage scoreboard on a subset of EARS, a studio speech dataset (about 6 GB);
# "moderate" is a mid-strength mix of the degradations in src/voice_cleanup/data/degrade.py
uv run python scripts/fetch_clean_corpora.py --ears-only
uv run python scripts/run_restoration_scoreboard.py --recipe moderate --limit 6
```

Results go to `runs/`. The held-out evaluation took 42 minutes on the RTX 4090. [docs/data.md](docs/data.md) lists each dataset with its use, size, licence and fetch script.

Known issues:

- The locked environment installs two cuDNN 9 builds, PyTorch's own and the one faster-whisper needs. PyTorch's GPU convolutions then fail with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. `scripts/eval_heldout.py` sets `torch.backends.cudnn.enabled = False` to avoid this. It is the only script re-run in this environment, and other GPU scripts may need the same line.
- The tests that need real audio skip until the podcast test set has been downloaded with `scripts/download_test_set.py` (see [docs/data.md](docs/data.md)).

## Data, models and licences

This repository contains code and documentation only. No audio, datasets or model weights are included. The scripts download them from their original sources, and each carries its own licence. Several are non-commercial: DAPS and EARS (CC BY-NC 4.0), the NISQA weights (CC BY-NC-SA 4.0), the SQUIM model (CC BY-NC 4.0) and the weights trained here (CC BY-NC 4.0). Running the full pipeline is therefore non-commercial in practice, whatever the licence on this code. PESQ's ITU-T terms also restrict commercial use of its scores. Details and attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How it was built

Built with Claude Code (Anthropic). Claude wrote most of the code and documentation. I chose the experiments and judged the results by listening.

## Licence

The code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). Third-party code adapted here keeps its original licence; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
