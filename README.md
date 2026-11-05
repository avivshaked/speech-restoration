# Voice Cleanup

Experiments in restoring speech recorded on consumer devices in ordinary rooms so it sounds like a studio recording. I fine-tuned an open generative restoration model, trained a small mastering model from scratch, and built an evaluation harness to judge both. The fine-tuned model copes with noisy, reverberant recordings from phones and tablets but not with severe bandwidth loss (the "old radio voice"). That is where the project stopped.

## Where it stands

### What works

- The restorer is [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) fine-tuned on [DAPS](https://zenodo.org/records/4660670), which has the same scripts recorded in a studio and on phones and tablets in real rooms. Four speakers were held out. On 12 held-out clips, mean Whisper word error rate (WER) fell from 0.273 for the off-the-shelf model to 0.102 after about half an epoch. Training on to six epochs moved it between 0.067 and 0.219 without a trend.
- The producer is a 2.79M-parameter spectral U-Net, trained from scratch to reproduce a studio engineer's processing (EQ, compression, loudness, breath removal) on clean speech. On held-out speakers it raised loudness by 4.1 LU with WER 0.013 and speaker similarity 0.998. I preferred its output to the engineer's master it was trained on.
- The evaluation harness covers PESQ, ESTOI, SI-SDR, DNSMOS, SQUIM, NISQA, Whisper WER, speaker similarity and spectral balance. It has a registry for swapping restoration models in, and every run writes a loudness-matched A/B listening pack.

### What doesn't

- The restorer only ever saw DAPS conditions: background noise, room reverberation and device colouration. It was not trained on bandwidth loss, clipping or codec damage.
- A rebuild on full-band speech (VCTK and Hi-Fi TTS) with synthetic bandwidth and EQ damage improved WER and high-frequency recovery but failed the listening test (below).
- Restoration costs some intelligibility. On real DAPS device recordings the untreated audio already transcribed almost perfectly (WER 0.00 to 0.03), and the off-the-shelf restorer scored the same or worse on every clip. The fine-tuning runs didn't record the untreated input's WER, so I can't say whether the fine-tuned model beats doing nothing on that measure.

### Where it stopped

Comparing other open restorers on bandwidth-limited speech. [VoiceFixer](https://github.com/haoheliu/voicefixer) came closest. On a synthetic bandwidth-limited VCTK clip it brought the 8-16 kHz energy from -38.4 dB to -7.5 dB, level with the clean original, where off-the-shelf Resemble Enhance reached -20.1 dB. It also brightens speech that was already clean (by up to 4.4 dB), and it sounds too bright. I added a high-shelf filter to tame it and didn't take the comparison further.

## What the metrics missed

On moderately damaged synthetic speech, off-the-shelf Resemble Enhance had a WER of 0.773, against 0.059 for the untreated input. NISQA, which predicts a listener's score, gave the same output 4.01 out of 5.

The full-band rebuild was Resemble Enhance fine-tuned on VCTK and Hi-Fi TTS. On held-out speakers WER fell from 0.135 to 0.056, high-frequency recovery rose from 11.1 to 15.1 dB, and speaker similarity was 0.986. The same held-out samples did not sound natural or like the clean targets.

Neither WER, which compares transcripts, nor speaker similarity, which compares voice embeddings, measures naturalness. The MOS predictors were trained on other kinds of damage, and on this material they scored hallucinated speech highly. My best explanation for the rebuild is that Resemble Enhance conditions on the damaged input's spectrum, so it rebuilds the colouration it is given. Fixing severe damage probably needs a model with a stronger prior over speech, and the strongest published ones (Miipher, FINALLY) had no open weights when I checked in May 2026.

Every run writes a loudness-matched A/B pack, and a listening pass decided whether a model moved forward. All the listening was mine. I worked as a sound designer for seventeen years before moving into software, but this was still one listener with no formal protocol. Sample sizes are small throughout: 4 to 16 clips, one seed. The numbers and their caveats are in [docs/findings.md](docs/findings.md).

## Running the evaluation

Needs Linux or WSL2, an NVIDIA GPU (developed on a 24 GB RTX 4090), Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/avivshaked/speech-restoration.git
cd speech-restoration
uv sync
uv run pytest -m "not slow"

# Clean reference speech (EARS subset, about 6 GB) and third-party models, fetched from source
uv run python scripts/fetch_clean_corpora.py --ears-only
uv run python scripts/fetch_mpsenet.py
uv run python scripts/fetch_resemble_enhance.py
uv run python scripts/fetch_nisqa.py

# Degrade clean speech, restore it with each model, score against the original
uv run python scripts/run_restoration_scoreboard.py --recipe moderate --limit 6
```

One fast test fails unless the system `libaio` library is installed, because DeepSpeed compiles against it. The tests that need real audio skip until `scripts/download_test_set.py` has run.

Scores, a markdown scoreboard and the A/B listening pack go to `runs/restoration-scoreboard/<timestamp>/`. [docs/data.md](docs/data.md) lists each dataset with its use, size, licence and fetch script. The trained checkpoints are not published, so reproducing the fine-tuned results means retraining (`scripts/finetune_re_daps.py`, `scripts/train_producer.py`).

## Data, models and licences

This repository contains code and documentation only. No audio, datasets or model weights are included. The scripts download them from their original sources, and each carries its own licence. Several are non-commercial: DAPS and EARS (CC BY-NC 4.0), the NISQA weights (CC BY-NC-SA 4.0) and the SQUIM model (CC BY-NC 4.0). Running the full pipeline is therefore non-commercial in practice, whatever the licence on this code. PESQ's ITU-T terms also restrict commercial use of its scores. Details and attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How it was built

Built with Claude Code (Anthropic). Claude wrote most of the code and documentation. I chose the experiments and judged the results by listening.

## Licence

The code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). Third-party code adapted here keeps its original licence; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
