# Data

Nothing in this repository is audio. Every dataset is downloaded from its original source by a script, into `data/`, which git ignores. Check each licence at the source before use; [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) records what was confirmed and when.

| You want to | Dataset | Download size | Licence | Fetch with |
|---|---|---|---|---|
| Run the restoration scoreboard | [EARS](https://github.com/facebookresearch/ears_dataset), 10-speaker evaluation subset (48 kHz, anechoic) | about 6 GB (about 592 MB per speaker; all 107 speakers about 63 GB) | CC BY-NC 4.0 | `scripts/fetch_clean_corpora.py --ears-only` |
| Fine-tune the restorer, train the producer | [DAPS](https://zenodo.org/records/4660670): the same scripts recorded in a studio and on devices in real rooms | about 16 GB | CC BY-NC 4.0 | `scripts/fetch_clean_corpora.py --daps-only` |
| Reproduce the full-band rebuild | [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) (48 kHz, 110 speakers) | about 11 GB | CC BY 4.0 | `scripts/fetch_tier1_corpora.py --vctk-only` |
| | [Hi-Fi TTS](https://www.openslr.org/109/) (44.1 kHz, 10 speakers) | about 41 GB | CC BY 4.0 | `scripts/fetch_tier1_corpora.py --hifi-only` |
| Compare restorers on synthetic damage | three VCTK speakers, synthetically degraded | needs VCTK | CC BY 4.0 | `scripts/fetch_bazaar_refs.py` |
| Compare restorers on real bandwidth loss | four public-domain old-time-radio clips from archive.org | small | Public Domain Mark or CC0, re-checked at download | `scripts/fetch_pd_radio_refs.py` |
| Score real podcast audio with no clean reference | 10 Creative Commons podcast clips from archive.org | small | per item, mostly CC BY-NC-ND | `scripts/download_test_set.py` |

## The podcast test set

`scripts/download_test_set.py` reads `manifests/podcast-real-candidates.json`, downloads each item marked `kept`, and cuts a 20 s segment from the middle. The manifest records each item's archive.org identifier, URL and licence as the uploader declared it. `scripts/fetch_test_set_candidates.py` regenerates a manifest from archive.org's live search, but search results change over time, so a regenerated set will differ.

Most of these clips are CC BY-NC-ND. They can be used for evaluation, but processed versions of them should not be published.

## Model weights

The evaluation also downloads third-party model weights. The fetch scripts for Resemble Enhance, MP-SENet and NISQA pin an upstream commit and check SHA-256 hashes, and the DNSMOS model is hash-checked on download. Whisper, WavLM, SQUIM and VoiceFixer weights are fetched by their own libraries on first use. Their licences are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The weights trained in this project are not published.
