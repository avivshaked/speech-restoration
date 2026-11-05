# Third-party notices

This repository's own code and documentation are licensed under the Apache License 2.0 (see `LICENSE`). It contains no third-party audio, datasets or model weights. The scripts download those from their original sources at run time, and they remain under their own licences, listed below. Licences were checked against the upstream source in November 2026 where marked "confirmed". Check the source yourself before relying on any of them.

## Code adapted into this repository

Short passages adapted from the projects below are marked in the source with a comment naming the upstream. Each is used under the MIT licence, whose notice is reproduced at the end of this file.

| Upstream | Where it is used here | Copyright notice |
|---|---|---|
| [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) (`DNSMOS/dnsmos_local.py`) | `src/voice_cleanup/eval/non_intrusive.py`: DNSMOS windowing and polynomial calibration constants | Copyright (c) Microsoft Corporation. |
| [yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet) (`inference.py`) | `src/voice_cleanup/eval/mpsenet.py`: input RMS normalisation | Copyright (c) 2023 Yexin Lu |
| [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse) (`enhancement.py`) | `src/voice_cleanup/eval/sgmse_enh.py`: inference call sequence | Copyright (c) 2022 Signal Processing (SP), Universität Hamburg |
| [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | `scripts/fetch_resemble_enhance.py`: one line quoted as a patch target for NumPy 2 compatibility | Copyright (c) 2023 Resemble AI |

## Models and code downloaded at run time

| Component | Used for | Licence | Status |
|---|---|---|---|
| Resemble Enhance (code and checkpoint) | generative restorer | MIT (code); MIT (Hugging Face model card) | confirmed |
| MP-SENet (code and checkpoints) | noise-suppression baseline | MIT (code); checkpoints not stated | code confirmed |
| VoiceFixer 0.1.3 (PyPI) and its checkpoints | restorer comparison | MIT (code); checkpoints not stated | code confirmed |
| SGMSE+ | restorer comparison (not run) | MIT (code); checkpoints not stated. One CUDA kernel in the upstream repo is under the NVIDIA Source Code License-NC | code confirmed |
| NISQA | predicted quality score | MIT (code); **CC BY-NC-SA 4.0 (weights)** | confirmed |
| DNSMOS P.835 ONNX models | predicted quality score | repository code MIT; models not separately stated | partly confirmed |
| SQUIM (torchaudio) | predicted quality score | **CC BY-NC 4.0 (weights)** | confirmed in torchaudio source |
| Whisper large-v3 via faster-whisper | word error rate | MIT (faster-whisper conversion); Apache-2.0 (upstream model card) | confirmed |
| WavLM base plus SV (Microsoft) | speaker similarity | points to the UniSpeech licence | not checked |
| PESQ (`pesq` on PyPI) | intrusive quality score | package says MIT, but bundles ITU-T P.862 code whose notice restricts redistribution and commercial use of results | confirmed in package source |
| ffmpeg static build (via imageio-ffmpeg) | codec degradation | GPL v3 binary, called as a separate process and never distributed here | confirmed |

## Datasets downloaded at run time

| Dataset | Used for | Licence | Status |
|---|---|---|---|
| [DAPS](https://zenodo.org/records/4660670) | restorer fine-tuning and evaluation, producer training | **CC BY-NC 4.0** | confirmed |
| [EARS](https://github.com/facebookresearch/ears_dataset) | clean reference for the scoreboard | **CC BY-NC 4.0** | confirmed |
| [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) | full-band rebuild, synthetic comparison clips | CC BY 4.0 (licence file on DataShare) | confirmed |
| [Hi-Fi TTS](https://www.openslr.org/109/) | full-band rebuild | CC BY 4.0 | confirmed |
| archive.org podcast clips (`manifests/podcast-real-candidates.json`) | real-world, no-reference test set | per item, as recorded in the manifest; most are CC BY-NC-ND | per uploader metadata |
| archive.org old-time radio (`scripts/fetch_pd_radio_refs.py`) | real bandwidth-limited references | Public Domain Mark or CC0, per item | per uploader metadata |

Weights trained on DAPS or EARS inherit their non-commercial terms. No such weights are published here.

## MIT licence text

The following notice applies to the adapted passages listed under "Code adapted into this repository", with the copyright line given for each upstream above.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
