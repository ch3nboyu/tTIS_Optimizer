# An optimization and simulation system for transcranial direct current stimulation and temporal interference stimulation based on personalized modeling

‌

## Description

Since the introduction of transcranial temporal interference stimulation(tTIS), there has been an ever-growing interest in this novel method, as it theoretically allows non-invasive stimulation of deep brain target regions.

However, there has been a pressing challenge for finding optimal electrode combinations for stimulating region of interest(ROI). Most of prebvious methods use exhaustive search to find the best match, but faster and, at the same time, reliable solutions are required.

In this study, the electrode combinations as well as the injected current for a two-electrode pair stimulation were optimized using a genetic algorithm(GA), considering the right hippocampus as the region of interest (ROI).

## Installation

1. Install dependent packages

```bash
conda env create simNIBS_environment_win.yml
simnibs_env
pip install -f https://github.com/simnibs/simnibs/releases/latest simnibs
```

2. Prepare data

Download [data](https://github.com/simnibs/example-dataset/releases/latest/download/simnibs4_examples.zip)
