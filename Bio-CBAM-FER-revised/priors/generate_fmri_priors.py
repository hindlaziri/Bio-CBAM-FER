"""Backward-compatible CLI for the documented fMRI-to-face TPS pipeline.

This command does not fabricate "fMRI-derived" maps from hard-coded Gaussian
face regions. It requires an activation map and an explicit correspondence CSV.
Run ``python -m priors.generate_fmri_priors --help`` for arguments.
"""

from .fmri_pipeline import main


if __name__ == "__main__":
    main()
