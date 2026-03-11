# fMRI-Derived Spatial Priors

This directory contains fMRI-guided spatial attention maps that guide the Bio-CBAM model towards neurobiologically relevant facial regions for emotion recognition.

## Prior Files

### Emotion-Specific Priors

- `prior_angry.npy`: Prior for angry emotion (emphasis on eyes and eyebrows)
- `prior_happy.npy`: Prior for happy emotion (emphasis on mouth and eyes)
- `prior_sad.npy`: Prior for sad emotion (emphasis on eyes and mouth)
- `prior_fear.npy`: Prior for fear emotion (emphasis on eyes)
- `prior_neutral.npy`: Prior for neutral emotion (balanced across regions)
- `prior_disgust.npy`: Prior for disgust emotion (emphasis on nose and mouth)
- `prior_surprise.npy`: Prior for surprise emotion (emphasis on eyes and mouth)

### Combined Priors

- `prior_combined.npy`: Combined prior from all facial regions
- `prior_facial.npy`: General facial prior emphasizing emotion-relevant regions

## Generating Priors

To generate fMRI priors:

```bash
python generate_fmri_priors.py
```

This will create all prior files in the `priors/` directory.

## Using Priors in Training

The priors are automatically loaded and used during model training:

```python
from models import create_bio_cbam
import numpy as np

# Load prior
prior = np.load('priors/prior_happy.npy')

# Create model with fMRI guidance
model = create_bio_cbam(num_classes=7, use_fmri_prior=True)

# Use in forward pass
logits, attention_maps = model(images, landmarks=None)
```

## Neurobiological Basis

The priors are based on fMRI studies showing:

1. **Amygdala activation**: Responds to emotional faces, particularly fear and anger
2. **Orbitofrontal cortex**: Processes emotional value and decision-making
3. **Insula**: Processes disgust and emotional awareness
4. **Superior temporal sulcus**: Processes biological motion and facial expressions

## File Format

Each `.npy` file contains:
- Shape: (224, 224)
- Data type: float32
- Range: [0, 1] (normalized)
- Values: Spatial attention weights for each pixel

## Customization

To customize priors, modify `generate_fmri_priors.py`:

```python
# Adjust region centers and sigmas
FACIAL_REGIONS = {
    'eyes': {'center': (0.5, 0.35), 'sigma': 0.15},
    'mouth': {'center': (0.5, 0.65), 'sigma': 0.12},
    ...
}

# Adjust emotion-specific weights
emotion_weights = {
    'angry': {'eyes': 0.5, 'eyebrows': 0.3, 'mouth': 0.2},
    ...
}
```

## References

1. Adolphs, R. (2002). Neural systems for recognizing emotion. Current Opinion in Neurobiology
2. Haxby, J. V., et al. (2000). The functional architecture of human object recognition. PNAS
3. Vuilleumier, P., & Driver, J. (2007). Modulating spatial attention by emotion. Current Biology
