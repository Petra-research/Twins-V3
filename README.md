Developmental Alignment: HRLS vs Pure Emergence

100 twin-pair experiments studying affective divergence in continuous neural field systems.

**Paper:** [EA Forum](https://forum.effectivealtruism.org/posts/y9XP2B23yGorKz2Wy/shared-dynamics-divergent-feelings-100-hrls-vs-pure-surge)

## Key Finding
- **Activity correlation:** r≈0.21 (stable coupling)
- **Emotion correlation:** r≈0.008 (divergent temperament)

Systems with identical architecture develop different emotional signatures before any reward/suppression training.

## Architecture
- 10k-neuron continuous cortex field
- Oja plasticity (normalized Hebbian)
- Energy/sleep dynamics
- HRLS developmental scaffolding vs pure emergence

## Run
```bash
python twins_v3_continuous.py --mode free --steps 50000
