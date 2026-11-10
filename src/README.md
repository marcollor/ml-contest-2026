# Training adaptative_model

Run `python src/prepare_data.py`, then `python src/train.py` from the repository
root. The trainer uses the fixed V024 reconstruction with 80 epochs, batch
size 256, seed 20261114, AdamW with cosine decay, inverse-frequency class weights,
and 25% clean samples. Clean and perturbed development scores select the epoch
with the best lower score. Run `python src/train.py --check-only` to check setup.

The output is `model/retrained.pt`, which contains the selected weights, recipe,
file fingerprints and runtime versions. The history records loss, accuracy
and score per epoch, with class recall for the selected epoch. Export and
evaluate the selected checkpoint with:

```bash
python -m tools.export_onnx --trained
python -m tools.export_c --trained
python src/evaluate.py --trained
```
