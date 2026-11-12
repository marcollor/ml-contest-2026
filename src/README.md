# Training TemporalCNN

Use Python 3.11+ and run the commands below from the repository root.
The trainer uses robust per-channel preprocessing, 80 epochs, batch size 256,
seed 20261114, AdamW with cosine decay, inverse-frequency class weights, and
25% clean samples. It selects the checkpoint by maximizing the lower of the
clean and perturbed development scores.

```bash
python -m pip install -r src/requirements.txt
python src/prepare_data.py
python src/train.py --check-only
python src/train.py
```

Training saves the selected weights, recipe, file fingerprints, and runtime
versions in `model/retrained.pt`. Export and evaluate the new checkpoint on
Linux or WSL with a C compiler available:

```bash
python -m tools.export_onnx --trained
python -m tools.export_c --trained
python src/evaluate.py --trained
```
