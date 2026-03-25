# Run one experiment

```bash
python train.py --reward mixture --grid-size 20 --steps 1000 --out-dir outputs/mixture
```

# Distillation
```bash
python distill.py   --checkpoint outputs/mixture/checkpoint.pt   --steps 300  --f-warmup-steps 0  --f-updates-per-step 25  --out-dir outputs/distilled_mixture
```
