# Intermediate dataset validation

Validate the detection-driven output structure, timestamps, calibration, ego poses, and labels.

```bash
python -m validation.intermediate.validate_intermediate --stage1 /path/to/dataset
```

The validator writes a YAML report and diagnostic figures into the dataset directory.
