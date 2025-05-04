# GGFAI Data Directory

This directory contains data files used by the GGFAI framework:

## Files

- `active_learning.txt`: Contains uncertain predictions flagged for manual review by the intent engine. These can be used to improve model performance over time.

## Usage

The data in this directory is used by various components of the GGFAI framework:

1. **Intent Engine**: Uses `active_learning.txt` to log uncertain predictions for later review and model improvement.

2. **Model Adapter**: May use this directory to store model-specific data or cached results.

3. **Analytics Tracker**: May store analytics data in this directory.

## Adding New Data

When adding new data files to this directory, please follow these guidelines:

1. Use clear, descriptive filenames
2. Include a header comment explaining the file's purpose and format
3. Update this README.md with information about the new file