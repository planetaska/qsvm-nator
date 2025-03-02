# First Look at ESC Dataset

These scripts will help introduce you to the ESC dataset
(Environmental Sound Classification).

## Instructions
Blindly run the scripts in this order:
1. `esc_download.py`:
   - fetches and arranges audio files (into ESC10 and 50)
2. `esc_feature_explorer.py`:
   - sample a random audio sample
   - visualize possible features
3. `esc_feature_extractor.py`:
    - pick specific features for dataset
    - to start... pick ESC10, all features, both mean & std
    _(this may take a few minutes)_
4. `esc_feature_analysis_svm.py`:
    - use dataset with svm
    - view feature importance

## Next

### Choose Best Features

Looking at the feature importance (essentially, magnitude of weights)
by fitting an SVM
gives us a single perspective of which features matter most.

### ESC10 vs ESC50

The most important features will likely be different.
It may be sufficient to make a generalization based on ESC10.

### Empirically Evaluate

Either way, to really know we must empirically evaluate.

This would mean combinatorically evaluating all features against test accuracy and other metrics.

After that, we'd have to make a judgement call with respect to model complexity versus performance.