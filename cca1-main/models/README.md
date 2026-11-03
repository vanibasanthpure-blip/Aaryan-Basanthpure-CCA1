# Models

Trained model artifacts are not committed (see `.gitignore`) -- they are
regenerated deterministically (fixed random seed, see `src/utils.py:SEED`) by:

```bash
python src/train.py --approach all
```

This produces:

- `svm_rr_hrv.joblib` -- scikit-learn Pipeline (StandardScaler + SVC)
- `rf_dwt.joblib` -- scikit-learn RandomForestClassifier
- `cnn_bilstm.pt` -- PyTorch `state_dict` for `src.model_cnn_bilstm.CNNBiLSTM`

`src/predict.py` and `app/streamlit_app.py` load whichever of these three
files are present.
