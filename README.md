# network-intrusion-detection
Use network traffic features to classify whether traffic is benign or malicious, and optionally identify the attack category. Public benchmark datasets like NSL-KDD and CICIDS2017 are widely used for this kind of work. NSL-KDD was created to address some of the major redundancy problems in KDD’99, and CICIDS2017 was built with more modern attack scenarios and extracted flow features.

## Project structure

```text
data/
	KDDTrain+.txt
	KDDTest+.txt
notebooks/
	nsl_kdd_mlp.ipynb
report/
	final_report.tex
results/
	generated model, metric, and plot files
src/
	columns.py
	dataset.py
	download_data.py
	evaluate.py
	evaluate_autoencoder.py
	model.py
	autoencoder.py
	preprocess.py
	train.py
	train_autoencoder.py
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run order

Run these commands from the project root.

1. Download the NSL-KDD dataset:

```bash
python src/download_data.py
```

2. Preprocess the data:

```bash
python src/preprocess.py
```

This creates `results/preprocessor.joblib` locally.

3. Train the supervised MLP baseline:

```bash
python src/train.py
```

This creates `results/mlp_model.pt` and `results/learning_curve.png` locally.

4. Evaluate the supervised MLP:

```bash
python src/evaluate.py
```

This creates `results/metrics.txt` and `results/confusion_matrix.png` locally.

5. Train the autoencoder:

```bash
python src/train_autoencoder.py
```

This creates `results/autoencoder_model.pt` and `results/autoencoder_learning_curve.png` locally.

6. Evaluate the autoencoder:

```bash
python src/evaluate_autoencoder.py
```

This creates `results/autoencoder_metrics.txt` and `results/autoencoder_confusion_matrix.png` locally.

## File guide

- `src/download_data.py` downloads the NSL-KDD training and testing files.
- `src/columns.py` stores the NSL-KDD column names because the raw text files do not include headers.
- `src/preprocess.py` loads the data, converts labels to binary labels, one-hot encodes categorical features, scales numeric features, and saves the fitted preprocessor.
- `src/dataset.py` converts processed arrays into a PyTorch dataset.
- `src/model.py` defines the supervised MLP classifier.
- `src/train.py` trains the supervised MLP and saves the trained model plus learning curve.
- `src/evaluate.py` evaluates the supervised MLP and saves metrics plus a confusion matrix.
- `src/autoencoder.py` defines the autoencoder model.
- `src/train_autoencoder.py` trains the autoencoder on normal traffic.
- `src/evaluate_autoencoder.py` evaluates the autoencoder using reconstruction error.

## Generated files

Dataset files and model outputs are generated locally and ignored by Git. Recreate them by running the commands above.
