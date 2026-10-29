# Heart Disease Neural Network in Dart

A small educational binary-classification project that implements a fully
connected neural network and backpropagation **from scratch in Dart**. It trains
on 11 clinical features to predict whether a record is labelled with heart
disease.

> This software is a university demonstration, not a medical device. Its output
> must not be used for diagnosis or treatment decisions.

## What the project demonstrates

- Deterministic CSV loading, categorical encoding, and an 80/20 train/test split
- Min-max scaling fitted only on training data to avoid test-data leakage
- Xavier initialization, sigmoid activations, binary cross-entropy, and SGD
- Generic backpropagation through one or more hidden layers
- Reproducible experiments through a configurable random seed
- Unit tests for data preparation and neural-network learning

## Project structure

```text
bin/train.dart                 Training command-line program
lib/src/dataset.dart           CSV parsing, encoding, splitting, normalization
lib/src/neural_network.dart    Feed-forward network and backpropagation
test/                          Automated tests
heart.csv                      Heart-disease dataset (918 records)
```

## Requirements and setup

Install Dart SDK 3.3 or newer, then run:

```bash
dart pub get
dart run bin/train.dart
```

Optional arguments:

```bash
dart run bin/train.dart --epochs 300 --seed 42 --data heart.csv
```

The command prints average training loss plus train and test accuracy every ten
epochs. The seeded split and initialization make the same command reproducible.

## Test and analyze

```bash
dart format --output=none --set-exit-if-changed .
dart analyze
dart test
```

## Dataset

The CSV contains `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`,
`FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, and
the `HeartDisease` label. Before publishing or submitting the project, add the
dataset's original source URL and license here if your copy came from an
external repository.

## Limitations

- Accuracy from one train/test split is not clinical validation.
- The current output supports binary classification only.
- The parser expects this simple CSV schema and does not handle quoted fields.
- A formal report should also evaluate a confusion matrix, precision, recall,
  F1 score, and cross-validation.

## License

Source code is available under the [MIT License](LICENSE). The dataset may have
separate terms from its original provider.
