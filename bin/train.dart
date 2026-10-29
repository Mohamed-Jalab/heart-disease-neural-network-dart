import 'dart:io';
import 'dart:math';

import 'package:heart_nn/heart_nn.dart';

void main(List<String> arguments) {
  final epochs = _integerOption(arguments, '--epochs', fallback: 200);
  final seed = _integerOption(arguments, '--seed', fallback: 42);
  final datasetPath = _stringOption(arguments, '--data', fallback: 'heart.csv');
  final split = HeartDataset.splitAndNormalize(
    HeartDataset.load(datasetPath),
    seed: seed,
  );
  final network = NeuralNetwork(
    layerSizes: [HeartDataset.featureNames.length, 12, 1],
    learningRate: 0.1,
    seed: seed,
  );
  final random = Random(seed);

  stdout.writeln(
    'Training on ${split.training.length} samples; '
    'testing on ${split.testing.length} samples.',
  );
  for (var epoch = 1; epoch <= epochs; epoch++) {
    split.training.shuffle(random);
    var totalLoss = 0.0;
    for (final sample in split.training) {
      totalLoss +=
          network.trainSample(sample.features, sample.label.toDouble());
    }
    if (epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
      final trainingAccuracy = _accuracy(network, split.training);
      final testingAccuracy = _accuracy(network, split.testing);
      stdout.writeln(
        'Epoch ${epoch.toString().padLeft(3)} | '
        'loss ${(totalLoss / split.training.length).toStringAsFixed(4)} | '
        'train ${(trainingAccuracy * 100).toStringAsFixed(1)}% | '
        'test ${(testingAccuracy * 100).toStringAsFixed(1)}%',
      );
    }
  }
}

double _accuracy(NeuralNetwork network, List<HeartSample> samples) {
  final correct = samples
      .where((sample) => network.predict(sample.features) == sample.label)
      .length;
  return correct / samples.length;
}

int _integerOption(List<String> arguments, String name,
    {required int fallback}) {
  final value = _stringOption(arguments, name, fallback: '$fallback');
  final parsed = int.tryParse(value);
  if (parsed == null || parsed <= 0) {
    throw FormatException('$name must be a positive integer');
  }
  return parsed;
}

String _stringOption(List<String> arguments, String name,
    {required String fallback}) {
  final index = arguments.indexOf(name);
  if (index < 0) return fallback;
  if (index + 1 >= arguments.length) {
    throw FormatException('$name needs a value');
  }
  return arguments[index + 1];
}
