import 'dart:io';
import 'dart:math';

class HeartSample {
  const HeartSample(this.features, this.label);

  final List<double> features;
  final int label;
}

class DatasetSplit {
  const DatasetSplit({required this.training, required this.testing});

  final List<HeartSample> training;
  final List<HeartSample> testing;
}

abstract final class HeartDataset {
  static const featureNames = [
    'Age',
    'Sex',
    'ChestPainType',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'RestingECG',
    'MaxHR',
    'ExerciseAngina',
    'Oldpeak',
    'ST_Slope',
  ];

  static const _categories = <int, List<String>>{
    1: ['F', 'M'],
    2: ['ASY', 'ATA', 'NAP', 'TA'],
    6: ['LVH', 'Normal', 'ST'],
    8: ['N', 'Y'],
    10: ['Down', 'Flat', 'Up'],
  };

  static List<HeartSample> load(String path) {
    final lines = File(path)
        .readAsLinesSync()
        .where((line) => line.trim().isNotEmpty)
        .toList();
    if (lines.length < 2) throw const FormatException('Dataset has no samples');
    final header = lines.first.split(',');
    if (header.length != 12 || header.last.trim() != 'HeartDisease') {
      throw const FormatException('Unexpected heart dataset header');
    }
    return [
      for (var index = 1; index < lines.length; index++)
        _parseLine(lines[index], index + 1),
    ];
  }

  static DatasetSplit splitAndNormalize(
    List<HeartSample> samples, {
    double trainingFraction = 0.8,
    int seed = 42,
  }) {
    if (samples.length < 2) {
      throw ArgumentError('At least two samples are required');
    }
    if (trainingFraction <= 0 || trainingFraction >= 1) {
      throw ArgumentError.value(trainingFraction, 'trainingFraction');
    }
    final shuffled = List<HeartSample>.of(samples)..shuffle(Random(seed));
    final splitIndex = (shuffled.length * trainingFraction).floor();
    final rawTraining = shuffled.sublist(0, splitIndex);
    final rawTesting = shuffled.sublist(splitIndex);
    final minimums = List<double>.filled(featureNames.length, double.infinity);
    final maximums =
        List<double>.filled(featureNames.length, double.negativeInfinity);

    for (final sample in rawTraining) {
      for (var i = 0; i < sample.features.length; i++) {
        minimums[i] = min(minimums[i], sample.features[i]);
        maximums[i] = max(maximums[i], sample.features[i]);
      }
    }

    HeartSample normalize(HeartSample sample) => HeartSample(
          List.generate(sample.features.length, (i) {
            final range = maximums[i] - minimums[i];
            return range == 0 ? 0 : (sample.features[i] - minimums[i]) / range;
          }),
          sample.label,
        );
    return DatasetSplit(
      training: rawTraining.map(normalize).toList(),
      testing: rawTesting.map(normalize).toList(),
    );
  }

  static HeartSample _parseLine(String line, int lineNumber) {
    final values = line.split(',').map((value) => value.trim()).toList();
    if (values.length != 12) {
      throw FormatException('Expected 12 values on line $lineNumber');
    }
    try {
      final features = <double>[];
      for (var i = 0; i < 11; i++) {
        final categories = _categories[i];
        if (categories == null) {
          features.add(double.parse(values[i]));
        } else {
          final encoded = categories.indexOf(values[i]);
          if (encoded < 0) {
            throw FormatException('Unknown category ${values[i]}');
          }
          features.add(encoded.toDouble());
        }
      }
      final label = int.parse(values.last);
      if (label != 0 && label != 1) {
        throw const FormatException('Invalid label');
      }
      return HeartSample(features, label);
    } on FormatException catch (error) {
      throw FormatException(
          'Invalid data on line $lineNumber: ${error.message}');
    }
  }
}
