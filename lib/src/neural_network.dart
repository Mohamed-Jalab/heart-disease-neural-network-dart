import 'dart:math';

/// A fully connected neural network with sigmoid activations.
///
/// It intentionally uses no machine-learning package so the forward pass and
/// backpropagation algorithm remain easy to study.
class NeuralNetwork {
  NeuralNetwork({
    required List<int> layerSizes,
    this.learningRate = 0.1,
    int seed = 42,
  }) : _random = Random(seed) {
    if (layerSizes.length < 2 || layerSizes.any((size) => size <= 0)) {
      throw ArgumentError(
          'layerSizes must contain at least two positive sizes');
    }
    if (layerSizes.last != 1) {
      throw ArgumentError('Binary classification requires one output neuron');
    }
    if (learningRate <= 0) {
      throw ArgumentError.value(
          learningRate, 'learningRate', 'must be positive');
    }

    for (var layer = 1; layer < layerSizes.length; layer++) {
      final inputCount = layerSizes[layer - 1];
      final limit = sqrt(6 / (inputCount + layerSizes[layer]));
      weights.add(List.generate(
        layerSizes[layer],
        (_) => List.generate(inputCount, (_) => _uniform(-limit, limit)),
      ));
      biases.add(List.filled(layerSizes[layer], 0.0));
    }
  }

  final Random _random;
  final double learningRate;
  final List<List<List<double>>> weights = [];
  final List<List<double>> biases = [];

  double predictProbability(List<double> input) => _forward(input).last.single;

  int predict(List<double> input, {double threshold = 0.5}) =>
      predictProbability(input) >= threshold ? 1 : 0;

  /// Runs one stochastic-gradient-descent update and returns cross-entropy loss.
  double trainSample(List<double> input, double target) {
    if (target != 0 && target != 1) {
      throw ArgumentError.value(target, 'target', 'must be either 0 or 1');
    }
    final activations = _forward(input);
    final deltas = List<List<double>>.filled(weights.length, const []);
    final prediction = activations.last.single;

    // With sigmoid and cross entropy, dLoss/dZ is prediction - target.
    deltas.last = [prediction - target];
    for (var layer = weights.length - 2; layer >= 0; layer--) {
      deltas[layer] = List.generate(weights[layer].length, (neuron) {
        var propagated = 0.0;
        for (var next = 0; next < weights[layer + 1].length; next++) {
          propagated +=
              weights[layer + 1][next][neuron] * deltas[layer + 1][next];
        }
        final output = activations[layer + 1][neuron];
        return propagated * output * (1 - output);
      });
    }

    for (var layer = 0; layer < weights.length; layer++) {
      for (var neuron = 0; neuron < weights[layer].length; neuron++) {
        for (var inputIndex = 0;
            inputIndex < weights[layer][neuron].length;
            inputIndex++) {
          weights[layer][neuron][inputIndex] -= learningRate *
              deltas[layer][neuron] *
              activations[layer][inputIndex];
        }
        biases[layer][neuron] -= learningRate * deltas[layer][neuron];
      }
    }

    return -target * log(max(prediction, 1e-15)) -
        (1 - target) * log(max(1 - prediction, 1e-15));
  }

  List<List<double>> _forward(List<double> input) {
    if (input.length != weights.first.first.length) {
      throw ArgumentError(
        'Expected ${weights.first.first.length} inputs, got ${input.length}',
      );
    }
    final activations = <List<double>>[List.of(input)];
    for (var layer = 0; layer < weights.length; layer++) {
      activations.add(List.generate(weights[layer].length, (neuron) {
        var sum = biases[layer][neuron];
        for (var i = 0; i < weights[layer][neuron].length; i++) {
          sum += weights[layer][neuron][i] * activations.last[i];
        }
        return _sigmoid(sum);
      }));
    }
    return activations;
  }

  double _uniform(double minimum, double maximum) =>
      minimum + _random.nextDouble() * (maximum - minimum);

  static double _sigmoid(double value) {
    if (value >= 0) return 1 / (1 + exp(-value));
    final exponential = exp(value);
    return exponential / (1 + exponential);
  }
}
