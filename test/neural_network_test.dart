import 'package:heart_nn/heart_nn.dart';
import 'package:test/test.dart';

void main() {
  group('NeuralNetwork', () {
    test('rejects inputs with the wrong length', () {
      final network = NeuralNetwork(layerSizes: [2, 1]);
      expect(() => network.predict([1]), throwsArgumentError);
    });

    test('learns the OR truth table', () {
      final network = NeuralNetwork(
        layerSizes: [2, 3, 1],
        learningRate: 0.5,
        seed: 7,
      );
      const samples = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 1.0),
      ];
      for (var epoch = 0; epoch < 2000; epoch++) {
        for (final sample in samples) {
          network.trainSample(sample.$1, sample.$2);
        }
      }
      for (final sample in samples) {
        expect(network.predict(sample.$1), sample.$2.toInt());
      }
    });
  });
}
