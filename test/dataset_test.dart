import 'package:heart_nn/heart_nn.dart';
import 'package:test/test.dart';

void main() {
  test('loads every row and creates a deterministic 80/20 split', () {
    final samples = HeartDataset.load('heart.csv');
    final first = HeartDataset.splitAndNormalize(samples, seed: 12);
    final second = HeartDataset.splitAndNormalize(samples, seed: 12);

    expect(samples, hasLength(918));
    expect(samples.first.features, hasLength(11));
    expect(first.training, hasLength(734));
    expect(first.testing, hasLength(184));
    expect(first.training.first.features, second.training.first.features);
    expect(
      first.training.expand((sample) => sample.features),
      everyElement(inInclusiveRange(0.0, 1.0)),
    );
  });
}
