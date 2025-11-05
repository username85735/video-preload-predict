#!/usr/bin/env python3
"""
Test model with normalized inputs to find the correct input scaling
"""

import struct
import numpy as np

class VideoPreloadModel:
    """ByteNN video preload prediction model"""

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = f.read()

        self.header_size = struct.unpack('<I', data[16:20])[0]
        self.metadata_offset = struct.unpack('<I', data[20:24])[0]
        weights_data = data[self.header_size:self.metadata_offset]
        self.weights_int8 = np.frombuffer(weights_data, dtype=np.int8)
        self.architecture = [28, 431, 496, 1]
        self._extract_layers()

    def _extract_layers(self):
        self.layers = []
        offset = 0
        for i in range(len(self.architecture) - 1):
            n_in = self.architecture[i]
            n_out = self.architecture[i + 1]
            weight_size = n_in * n_out
            weights_float = self.weights_int8[offset:offset + weight_size].astype(np.float32) / 127.0
            W = weights_float.reshape(n_in, n_out)
            offset += weight_size
            b = self.weights_int8[offset:offset + n_out].astype(np.float32) / 127.0
            offset += n_out
            self.layers.append({'weights': W, 'biases': b})

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict(self, features):
        x = np.array(features, dtype=np.float32)
        x = x @ self.layers[0]['weights'] + self.layers[0]['biases']
        x = self.relu(x)
        x = x @ self.layers[1]['weights'] + self.layers[1]['biases']
        x = self.relu(x)
        x = x @ self.layers[2]['weights'] + self.layers[2]['biases']
        return self.sigmoid(x[0])

def test_various_scalings():
    """Test with different input scalings"""
    print("="*80)
    print("TESTING DIFFERENT INPUT SCALINGS")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Try different normalizations
    scalings = [
        ('Raw values (0-100 range)', 1.0),
        ('Normalized [0,1]', 0.01),
        ('Normalized [-1,1]', 0.02),
        ('Standardized (z-score)', 0.1),
        ('Very small values', 0.001),
    ]

    base_features = np.array([
        30, 5, 1500, 720, 1280, 1, 1, 30, 128, 1,
        50, 20, 3, 0.95, 100, 0.01, 1, 4096, 2, 80,
        0, 14, 0.8, 1, 1, 0.9, 0.85, 0.9
    ])

    print("\nBase feature vector (ideal conditions for preload):")
    print(f"  Values: {base_features[:10]}...")

    for name, scale in scalings:
        scaled = base_features * scale
        output = model.predict(scaled)
        print(f"\n{name} (scale={scale}):")
        print(f"  Input mean: {np.mean(scaled):.4f}, std: {np.std(scaled):.4f}")
        print(f"  Output: {output:.6f} → {'PRELOAD' if output > 0.5 else 'SKIP'}")

def test_systematic_range():
    """Test systematic range of inputs"""
    print("\n" + "="*80)
    print("SYSTEMATIC INPUT RANGE TESTING")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Test with all features set to same value, varying from -5 to 5
    print("\nTesting with all features set to constant value:")
    print(f"{'Value':>8} {'Output':>10} {'Decision':>10}")
    print("-" * 30)

    test_values = [-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5, 10]

    outputs = []
    for val in test_values:
        features = np.ones(28) * val
        output = model.predict(features)
        outputs.append(output)
        print(f"{val:8.2f} {output:10.6f} {'PRELOAD' if output > 0.5 else 'SKIP':>10}")

    # Find the threshold
    print("\nAnalysis:")
    print(f"  Range: [{min(outputs):.6f}, {max(outputs):.6f}]")
    print(f"  Variance: {np.var(outputs):.6f}")

def test_feature_sensitivity():
    """Test sensitivity to individual features"""
    print("\n" + "="*80)
    print("INDIVIDUAL FEATURE SENSITIVITY")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Start with all zeros
    base = np.zeros(28)
    baseline_output = model.predict(base)

    print(f"\nBaseline (all zeros): {baseline_output:.6f}")
    print("\nTesting each feature individually (setting to 1.0):")
    print(f"{'Feature':>8} {'Output':>10} {'Change':>10} {'Impact':>10}")
    print("-" * 42)

    for i in range(28):
        features = base.copy()
        features[i] = 1.0
        output = model.predict(features)
        change = output - baseline_output
        impact = 'HIGH' if abs(change) > 0.1 else ('MED' if abs(change) > 0.01 else 'LOW')
        print(f"{i:8d} {output:10.6f} {change:+10.6f} {impact:>10}")

def test_random_exploration():
    """Test with many random inputs to understand output distribution"""
    print("\n" + "="*80)
    print("RANDOM INPUT EXPLORATION (1000 samples)")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    np.random.seed(42)
    n_samples = 1000

    print("\nTesting different distributions:")

    distributions = [
        ('Uniform [-1, 1]', lambda: np.random.uniform(-1, 1, 28)),
        ('Uniform [0, 1]', lambda: np.random.uniform(0, 1, 28)),
        ('Normal (0, 1)', lambda: np.random.randn(28)),
        ('Normal (0, 0.1)', lambda: np.random.randn(28) * 0.1),
        ('Normal (0, 0.5)', lambda: np.random.randn(28) * 0.5),
    ]

    for name, generator in distributions:
        outputs = []
        for _ in range(n_samples):
            features = generator()
            output = model.predict(features)
            outputs.append(output)

        outputs = np.array(outputs)

        near_zero = np.sum(outputs < 0.1)
        near_one = np.sum(outputs > 0.9)
        middle = np.sum((outputs >= 0.4) & (outputs <= 0.6))

        print(f"\n{name}:")
        print(f"  Mean output: {np.mean(outputs):.4f}")
        print(f"  Std output: {np.std(outputs):.4f}")
        print(f"  Min/Max: [{np.min(outputs):.4f}, {np.max(outputs):.4f}]")
        print(f"  Near 0 (<0.1): {near_zero} ({100*near_zero/n_samples:.1f}%)")
        print(f"  Near 1 (>0.9): {near_one} ({100*near_one/n_samples:.1f}%)")
        print(f"  Middle (0.4-0.6): {middle} ({100*middle/n_samples:.1f}%)")
        print(f"  → Preload rate: {100*np.sum(outputs > 0.5)/n_samples:.1f}%")

def visualize_output_distribution():
    """Create a detailed histogram of outputs"""
    print("\n" + "="*80)
    print("OUTPUT DISTRIBUTION VISUALIZATION")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    np.random.seed(123)
    n_samples = 5000

    # Use normal distribution with mean 0, std 0.5
    outputs = []
    for _ in range(n_samples):
        features = np.random.randn(28) * 0.5
        output = model.predict(features)
        outputs.append(output)

    outputs = np.array(outputs)

    print(f"\nTested {n_samples} random inputs (Normal(0, 0.5))")
    print(f"\nOutput histogram (20 bins):")

    hist, bin_edges = np.histogram(outputs, bins=20)
    max_count = max(hist)

    for i in range(len(hist)):
        bar = '█' * int(50 * hist[i] / max_count) if max_count > 0 else ''
        pct = 100 * hist[i] / n_samples
        print(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]:4d} ({pct:4.1f}%) {bar}")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(outputs):.6f}")
    print(f"  Median: {np.median(outputs):.6f}")
    print(f"  Std: {np.std(outputs):.6f}")
    print(f"  25th percentile: {np.percentile(outputs, 25):.6f}")
    print(f"  75th percentile: {np.percentile(outputs, 75):.6f}")

    # Binary analysis
    bins = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
    for i in range(len(bins) - 1):
        count = np.sum((outputs >= bins[i]) & (outputs < bins[i+1]))
        pct = 100 * count / n_samples
        if count > 0:
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:4d} ({pct:5.2f}%)")

def main():
    print("="*80)
    print("VIDEO PRELOAD MODEL - NORMALIZED INPUT TESTING")
    print("="*80)

    test_various_scalings()
    test_systematic_range()
    test_feature_sensitivity()
    test_random_exploration()
    visualize_output_distribution()

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
