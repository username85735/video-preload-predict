#!/usr/bin/env python3
"""
Deep investigation of model internals - probe everything we can!
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class VideoPreloadModel:
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
            weights_int8 = self.weights_int8[offset:offset + weight_size]
            weights_float = weights_int8.astype(np.float32) / 127.0
            W = weights_float.reshape(n_in, n_out)
            offset += weight_size
            biases_int8 = self.weights_int8[offset:offset + n_out]
            b = biases_int8.astype(np.float32) / 127.0
            offset += n_out
            self.layers.append({
                'weights': W,
                'biases': b,
                'weights_int8': weights_int8,
                'biases_int8': biases_int8,
                'input_dim': n_in,
                'output_dim': n_out
            })

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict_with_activations(self, features):
        """Predict and return all intermediate activations"""
        x = np.array(features, dtype=np.float32)
        activations = [x.copy()]

        # Layer 0
        z0 = x @ self.layers[0]['weights'] + self.layers[0]['biases']
        a0 = self.relu(z0)
        activations.append(a0.copy())

        # Layer 1
        z1 = a0 @ self.layers[1]['weights'] + self.layers[1]['biases']
        a1 = self.relu(z1)
        activations.append(a1.copy())

        # Layer 2
        z2 = a1 @ self.layers[2]['weights'] + self.layers[2]['biases']
        output = self.sigmoid(z2[0])

        return output, activations, [z0, z1, z2]

    def predict(self, features):
        x = np.array(features, dtype=np.float32)
        x = x @ self.layers[0]['weights'] + self.layers[0]['biases']
        x = self.relu(x)
        x = x @ self.layers[1]['weights'] + self.layers[1]['biases']
        x = self.relu(x)
        x = x @ self.layers[2]['weights'] + self.layers[2]['biases']
        return self.sigmoid(x[0])


def analyze_weight_distributions(model):
    """Deep analysis of weight distributions across layers"""
    print("="*80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("="*80)

    for i, layer in enumerate(model.layers):
        W = layer['weights']
        b = layer['biases']
        W_int8 = layer['weights_int8']

        print(f"\nLayer {i}: {layer['input_dim']} -> {layer['output_dim']}")
        print(f"{'='*60}")

        # Basic statistics
        print(f"  Weights (float32):")
        print(f"    Shape: {W.shape}")
        print(f"    Min: {np.min(W):.6f}, Max: {np.max(W):.6f}")
        print(f"    Mean: {np.mean(W):.6f}, Std: {np.std(W):.6f}")
        print(f"    Median: {np.median(W):.6f}")

        # Sparsity
        near_zero = np.sum(np.abs(W) < 0.01)
        sparsity = near_zero / W.size
        print(f"    Near-zero (<0.01): {near_zero} ({100*sparsity:.2f}%)")

        # Value distribution
        print(f"  Biases (float32):")
        print(f"    Min: {np.min(b):.6f}, Max: {np.max(b):.6f}")
        print(f"    Mean: {np.mean(b):.6f}, Std: {np.std(b):.6f}")

        # Quantized analysis
        print(f"  Quantized (INT8):")
        print(f"    Weights using full range: {np.min(W_int8)} to {np.max(W_int8)}")
        unique_vals = len(np.unique(W_int8))
        print(f"    Unique weight values: {unique_vals}/256")

        # Find strongest connections
        abs_weights = np.abs(W)
        top_5_indices = np.argsort(abs_weights.flatten())[-5:]
        print(f"  Top 5 strongest weights:")
        for idx in top_5_indices[::-1]:
            i_in, i_out = np.unravel_index(idx, W.shape)
            print(f"    [{i_in:3d}, {i_out:3d}]: {W[i_in, i_out]:+.6f}")


def analyze_feature_importance_comprehensive(model):
    """Comprehensive feature importance analysis"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    n_features = 28
    baseline = np.zeros(n_features)
    baseline_output = model.predict(baseline)

    print(f"\nBaseline (all zeros): {baseline_output:.6f}")
    print("\n{'Feature':<8} {'Output':<12} {'Delta':<12} {'Impact':<10} {'Abs Delta':<12}")
    print("-" * 70)

    feature_impacts = []

    for i in range(n_features):
        # Positive perturbation
        features_pos = baseline.copy()
        features_pos[i] = 1.0
        output_pos = model.predict(features_pos)
        delta_pos = output_pos - baseline_output

        # Negative perturbation
        features_neg = baseline.copy()
        features_neg[i] = -1.0
        output_neg = model.predict(features_neg)
        delta_neg = output_neg - baseline_output

        # Use the larger absolute change
        if abs(delta_pos) > abs(delta_neg):
            delta = delta_pos
            output = output_pos
            direction = "+"
        else:
            delta = delta_neg
            output = output_neg
            direction = "-"

        abs_delta = abs(delta)

        if abs_delta > 0.5:
            impact = "CRITICAL"
        elif abs_delta > 0.1:
            impact = "HIGH"
        elif abs_delta > 0.01:
            impact = "MEDIUM"
        else:
            impact = "LOW"

        feature_impacts.append({
            'feature': i,
            'output': output,
            'delta': delta,
            'abs_delta': abs_delta,
            'impact': impact,
            'direction': direction
        })

        print(f"{i:<8} {output:<12.6f} {delta:<+12.6f} {impact:<10} {abs_delta:<12.6f}")

    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: x['abs_delta'], reverse=True)

    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)

    for rank, fi in enumerate(feature_impacts[:10], 1):
        print(f"{rank:2d}. Feature {fi['feature']:2d}: {fi['impact']:<10} "
              f"(Δ = {fi['delta']:+.6f}, dir={fi['direction']})")

    return feature_impacts


def analyze_neuron_activations(model):
    """Analyze activation patterns in hidden layers"""
    print("\n" + "="*80)
    print("NEURON ACTIVATION ANALYSIS")
    print("="*80)

    np.random.seed(42)
    n_samples = 1000

    print(f"\nTesting with {n_samples} random inputs...")

    # Collect activations for each layer
    layer_activations = [[] for _ in range(2)]  # 2 hidden layers

    for _ in range(n_samples):
        features = np.random.randn(28) * 0.5
        _, activations, _ = model.predict_with_activations(features)

        # Store hidden layer activations (indices 1 and 2)
        layer_activations[0].append(activations[1])
        layer_activations[1].append(activations[2])

    # Analyze each hidden layer
    for layer_idx, layer_acts in enumerate(layer_activations):
        acts = np.array(layer_acts)  # Shape: (n_samples, n_neurons)

        print(f"\nHidden Layer {layer_idx} ({acts.shape[1]} neurons):")
        print("="*60)

        # Check for dead neurons
        max_activations = np.max(acts, axis=0)
        dead_neurons = np.sum(max_activations == 0)
        print(f"  Dead neurons (never activate): {dead_neurons} ({100*dead_neurons/acts.shape[1]:.2f}%)")

        # Average activation
        mean_acts = np.mean(acts, axis=0)
        print(f"  Average activation per neuron:")
        print(f"    Mean: {np.mean(mean_acts):.6f}")
        print(f"    Std: {np.std(mean_acts):.6f}")
        print(f"    Max: {np.max(mean_acts):.6f}")
        print(f"    Min: {np.min(mean_acts):.6f}")

        # Sparsity (how often neurons are zero)
        sparsity = np.sum(acts == 0) / acts.size
        print(f"  Sparsity (zero activations): {100*sparsity:.2f}%")

        # Find most/least active neurons
        activation_rates = np.mean(acts > 0, axis=0)
        most_active = np.argsort(activation_rates)[-5:]
        least_active = np.argsort(activation_rates)[:5]

        print(f"  Top 5 most active neurons:")
        for idx in most_active[::-1]:
            print(f"    Neuron {idx:3d}: active {100*activation_rates[idx]:.1f}% of time, "
                  f"avg={mean_acts[idx]:.4f}")

        print(f"  Top 5 least active neurons:")
        for idx in least_active:
            if activation_rates[idx] > 0:
                print(f"    Neuron {idx:3d}: active {100*activation_rates[idx]:.1f}% of time, "
                      f"avg={mean_acts[idx]:.4f}")
            else:
                print(f"    Neuron {idx:3d}: DEAD (never activates)")


def find_decision_boundary(model):
    """Map the decision boundary between preload/skip"""
    print("\n" + "="*80)
    print("DECISION BOUNDARY ANALYSIS")
    print("="*80)

    print("\nFinding critical input combinations that flip decisions...")

    # Test combinations of the critical features (2, 3, 7, 8)
    critical_features = [2, 3, 7, 8]

    print(f"\nTesting critical features: {critical_features}")

    # Grid search over these features
    test_values = [-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]

    results = []

    print("\nTesting all combinations (this may take a moment)...")

    for val0 in test_values:
        for val1 in test_values:
            features = np.zeros(28)
            features[critical_features[0]] = val0
            features[critical_features[1]] = val1
            output = model.predict(features)

            if 0.4 < output < 0.6:  # Near decision boundary
                results.append({
                    'f2': val0,
                    'f3': val1,
                    'output': output
                })

    if results:
        print(f"\nFound {len(results)} points near decision boundary (0.4-0.6):")
        for r in results[:10]:
            print(f"  Feature 2={r['f2']:+.2f}, Feature 3={r['f3']:+.2f} → {r['output']:.6f}")
    else:
        print("\nNo points found near decision boundary!")
        print("This confirms the model produces very binary outputs.")

    # Try to find the exact threshold
    print("\n" + "="*60)
    print("Finding threshold for single feature activation:")
    print("="*60)

    for feat in critical_features:
        print(f"\nFeature {feat}:")

        # Binary search for threshold
        low, high = 0.0, 2.0

        for _ in range(20):  # 20 iterations for precision
            mid = (low + high) / 2
            features = np.zeros(28)
            features[feat] = mid
            output = model.predict(features)

            if output < 0.5:
                low = mid
            else:
                high = mid

        threshold = (low + high) / 2
        features = np.zeros(28)
        features[feat] = threshold
        output_at_threshold = model.predict(features)

        print(f"  Threshold ≈ {threshold:.6f}")
        print(f"  Output at threshold: {output_at_threshold:.6f}")


def analyze_weight_importance_by_magnitude(model):
    """Analyze which weights have the most influence by magnitude"""
    print("\n" + "="*80)
    print("WEIGHT MAGNITUDE IMPORTANCE ANALYSIS")
    print("="*80)

    for layer_idx, layer in enumerate(model.layers):
        W = layer['weights']

        print(f"\nLayer {layer_idx}: {layer['input_dim']} -> {layer['output_dim']}")
        print("="*60)

        # Find strongest connections per input
        print(f"  Strongest connections from each input:")

        if layer_idx == 0:  # Only analyze first layer for input features
            for input_idx in range(min(28, layer['input_dim'])):
                weights_from_input = W[input_idx, :]
                abs_weights = np.abs(weights_from_input)

                top_3_outputs = np.argsort(abs_weights)[-3:]
                max_weight = np.max(abs_weights)
                sum_abs_weights = np.sum(abs_weights)

                if max_weight > 0.5:  # Only show significant ones
                    print(f"    Input {input_idx:2d}: max_weight={max_weight:.4f}, "
                          f"sum_abs={sum_abs_weights:.4f}, top_targets={top_3_outputs[::-1][:3]}")


def test_extreme_cases(model):
    """Test extreme and edge cases"""
    print("\n" + "="*80)
    print("EXTREME CASE TESTING")
    print("="*80)

    test_cases = [
        ("All zeros", np.zeros(28)),
        ("All ones", np.ones(28)),
        ("All -1", -np.ones(28)),
        ("All 10", np.ones(28) * 10),
        ("All -10", -np.ones(28) * 10),
        ("First half +1, second half -1", np.concatenate([np.ones(14), -np.ones(14)])),
        ("Alternating +1/-1", np.array([1 if i % 2 == 0 else -1 for i in range(28)])),
        ("Random large", np.random.randn(28) * 10),
        ("Random tiny", np.random.randn(28) * 0.001),
    ]

    print("\n{'Case':<40} {'Output':<12} {'Decision':<10}")
    print("-" * 65)

    for name, features in test_cases:
        output = model.predict(features)
        decision = "PRELOAD" if output > 0.5 else "SKIP"
        print(f"{name:<40} {output:<12.6f} {decision:<10}")


def main():
    print("="*80)
    print("DEEP MODEL INVESTIGATION")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')
    print(f"\nModel loaded: {' -> '.join(map(str, model.architecture))}")
    print(f"Total parameters: {len(model.weights_int8):,}")

    # Run all investigations
    analyze_weight_distributions(model)

    feature_impacts = analyze_feature_importance_comprehensive(model)

    analyze_neuron_activations(model)

    find_decision_boundary(model)

    analyze_weight_importance_by_magnitude(model)

    test_extreme_cases(model)

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

    return model, feature_impacts


if __name__ == '__main__':
    model, feature_impacts = main()
