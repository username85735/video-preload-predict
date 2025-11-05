#!/usr/bin/env python3
"""
Deep investigation of the 66 dead neurons
"""

import struct
import numpy as np

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
            self.layers.append({'weights': W, 'biases': b, 'input_dim': n_in, 'output_dim': n_out})

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict_with_activations(self, features):
        x = np.array(features, dtype=np.float32)
        activations = [x.copy()]
        z0 = x @ self.layers[0]['weights'] + self.layers[0]['biases']
        a0 = self.relu(z0)
        activations.append(a0.copy())
        z1 = a0 @ self.layers[1]['weights'] + self.layers[1]['biases']
        a1 = self.relu(z1)
        activations.append(a1.copy())
        z2 = a1 @ self.layers[2]['weights'] + self.layers[2]['biases']
        output = self.sigmoid(z2[0])
        return output, activations, [z0, z1, z2]


def find_dead_neurons(model, n_samples=10000):
    """Find neurons that never activate"""
    print("="*80)
    print("DEAD NEURON INVESTIGATION")
    print("="*80)

    np.random.seed(42)
    print(f"\nTesting with {n_samples} random inputs to find dead neurons...")

    # Track max activation for each neuron
    max_activations_layer0 = np.zeros(431)
    max_activations_layer1 = np.zeros(496)

    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        features = np.random.randn(28) * 0.5
        _, activations, pre_activations = model.predict_with_activations(features)

        max_activations_layer0 = np.maximum(max_activations_layer0, activations[1])
        max_activations_layer1 = np.maximum(max_activations_layer1, activations[2])

    # Find dead neurons
    dead_layer0 = np.where(max_activations_layer0 == 0)[0]
    dead_layer1 = np.where(max_activations_layer1 == 0)[0]

    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)

    print(f"\nLayer 0 (431 neurons):")
    print(f"  Dead neurons: {len(dead_layer0)} ({100*len(dead_layer0)/431:.2f}%)")
    if len(dead_layer0) > 0:
        print(f"  Dead neuron indices: {dead_layer0[:20]}...")

    print(f"\nLayer 1 (496 neurons):")
    print(f"  Dead neurons: {len(dead_layer1)} ({100*len(dead_layer1)/496:.2f}%)")
    if len(dead_layer1) > 0:
        print(f"  Dead neuron indices: {dead_layer1}")

    return dead_layer0, dead_layer1


def analyze_dead_neuron_weights(model, dead_neurons_layer1):
    """Analyze the weights connected to dead neurons"""
    print("\n" + "="*80)
    print("DEAD NEURON WEIGHT ANALYSIS")
    print("="*80)

    if len(dead_neurons_layer1) == 0:
        print("\nNo dead neurons found!")
        return

    print(f"\nAnalyzing weights for {len(dead_neurons_layer1)} dead neurons in layer 1...")

    # Look at incoming weights (from layer 0 to dead neurons in layer 1)
    W1 = model.layers[1]['weights']  # Shape: (431, 496)
    b1 = model.layers[1]['biases']   # Shape: (496,)

    print("\nIncoming weights to dead neurons:")
    print(f"{'Neuron':<8} {'Bias':<10} {'Max Weight':<12} {'Min Weight':<12} {'Mean':<12}")
    print("-" * 60)

    for neuron_idx in dead_neurons_layer1[:20]:  # Show first 20
        weights_to_neuron = W1[:, neuron_idx]
        bias = b1[neuron_idx]

        print(f"{neuron_idx:<8} {bias:<10.6f} {np.max(weights_to_neuron):<12.6f} "
              f"{np.min(weights_to_neuron):<12.6f} {np.mean(weights_to_neuron):<12.6f}")

    # Analyze biases of dead neurons
    dead_biases = b1[dead_neurons_layer1]
    print(f"\nDead neuron biases statistics:")
    print(f"  Mean: {np.mean(dead_biases):.6f}")
    print(f"  Std: {np.std(dead_biases):.6f}")
    print(f"  Min: {np.min(dead_biases):.6f}")
    print(f"  Max: {np.max(dead_biases):.6f}")

    # Check if dead neurons have large negative biases
    very_negative = np.sum(dead_biases < -0.5)
    print(f"  Dead neurons with very negative bias (<-0.5): {very_negative} "
          f"({100*very_negative/len(dead_biases):.1f}%)")

    # Look at outgoing weights (from dead neurons to output)
    W2 = model.layers[2]['weights']  # Shape: (496, 1)

    print("\nOutgoing weights from dead neurons to output:")
    dead_output_weights = W2[dead_neurons_layer1, 0]
    print(f"  Mean: {np.mean(dead_output_weights):.6f}")
    print(f"  Std: {np.std(dead_output_weights):.6f}")
    print(f"  Max abs: {np.max(np.abs(dead_output_weights)):.6f}")

    print("\nTop 10 dead neurons with largest outgoing weights:")
    sorted_indices = np.argsort(np.abs(dead_output_weights))[::-1]
    for i in sorted_indices[:10]:
        neuron_idx = dead_neurons_layer1[i]
        weight = dead_output_weights[i]
        print(f"  Neuron {neuron_idx}: output_weight = {weight:+.6f}")


def test_if_dead_neurons_can_activate(model, dead_neurons_layer1):
    """Try to find inputs that would activate dead neurons"""
    print("\n" + "="*80)
    print("ATTEMPTING TO ACTIVATE DEAD NEURONS")
    print("="*80)

    if len(dead_neurons_layer1) == 0:
        print("\nNo dead neurons to test!")
        return

    print(f"\nTrying extreme inputs to activate dead neurons...")

    # Try various extreme inputs
    test_inputs = [
        ("Very large positive", np.ones(28) * 100),
        ("Very large negative", -np.ones(28) * 100),
        ("Mix of extremes", np.array([100 if i % 2 == 0 else -100 for i in range(28)])),
    ]

    for name, features in test_inputs:
        _, activations, pre_activations = model.predict_with_activations(features)

        layer1_acts = activations[2]
        activated_dead = 0

        for neuron_idx in dead_neurons_layer1:
            if layer1_acts[neuron_idx] > 0:
                activated_dead += 1

        print(f"  {name}: activated {activated_dead}/{len(dead_neurons_layer1)} dead neurons")

    print("\nConclusion: Dead neurons have likely been intentionally disabled")
    print("  - Possible reasons:")
    print("    1. Post-training pruning for compression")
    print("    2. Neurons became dead during quantization")
    print("    3. Training dynamics led to dead ReLUs")


def compare_active_vs_dead_neurons(model, dead_neurons_layer1):
    """Compare properties of active vs dead neurons"""
    print("\n" + "="*80)
    print("ACTIVE VS DEAD NEURON COMPARISON")
    print("="*80)

    all_neurons = np.arange(496)
    active_neurons = np.setdiff1d(all_neurons, dead_neurons_layer1)

    W1 = model.layers[1]['weights']
    b1 = model.layers[1]['biases']

    print(f"\nComparing {len(active_neurons)} active vs {len(dead_neurons_layer1)} dead neurons:")
    print("="*60)

    # Compare biases
    active_biases = b1[active_neurons]
    dead_biases = b1[dead_neurons_layer1]

    print("\nBias comparison:")
    print(f"  Active neurons - Mean: {np.mean(active_biases):+.6f}, Std: {np.std(active_biases):.6f}")
    print(f"  Dead neurons   - Mean: {np.mean(dead_biases):+.6f}, Std: {np.std(dead_biases):.6f}")

    # Compare incoming weight magnitudes
    active_weights = W1[:, active_neurons]
    dead_weights = W1[:, dead_neurons_layer1]

    print("\nIncoming weight magnitude comparison:")
    print(f"  Active neurons - Mean |W|: {np.mean(np.abs(active_weights)):.6f}")
    print(f"  Dead neurons   - Mean |W|: {np.mean(np.abs(dead_weights)):.6f}")

    # Histogram of biases
    print("\nBias distribution:")
    bins = np.linspace(-1, 1, 11)
    active_hist, _ = np.histogram(active_biases, bins=bins)
    dead_hist, _ = np.histogram(dead_biases, bins=bins)

    print(f"{'Range':<15} {'Active':<10} {'Dead':<10}")
    print("-" * 35)
    for i in range(len(bins) - 1):
        print(f"[{bins[i]:+.2f}, {bins[i+1]:+.2f}): {active_hist[i]:<10} {dead_hist[i]:<10}")


def main():
    print("="*80)
    print("INVESTIGATING THE 66 DEAD NEURONS")
    print("="*80)

    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Find dead neurons with large sample
    dead_layer0, dead_layer1 = find_dead_neurons(model, n_samples=10000)

    if len(dead_layer1) > 0:
        # Analyze their weights
        analyze_dead_neuron_weights(model, dead_layer1)

        # Try to activate them
        test_if_dead_neurons_can_activate(model, dead_layer1)

        # Compare with active neurons
        compare_active_vs_dead_neurons(model, dead_layer1)

    print("\n" + "="*80)
    print("DEAD NEURON INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
