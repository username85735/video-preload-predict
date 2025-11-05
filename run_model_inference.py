#!/usr/bin/env python3
"""
Run inference on the video preload model with various inputs
"""

import struct
import numpy as np

class VideoPreloadModel:
    """ByteNN video preload prediction model"""

    def __init__(self, model_path):
        """Load the model from file"""
        with open(model_path, 'rb') as f:
            data = f.read()

        # Parse header
        self.header_size = struct.unpack('<I', data[16:20])[0]
        self.metadata_offset = struct.unpack('<I', data[20:24])[0]

        # Extract quantized weights
        weights_data = data[self.header_size:self.metadata_offset]
        self.weights_int8 = np.frombuffer(weights_data, dtype=np.int8)

        # Architecture: 28 -> 431 -> 496 -> 1
        self.architecture = [28, 431, 496, 1]

        # Extract layer weights
        self._extract_layers()

        print("Model loaded successfully!")
        print(f"Architecture: {' -> '.join(map(str, self.architecture))}")
        print(f"Total parameters: {len(self.weights_int8):,}")

    def _extract_layers(self):
        """Extract individual layer weights and biases"""
        self.layers = []
        offset = 0

        for i in range(len(self.architecture) - 1):
            n_in = self.architecture[i]
            n_out = self.architecture[i + 1]

            # Extract and dequantize weights
            weight_size = n_in * n_out
            weights_int8 = self.weights_int8[offset:offset + weight_size]
            weights_float = weights_int8.astype(np.float32) / 127.0
            W = weights_float.reshape(n_in, n_out)
            offset += weight_size

            # Extract and dequantize biases
            biases_int8 = self.weights_int8[offset:offset + n_out]
            b = biases_int8.astype(np.float32) / 127.0
            offset += n_out

            self.layers.append({
                'weights': W,
                'biases': b,
                'input_dim': n_in,
                'output_dim': n_out
            })

            print(f"  Layer {i}: {n_in} -> {n_out}, W shape: {W.shape}, b shape: {b.shape}")

    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict(self, features, verbose=False):
        """
        Run inference on input features

        Args:
            features: numpy array of shape (28,)
            verbose: print intermediate activations

        Returns:
            float: prediction score
        """
        if len(features) != 28:
            raise ValueError(f"Expected 28 features, got {len(features)}")

        x = np.array(features, dtype=np.float32)

        if verbose:
            print("\nForward pass:")
            print(f"  Input: shape={x.shape}, mean={np.mean(x):.4f}, std={np.std(x):.4f}")

        # Layer 0: input -> hidden1
        x = x @ self.layers[0]['weights'] + self.layers[0]['biases']
        if verbose:
            print(f"  After layer 0 (linear): mean={np.mean(x):.4f}, std={np.std(x):.4f}, min={np.min(x):.4f}, max={np.max(x):.4f}")

        x = self.relu(x)
        if verbose:
            print(f"  After ReLU: mean={np.mean(x):.4f}, std={np.std(x):.4f}, non-zero={np.count_nonzero(x)}/{len(x)}")

        # Layer 1: hidden1 -> hidden2
        x = x @ self.layers[1]['weights'] + self.layers[1]['biases']
        if verbose:
            print(f"  After layer 1 (linear): mean={np.mean(x):.4f}, std={np.std(x):.4f}, min={np.min(x):.4f}, max={np.max(x):.4f}")

        x = self.relu(x)
        if verbose:
            print(f"  After ReLU: mean={np.mean(x):.4f}, std={np.std(x):.4f}, non-zero={np.count_nonzero(x)}/{len(x)}")

        # Layer 2: hidden2 -> output
        x = x @ self.layers[2]['weights'] + self.layers[2]['biases']
        if verbose:
            print(f"  After layer 2 (linear): {x[0]:.4f}")

        # Apply sigmoid to get probability
        output = self.sigmoid(x[0])
        if verbose:
            print(f"  After sigmoid: {output:.6f}")

        return output

def generate_test_scenarios():
    """Generate various test scenarios"""
    scenarios = []

    # Scenario 1: All zeros (baseline)
    scenarios.append({
        'name': 'All zeros (baseline)',
        'features': np.zeros(28)
    })

    # Scenario 2: All ones
    scenarios.append({
        'name': 'All ones',
        'features': np.ones(28)
    })

    # Scenario 3: All negative ones
    scenarios.append({
        'name': 'All negative ones',
        'features': -np.ones(28)
    })

    # Scenario 4: High-quality video on WiFi (should preload)
    # Features: short duration, small size, high bandwidth, WiFi, good history
    good_conditions = np.array([
        30.0,      # duration: 30 seconds (short)
        5.0,       # file size: 5 MB (small)
        1500.0,    # bitrate: 1.5 Mbps
        720.0,     # width: 720p
        1280.0,    # height
        1.0,       # codec: H264 (encoded as 1)
        1.0,       # format: MP4
        30.0,      # fps: 30
        128.0,     # audio bitrate
        1.0,       # has thumbnail
        50.0,      # bandwidth: 50 Mbps (WiFi)
        20.0,      # latency: 20ms (good)
        3.0,       # network type: WiFi (encoded as 3)
        0.95,      # connection stability
        100.0,     # TTFB
        0.01,      # packet loss: 1%
        1.0,       # device: mobile
        4096.0,    # available memory: 4GB
        2.0,       # CPU speed
        80.0,      # battery: 80%
        0.0,       # power saving: off
        14.0,      # time: 2pm
        0.8,       # viewing history score (high)
        1.0,       # category: popular
        1.0,       # autoplay: on
        0.9,       # engagement score (high)
        0.85,      # preload success rate (high)
        0.9        # app usage pattern (active user)
    ])
    scenarios.append({
        'name': 'Ideal conditions (WiFi, short video, active user)',
        'features': good_conditions
    })

    # Scenario 5: Poor conditions (should NOT preload)
    # Long video, slow network, low battery
    bad_conditions = np.array([
        600.0,     # duration: 10 minutes (long)
        50.0,      # file size: 50 MB (large)
        3000.0,    # bitrate: 3 Mbps (high)
        1920.0,    # width: 1080p
        1080.0,    # height
        1.0,       # codec
        1.0,       # format
        60.0,      # fps: 60 (high quality)
        256.0,     # audio bitrate
        1.0,       # has thumbnail
        1.5,       # bandwidth: 1.5 Mbps (slow 4G)
        200.0,     # latency: 200ms (poor)
        1.0,       # network type: 4G (encoded as 1)
        0.5,       # connection stability (poor)
        500.0,     # TTFB (slow)
        0.05,      # packet loss: 5%
        1.0,       # device: mobile
        512.0,     # available memory: 512MB (low)
        1.0,       # CPU speed (slow)
        10.0,      # battery: 10% (critical)
        1.0,       # power saving: on
        3.0,       # time: 3am
        0.2,       # viewing history score (low)
        5.0,       # category: niche
        0.0,       # autoplay: off
        0.1,       # engagement score (low)
        0.3,       # preload success rate (low)
        0.2        # app usage pattern (inactive)
    ])
    scenarios.append({
        'name': 'Poor conditions (slow network, long video, low battery)',
        'features': bad_conditions
    })

    # Scenario 6: Medium conditions
    medium_conditions = np.array([
        120.0,     # duration: 2 minutes
        15.0,      # file size: 15 MB
        2000.0,    # bitrate: 2 Mbps
        1280.0,    # width: 720p
        720.0,     # height
        1.0,       # codec
        1.0,       # format
        30.0,      # fps
        192.0,     # audio bitrate
        1.0,       # has thumbnail
        10.0,      # bandwidth: 10 Mbps
        50.0,      # latency: 50ms
        2.0,       # network type: 4G+ (encoded as 2)
        0.8,       # connection stability
        200.0,     # TTFB
        0.02,      # packet loss: 2%
        1.0,       # device: mobile
        2048.0,    # available memory: 2GB
        1.5,       # CPU speed
        50.0,      # battery: 50%
        0.0,       # power saving: off
        12.0,      # time: noon
        0.5,       # viewing history score
        3.0,       # category
        1.0,       # autoplay: on
        0.5,       # engagement score
        0.6,       # preload success rate
        0.5        # app usage pattern
    ])
    scenarios.append({
        'name': 'Medium conditions (moderate everything)',
        'features': medium_conditions
    })

    # Scenario 7: Random inputs
    np.random.seed(42)
    for i in range(5):
        scenarios.append({
            'name': f'Random scenario {i+1}',
            'features': np.random.randn(28)
        })

    # Scenario 8: Small variations around zero
    for i in range(5):
        scenarios.append({
            'name': f'Small variation {i+1}',
            'features': np.random.randn(28) * 0.1
        })

    # Scenario 9: Large positive values
    scenarios.append({
        'name': 'Large positive values',
        'features': np.ones(28) * 10.0
    })

    # Scenario 10: Large negative values
    scenarios.append({
        'name': 'Large negative values',
        'features': -np.ones(28) * 10.0
    })

    return scenarios

def run_batch_inference(model, scenarios):
    """Run inference on all scenarios"""
    print("\n" + "="*80)
    print("BATCH INFERENCE RESULTS")
    print("="*80)

    results = []

    for i, scenario in enumerate(scenarios):
        output = model.predict(scenario['features'])
        results.append({
            'name': scenario['name'],
            'output': output,
            'decision': 'PRELOAD' if output > 0.5 else 'SKIP'
        })

        print(f"\n{i+1}. {scenario['name']}")
        print(f"   Output: {output:.6f}")
        print(f"   Decision: {results[-1]['decision']}")

    return results

def analyze_outputs(results):
    """Analyze the distribution of outputs"""
    print("\n" + "="*80)
    print("OUTPUT ANALYSIS")
    print("="*80)

    outputs = [r['output'] for r in results]

    print(f"\nOutput statistics:")
    print(f"  Count: {len(outputs)}")
    print(f"  Min: {np.min(outputs):.6f}")
    print(f"  Max: {np.max(outputs):.6f}")
    print(f"  Mean: {np.mean(outputs):.6f}")
    print(f"  Median: {np.median(outputs):.6f}")
    print(f"  Std: {np.std(outputs):.6f}")

    # Check for binary-like behavior
    print(f"\nBinary behavior analysis:")
    near_zero = sum(1 for o in outputs if o < 0.1)
    near_one = sum(1 for o in outputs if o > 0.9)
    middle = sum(1 for o in outputs if 0.4 <= o <= 0.6)

    print(f"  Near 0 (< 0.1): {near_zero} ({100*near_zero/len(outputs):.1f}%)")
    print(f"  Near 1 (> 0.9): {near_one} ({100*near_one/len(outputs):.1f}%)")
    print(f"  Middle (0.4-0.6): {middle} ({100*middle/len(outputs):.1f}%)")

    # Decision distribution
    preload_count = sum(1 for r in results if r['decision'] == 'PRELOAD')
    skip_count = len(results) - preload_count

    print(f"\nDecision distribution:")
    print(f"  PRELOAD: {preload_count} ({100*preload_count/len(results):.1f}%)")
    print(f"  SKIP: {skip_count} ({100*skip_count/len(results):.1f}%)")

    # Histogram
    print(f"\nOutput histogram (10 bins):")
    hist, bin_edges = np.histogram(outputs, bins=10)
    for i in range(len(hist)):
        bar = '#' * int(50 * hist[i] / max(hist)) if max(hist) > 0 else ''
        print(f"  [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {hist[i]:3d} {bar}")

def test_specific_case_verbose(model):
    """Test one specific case with verbose output"""
    print("\n" + "="*80)
    print("DETAILED INFERENCE EXAMPLE")
    print("="*80)

    # Test ideal conditions
    ideal = np.array([
        30.0, 5.0, 1500.0, 720.0, 1280.0, 1.0, 1.0, 30.0, 128.0, 1.0,
        50.0, 20.0, 3.0, 0.95, 100.0, 0.01, 1.0, 4096.0, 2.0, 80.0,
        0.0, 14.0, 0.8, 1.0, 1.0, 0.9, 0.85, 0.9
    ])

    print("\nTest case: Ideal preload conditions")
    output = model.predict(ideal, verbose=True)
    print(f"\nFinal prediction: {output:.6f}")
    print(f"Decision: {'PRELOAD' if output > 0.5 else 'SKIP'}")

def main():
    print("="*80)
    print("VIDEO PRELOAD MODEL - INFERENCE TESTING")
    print("="*80)

    # Load model
    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Generate test scenarios
    scenarios = generate_test_scenarios()
    print(f"\nGenerated {len(scenarios)} test scenarios")

    # Run batch inference
    results = run_batch_inference(model, scenarios)

    # Analyze outputs
    analyze_outputs(results)

    # Test one case with verbose output
    test_specific_case_verbose(model)

    print("\n" + "="*80)
    print("INFERENCE TESTING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
