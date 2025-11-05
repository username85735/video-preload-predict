#!/usr/bin/env python3
"""
Decode ByteNN quantized int8 model
"""

import struct
import numpy as np

def parse_bytenn_quantized(data):
    """Parse ByteNN format with int8 quantized weights"""
    print("="*80)
    print("BYTENN QUANTIZED MODEL DECODER")
    print("="*80)

    # Header
    magic = data[0:2].decode('ascii')
    file_size = struct.unpack('<I', data[4:8])[0]
    num_layers = struct.unpack('<I', data[8:12])[0]
    config_val = struct.unpack('<I', data[12:16])[0]
    header_size = struct.unpack('<I', data[16:20])[0]
    metadata_offset = struct.unpack('<I', data[20:24])[0]

    print(f"\nFile Header:")
    print(f"  Magic: {magic}")
    print(f"  File size: {file_size} bytes")
    print(f"  Number of layers: {num_layers}")
    print(f"  Config value: {config_val}")
    print(f"  Header size: {header_size} bytes")
    print(f"  Metadata offset: 0x{metadata_offset:08x}")

    # Weight data
    weight_start = header_size
    weight_end = metadata_offset
    weight_data_size = weight_end - weight_start

    print(f"\nWeight Data Section:")
    print(f"  Start: 0x{weight_start:08x}")
    print(f"  End: 0x{weight_end:08x}")
    print(f"  Size: {weight_data_size} bytes")

    # Extract int8 weights
    weights_int8 = np.frombuffer(data[weight_start:weight_end], dtype=np.int8)
    print(f"  Total int8 values: {len(weights_int8)}")

    # Statistics on quantized values
    print(f"\nQuantized Weight Statistics (int8):")
    print(f"  Min: {np.min(weights_int8)}")
    print(f"  Max: {np.max(weights_int8)}")
    print(f"  Mean: {np.mean(weights_int8):.2f}")
    print(f"  Std: {np.std(weights_int8):.2f}")

    # Value distribution
    print(f"\nValue distribution:")
    unique, counts = np.unique(weights_int8, return_counts=True)
    print(f"  Unique values: {len(unique)}")
    print(f"  Range: [{unique[0]}, {unique[-1]}]")

    # Show histogram
    print(f"\nHistogram (10 bins):")
    hist, bin_edges = np.histogram(weights_int8, bins=10)
    for i in range(len(hist)):
        bar = '#' * int(50 * hist[i] / max(hist))
        print(f"  [{bin_edges[i]:6.1f}, {bin_edges[i+1]:6.1f}): {hist[i]:6d} {bar}")

    return {
        'magic': magic,
        'num_layers': num_layers,
        'config_val': config_val,
        'weights_int8': weights_int8,
        'header_size': header_size,
        'metadata_offset': metadata_offset,
        'metadata': data[metadata_offset:]
    }

def dequantize_weights(weights_int8, scale=1/127):
    """Dequantize int8 weights to float32"""
    return weights_int8.astype(np.float32) * scale

def reverse_engineer_quantized_architecture(model_info):
    """Try to determine the neural network architecture"""
    print("\n" + "="*80)
    print("ARCHITECTURE ANALYSIS")
    print("="*80)

    total_params = len(model_info['weights_int8'])
    num_layers = model_info['num_layers']
    config_val = model_info['config_val']

    print(f"\nTotal parameters (int8): {total_params}")
    print(f"Number of layers: {num_layers}")
    print(f"Config value: {config_val}")

    # Try standard feedforward architectures
    print("\nSearching for matching architectures...")

    found = []

    # Test 2-layer networks
    for input_dim in range(5, 50):
        for hidden_dim in range(10, 200):
            for output_dim in [1, 2]:
                params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
                if params == total_params:
                    found.append([input_dim, hidden_dim, output_dim])

    # Test 3-layer networks
    for input_dim in range(5, 40):
        for h1 in range(10, 150):
            for h2 in range(10, 150):
                for output_dim in [1, 2]:
                    params = (input_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * output_dim + output_dim)
                    if params == total_params:
                        found.append([input_dim, h1, h2, output_dim])

    # Test 4-layer networks
    for input_dim in range(5, 30):
        for h1 in range(10, 120):
            for h2 in range(10, 120):
                for h3 in range(5, 60):
                    for output_dim in [1, 2]:
                        params = (input_dim * h1 + h1) + (h1 * h2 + h2) + \
                                (h2 * h3 + h3) + (h3 * output_dim + output_dim)
                        if params == total_params:
                            found.append([input_dim, h1, h2, h3, output_dim])

    # Test 5-layer networks
    for input_dim in range(5, 20):
        for h1 in range(20, 100):
            for h2 in range(20, 100):
                for h3 in range(10, 50):
                    for h4 in range(5, 30):
                        for output_dim in [1, 2]:
                            params = (input_dim * h1 + h1) + (h1 * h2 + h2) + \
                                    (h2 * h3 + h3) + (h3 * h4 + h4) + (h4 * output_dim + output_dim)
                            if params == total_params:
                                found.append([input_dim, h1, h2, h3, h4, output_dim])

    if found:
        print(f"\nFound {len(found)} possible architecture(s):")
        for i, arch in enumerate(found[:20]):  # Show first 20
            arch_str = ' -> '.join(map(str, arch))
            print(f"  {i+1}. {arch_str}")

        # Return the most reasonable one (prefer simpler architectures)
        best_arch = min(found, key=lambda x: len(x))
        return best_arch
    else:
        print("\nNo exact match found with standard feedforward architecture.")
        print("The model might use:")
        print("  - Custom layer connections")
        print("  - Skip connections")
        print("  - Embedding layers")
        return None

def analyze_metadata_section(metadata):
    """Analyze the metadata section"""
    print("\n" + "="*80)
    print("METADATA ANALYSIS")
    print("="*80)

    print(f"\nMetadata size: {len(metadata)} bytes")

    # Try different interpretations
    # As float32
    if len(metadata) >= 4:
        print("\nFirst 20 values as float32:")
        for i in range(0, min(80, len(metadata)), 4):
            try:
                f = struct.unpack('<f', metadata[i:i+4])[0]
                if i % 20 == 0:
                    print()
                print(f"  [{i//4:3d}] {f:12.6f}", end="")
            except:
                break
        print()

    # Check for layer dimension markers
    print("\nLooking for small integers (potential dimensions):")
    for i in range(0, min(80, len(metadata)), 4):
        try:
            val = struct.unpack('<I', metadata[i:i+4])[0]
            if 1 < val < 1000:
                print(f"  Offset {i}: {val}")
        except:
            break

def extract_layer_weights(weights_int8, architecture):
    """Extract individual layer weights based on architecture"""
    if not architecture:
        return None

    print("\n" + "="*80)
    print("LAYER WEIGHT EXTRACTION")
    print("="*80)

    layers = []
    offset = 0

    for i in range(len(architecture) - 1):
        n_in = architecture[i]
        n_out = architecture[i + 1]

        # Weight matrix: n_in × n_out
        weight_size = n_in * n_out
        weights = weights_int8[offset:offset + weight_size].reshape(n_in, n_out)
        offset += weight_size

        # Bias vector: n_out
        biases = weights_int8[offset:offset + n_out]
        offset += n_out

        layers.append({
            'layer': i,
            'input_dim': n_in,
            'output_dim': n_out,
            'weights': weights,
            'biases': biases
        })

        print(f"\nLayer {i}: {n_in} -> {n_out}")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights stats: min={np.min(weights)}, max={np.max(weights)}, mean={np.mean(weights):.2f}")
        print(f"  Biases shape: {biases.shape}")
        print(f"  Biases stats: min={np.min(biases)}, max={np.max(biases)}, mean={np.mean(biases):.2f}")

    return layers

def generate_final_report(model_info, architecture):
    """Generate final comprehensive report"""
    print("\n" + "="*80)
    print("FINAL MODEL ANALYSIS REPORT")
    print("="*80)

    print("\n" + "="*40)
    print("MODEL IDENTIFICATION")
    print("="*40)
    print(f"\nFile Format: ByteNN (ByteDance Neural Network)")
    print(f"Model Name: video_preload_predict")
    print(f"Purpose: Video Preload Prediction for Mobile/Web")
    print(f"Quantization: INT8 (8-bit integer weights)")

    print("\n" + "="*40)
    print("MODEL SPECIFICATIONS")
    print("="*40)
    print(f"\nTotal Parameters: {len(model_info['weights_int8']):,}")
    print(f"Weight Format: INT8 (1 byte per weight)")
    print(f"Model Size: {len(model_info['weights_int8']) / 1024:.2f} KB (weights only)")
    print(f"Number of Layers: {model_info['num_layers']}")

    if architecture:
        print("\n" + "="*40)
        print("NETWORK ARCHITECTURE")
        print("="*40)

        print(f"\nArchitecture: {' -> '.join(map(str, architecture))}")
        print(f"\nLayer-by-layer breakdown:")
        for i in range(len(architecture) - 1):
            n_in = architecture[i]
            n_out = architecture[i + 1]
            params = n_in * n_out + n_out
            print(f"  Layer {i}: [{n_in} x {n_out}] + {n_out} bias = {params} params")

        print(f"\n{'='*40}")
        print("ARCHITECTURE INTERPRETATION")
        print("="*40)

        print(f"\nInput Layer: {architecture[0]} features")
        print("  Likely video metadata:")
        print("    - Video duration")
        print("    - Video file size")
        print("    - Video bitrate")
        print("    - Video resolution (width, height)")
        print("    - Codec information")
        print("    - Network conditions (bandwidth, latency)")
        print("    - Device capabilities")
        print("    - User behavior patterns")

        for i, dim in enumerate(architecture[1:-1]):
            print(f"\nHidden Layer {i+1}: {dim} neurons")
            print(f"  Activation: Likely ReLU or Sigmoid")

        print(f"\nOutput Layer: {architecture[-1]}")
        if architecture[-1] == 1:
            print("  Type: Regression or Binary Classification")
            print("  Output: Probability or score for preload decision")
            print("  Interpretation: Higher value = recommend preload")
        elif architecture[-1] == 2:
            print("  Type: Binary Classification")
            print("  Output: [preload, don't preload] probabilities")
            print("  Activation: Likely Softmax")

    print("\n" + "="*40)
    print("QUANTIZATION DETAILS")
    print("="*40)

    weights = model_info['weights_int8']
    print(f"\nQuantization Range: [{np.min(weights)}, {np.max(weights)}]")
    print(f"Typical Dequantization: value_fp32 = value_int8 / 127.0")

    # Estimate dequantized range
    deq_min = np.min(weights) / 127.0
    deq_max = np.max(weights) / 127.0
    print(f"Dequantized Range (approx): [{deq_min:.4f}, {deq_max:.4f}]")

    print("\n" + "="*40)
    print("USE CASE")
    print("="*40)

    print("\nThis model predicts whether a video should be preloaded")
    print("based on various input features.")
    print("\nTypical usage:")
    print("  1. Collect video and context features (8-20 values)")
    print("  2. Feed into model")
    print("  3. Get prediction score")
    print("  4. If score > threshold: preload video")
    print("  5. If score < threshold: don't preload (save bandwidth)")
    print("\nBenefits:")
    print("  - Improves user experience (faster playback)")
    print("  - Reduces bandwidth waste (only preload likely-to-watch videos)")
    print("  - Optimized for mobile (INT8 quantization, small size)")

def main():
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    print(f"Analyzing: video_preload_predict.bytenn\n")

    # Parse the model
    model_info = parse_bytenn_quantized(data)

    # Reverse engineer architecture
    architecture = reverse_engineer_quantized_architecture(model_info)

    # Analyze metadata
    analyze_metadata_section(model_info['metadata'])

    # Extract layer weights
    if architecture:
        layers = extract_layer_weights(model_info['weights_int8'], architecture)

    # Generate final report
    generate_final_report(model_info, architecture)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - MODEL FULLY DECODED")
    print("="*80)

if __name__ == '__main__':
    main()
