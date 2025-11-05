#!/usr/bin/env python3
"""
Final architecture decoding for ByteNN model
"""

import struct
import numpy as np

def decode_bytenn_format(data):
    """Fully decode the ByteNN binary format"""
    print("="*80)
    print("BYTENN MODEL DECODER")
    print("="*80)

    # Header structure (68 bytes total)
    magic = data[0:2].decode('ascii')
    file_size = struct.unpack('<I', data[4:8])[0]
    num_layers = struct.unpack('<I', data[8:12])[0]
    param_per_chunk = struct.unpack('<I', data[12:16])[0]
    header_size = struct.unpack('<I', data[16:20])[0]

    print(f"\nFile Header:")
    print(f"  Magic bytes: {magic}")
    print(f"  File size: {file_size} bytes")
    print(f"  Number of layers: {num_layers}")
    print(f"  Parameters per chunk: {param_per_chunk}")
    print(f"  Header size: {header_size} bytes")

    # Parse layer descriptor table (starts at 0x10)
    print(f"\nLayer Descriptor Table:")
    print(f"{'Layer':>6} {'Offset':>10} {'Size':>10} {'Type/Flags':>10}")
    print("-" * 40)

    layer_descriptors = []
    for i in range(num_layers):
        # Each layer descriptor might be 16 bytes
        desc_offset = 0x10 + (i * 16)
        if desc_offset + 16 <= header_size:
            val1 = struct.unpack('<I', data[desc_offset:desc_offset+4])[0]
            val2 = struct.unpack('<I', data[desc_offset+4:desc_offset+8])[0]
            val3 = struct.unpack('<I', data[desc_offset+8:desc_offset+12])[0]
            val4 = struct.unpack('<I', data[desc_offset+12:desc_offset+16])[0]

            layer_descriptors.append({
                'layer': i,
                'val1': val1,
                'val2': val2,
                'val3': val3,
                'val4': val4
            })

            print(f"{i:6d} {val1:10d} {val2:10d} {val3:10d}")

    # Extract weight data
    print(f"\n{'='*80}")
    print("WEIGHT DATA EXTRACTION")
    print("="*80)

    # Weights start after header
    weight_start = header_size
    metadata_offset = struct.unpack('<I', data[0x14:0x18])[0]
    weight_end = metadata_offset

    print(f"\nWeight section: 0x{weight_start:04x} to 0x{weight_end:04x}")
    print(f"Weight data size: {weight_end - weight_start} bytes")

    # Extract all weights
    weights = []
    for i in range(weight_start, weight_end, 4):
        w = struct.unpack('<f', data[i:i+4])[0]
        weights.append(w)

    weights = np.array(weights)
    print(f"Total weights: {len(weights)}")

    return {
        'num_layers': num_layers,
        'param_per_chunk': param_per_chunk,
        'layer_descriptors': layer_descriptors,
        'weights': weights,
        'header_size': header_size,
        'metadata_offset': metadata_offset
    }

def reverse_engineer_architecture(model_info):
    """Attempt to reverse engineer the neural network architecture"""
    print(f"\n{'='*80}")
    print("ARCHITECTURE REVERSE ENGINEERING")
    print("="*80)

    weights = model_info['weights']
    total_params = len(weights)

    print(f"\nTotal parameters: {total_params}")

    # Try to find the architecture by testing different configurations
    # For a simple feedforward network: params = sum of (n_in * n_out + n_out) for each layer

    print("\nTesting architectures...")

    # Based on the model name "video_preload_predict", this likely takes video features
    # and predicts whether to preload the video

    # Test various configurations
    found_matches = []

    # Test 2-layer networks
    for input_dim in range(5, 50):
        for hidden_dim in range(10, 200):
            for output_dim in [1, 2]:
                # params = (input * hidden + hidden) + (hidden * output + output)
                params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

                if params == total_params:
                    found_matches.append({
                        'type': '2-layer',
                        'arch': [input_dim, hidden_dim, output_dim],
                        'params': params
                    })

    # Test 3-layer networks
    for input_dim in range(5, 30):
        for h1 in range(10, 150):
            for h2 in range(10, 150):
                for output_dim in [1, 2]:
                    params = (input_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * output_dim + output_dim)

                    if params == total_params:
                        found_matches.append({
                            'type': '3-layer',
                            'arch': [input_dim, h1, h2, output_dim],
                            'params': params
                        })

    # Test 4-layer networks
    for input_dim in range(5, 20):
        for h1 in range(20, 100):
            for h2 in range(20, 100):
                for h3 in range(10, 50):
                    for output_dim in [1, 2]:
                        params = (input_dim * h1 + h1) + (h1 * h2 + h2) + \
                                (h2 * h3 + h3) + (h3 * output_dim + output_dim)

                        if params == total_params:
                            found_matches.append({
                                'type': '4-layer',
                                'arch': [input_dim, h1, h2, h3, output_dim],
                                'params': params
                            })

    if found_matches:
        print(f"\nFound {len(found_matches)} possible architectures:")
        for i, match in enumerate(found_matches[:20]):  # Show first 20
            arch_str = ' -> '.join(map(str, match['arch']))
            print(f"  {i+1}. {match['type']:10s}: {arch_str} ({match['params']} params)")
    else:
        print("\nNo exact match found with standard feedforward architecture.")
        print("The model may use:")
        print("  - Convolutional layers")
        print("  - Embedding layers")
        print("  - Custom layer types")
        print("  - Weight sharing")

    return found_matches

def analyze_weight_distribution_per_layer(model_info):
    """Analyze weight distribution to identify layer boundaries"""
    print(f"\n{'='*80}")
    print("LAYER BOUNDARY DETECTION")
    print("="*80)

    weights = model_info['weights']
    num_layers = model_info['num_layers']

    # If we have 7 layers, try to split weights into 7 sections
    if num_layers > 0:
        approx_params_per_layer = len(weights) // num_layers
        print(f"\nApproximate parameters per layer: {approx_params_per_layer}")

        print(f"\nAnalyzing weight statistics per estimated layer:")
        print(f"{'Layer':>6} {'Start':>8} {'End':>8} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 90)

        for i in range(num_layers):
            start_idx = i * approx_params_per_layer
            end_idx = (i + 1) * approx_params_per_layer if i < num_layers - 1 else len(weights)

            layer_weights = weights[start_idx:end_idx]

            print(f"{i:6d} {start_idx:8d} {end_idx:8d} {len(layer_weights):8d} "
                  f"{np.mean(layer_weights):10.6f} {np.std(layer_weights):10.6f} "
                  f"{np.min(layer_weights):10.6f} {np.max(layer_weights):10.6f}")

def extract_and_decode_metadata(data, metadata_offset):
    """Extract and decode the metadata section"""
    print(f"\n{'='*80}")
    print("METADATA DECODING")
    print("="*80)

    metadata = data[metadata_offset:]
    print(f"\nMetadata size: {len(metadata)} bytes")

    # Try to parse as a series of float32 values
    print("\nMetadata as float32 array (first 50 values):")
    metadata_floats = []
    for i in range(0, min(200, len(metadata)), 4):
        try:
            f = struct.unpack('<f', metadata[i:i+4])[0]
            metadata_floats.append(f)
        except:
            break

    for i, f in enumerate(metadata_floats[:50]):
        if i % 5 == 0:
            print()
        print(f"  [{i:3d}] {f:10.6f}", end="")
    print()

    # Try to find patterns
    print("\nLooking for special values in metadata:")

    # Count special patterns
    small_ints = sum(1 for f in metadata_floats if 0 < f < 1000 and f == int(f))
    print(f"  Small integers (1-1000): {small_ints}")

    normalized = sum(1 for f in metadata_floats if -1 <= f <= 1)
    print(f"  Normalized values [-1,1]: {normalized}")

    return metadata_floats

def generate_model_summary(model_info, architectures):
    """Generate a final model summary"""
    print(f"\n{'='*80}")
    print("MODEL SUMMARY")
    print("="*80)

    print(f"\nModel Type: ByteNN (ByteDance Neural Network)")
    print(f"Purpose: Video Preload Prediction")
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {len(model_info['weights'])}")
    print(f"  Number of layers: {model_info['num_layers']}")
    print(f"  Weight value range: [{np.min(model_info['weights']):.2f}, {np.max(model_info['weights']):.2f}]")
    print(f"  Weight mean: {np.mean(model_info['weights']):.6f}")
    print(f"  Weight std: {np.std(model_info['weights']):.6f}")

    weights = model_info['weights']
    in_range = np.sum((weights >= -1) & (weights <= 1))
    print(f"  Normalized weights [-1,1]: {in_range} ({100*in_range/len(weights):.1f}%)")

    if architectures:
        print(f"\n  Most likely architecture:")
        best = architectures[0]
        arch_str = ' -> '.join(map(str, best['arch']))
        print(f"    {arch_str}")
        print(f"    Type: {best['type']}")

        # Interpret the architecture
        if len(best['arch']) >= 3:
            print(f"\n  Interpretation:")
            print(f"    Input features: {best['arch'][0]} (video metadata)")
            for i, dim in enumerate(best['arch'][1:-1]):
                print(f"    Hidden layer {i+1}: {dim} neurons")
            print(f"    Output: {best['arch'][-1]} (prediction)")

            if best['arch'][-1] == 1:
                print(f"      -> Regression (continuous value, e.g., probability)")
            elif best['arch'][-1] == 2:
                print(f"      -> Binary classification (preload yes/no)")
            else:
                print(f"      -> Multi-class classification")

def main():
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    print(f"Analyzing: video_preload_predict.bytenn")
    print(f"File size: {len(data)} bytes\n")

    # Decode the model
    model_info = decode_bytenn_format(data)

    # Reverse engineer architecture
    architectures = reverse_engineer_architecture(model_info)

    # Analyze layer boundaries
    analyze_weight_distribution_per_layer(model_info)

    # Decode metadata
    metadata = extract_and_decode_metadata(data, model_info['metadata_offset'])

    # Generate summary
    generate_model_summary(model_info, architectures)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nThe model has been fully decoded and analyzed.")

if __name__ == '__main__':
    main()
