#!/usr/bin/env python3
"""
Final attempt to decode the exact architecture
"""

import struct
import numpy as np

def advanced_architecture_search(total_params, config_val=478):
    """Advanced architecture search with config value hint"""
    print("="*80)
    print("ADVANCED ARCHITECTURE SEARCH")
    print("="*80)

    print(f"\nTotal parameters: {total_params}")
    print(f"Config value from header: {config_val}")

    # The config value 478 is likely important
    # Let's try architectures that involve 478

    print(f"\n{'='*40}")
    print("METHOD 1: Using config value as hidden dimension")
    print("="*40)

    found = []

    # Try architectures with 478 as a hidden layer dimension
    for input_dim in range(5, 50):
        for output_dim in [1, 2]:
            # Simple: input -> 478 -> output
            params = input_dim * config_val + config_val + config_val * output_dim + output_dim
            if params == total_params:
                found.append(('Method 1a', [input_dim, config_val, output_dim]))

            # With additional hidden layer
            for h2 in range(10, 500, 10):
                # input -> 478 -> h2 -> output
                params = (input_dim * config_val + config_val) + \
                        (config_val * h2 + h2) + \
                        (h2 * output_dim + output_dim)
                if params == total_params:
                    found.append(('Method 1b', [input_dim, config_val, h2, output_dim]))

                # input -> h2 -> 478 -> output
                params = (input_dim * h2 + h2) + \
                        (h2 * config_val + config_val) + \
                        (config_val * output_dim + output_dim)
                if params == total_params:
                    found.append(('Method 1c', [input_dim, h2, config_val, output_dim]))

    print(f"\n{'='*40}")
    print("METHOD 2: Config as multiplier or group size")
    print("="*40)

    # Maybe 478 represents number of groups/channels
    # Or total_params / 478 gives layer size

    ratio = total_params / config_val
    print(f"\nTotal params / config = {ratio:.2f}")

    if ratio == int(ratio):
        print(f"  Exact ratio! Could be {int(ratio)} weights per {config_val} groups")

    print(f"\n{'='*40}")
    print("METHOD 3: Factorization analysis")
    print("="*40)

    # Factor 227268
    print(f"\nFactors of {total_params}:")
    factors = []
    for i in range(2, min(1000, int(np.sqrt(total_params)) + 1)):
        if total_params % i == 0:
            factors.append((i, total_params // i))

    print("  Small factors:")
    for f1, f2 in factors[:20]:
        print(f"    {f1} x {f2}")

    print(f"\n{'='*40}")
    print("METHOD 4: Alternative layer counting")
    print("="*40)

    # Maybe "7 layers" includes input/output or counts differently
    # Try architectures with different effective layer counts

    for num_hidden in range(1, 6):
        print(f"\n  With {num_hidden} hidden layer(s):")

        if num_hidden == 1:
            for i in range(5, 50):
                for h in range(50, 600):
                    for o in [1, 2]:
                        p = i * h + h + h * o + o
                        if p == total_params:
                            found.append((f'Method 4-{num_hidden}hidden', [i, h, o]))

        elif num_hidden == 2:
            for i in range(5, 30):
                for h1 in range(50, 500):
                    for h2 in range(50, 500):
                        for o in [1, 2]:
                            p = (i * h1 + h1) + (h1 * h2 + h2) + (h2 * o + o)
                            if p == total_params:
                                found.append((f'Method 4-{num_hidden}hidden', [i, h1, h2, o]))

    print(f"\n{'='*40}")
    print("METHOD 5: Recurrent or special architectures")
    print("="*40)

    # LSTM: 4 * (input * hidden + hidden * hidden + hidden)
    # GRU: 3 * (input * hidden + hidden * hidden + hidden)

    print("\n  Testing LSTM architecture:")
    for input_dim in range(5, 30):
        for hidden_dim in range(50, 300):
            for output_dim in [1, 2]:
                # LSTM params
                lstm_params = 4 * (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)
                # Output layer
                output_params = hidden_dim * output_dim + output_dim
                total = lstm_params + output_params

                if total == total_params:
                    found.append(('LSTM', [input_dim, f'LSTM({hidden_dim})', output_dim]))

    print("\n  Testing GRU architecture:")
    for input_dim in range(5, 30):
        for hidden_dim in range(50, 300):
            for output_dim in [1, 2]:
                # GRU params
                gru_params = 3 * (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)
                # Output layer
                output_params = hidden_dim * output_dim + output_dim
                total = gru_params + output_params

                if total == total_params:
                    found.append(('GRU', [input_dim, f'GRU({hidden_dim})', output_dim]))

    return found

def display_results(found_architectures):
    """Display all found architectures"""
    print("\n" + "="*80)
    print("FOUND ARCHITECTURES")
    print("="*80)

    if not found_architectures:
        print("\nNo exact architecture match found.")
        print("\nPossible reasons:")
        print("  1. Model uses weight sharing")
        print("  2. Model uses custom layer types")
        print("  3. Model uses convolutional layers")
        print("  4. Parameters include additional metadata")
        return None

    print(f"\nFound {len(found_architectures)} possible architecture(s):\n")

    # Group by method
    by_method = {}
    for method, arch in found_architectures:
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(arch)

    for method, archs in by_method.items():
        print(f"{method}:")
        for arch in archs:
            arch_str = ' -> '.join(map(str, arch))
            print(f"  {arch_str}")
        print()

    # Return the most likely one
    return found_architectures[0][1]

def analyze_with_best_architecture(data, architecture):
    """Analyze the model with the best architecture guess"""
    print("="*80)
    print(f"DETAILED ANALYSIS WITH ARCHITECTURE: {' -> '.join(map(str, architecture))}")
    print("="*80)

    # Extract weights
    header_size = struct.unpack('<I', data[16:20])[0]
    metadata_offset = struct.unpack('<I', data[20:24])[0]

    weights_int8 = np.frombuffer(data[header_size:metadata_offset], dtype=np.int8)

    print(f"\nLayer-by-layer breakdown:")
    offset = 0

    for i in range(len(architecture) - 1):
        n_in = architecture[i]
        n_out = architecture[i + 1]

        # Handle special layer types
        if isinstance(n_in, str) or isinstance(n_out, str):
            print(f"\nLayer {i}: {n_in} -> {n_out} (special layer type)")
            continue

        weight_size = n_in * n_out
        bias_size = n_out
        layer_params = weight_size + bias_size

        if offset + layer_params <= len(weights_int8):
            layer_weights = weights_int8[offset:offset + weight_size]
            layer_biases = weights_int8[offset + weight_size:offset + layer_params]

            print(f"\nLayer {i}: {n_in} -> {n_out}")
            print(f"  Weights: {weight_size} ({n_in} x {n_out})")
            print(f"    Range: [{np.min(layer_weights)}, {np.max(layer_weights)}]")
            print(f"    Mean: {np.mean(layer_weights):.2f}, Std: {np.std(layer_weights):.2f}")
            print(f"  Biases: {bias_size}")
            print(f"    Range: [{np.min(layer_biases)}, {np.max(layer_biases)}]")
            print(f"    Mean: {np.mean(layer_biases):.2f}, Std: {np.std(layer_biases):.2f}")
            print(f"  Total params: {layer_params}")

            offset += layer_params
        else:
            print(f"\nLayer {i}: {n_in} -> {n_out} (EXCEEDS DATA SIZE)")
            break

    print(f"\nTotal params used: {offset} / {len(weights_int8)}")

    if offset != len(weights_int8):
        print(f"\nWARNING: Mismatch! {len(weights_int8) - offset} parameters unaccounted for")

def main():
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    total_params = 227268
    config_val = 478

    # Search for architecture
    found = advanced_architecture_search(total_params, config_val)

    # Display results
    best_arch = display_results(found)

    # If we found a match, analyze in detail
    if best_arch:
        analyze_with_best_architecture(data, best_arch)

        print("\n" + "="*80)
        print("FINAL CONCLUSION")
        print("="*80)

        arch_str = ' -> '.join(map(str, best_arch))
        print(f"\nMost Likely Architecture: {arch_str}")

        print(f"\nInput features ({best_arch[0]}):")
        print("  Possible video features:")
        print("    - Duration, file size, bitrate, resolution")
        print("    - Network bandwidth, latency")
        print("    - Device info, user history, etc.")

        print(f"\nOutput ({best_arch[-1]}):")
        if best_arch[-1] == 1:
            print("  Single value: preload probability/score")
        else:
            print("  Binary classification: [no_preload, preload]")

    print("\n" + "="*80)
    print("ARCHITECTURE DECODING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
