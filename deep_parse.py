#!/usr/bin/env python3
"""
Deep parsing of the ByteNN model structure
"""

import struct
import numpy as np

def parse_bytenn_header(data):
    """Parse the complete ByteNN header structure"""
    print("="*80)
    print("BYTENN HEADER STRUCTURE")
    print("="*80)

    offset = 0

    # Magic and file info
    magic = data[0:2].decode('ascii')
    reserved1 = struct.unpack('<H', data[2:4])[0]
    file_size = struct.unpack('<I', data[4:8])[0]

    print(f"Magic: {magic}")
    print(f"Reserved1: {reserved1}")
    print(f"File size: {file_size} bytes")

    # Main parameters
    num_sections = struct.unpack('<I', data[8:12])[0]
    total_params = struct.unpack('<I', data[12:16])[0]
    header_size = struct.unpack('<I', data[16:20])[0]

    print(f"\nMain parameters:")
    print(f"  Number of sections/layers: {num_sections}")
    print(f"  Total parameters: {total_params}")
    print(f"  Header size: {header_size} bytes")

    # Read section table
    print(f"\nSection table (starting at offset 0x10):")
    section_info = []

    # It looks like the header has offset pointers
    # Let's carefully parse the structure
    offset = 0x10

    # Based on the hex dump, the pattern seems to be at 0x10-0x43
    # Let's try to understand the structure better

    print("\nDetailed header dump:")
    for i in range(0, 0x50, 4):
        val = struct.unpack('<I', data[i:i+4])[0]
        print(f"  0x{i:02x}: {val:10d} (0x{val:08x})")

    return {
        'magic': magic,
        'num_sections': num_sections,
        'total_params': total_params,
        'header_size': header_size,
        'file_size': file_size
    }

def extract_weights_by_section(data, start_offset=0x50):
    """Extract weights and try to determine layer dimensions"""
    print("\n" + "="*80)
    print("WEIGHT EXTRACTION AND LAYER ANALYSIS")
    print("="*80)

    # Extract all floats as a continuous array
    floats = []
    offset = start_offset

    while offset < len(data) - 3:
        try:
            f = struct.unpack('<f', data[offset:offset+4])[0]
            if abs(f) < 1e10 and not np.isnan(f) and not np.isinf(f):
                floats.append(f)
            offset += 4
        except:
            break

    weights = np.array(floats)
    print(f"Total weights extracted: {len(weights)}")

    # Try to factor the total number of weights to guess layer dimensions
    print(f"\nAttempting to determine layer architecture...")
    print(f"Total parameters: {len(weights)}")

    # Common video prediction model architectures
    # Try to find factorizations that make sense

    # Look for common layer sizes
    common_sizes = [8, 16, 32, 64, 128, 256, 512]

    print(f"\nPossible factorizations:")
    for size in common_sizes:
        if len(weights) % size == 0:
            print(f"  {size} × {len(weights) // size}")

    # Statistical analysis per chunk
    chunk_size = 478  # From header analysis
    num_chunks = len(weights) // chunk_size

    print(f"\nAnalyzing in chunks of {chunk_size} (from header):")
    print(f"  Number of complete chunks: {num_chunks}")

    if num_chunks > 0:
        for i in range(min(num_chunks, 10)):  # Show first 10 chunks
            chunk = weights[i*chunk_size:(i+1)*chunk_size]
            print(f"  Chunk {i}: mean={np.mean(chunk):7.4f}, std={np.std(chunk):7.4f}, "
                  f"min={np.min(chunk):7.4f}, max={np.max(chunk):7.4f}")

    return weights

def analyze_model_architecture(weights, header_info):
    """Try to reverse engineer the model architecture"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*80)

    total_params = len(weights)
    num_sections = header_info.get('num_sections', 7)

    print(f"Total parameters: {total_params}")
    print(f"Number of sections: {num_sections}")

    # For a video preload prediction model, we expect:
    # - Input features (video metadata: duration, size, bitrate, etc.)
    # - Hidden layers
    # - Output (probability of preload success or predicted performance)

    # Let's try different architectures
    print("\nTrying common neural network architectures:")

    # Simple feedforward network
    print("\n1. Feedforward Network Hypothesis:")

    # Try to find a factorization that makes sense
    # Format: input_size -> hidden1 -> hidden2 -> ... -> output_size

    # Common input sizes for video features: 10-50
    # Output: likely 1 (regression) or 2-10 (classification)

    for input_size in [8, 10, 16, 20, 32]:
        for output_size in [1, 2, 4, 8]:
            for h1 in [16, 32, 64, 128]:
                for h2 in [16, 32, 64, 128]:
                    # Calculate params for: input -> h1 -> h2 -> output
                    params = (input_size * h1 + h1) + \
                            (h1 * h2 + h2) + \
                            (h2 * output_size + output_size)

                    if abs(params - total_params) < 100:  # Close match
                        print(f"  {input_size} -> {h1} -> {h2} -> {output_size} = {params} params")

    # Also try 3 hidden layers
    print("\n2. Deeper Network Hypothesis (3 hidden layers):")
    for input_size in [8, 10, 16]:
        for output_size in [1, 2]:
            for h1 in [32, 64]:
                for h2 in [32, 64]:
                    for h3 in [16, 32]:
                        params = (input_size * h1 + h1) + \
                                (h1 * h2 + h2) + \
                                (h2 * h3 + h3) + \
                                (h3 * output_size + output_size)

                        if abs(params - total_params) < 100:
                            print(f"  {input_size} -> {h1} -> {h2} -> {h3} -> {output_size} = {params} params")

def analyze_weight_patterns(weights):
    """Analyze weight patterns to understand layer boundaries"""
    print("\n" + "="*80)
    print("WEIGHT PATTERN ANALYSIS")
    print("="*80)

    # Look for discontinuities in the weight distribution
    # Layer boundaries often show up as sudden changes in statistics

    window_size = 100
    stats = []

    for i in range(0, len(weights) - window_size, window_size):
        window = weights[i:i+window_size]
        stats.append({
            'offset': i,
            'mean': np.mean(window),
            'std': np.std(window),
            'min': np.min(window),
            'max': np.max(window)
        })

    print(f"\nRolling statistics (window size = {window_size}):")
    print(f"{'Offset':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)

    for i, s in enumerate(stats[:50]):  # First 50 windows
        print(f"{s['offset']:8d} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f}")

    # Look for large changes in std (potential layer boundaries)
    print("\nPotential layer boundaries (large std changes):")
    for i in range(1, len(stats)):
        std_change = abs(stats[i]['std'] - stats[i-1]['std'])
        if std_change > 0.1:  # Significant change
            print(f"  Offset {stats[i]['offset']}: std change = {std_change:.4f}")

def extract_metadata_section(data):
    """Try to extract any metadata or layer description section"""
    print("\n" + "="*80)
    print("METADATA SECTION ANALYSIS")
    print("="*80)

    # Based on offset analysis, there seems to be data near the end
    metadata_offset = struct.unpack('<I', data[0x14:0x18])[0]
    print(f"Metadata section at offset: 0x{metadata_offset:08x} ({metadata_offset})")

    if metadata_offset < len(data):
        metadata_size = len(data) - metadata_offset
        print(f"Metadata size: {metadata_size} bytes")

        metadata = data[metadata_offset:]
        print(f"\nMetadata hex dump (first 256 bytes):")
        for i in range(0, min(256, len(metadata)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in metadata[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in metadata[i:i+16])
            print(f"{i:08x}  {hex_str:<48}  {ascii_str}")

        # Try to interpret as structured data
        print("\nAttempting to parse as structured data:")
        try:
            # Try to read as a series of integers or floats
            for i in range(0, min(64, len(metadata)), 4):
                as_int = struct.unpack('<I', metadata[i:i+4])[0]
                as_float = struct.unpack('<f', metadata[i:i+4])[0]
                print(f"  0x{i:02x}: int={as_int:10d}, float={as_float:10.4f}")
        except:
            pass

def main():
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    header_info = parse_bytenn_header(data)
    weights = extract_weights_by_section(data)
    analyze_model_architecture(weights, header_info)
    analyze_weight_patterns(weights)
    extract_metadata_section(data)

    print("\n" + "="*80)
    print("DEEP ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
