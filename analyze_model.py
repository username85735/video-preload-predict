#!/usr/bin/env python3
"""
Comprehensive analysis of video_preload_predict.bytenn model file
"""

import struct
import numpy as np
import json
from collections import defaultdict

def read_header(data):
    """Parse the file header"""
    print("="*80)
    print("FILE HEADER ANALYSIS")
    print("="*80)

    # Read magic bytes
    magic = data[0:2]
    print(f"Magic bytes: {magic.hex()} ('{magic.decode('ascii', errors='ignore')}')")

    # Read file size marker
    file_size = struct.unpack('<I', data[4:8])[0]
    print(f"File size marker at offset 0x04: {file_size} bytes")

    # Read other header values
    val1 = struct.unpack('<I', data[8:12])[0]
    val2 = struct.unpack('<I', data[12:16])[0]
    val3 = struct.unpack('<I', data[16:20])[0]
    val4 = struct.unpack('<I', data[20:24])[0]
    val5 = struct.unpack('<I', data[24:28])[0]

    print(f"Value at 0x08: {val1}")
    print(f"Value at 0x0C: {val2}")
    print(f"Value at 0x10: {val3}")
    print(f"Value at 0x14: {val4}")
    print(f"Value at 0x18: {val5}")

    # Try to find patterns in the header
    print("\nFirst 256 bytes (hex):")
    for i in range(0, 256, 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"{i:08x}  {hex_str:<48}  {ascii_str}")

    return {
        'magic': magic,
        'file_size': file_size,
        'header_values': [val1, val2, val3, val4, val5]
    }

def analyze_offsets(data):
    """Look for offset pointers in the header"""
    print("\n" + "="*80)
    print("OFFSET ANALYSIS")
    print("="*80)

    # The header seems to contain offset pointers
    # Let's look for patterns of 4-byte values that might be offsets
    offsets = []
    for i in range(0, 100, 4):
        val = struct.unpack('<I', data[i:i+4])[0]
        if val < len(data) and val > 0:
            offsets.append((i, val))

    print("Potential offset pointers in first 100 bytes:")
    for pos, offset in offsets:
        print(f"  Offset 0x{pos:02x}: points to 0x{offset:08x} ({offset})")

    return offsets

def analyze_data_sections(data, offsets):
    """Analyze different sections of the file"""
    print("\n" + "="*80)
    print("DATA SECTIONS ANALYSIS")
    print("="*80)

    # Look at the data after the header
    # Based on offset patterns, try to identify sections

    # Common offset at 0x14 (value: 0x0378)
    section1_offset = struct.unpack('<I', data[0x14:0x18])[0]
    print(f"\nSection 1 at offset 0x{section1_offset:08x}:")
    print(f"  First 64 bytes: {data[section1_offset:section1_offset+64].hex()}")

    # Check offset at 0x1c
    section2_offset = struct.unpack('<I', data[0x1c:0x20])[0]
    print(f"\nSection 2 at offset 0x{section2_offset:08x}:")
    print(f"  First 64 bytes: {data[section2_offset:section2_offset+64].hex()}")

def analyze_floats(data, start=0x100, count=100):
    """Try to interpret data as floating point numbers"""
    print("\n" + "="*80)
    print("FLOATING POINT ANALYSIS")
    print("="*80)

    floats = []
    for i in range(start, min(start + count*4, len(data)), 4):
        try:
            f = struct.unpack('<f', data[i:i+4])[0]
            floats.append((i, f))
        except:
            pass

    print(f"\nFirst {min(count, len(floats))} float values starting at 0x{start:04x}:")
    for i, (pos, val) in enumerate(floats[:count]):
        if i % 5 == 0:
            print()
        print(f"  0x{pos:04x}: {val:10.6f}", end="")
    print()

    # Statistics on float values
    vals = [f for _, f in floats]
    if vals:
        print(f"\nFloat statistics:")
        print(f"  Count: {len(vals)}")
        print(f"  Min: {min(vals):.6f}")
        print(f"  Max: {max(vals):.6f}")
        print(f"  Mean: {np.mean(vals):.6f}")
        print(f"  Std: {np.std(vals):.6f}")

    return floats

def find_patterns(data):
    """Look for repeating patterns that might indicate structure"""
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)

    # Look for repeating byte sequences
    sequences = defaultdict(list)
    seq_len = 4

    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        sequences[seq].append(i)

    # Find most common sequences
    common = sorted(sequences.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    print(f"\nMost common {seq_len}-byte sequences:")
    for seq, positions in common:
        if len(positions) > 5:  # Only show if appears more than 5 times
            print(f"  {seq.hex()}: appears {len(positions)} times")
            if len(positions) <= 10:
                print(f"    Positions: {[hex(p) for p in positions[:10]]}")

def analyze_as_neural_network(data):
    """Try to interpret as a neural network model"""
    print("\n" + "="*80)
    print("NEURAL NETWORK STRUCTURE ANALYSIS")
    print("="*80)

    # ByteNN might be "ByteDance Neural Network" or similar
    # Look for typical NN patterns:
    # - Layer definitions
    # - Weight matrices
    # - Bias vectors
    # - Activation functions

    # The header might contain:
    # - Number of layers
    # - Layer dimensions
    # - Offsets to weight data

    num_layers = struct.unpack('<I', data[8:12])[0]
    print(f"Potential number of layers: {num_layers}")

    # Try to parse layer information
    print("\nAttempting to parse layer structure...")

    # Starting after the main header, look for layer definitions
    offset = 0x30  # Start after what seems to be the main header

    for i in range(min(num_layers, 20)):  # Limit to 20 to avoid infinite loops
        try:
            # Try to read layer parameters
            layer_type = struct.unpack('<I', data[offset:offset+4])[0]
            layer_offset = struct.unpack('<I', data[offset+4:offset+8])[0]

            print(f"  Layer {i}: type={layer_type}, data_offset=0x{layer_offset:08x}")
            offset += 8

            if offset >= len(data):
                break
        except:
            break

def deep_structure_analysis(data):
    """Perform deep structural analysis"""
    print("\n" + "="*80)
    print("DEEP STRUCTURE ANALYSIS")
    print("="*80)

    # Analyze the header more carefully
    # Looking at the hex dump, there seem to be multiple offset pointers

    # Parse what might be a table of contents
    print("\nParsing potential table of contents:")

    # The pattern suggests offsets might be stored as pairs or triplets
    for i in range(0x10, 0x50, 0x10):
        v1 = struct.unpack('<I', data[i:i+4])[0]
        v2 = struct.unpack('<I', data[i+4:i+8])[0]
        v3 = struct.unpack('<I', data[i+8:i+12])[0]
        v4 = struct.unpack('<I', data[i+12:i+16])[0]

        print(f"  0x{i:02x}: {v1:10d} (0x{v1:08x})  {v2:10d} (0x{v2:08x})  {v3:10d} (0x{v3:08x})  {v4:10d} (0x{v4:08x})")

def extract_all_floats(data, start_offset=0x50):
    """Extract all float data and analyze distributions"""
    print("\n" + "="*80)
    print("COMPLETE FLOAT EXTRACTION AND ANALYSIS")
    print("="*80)

    # Extract all valid floats from the data section
    floats = []
    positions = []

    for i in range(start_offset, len(data) - 3, 4):
        try:
            f = struct.unpack('<f', data[i:i+4])[0]
            # Filter out invalid floats (NaN, Inf, or unreasonably large values)
            if abs(f) < 1e10 and not np.isnan(f) and not np.isinf(f):
                floats.append(f)
                positions.append(i)
        except:
            pass

    print(f"Total valid floats extracted: {len(floats)}")
    print(f"Data region: 0x{start_offset:04x} to 0x{len(data):04x}")

    if floats:
        floats_array = np.array(floats)
        print(f"\nFloat statistics:")
        print(f"  Min: {np.min(floats_array):.6f}")
        print(f"  Max: {np.max(floats_array):.6f}")
        print(f"  Mean: {np.mean(floats_array):.6f}")
        print(f"  Median: {np.median(floats_array):.6f}")
        print(f"  Std: {np.std(floats_array):.6f}")

        # Check if values are normalized (common in neural networks)
        in_range = np.sum((floats_array >= -1) & (floats_array <= 1))
        print(f"  Values in [-1, 1]: {in_range} ({100*in_range/len(floats):.1f}%)")

        # Distribution analysis
        print(f"\nValue distribution:")
        print(f"  Negative values: {np.sum(floats_array < 0)}")
        print(f"  Zero values: {np.sum(floats_array == 0)}")
        print(f"  Positive values: {np.sum(floats_array > 0)}")

        # Histogram
        print(f"\nValue histogram (10 bins):")
        hist, bin_edges = np.histogram(floats_array, bins=10)
        for i in range(len(hist)):
            bar = '#' * int(50 * hist[i] / max(hist))
            print(f"  [{bin_edges[i]:8.3f}, {bin_edges[i+1]:8.3f}): {hist[i]:6d} {bar}")

    return floats, positions

def main():
    print("Video Preload Predict Model Analysis")
    print("File: video_preload_predict.bytenn")
    print()

    # Read the entire file
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    print(f"File size: {len(data)} bytes")
    print()

    # Perform various analyses
    header_info = read_header(data)
    offsets = analyze_offsets(data)
    analyze_data_sections(data, offsets)
    deep_structure_analysis(data)
    analyze_floats(data, start=0x100, count=50)
    floats, positions = extract_all_floats(data, start_offset=0x50)
    find_patterns(data)
    analyze_as_neural_network(data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
