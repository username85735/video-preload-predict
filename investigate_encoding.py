#!/usr/bin/env python3
"""
Investigate the actual encoding of the model data
"""

import struct
import numpy as np

def hexdump(data, offset=0, length=256):
    """Print a hex dump of data"""
    for i in range(0, min(length, len(data)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"{offset+i:08x}  {hex_str:<48}  {ascii_str}")

def analyze_header_carefully(data):
    """Carefully analyze the header byte by byte"""
    print("="*80)
    print("DETAILED HEADER ANALYSIS")
    print("="*80)

    print("\nFirst 128 bytes with interpretation:")
    hexdump(data, 0, 128)

    print("\n\nHeader fields analysis:")
    print(f"Offset 0x00-0x01: {data[0:2].hex()} = '{data[0:2].decode('ascii', errors='ignore')}' (magic)")
    print(f"Offset 0x02-0x03: {struct.unpack('<H', data[2:4])[0]} (reserved/version?)")
    print(f"Offset 0x04-0x07: {struct.unpack('<I', data[4:8])[0]} (file size)")
    print(f"Offset 0x08-0x0b: {struct.unpack('<I', data[8:12])[0]} (field 1)")
    print(f"Offset 0x0c-0x0f: {struct.unpack('<I', data[12:16])[0]} (field 2)")
    print(f"Offset 0x10-0x13: {struct.unpack('<I', data[16:20])[0]} (field 3 - header size?)")
    print(f"Offset 0x14-0x17: {struct.unpack('<I', data[20:24])[0]} (field 4)")

    # Check where actual data starts
    print("\n\nLooking for data start...")

    # The header is 68 bytes (0x44), let's see what's right after
    print(f"\nBytes 0x44-0x54 (right after header):")
    hexdump(data, 0x44, 32)

    # Try to interpret as different data types
    print("\n\nTrying different interpretations at offset 0x44:")

    # As uint8
    uint8_vals = [data[0x44 + i] for i in range(20)]
    print(f"  As uint8: {uint8_vals}")

    # As int8
    int8_vals = [struct.unpack('b', data[0x44 + i:0x44 + i + 1])[0] for i in range(20)]
    print(f"  As int8: {int8_vals}")

    # As float16 (half precision)
    try:
        float16_vals = []
        for i in range(0, 20, 2):
            half = np.frombuffer(data[0x44 + i:0x44 + i + 2], dtype=np.float16)[0]
            float16_vals.append(float(half))
        print(f"  As float16: {float16_vals}")
    except:
        print(f"  As float16: [error]")

    # As float32
    float32_vals = []
    for i in range(0, 20, 4):
        f = struct.unpack('<f', data[0x44 + i:0x44 + i + 4])[0]
        float32_vals.append(f)
    print(f"  As float32: {float32_vals}")

def find_float_data(data):
    """Search for regions that look like valid float32 data"""
    print("\n" + "="*80)
    print("SEARCHING FOR VALID FLOAT32 REGIONS")
    print("="*80)

    # Scan through the file looking for regions with valid floats
    window_size = 100
    best_regions = []

    for offset in range(0, len(data) - window_size * 4, 100):
        valid_count = 0
        floats = []

        for i in range(window_size):
            try:
                f = struct.unpack('<f', data[offset + i*4:offset + i*4 + 4])[0]
                if not np.isnan(f) and not np.isinf(f) and abs(f) < 1e6:
                    valid_count += 1
                    floats.append(f)
            except:
                pass

        if valid_count > 80:  # At least 80% valid floats
            best_regions.append({
                'offset': offset,
                'valid_count': valid_count,
                'mean': np.mean(floats) if floats else 0,
                'std': np.std(floats) if floats else 0
            })

    print(f"\nFound {len(best_regions)} regions with >80% valid floats:")
    for i, region in enumerate(best_regions[:10]):
        print(f"  Region {i+1}: offset 0x{region['offset']:08x}, "
              f"valid={region['valid_count']}, mean={region['mean']:.4f}, std={region['std']:.4f}")

    return best_regions

def try_quantized_formats(data, offset=0x44):
    """Try interpreting as quantized integer formats"""
    print("\n" + "="*80)
    print("TESTING QUANTIZED INTEGER FORMATS")
    print("="*80)

    # Many mobile ML models use int8 quantization
    print("\nAs int8 with dequantization:")

    int8_data = struct.unpack(f'<{100}b', data[offset:offset+100])
    print(f"  First 100 int8 values: {int8_data[:20]}...")

    # Try to find scale and zero point
    # Dequantized = (quantized - zero_point) * scale

    # Common scale values
    for scale in [0.1, 0.01, 0.001, 1/127, 1/255]:
        dequantized = [x * scale for x in int8_data[:20]]
        print(f"  With scale={scale:.6f}: {[f'{x:.4f}' for x in dequantized[:10]]}")

def check_metadata_for_layer_info(data):
    """Check if metadata section contains layer configuration"""
    print("\n" + "="*80)
    print("METADATA SECTION FOR LAYER CONFIGURATION")
    print("="*80)

    metadata_offset = struct.unpack('<I', data[0x14:0x18])[0]
    metadata = data[metadata_offset:]

    print(f"\nMetadata at offset 0x{metadata_offset:08x} ({len(metadata)} bytes):")
    hexdump(metadata, metadata_offset, 128)

    # Try to parse as structured data
    print("\n\nTrying to parse metadata structure:")

    # Maybe it contains dimension information
    # Try reading as a series of uint32
    print("\nAs uint32 values:")
    for i in range(0, min(64, len(metadata)), 4):
        val = struct.unpack('<I', metadata[i:i+4])[0]
        if val < 10000:  # Reasonable dimension size
            print(f"  0x{i:02x}: {val}")

def analyze_complete_file_structure(data):
    """Analyze the complete file structure"""
    print("\n" + "="*80)
    print("COMPLETE FILE STRUCTURE ANALYSIS")
    print("="*80)

    # Get offsets from header
    field1 = struct.unpack('<I', data[8:12])[0]   # 7
    field2 = struct.unpack('<I', data[12:16])[0]  # 478
    field3 = struct.unpack('<I', data[16:20])[0]  # 68
    field4 = struct.unpack('<I', data[20:24])[0]  # 227336

    print(f"\nKey offsets from header:")
    print(f"  Field 1 (0x08): {field1}")
    print(f"  Field 2 (0x0c): {field2}")
    print(f"  Field 3 (0x10): {field3} (header size)")
    print(f"  Field 4 (0x14): {field4} (metadata offset)")

    # Calculate sizes
    header_size = field3
    metadata_offset = field4
    weight_data_size = metadata_offset - header_size

    print(f"\nCalculated sections:")
    print(f"  Header: 0x{0:08x} to 0x{header_size:08x} ({header_size} bytes)")
    print(f"  Weight data: 0x{header_size:08x} to 0x{metadata_offset:08x} ({weight_data_size} bytes)")
    print(f"  Metadata: 0x{metadata_offset:08x} to 0x{len(data):08x} ({len(data) - metadata_offset} bytes)")

    # How many floats could fit in weight data?
    max_float32 = weight_data_size // 4
    max_float16 = weight_data_size // 2
    max_int8 = weight_data_size

    print(f"\nCapacity analysis:")
    print(f"  As float32: {max_float32} parameters")
    print(f"  As float16: {max_float16} parameters")
    print(f"  As int8: {max_int8} parameters")

    # The header says field2=478, which might be related
    if field2 > 0:
        print(f"\nIf field2 ({field2}) represents dimensions:")
        print(f"  {max_float32} / {field2} = {max_float32 / field2:.1f}")
        print(f"  {max_float16} / {field2} = {max_float16 / field2:.1f}")

def main():
    with open('video_preload_predict.bytenn', 'rb') as f:
        data = f.read()

    print(f"Investigating: video_preload_predict.bytenn ({len(data)} bytes)\n")

    analyze_header_carefully(data)
    analyze_complete_file_structure(data)
    regions = find_float_data(data)
    try_quantized_formats(data)
    check_metadata_for_layer_info(data)

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
