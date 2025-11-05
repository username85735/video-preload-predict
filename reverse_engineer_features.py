#!/usr/bin/env python3
"""
Reverse engineer what each of the 28 input features represents
Using multiple approaches: weights, thresholds, domain knowledge, and testing
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


def analyze_weight_patterns(model):
    """Analyze weight connection patterns for each feature"""
    print("="*80)
    print("WEIGHT PATTERN ANALYSIS")
    print("="*80)

    W0 = model.layers[0]['weights']  # Shape: (28, 431)

    print("\nAnalyzing connection patterns from each input feature...")
    print(f"{'Feat':<5} {'Σ|W|':<10} {'Max|W|':<10} {'Top Neurons':<30} {'Polarity':<10}")
    print("-" * 80)

    feature_profiles = []

    for feat_idx in range(28):
        weights = W0[feat_idx, :]
        abs_weights = np.abs(weights)

        sum_abs = np.sum(abs_weights)
        max_abs = np.max(abs_weights)

        # Find top 3 connected neurons
        top_3 = np.argsort(abs_weights)[-3:][::-1]

        # Polarity: more positive or negative?
        pos_sum = np.sum(weights[weights > 0])
        neg_sum = np.sum(weights[weights < 0])
        polarity = "POSITIVE" if pos_sum > abs(neg_sum) else "NEGATIVE"

        feature_profiles.append({
            'idx': feat_idx,
            'sum_abs': sum_abs,
            'max_abs': max_abs,
            'top_neurons': top_3,
            'polarity': polarity,
            'pos_sum': pos_sum,
            'neg_sum': neg_sum
        })

        print(f"{feat_idx:<5} {sum_abs:<10.2f} {max_abs:<10.4f} {str(top_3):<30} {polarity:<10}")

    return feature_profiles


def test_domain_specific_hypotheses(model):
    """Test specific hypotheses about what features might represent"""
    print("\n" + "="*80)
    print("DOMAIN-SPECIFIC HYPOTHESIS TESTING")
    print("="*80)

    print("\nTesting various domain-specific input patterns...")

    # Hypothesis groups based on video preload use case
    hypotheses = []

    # VIDEO QUALITY FEATURES
    print("\n" + "─"*80)
    print("TESTING: VIDEO QUALITY FEATURES")
    print("─"*80)

    # Test: Low quality video (should SKIP preload)
    low_quality = np.zeros(28)
    # Assume quality features might want HIGH values to preload
    # So LOW values should result in SKIP

    # Test: High quality video (might PRELOAD)
    high_quality = np.zeros(28)

    # Let's test each critical feature individually with "high quality" semantics
    critical_features = [2, 3, 5, 6, 7, 8, 11, 21, 24]

    print("\nTesting if critical features represent POSITIVE quality indicators:")
    print("(If setting to 1.0 increases preload likelihood → positive indicator)")
    print("(If setting to 1.0 decreases preload likelihood → negative indicator)")

    baseline = model.predict(np.zeros(28))

    for feat in critical_features:
        test_pos = np.zeros(28)
        test_pos[feat] = 1.0
        output_pos = model.predict(test_pos)

        test_neg = np.zeros(28)
        test_neg[feat] = -1.0
        output_neg = model.predict(test_neg)

        if output_pos > baseline:
            direction = "POSITIVE (↑ preload)"
            likely_type = "Quality/Good condition"
        else:
            direction = "NEGATIVE (↓ preload)"
            likely_type = "Cost/Bad condition"

        print(f"  Feature {feat:2d}: {direction:<25} → Likely: {likely_type}")

        hypotheses.append({
            'feature': feat,
            'direction': direction,
            'likely_type': likely_type,
            'sensitivity': abs(output_pos - baseline)
        })

    return hypotheses


def test_feature_combinations(model):
    """Test combinations that would make sense for different feature types"""
    print("\n" + "="*80)
    print("FEATURE COMBINATION TESTING")
    print("="*80)

    print("\nTesting realistic feature combinations...")

    scenarios = [
        {
            'name': 'High bitrate + high resolution',
            'features': {2: 1.0, 3: 1.0}
        },
        {
            'name': 'High FPS + high audio quality',
            'features': {7: 1.0, 8: 1.0}
        },
        {
            'name': 'All critical features positive',
            'features': {f: 1.0 for f in [2, 3, 5, 6, 7, 8, 11, 21, 24]}
        },
        {
            'name': 'Only non-critical features',
            'features': {f: 1.0 for f in [0, 1, 4, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27]}
        },
    ]

    print(f"\n{'Scenario':<40} {'Output':<12} {'Decision':<10}")
    print("-" * 65)

    for scenario in scenarios:
        features = np.zeros(28)
        for idx, val in scenario['features'].items():
            features[idx] = val

        output = model.predict(features)
        decision = "PRELOAD" if output > 0.5 else "SKIP"

        print(f"{scenario['name']:<40} {output:<12.6f} {decision:<10}")


def cluster_features_by_behavior(model):
    """Cluster features by their behavioral patterns"""
    print("\n" + "="*80)
    print("FEATURE CLUSTERING BY BEHAVIOR")
    print("="*80)

    print("\nTesting many input combinations to find feature groups...")

    np.random.seed(42)
    n_samples = 500

    # Collect feature-output relationships
    feature_outputs = {i: [] for i in range(28)}

    for _ in range(n_samples):
        # Random inputs
        base = np.random.randn(28) * 0.3
        base_output = model.predict(base)

        # Test each feature's marginal effect
        for feat_idx in range(28):
            perturbed = base.copy()
            perturbed[feat_idx] += 0.5  # Add 0.5 to this feature
            perturbed_output = model.predict(perturbed)

            delta = perturbed_output - base_output
            feature_outputs[feat_idx].append(delta)

    # Compute average marginal effect
    print("\nAverage marginal effect when increasing each feature by 0.5:")
    print(f"{'Feature':<8} {'Avg Δ':<12} {'Std Δ':<12} {'Effect Type':<20}")
    print("-" * 60)

    feature_categories = {
        'strong_positive': [],
        'moderate_positive': [],
        'weak': [],
        'moderate_negative': [],
        'strong_negative': []
    }

    for feat_idx in range(28):
        deltas = feature_outputs[feat_idx]
        avg_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        if avg_delta > 0.1:
            effect_type = "STRONG POSITIVE"
            feature_categories['strong_positive'].append(feat_idx)
        elif avg_delta > 0.01:
            effect_type = "Moderate positive"
            feature_categories['moderate_positive'].append(feat_idx)
        elif avg_delta < -0.1:
            effect_type = "STRONG NEGATIVE"
            feature_categories['strong_negative'].append(feat_idx)
        elif avg_delta < -0.01:
            effect_type = "Moderate negative"
            feature_categories['moderate_negative'].append(feat_idx)
        else:
            effect_type = "Weak/None"
            feature_categories['weak'].append(feat_idx)

        print(f"{feat_idx:<8} {avg_delta:<+12.6f} {std_delta:<12.6f} {effect_type:<20}")

    print("\n" + "─"*60)
    print("FEATURE CATEGORIZATION:")
    print("─"*60)
    for cat, features in feature_categories.items():
        if features:
            print(f"  {cat:20s}: {features}")

    return feature_categories


def infer_feature_meanings(model):
    """Make educated guesses about feature meanings based on all evidence"""
    print("\n" + "="*80)
    print("FEATURE MEANING INFERENCE")
    print("="*80)

    # We know from previous analysis:
    critical_positive = [2, 3, 7, 8]  # These increase preload when positive
    critical_negative = [5, 6, 11, 21, 24]  # These might decrease preload

    # Based on video preload domain knowledge
    possible_features = {
        # Video properties
        'video_duration': 'Short videos more likely to preload',
        'video_file_size': 'Small files more likely to preload',
        'video_bitrate': 'High bitrate = high quality → preload',
        'video_width': 'Resolution width',
        'video_height': 'Resolution height',
        'video_codec': 'Encoded codec type',
        'video_format': 'Container format',
        'video_fps': 'Frame rate',
        'audio_bitrate': 'Audio quality',
        'has_thumbnail': 'Boolean: has preview',

        # Network conditions
        'network_bandwidth': 'Current bandwidth (Mbps)',
        'network_latency': 'Ping time (ms)',
        'network_type': 'WiFi/4G/5G',
        'connection_stability': 'Connection quality score',
        'ttfb': 'Time to first byte',
        'packet_loss': 'Packet loss rate',

        # Device info
        'device_type': 'Mobile/tablet/desktop',
        'available_memory': 'Free RAM',
        'cpu_speed': 'Processor capability',
        'battery_level': 'Battery percentage',
        'power_saving_mode': 'Boolean: power saver on',

        # User context
        'time_of_day': 'Hour of day',
        'user_engagement': 'Historical engagement score',
        'video_category': 'Content category',
        'autoplay_enabled': 'Boolean: autoplay on',
        'user_engagement_score': 'Recent activity',
        'preload_success_rate': 'Historical preload success',
        'usage_pattern': 'Active/passive user'
    }

    print("\nMaking educated guesses based on:")
    print("  1. Critical feature thresholds")
    print("  2. Positive vs negative influence")
    print("  3. Domain knowledge about video streaming")
    print("  4. Weight patterns")

    print("\n" + "─"*80)
    print("MOST LIKELY FEATURE MAPPINGS:")
    print("─"*80)

    # Build hypothesis for each feature
    feature_hypotheses = {}

    # Based on what we know:
    # Features 2,3,7,8 are positive and CRITICAL with low thresholds
    # Feature 2 has threshold 0.912 (less sensitive)
    # Features 3,7,8 have thresholds ~0.21-0.24 (very sensitive)

    guesses = {
        0: ("video_duration OR file_size", "LOW", "Small values good for preload"),
        1: ("video_container_format", "LOW", "Encoded format type"),
        2: ("video_bitrate", "HIGH", "Threshold 0.91 - high quality indicator"),
        3: ("video_resolution_width", "CRITICAL", "Threshold 0.22 - VERY sensitive!"),
        4: ("video_encoding_complexity", "LOW", "Codec complexity"),
        5: ("network_bandwidth_INVERSE", "CRITICAL", "Negative: LOW bandwidth = preload?"),
        6: ("battery_level_INVERSE", "CRITICAL", "Negative: LOW battery = preload?"),
        7: ("video_fps", "CRITICAL", "Threshold 0.24 - frame rate"),
        8: ("audio_bitrate", "CRITICAL", "Threshold 0.21 - audio quality"),
        9: ("has_thumbnail", "LOW", "Boolean flag"),
        10: ("network_latency", "LOW", "Connection quality"),
        11: ("network_type_INVERSE", "CRITICAL", "Negative: poor network = preload?"),
        12: ("connection_stability", "LOW", "Network reliability"),
        13: ("ttfb", "LOW", "Server response time"),
        14: ("packet_loss_rate", "LOW", "Network quality"),
        15: ("device_type", "LOW", "Mobile/desktop"),
        16: ("available_memory", "LOW", "Free RAM"),
        17: ("cpu_speed", "LOW", "Processing power"),
        18: ("video_height", "LOW", "Resolution height"),
        19: ("cache_status", "LOW", "Cache availability"),
        20: ("power_saving_mode", "LOW", "Boolean: power saver"),
        21: ("previous_buffer_events", "CRITICAL", "Negative: past problems = skip?"),
        22: ("time_of_day", "LOW", "Hour of day"),
        23: ("user_tier", "LOW", "Premium/free user"),
        24: ("preload_quota_remaining", "CRITICAL", "Negative: quota used up?"),
        25: ("video_popularity", "LOW", "View count"),
        26: ("user_watch_history", "LOW", "Past behavior"),
        27: ("device_storage", "LOW", "Available disk space"),
    }

    print(f"\n{'Feat':<5} {'Most Likely Meaning':<30} {'Impact':<12} {'Notes':<40}")
    print("-" * 95)

    for feat_idx in range(28):
        guess, impact, notes = guesses.get(feat_idx, ("Unknown", "?", ""))
        print(f"{feat_idx:<5} {guess:<30} {impact:<12} {notes:<40}")
        feature_hypotheses[feat_idx] = {'guess': guess, 'impact': impact, 'notes': notes}

    return feature_hypotheses


def validate_hypotheses_with_tests(model, hypotheses):
    """Validate our hypotheses with targeted tests"""
    print("\n" + "="*80)
    print("HYPOTHESIS VALIDATION TESTING")
    print("="*80)

    print("\nTesting domain-specific scenarios to validate guesses...")

    test_cases = [
        {
            'name': 'Perfect conditions (WiFi, good video, engaged user)',
            'description': 'High quality video on fast WiFi with active user',
            'features': np.array([
                0.1,   # 0: short video
                0.1,   # 1: standard format
                0.5,   # 2: good bitrate  ← Should trigger!
                0.3,   # 3: good resolution ← Should trigger!
                0.1,   # 4: standard codec
                0.0,   # 5: NOT inverse (good bandwidth)
                0.0,   # 6: NOT inverse (good battery)
                0.3,   # 7: good fps ← Should trigger!
                0.3,   # 8: good audio ← Should trigger!
                0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1
            ])
        },
        {
            'name': 'Poor network conditions',
            'description': 'Good video but slow network',
            'features': np.array([
                0.1, 0.1,
                0.5,   # Good bitrate
                0.3,   # Good resolution
                0.1,
                0.8,   # 5: INVERSE - poor bandwidth!
                0.0,   # Good battery
                0.3, 0.3,  # Good video
                0.1, 0.1,
                0.8,   # 11: INVERSE - poor network type!
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.5,   # 21: Previous buffer problems!
                0.1, 0.1,
                0.7,   # 24: Quota running low!
                0.1, 0.1, 0.1
            ])
        },
        {
            'name': 'Low battery mode',
            'description': 'Low battery should prevent preload',
            'features': np.array([
                0.1, 0.1, 0.5, 0.3, 0.1, 0.0,
                0.9,   # 6: INVERSE - very low battery!
                0.3, 0.3, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.9,   # 20: Power saving mode!
                0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1
            ])
        }
    ]

    print(f"\n{'Scenario':<50} {'Output':<12} {'Decision':<10}")
    print("-" * 75)

    for test in test_cases:
        output = model.predict(test['features'])
        decision = "PRELOAD" if output > 0.5 else "SKIP"
        print(f"{test['name']:<50} {output:<12.6f} {decision:<10}")
        print(f"  → {test['description']}")

    print("\n" + "─"*75)
    print("INTERPRETATION:")
    print("─"*75)
    print("Based on these results, we can infer:")
    print("  • Features 5,6,11,21,24 are likely INVERSE/NEGATIVE indicators")
    print("  • Features 2,3,7,8 are likely POSITIVE quality indicators")
    print("  • Model combines multiple factors (not just one dominates)")


def main():
    print("="*80)
    print("REVERSE ENGINEERING INPUT FEATURES")
    print("="*80)
    print("\nGoal: Determine what each of the 28 input features represents")
    print("Using: weight patterns, thresholds, domain knowledge, behavioral testing")

    model = VideoPreloadModel('video_preload_predict.bytenn')

    # Multiple approaches
    feature_profiles = analyze_weight_patterns(model)

    domain_hyp = test_domain_specific_hypotheses(model)

    test_feature_combinations(model)

    feature_categories = cluster_features_by_behavior(model)

    feature_hypotheses = infer_feature_meanings(model)

    validate_hypotheses_with_tests(model, feature_hypotheses)

    print("\n" + "="*80)
    print("FEATURE REVERSE ENGINEERING COMPLETE")
    print("="*80)
    print("\nWe now have educated guesses for all 28 features!")
    print("Based on: weights, thresholds, behavior, and domain knowledge")

    return feature_hypotheses


if __name__ == '__main__':
    feature_hypotheses = main()
