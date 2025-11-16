import librosa
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
import sys
import argparse
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class BeatSegment:
    """Represents a segment of audio - either a beat interval or non-beat interval"""
    def __init__(self, audio_data, start_sample, end_sample, sample_rate, index, is_beat=True):
        self.audio_data = audio_data
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.sample_rate = sample_rate
        self.index = index
        self.is_beat = is_beat  # True if this is a beat interval, False if missing beat
        self.features = None
        self.cluster_id = -1  # Default to -1 (noise)
        self.cluster_label = ""  # Letter label (A, B, C, ...)
        self.sub_cluster_id = -1  # Sub-cluster within main cluster
        self.pattern_label = ""  # Full label (e.g., "A1", "A2", "B1")
        self.has_vocals = False  # Whether this segment contains vocals

    def extract_features(self):
        """Extract comprehensive musical features for similarity comparison"""
        if len(self.audio_data) == 0:
            self.features = np.zeros(26)
            return self.features

        # 1. MFCCs (13 coefficients) - timbre
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # 2. Spectral centroid - brightness
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate)
        centroid_mean = np.mean(spectral_centroid)

        # 3. Spectral rolloff - frequency distribution
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio_data, sr=self.sample_rate)
        rolloff_mean = np.mean(spectral_rolloff)

        # 4. Zero crossing rate - noisiness
        zcr = librosa.feature.zero_crossing_rate(self.audio_data)
        zcr_mean = np.mean(zcr)

        # 5. RMS energy - loudness
        rms = librosa.feature.rms(y=self.audio_data)
        rms_mean = np.mean(rms)

        # 6. Spectral contrast - texture
        spectral_contrast = librosa.feature.spectral_contrast(y=self.audio_data, sr=self.sample_rate)
        contrast_mean = np.mean(spectral_contrast, axis=1)

        # Combine all features
        self.features = np.concatenate([
            mfcc_mean,           # 13 features
            [centroid_mean],     # 1 feature
            [rolloff_mean],      # 1 feature
            [zcr_mean],          # 1 feature
            [rms_mean],          # 1 feature
            contrast_mean        # 7 features (default)
        ])

        # Normalize to prevent feature dominance
        if np.std(self.features) > 0:
            self.features = (self.features - np.mean(self.features)) / (np.std(self.features) + 1e-10)

        return self.features

    def get_duration(self):
        """Get duration in seconds"""
        return len(self.audio_data) / self.sample_rate

    def detect_vocals(self):
        """Detect if this segment contains vocals using improved spectral analysis"""
        if len(self.audio_data) == 0:
            self.has_vocals = False
            return False

        try:
            # Use harmonic-percussive source separation
            harmonic, percussive = librosa.effects.hpss(self.audio_data, margin=3.0)

            # Get spectral features
            spec_harmonic = np.abs(librosa.stft(harmonic))
            freqs = librosa.fft_frequencies(sr=self.sample_rate)

            # Human voice frequencies: 80-1100 Hz (fundamentals and lower harmonics)
            vocal_fundamental_mask = (freqs >= 80) & (freqs <= 1100)
            vocal_fundamental_energy = np.mean(spec_harmonic[vocal_fundamental_mask, :])

            # Mid-high frequencies for upper harmonics: 1100-8000 Hz
            vocal_harmonic_mask = (freqs >= 1100) & (freqs <= 8000)
            vocal_harmonic_energy = np.mean(spec_harmonic[vocal_harmonic_mask, :])

            # Very low frequencies (bass, kick drum, etc.): < 80 Hz
            bass_mask = freqs < 80
            bass_energy = np.mean(spec_harmonic[bass_mask, :])

            # Calculate harmonic-to-percussive ratio
            harmonic_energy = np.mean(np.abs(harmonic))
            percussive_energy = np.mean(np.abs(percussive))
            hp_ratio = harmonic_energy / (percussive_energy + 1e-10)

            # Check for spectral flux (vocals have more variation)
            spec_flux = np.mean(np.abs(np.diff(spec_harmonic, axis=1)))

            # Multiple criteria for vocal detection (balanced):
            # 1. Strong harmonic content relative to percussive
            # 2. Significant energy in vocal fundamental range
            # 3. Good balance between fundamental and upper harmonics
            # 4. Some spectral variation (not static)
            # 5. Not dominated by very low frequencies (bass)

            has_strong_harmonics = hp_ratio > 2.0  # Moderately strict
            has_vocal_fundamentals = vocal_fundamental_energy > 0.005  # Lower threshold
            has_balanced_harmonics = (vocal_harmonic_energy / (vocal_fundamental_energy + 1e-10)) > 0.2
            has_variation = spec_flux > 0.0005  # Lower threshold
            not_bass_heavy = bass_energy < (vocal_fundamental_energy * 3)  # More tolerant

            # Require at least 4 out of 5 criteria (more flexible)
            criteria_met = sum([has_strong_harmonics, has_vocal_fundamentals,
                               has_balanced_harmonics, has_variation, not_bass_heavy])

            self.has_vocals = criteria_met >= 4

            return self.has_vocals

        except Exception as e:
            print(f"Vocal detection error: {e}")
            self.has_vocals = False
            return False


class CandidatePattern:
    """Represents a candidate repeating pattern"""
    def __init__(self, segments, first_occurrence_index):
        self.segments = segments
        self.first_occurrence_index = first_occurrence_index
        self.repetition_count = len(segments)

    def get_representative_segment(self):
        """Return the first segment as representative"""
        return self.segments[0] if self.segments else None


class AudioAnalyzer:
    """Analyzes audio files for fixed-interval beat patterns"""

    def __init__(self):
        self.segments = []
        self.audio_data = None
        self.sample_rate = None
        self.energy_function = None
        self.beat_interval = None  # Fixed beat interval in samples
        self.beat_times = []  # List of beat positions in samples
        self.time_axis = None  # Time axis for energy function

    def analyze_file(self, file_path):
        """Analyze a WAV file for fixed-interval beat detection"""
        print(f"Loading audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=None)

        # Store for later use
        self.audio_data = y
        self.sample_rate = sr

        # Normalize: mean = 0, stddev = 1
        print("Normalizing audio...")
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # Calculate energy function with 50ms window using convolution (much faster)
        print("Calculating energy function...")
        window_ms = 50
        window_samples = int(window_ms * sr / 1000)

        # Use numpy convolution for fast computation
        y_squared = y_normalized ** 2
        window = np.ones(window_samples)
        self.energy_function = np.convolve(y_squared, window, mode='same')

        # Create time axis for plotting
        self.time_axis = np.arange(len(self.energy_function)) / sr

        # Find optimal fixed beat interval
        print("Finding optimal beat interval...")
        self.beat_interval = self._find_optimal_beat_interval()
        print(f"Optimal beat interval: {self.beat_interval / sr:.3f} seconds")

        # Detect beats with phase alignment
        print("Detecting beats with phase alignment...")
        self.beat_times = self._detect_beats_with_phase_shift()
        print(f"Found {len(self.beat_times)} beats")

        # Create segments from beat intervals
        print("Creating beat segments...")
        self._create_beat_segments()

        # Cluster beat intervals
        print("Clustering beat intervals...")
        self._cluster_beat_intervals()

        return len(self.segments), self.beat_interval / sr

    def _find_optimal_beat_interval(self):
        """Find the optimal fixed beat interval (0.3-1.5 seconds) using autocorrelation"""
        min_interval = int(0.3 * self.sample_rate)  # 0.3 seconds
        max_interval = int(1.5 * self.sample_rate)  # 1.5 seconds

        # Downsample energy function for faster computation
        downsample_factor = 100
        energy_downsampled = self.energy_function[::downsample_factor]

        # Try different intervals and find the one with maximum autocorrelation
        best_interval = min_interval
        best_score = -np.inf

        # Search with finer granularity for the 0.3-1s range (every 0.01s)
        test_intervals = np.arange(min_interval, max_interval + 1, int(0.01 * self.sample_rate))

        for interval_samples in test_intervals:
            interval_ds = interval_samples // downsample_factor

            if interval_ds >= len(energy_downsampled):
                continue

            # Calculate autocorrelation at this lag
            score = np.correlate(
                energy_downsampled[:-interval_ds],
                energy_downsampled[interval_ds:],
                mode='valid'
            ).mean()

            if score > best_score:
                best_score = score
                best_interval = interval_samples

        return best_interval

    def _detect_beats_with_phase_shift(self):
        """Detect beats allowing for phase shifts and missing beats"""
        beat_times = []

        # Start from beginning and find first beat (highest energy peak in first interval)
        first_interval_end = min(self.beat_interval * 2, len(self.energy_function))
        first_beat = np.argmax(self.energy_function[:first_interval_end])

        current_phase = first_beat
        beat_times.append(current_phase)

        # Continue finding beats with phase correction
        while current_phase + self.beat_interval < len(self.energy_function):
            # Expected next beat position
            expected_beat = current_phase + self.beat_interval

            # Search window around expected position (±20% of interval)
            search_window = int(self.beat_interval * 0.2)
            search_start = max(0, expected_beat - search_window)
            search_end = min(len(self.energy_function), expected_beat + search_window)

            # Find peak in search window
            search_region = self.energy_function[search_start:search_end]

            if len(search_region) == 0:
                break

            peak_in_window = np.argmax(search_region)
            actual_beat = search_start + peak_in_window

            # Check if this is a real beat (energy above threshold)
            avg_energy = np.mean(self.energy_function)
            threshold = 0.3 * avg_energy

            if self.energy_function[actual_beat] > threshold:
                beat_times.append(actual_beat)
                current_phase = actual_beat
            else:
                # Missing beat - mark position anyway but will be handled in segmentation
                beat_times.append(expected_beat)
                current_phase = expected_beat

        # Align beat intervals to have consistent lengths
        beat_times = self._align_beat_intervals(beat_times)

        return beat_times

    def _align_beat_intervals(self, beat_times):
        """Align beat positions so intervals have consistent lengths across multiple consecutive intervals"""
        if len(beat_times) < 3:
            return beat_times

        # Perform multiple passes for better alignment
        aligned_beats = list(beat_times)

        # Pass 1: Multi-interval realignment
        aligned_beats = self._multi_interval_realignment(aligned_beats)

        # Pass 2: Local pairwise adjustments
        aligned_beats = self._pairwise_realignment(aligned_beats)

        return aligned_beats

    def _multi_interval_realignment(self, beat_times):
        """Realign multiple consecutive intervals by redistributing time"""
        if len(beat_times) < 4:
            return beat_times

        aligned_beats = [beat_times[0]]
        tolerance = self.beat_interval * 0.08  # 8% tolerance

        i = 1
        while i < len(beat_times) - 1:
            # Look ahead to find a sequence of intervals that need alignment
            window_size = min(5, len(beat_times) - i)  # Check up to 5 consecutive intervals

            # Calculate total deviation over the window
            total_actual_time = beat_times[i + window_size - 1] - aligned_beats[-1]
            total_expected_time = self.beat_interval * window_size
            total_deviation = total_actual_time - total_expected_time

            # If total deviation is significant, realign all beats in this window
            if abs(total_deviation) > tolerance * window_size:
                # Redistribute the beats evenly over the total time span
                realigned_window = self._redistribute_beats_in_window(
                    aligned_beats[-1],
                    beat_times[i + window_size - 1],
                    window_size
                )

                # Verify that realignment improves consistency
                if self._is_better_alignment(beat_times[i:i+window_size], realigned_window):
                    aligned_beats.extend(realigned_window)
                    i += window_size
                    continue

            # No multi-interval adjustment needed, just add the beat
            aligned_beats.append(beat_times[i])
            i += 1

        # Add last beat if not already added
        if len(aligned_beats) < len(beat_times):
            aligned_beats.append(beat_times[-1])

        return aligned_beats

    def _redistribute_beats_in_window(self, start_time, end_time, num_intervals):
        """Redistribute beat positions evenly in a time window, aligning to energy peaks"""
        total_duration = end_time - start_time
        avg_interval = total_duration / num_intervals

        new_beats = []
        for i in range(1, num_intervals):
            # Expected position
            expected_pos = int(start_time + i * avg_interval)

            # Search for energy peak near expected position
            search_window = int(self.beat_interval * 0.15)
            search_start = max(0, expected_pos - search_window)
            search_end = min(len(self.energy_function), expected_pos + search_window)

            if search_end > search_start:
                search_region = self.energy_function[search_start:search_end]
                if len(search_region) > 0:
                    peak_offset = np.argmax(search_region)
                    new_beat = search_start + peak_offset
                    new_beats.append(new_beat)
                else:
                    new_beats.append(expected_pos)
            else:
                new_beats.append(expected_pos)

        return new_beats

    def _is_better_alignment(self, original_beats, new_beats):
        """Check if new alignment has lower deviation than original"""
        if len(original_beats) != len(new_beats):
            return False

        # Calculate total deviation for original
        original_deviations = [abs((original_beats[i] - original_beats[i-1]) - self.beat_interval)
                              for i in range(1, len(original_beats))]
        original_total_dev = sum(original_deviations)

        # Calculate total deviation for new alignment (need previous beat)
        # This is approximate since we don't have the previous beat context
        new_deviations = []
        for i in range(1, len(new_beats)):
            dev = abs((new_beats[i] - new_beats[i-1]) - self.beat_interval)
            new_deviations.append(dev)
        new_total_dev = sum(new_deviations)

        # New alignment is better if it reduces total deviation by at least 20%
        return new_total_dev < original_total_dev * 0.8

    def _pairwise_realignment(self, beat_times):
        """Fine-tune alignment by adjusting pairs of consecutive intervals"""
        if len(beat_times) < 3:
            return beat_times

        aligned_beats = [beat_times[0]]
        tolerance = self.beat_interval * 0.1

        i = 1
        while i < len(beat_times) - 1:
            current_interval = beat_times[i] - aligned_beats[-1]
            next_interval = beat_times[i+1] - beat_times[i]

            # Calculate deviations
            current_deviation = current_interval - self.beat_interval
            next_deviation = next_interval - self.beat_interval

            # Check if intervals compensate each other
            if abs(current_deviation + next_deviation) < tolerance and abs(current_deviation) > tolerance:
                # Find better beat position
                search_start = aligned_beats[-1] + self.beat_interval - int(self.beat_interval * 0.15)
                search_end = aligned_beats[-1] + self.beat_interval + int(self.beat_interval * 0.15)
                search_start = max(0, search_start)
                search_end = min(len(self.energy_function), search_end)

                if search_end > search_start:
                    search_region = self.energy_function[search_start:search_end]
                    if len(search_region) > 0:
                        peak_pos = np.argmax(search_region)
                        new_beat = search_start + peak_pos

                        new_interval1 = new_beat - aligned_beats[-1]
                        new_interval2 = beat_times[i+1] - new_beat
                        new_dev1 = abs(new_interval1 - self.beat_interval)
                        new_dev2 = abs(new_interval2 - self.beat_interval)

                        if new_dev1 < abs(current_deviation) and new_dev2 < abs(next_deviation):
                            aligned_beats.append(new_beat)
                            i += 1
                            continue

            aligned_beats.append(beat_times[i])
            i += 1

        # Add last beat
        if len(beat_times) > 1:
            aligned_beats.append(beat_times[-1])

        return aligned_beats

    def _create_beat_segments(self):
        """Create segments from detected beats, marking missing beats"""
        self.segments = []
        avg_energy = np.mean(self.energy_function)
        threshold = 0.3 * avg_energy

        for idx in range(len(self.beat_times) - 1):
            start_sample = self.beat_times[idx]
            end_sample = self.beat_times[idx + 1]

            # Check if this interval contains a real beat (high energy)
            interval_energy = self.energy_function[start_sample:end_sample]
            max_energy = np.max(interval_energy) if len(interval_energy) > 0 else 0
            is_beat = max_energy > threshold

            # Extract audio for this segment
            segment_audio = self.audio_data[start_sample:end_sample]

            segment = BeatSegment(
                segment_audio,
                start_sample,
                end_sample,
                self.sample_rate,
                idx,
                is_beat=is_beat
            )
            self.segments.append(segment)

    def _cluster_beat_intervals(self):
        """Cluster beat intervals based on musical features and assign labels"""
        # Only cluster beat intervals (not missing beats)
        beat_segments = [seg for seg in self.segments if seg.is_beat]

        if len(beat_segments) < 2:
            return

        # Extract features for all beat segments
        print("Extracting features...")
        features_list = []
        for seg in beat_segments:
            seg.extract_features()
            features_list.append(seg.features)

        features_matrix = np.array(features_list)

        # Standardize features before clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)

        print("Performing clustering...")

        # Determine optimal number of clusters (between 10 and min(30, num_segments/3))
        n_segments = len(beat_segments)
        max_clusters = min(30, max(10, n_segments // 3))
        target_clusters = min(15, max_clusters)  # Aim for 15 clusters

        print(f"  Targeting {target_clusters} clusters (max: {max_clusters})")

        # Use K-Means clustering for guaranteed number of clusters
        kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)

        n_clusters = len(set(cluster_ids))
        print(f"  Found {n_clusters} distinct patterns")

        # Assign cluster IDs to segments
        for seg, cluster_id in zip(beat_segments, cluster_ids):
            seg.cluster_id = cluster_id

        # Create labels (A, B, C, ...) based on order of first appearance
        cluster_first_appearance = {}
        for seg in beat_segments:
            if seg.cluster_id not in cluster_first_appearance:
                cluster_first_appearance[seg.cluster_id] = seg.index

        # Sort clusters by first appearance
        sorted_clusters = sorted(cluster_first_appearance.items(), key=lambda x: x[1])

        # Assign letter labels
        cluster_to_label = {}
        label_index = 0
        for cluster_id, _ in sorted_clusters:
            if cluster_id == -1:
                cluster_to_label[cluster_id] = "?"  # Noise/outliers
            else:
                cluster_to_label[cluster_id] = chr(ord('A') + label_index)
                label_index += 1

        # Apply labels to segments
        for seg in beat_segments:
            seg.cluster_label = cluster_to_label[seg.cluster_id]

        print(f"Found {label_index} distinct beat patterns")

        # Perform sub-clustering within each cluster to find identical beats
        print("Performing sub-clustering for identical beats...")
        self._sub_cluster_beats(beat_segments)

        # Detect vocals in each segment
        print("Detecting vocals...")
        for seg in beat_segments:
            seg.detect_vocals()

        # Print song sequence
        self._print_song_sequence()

    def _sub_cluster_beats(self, beat_segments):
        """Perform sub-clustering within each main cluster using FFT similarity"""
        # Group segments by cluster
        clusters = {}
        for seg in beat_segments:
            if seg.cluster_id not in clusters:
                clusters[seg.cluster_id] = []
            clusters[seg.cluster_id].append(seg)

        # Sub-cluster each main cluster
        for cluster_id, segs in clusters.items():
            if len(segs) < 2:
                segs[0].sub_cluster_id = 1
                segs[0].pattern_label = f"{segs[0].cluster_label}1"
                continue

            # Compute FFT amplitude vectors for all segments in this cluster
            fft_features = []
            for seg in segs:
                fft_amp = self._compute_fft_amplitude(seg.audio_data)
                fft_features.append(fft_amp)

            # Use cosine distance to find nearly identical beats
            # Build similarity matrix
            n = len(segs)
            assigned = [False] * n
            sub_cluster_assignments = [-1] * n
            next_num = 1

            # Cosine similarity threshold (relaxed - must be > 0.90 to be identical)
            similarity_threshold = 0.90

            for i in range(n):
                if assigned[i]:
                    continue

                # Start new sub-cluster
                sub_cluster_assignments[i] = next_num
                assigned[i] = True

                # Find all other beats that are nearly identical to this one
                for j in range(i + 1, n):
                    if assigned[j]:
                        continue

                    # Compute cosine similarity between FFT features
                    similarity = self._cosine_similarity(fft_features[i], fft_features[j])

                    # If very similar, assign to same sub-cluster
                    if similarity >= similarity_threshold:
                        sub_cluster_assignments[j] = next_num
                        assigned[j] = True

                next_num += 1

            # Assign sub-cluster IDs and create pattern labels
            for seg, sub_id in zip(segs, sub_cluster_assignments):
                seg.sub_cluster_id = sub_id
                seg.pattern_label = f"{seg.cluster_label}{seg.sub_cluster_id}"

    def _compute_fft_amplitude(self, audio_data):
        """Compute FFT amplitude vector for a beat segment"""
        # Perform FFT
        fft_result = np.fft.fft(audio_data)

        # Take only the positive frequencies (first half)
        n = len(fft_result)
        fft_positive = fft_result[:n//2]

        # Get amplitudes (magnitude)
        amplitudes = np.abs(fft_positive)

        # Normalize to unit length for better comparison
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm

        return amplitudes

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        # Ensure vectors are same length (pad shorter one with zeros)
        max_len = max(len(vec1), len(vec2))
        if len(vec1) < max_len:
            vec1 = np.pad(vec1, (0, max_len - len(vec1)))
        if len(vec2) < max_len:
            vec2 = np.pad(vec2, (0, max_len - len(vec2)))

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _print_song_sequence(self):
        """Print the song sequence showing pattern labels in order"""
        # Get only beat segments (not missing beats) in order
        beat_segs = [s for s in self.segments if s.is_beat]

        if not beat_segs:
            return

        labels = [seg.pattern_label for seg in beat_segs]

        print("\n" + "="*60)
        print("SONG SEQUENCE (FULL):")
        print("="*60)

        # Print full sequence in chunks of 20 for readability
        chunk_size = 20
        for i in range(0, len(labels), chunk_size):
            chunk = labels[i:i+chunk_size]
            print(" ".join(chunk))

        print("\n" + "="*60)
        print("SONG SEQUENCE (CONDENSED):")
        print("="*60)

        # Create condensed sequence with exponents and brackets
        condensed = self._condense_sequence(labels)
        print(condensed)

        print("="*60 + "\n")

    def _condense_sequence(self, labels):
        """Condense sequence using exponents and nested brackets"""
        if not labels:
            return ""

        # Apply condensation iteratively until no more patterns found
        prev_result = None
        current = labels
        depth = 0

        while True:
            result_str = self._find_and_condense_patterns(current, depth)

            # If nothing changed, we're done
            if result_str == prev_result:
                break

            prev_result = result_str
            # Parse result back into tokens for next iteration
            current = self._parse_condensed_to_tokens(result_str)
            depth += 1

            # Safety limit on iterations
            if depth > 5:
                break

        return result_str

    def _parse_condensed_to_tokens(self, condensed_str):
        """Parse a condensed string back into tokens for further condensation"""
        import re
        # Split by spaces but keep grouped expressions together
        tokens = []
        current_token = ""
        bracket_depth = 0

        for char in condensed_str:
            if char in '([{':
                bracket_depth += 1
                current_token += char
            elif char in ')]}':
                bracket_depth -= 1
                current_token += char
            elif char == ' ' and bracket_depth == 0:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens

    def _find_and_condense_patterns(self, sequence, depth):
        """Find and condense repeating patterns with brackets"""
        if len(sequence) <= 1:
            return " ".join(sequence) if sequence else ""

        brackets = ["()", "[]", "{}"]  # Different bracket levels
        bracket_pair = brackets[min(depth, len(brackets)-1)]
        open_b, close_b = bracket_pair[0], bracket_pair[1]

        result = []
        i = 0

        while i < len(sequence):
            # Check for consecutive identical elements
            j = i + 1
            while j < len(sequence) and sequence[j] == sequence[i]:
                j += 1

            count = j - i
            if count > 1:
                # Multiple identical elements -> use exponent
                result.append(f"{sequence[i]}^{count}")
                i = j
            else:
                # Single element - check for larger repeating patterns
                best_pattern_len = 0
                best_repeat_count = 0

                # Try different pattern lengths
                for pattern_len in range(2, (len(sequence) - i) // 2 + 1):
                    if i + pattern_len > len(sequence):
                        break

                    pattern = sequence[i:i+pattern_len]
                    repeat_count = 1

                    # Count how many times this pattern repeats
                    pos = i + pattern_len
                    while pos + pattern_len <= len(sequence):
                        if sequence[pos:pos+pattern_len] == pattern:
                            repeat_count += 1
                            pos += pattern_len
                        else:
                            break

                    # Keep the longest repeating pattern
                    if repeat_count >= 2 and pattern_len > best_pattern_len:
                        best_pattern_len = pattern_len
                        best_repeat_count = repeat_count

                if best_pattern_len > 0:
                    # Found a repeating pattern
                    pattern = sequence[i:i+best_pattern_len]
                    condensed_pattern = " ".join(pattern)
                    result.append(f"{open_b}{condensed_pattern}{close_b}^{best_repeat_count}")
                    i += best_pattern_len * best_repeat_count
                else:
                    # No pattern found, just add the element
                    result.append(sequence[i])
                    i += 1

        return " ".join(result)



class AudioPlayer:
    """Handles seamless looping audio playback"""

    def __init__(self):
        self.playing = False
        self.stream = None
        self.audio_queue = queue.Queue()
        self.current_audio = None
        self.sample_rate = None
        self.playback_thread = None
        self.next_segment = None
        self.should_loop = True
        self.play_count = 0

    def play_loop(self, segment, callback=None):
        """Play a segment in a seamless loop"""
        self.stop()

        self.current_audio = segment.audio_data
        self.sample_rate = segment.sample_rate
        self.playing = True
        self.should_loop = True
        self.play_count = 0
        self.completion_callback = callback

        # Start playback in a separate thread
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def queue_next(self, segment):
        """Queue a segment to play after current one completes one loop"""
        self.next_segment = segment
        self.should_loop = False  # Stop looping after current iteration

    def play_once(self, segment):
        """Play a segment once (no loop)"""
        self.stop()

        self.current_audio = segment.audio_data
        self.sample_rate = segment.sample_rate

        # Play once in a separate thread
        self.playback_thread = threading.Thread(target=self._playback_once)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def play_sequence_once(self, segments):
        """Play multiple segments in sequence once"""
        self.stop()

        # Concatenate all segment audio
        if not segments:
            return

        self.sample_rate = segments[0].sample_rate
        audio_parts = [seg.audio_data for seg in segments]
        self.current_audio = np.concatenate(audio_parts)

        # Play once in a separate thread
        self.playback_thread = threading.Thread(target=self._playback_once)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def play_sequence_loop(self, segments):
        """Play multiple segments in sequence in a loop"""
        self.stop()

        # Concatenate all segment audio
        if not segments:
            return

        self.sample_rate = segments[0].sample_rate
        audio_parts = [seg.audio_data for seg in segments]
        self.current_audio = np.concatenate(audio_parts)
        self.playing = True

        # Start looping playback in a separate thread
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def _playback_once(self):
        """Internal method for single playback"""
        try:
            sd.play(self.current_audio, self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")

    def _playback_loop(self):
        """Internal method for continuous playback using callback for seamless looping"""
        position = 0
        loop_completed = False

        def callback(outdata, frames, time, status):
            nonlocal position, loop_completed

            if status:
                print(status)

            if not self.playing:
                raise sd.CallbackStop()

            audio_len = len(self.current_audio)

            # Fill the output buffer, looping seamlessly
            remaining = frames
            offset = 0

            while remaining > 0:
                available = audio_len - position
                to_copy = min(remaining, available)

                outdata[offset:offset + to_copy, 0] = self.current_audio[position:position + to_copy]

                position += to_copy
                offset += to_copy
                remaining -= to_copy

                # Check if we reached the end
                if position >= audio_len:
                    self.play_count += 1

                    # If we shouldn't loop and have completed one iteration, signal to stop
                    if not self.should_loop and self.play_count >= 1:
                        loop_completed = True
                        self.playing = False
                        raise sd.CallbackStop()

                    position = 0

        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=callback,
                blocksize=2048
            ):
                while self.playing:
                    sd.sleep(100)

            # After playback ends, check if we have a queued segment
            if loop_completed and self.next_segment:
                next_seg = self.next_segment
                self.next_segment = None
                if hasattr(self, 'completion_callback') and self.completion_callback:
                    self.completion_callback(next_seg)

        except Exception as e:
            if self.playing:  # Only print error if not intentionally stopped
                print(f"Playback error: {e}")

    def stop(self):
        """Stop playback"""
        was_playing = self.playing
        self.playing = False
        self.next_segment = None  # Clear any queued segment

        # Give callback time to stop
        if was_playing:
            sd.sleep(50)

        try:
            sd.stop()
        except:
            pass  # Ignore errors when stopping

        # Wait briefly for thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=0.5)


class BeatAnalyzerGUI:
    """GUI for the beat analyzer application"""

    def __init__(self, root, initial_files=None):
        self.root = root
        self.root.title("Beat Pattern Analyzer")
        self.root.geometry("1200x800")

        self.analyzer = AudioAnalyzer()
        self.player = AudioPlayer()
        self.current_files = []
        self.current_segment = None

        # Matplotlib figure for energy function
        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = None
        self.toolbar = None

        self._create_widgets()

        # Load and analyze initial files if provided
        if initial_files:
            self.current_files = list(initial_files)
            self.file_label.config(text=f"{len(initial_files)} file(s) selected")
            # Schedule analysis to run after GUI is fully initialized
            self.root.after(100, self._analyze_files)

    def _create_widgets(self):
        """Create GUI widgets"""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="WAV Files:").pack(side=tk.LEFT)

        select_btn = ttk.Button(
            file_frame,
            text="Select Files",
            command=self._select_files
        )
        select_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(file_frame, text="No files selected")
        self.file_label.pack(side=tk.LEFT, padx=5)

        # Status frame
        status_frame = ttk.Frame(self.root, padding="10")
        status_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        # Energy function graph frame
        graph_frame = ttk.Frame(self.root, padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(graph_frame, text="Energy Function:").pack(anchor=tk.W)

        # Create matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()

        # Song sequence frame
        sequence_frame = ttk.Frame(self.root, padding="10")
        sequence_frame.pack(fill=tk.BOTH, expand=True)

        # Header with label and play button
        seq_header = ttk.Frame(sequence_frame)
        seq_header.pack(fill=tk.X)

        ttk.Label(seq_header, text="Song Sequence (Condensed):").pack(side=tk.LEFT)

        self.play_seq_btn = ttk.Button(
            seq_header,
            text="Play Sequence",
            command=self._play_sequence
        )
        self.play_seq_btn.pack(side=tk.RIGHT, padx=5)

        # Text widget for song sequence (multiline, scrollable)
        self.sequence_text = tk.Text(sequence_frame, height=4, wrap=tk.WORD, font=('Courier', 10))
        self.sequence_text.pack(fill=tk.BOTH, expand=True)

        seq_scrollbar = ttk.Scrollbar(sequence_frame, orient=tk.VERTICAL, command=self.sequence_text.yview)
        self.sequence_text.configure(yscrollcommand=seq_scrollbar.set)
        seq_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Beat segments list frame
        list_frame = ttk.Frame(self.root, padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(list_frame, text="Beat Segments:").pack(anchor=tk.W)

        # Create treeview for segments
        columns = ("Index", "Time", "Type", "Pattern", "Duration")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("Index", text="Index")
        self.tree.heading("Time", text="Time (s)")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Pattern", text="Pattern")
        self.tree.heading("Duration", text="Duration (s)")

        self.tree.column("Index", width=50)
        self.tree.column("Time", width=80)
        self.tree.column("Type", width=80)
        self.tree.column("Pattern", width=60)
        self.tree.column("Duration", width=80)

        # Configure tags for coloring
        self.tree.tag_configure('beat', background='lightgreen')
        self.tree.tag_configure('beat_vocal', background='lightblue')  # Blue for beats with vocals
        self.tree.tag_configure('no_beat', background='lightcoral')

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind click to play
        self.tree.bind("<ButtonRelease-1>", self._on_segment_click)

        # Playback controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        self.stop_btn = ttk.Button(
            control_frame,
            text="Stop",
            command=self._stop_playback
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.playback_label = ttk.Label(control_frame, text="Not playing")
        self.playback_label.pack(side=tk.LEFT, padx=10)

    def _select_files(self):
        """Open file dialog to select WAV files"""
        files = filedialog.askopenfilenames(
            title="Select WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if files:
            self.current_files = list(files)
            self.file_label.config(text=f"{len(files)} file(s) selected")
            # Auto-start analysis when files are selected
            self._analyze_files()

    def _on_segment_click(self, event):
        """Handle click on segment - play it in a loop"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        idx = int(item["values"][0])

        self._start_looping_segment(idx)

    def _start_looping_segment(self, idx):
        """Start looping a specific segment"""
        if idx < len(self.analyzer.segments):
            self.current_segment = idx
            segment = self.analyzer.segments[idx]

            self.player.play_loop(segment)
            segment_type = "Beat" if segment.is_beat else "No Beat"
            self.playback_label.config(text=f"Playing segment {idx} ({segment_type}) in loop")

    def _stop_playback(self):
        """Stop playback"""
        self.player.stop()
        self.current_segment = None
        self.playback_label.config(text="Not playing")

    def _play_sequence(self):
        """Play the song sequence using first occurrence of each pattern"""
        if not self.analyzer.segments:
            return

        # Get beat segments with pattern labels
        beat_segs = [s for s in self.analyzer.segments if s.is_beat]
        if not beat_segs:
            return

        # Create a mapping of pattern_label -> first segment with that label
        pattern_map = {}
        for seg in beat_segs:
            if seg.pattern_label not in pattern_map:
                pattern_map[seg.pattern_label] = seg

        # Get the sequence of pattern labels
        labels = [s.pattern_label for s in beat_segs]

        # Build list of segments to play (using first occurrence of each pattern)
        sequence_to_play = []
        for label in labels:
            if label in pattern_map:
                sequence_to_play.append(pattern_map[label])

        # Play the sequence
        if sequence_to_play:
            self.player.play_sequence_once(sequence_to_play)
            self.playback_label.config(text=f"Playing sequence ({len(sequence_to_play)} beats)")

    def _plot_energy_function(self):
        """Plot the energy function with beat markers"""
        self.ax.clear()

        if self.analyzer.energy_function is None or self.analyzer.time_axis is None:
            return

        # Plot energy function
        self.ax.plot(self.analyzer.time_axis, self.analyzer.energy_function, 'b-', linewidth=0.5, label='Energy')

        # Plot beat markers
        if self.analyzer.beat_times:
            beat_times_sec = np.array(self.analyzer.beat_times) / self.analyzer.sample_rate
            beat_energies = self.analyzer.energy_function[self.analyzer.beat_times]

            self.ax.plot(beat_times_sec, beat_energies, 'ro', markersize=3, label='Beats')

            # Add vertical lines for beats
            for i, (beat_time, segment) in enumerate(zip(beat_times_sec[:-1], self.analyzer.segments)):
                color = 'green' if segment.is_beat else 'red'
                alpha = 0.3 if segment.is_beat else 0.2
                next_beat = beat_times_sec[i + 1] if i + 1 < len(beat_times_sec) else self.analyzer.time_axis[-1]
                self.ax.axvspan(beat_time, next_beat, color=color, alpha=alpha)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Energy')
        self.ax.set_title('Energy Function with Beat Detection')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def _analyze_files(self):
        """Analyze selected files"""
        if not self.current_files:
            self.status_label.config(text="No files selected!")
            return

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Analyze first file (extend for multiple files if needed)
        file_path = self.current_files[0]
        self.status_label.config(text=f"Analyzing {file_path}...")
        self.root.update()

        try:
            num_segments, beat_interval = self.analyzer.analyze_file(file_path)

            # Plot energy function
            self._plot_energy_function()

            # Populate treeview with segments
            for idx, segment in enumerate(self.analyzer.segments):
                time_sec = segment.start_sample / self.analyzer.sample_rate
                duration = segment.get_duration()

                # Determine segment type with vocal indicator
                if segment.is_beat:
                    segment_type = "Beat+Vocal" if segment.has_vocals else "Beat"
                    pattern_label = segment.pattern_label if hasattr(segment, 'pattern_label') and segment.pattern_label else segment.cluster_label
                else:
                    segment_type = "No Beat"
                    pattern_label = "-"

                # Choose tag based on vocals
                if not segment.is_beat:
                    tag = 'no_beat'
                elif segment.has_vocals:
                    tag = 'beat_vocal'
                else:
                    tag = 'beat'

                self.tree.insert(
                    "",
                    tk.END,
                    values=(idx, f"{time_sec:.2f}", segment_type, pattern_label, f"{duration:.2f}"),
                    tags=(tag,)
                )

            # Get condensed song sequence for display
            beat_segs = [s for s in self.analyzer.segments if s.is_beat]
            if beat_segs:
                labels = [s.pattern_label for s in beat_segs]
                condensed_seq = self.analyzer._condense_sequence(labels)

                # Update sequence text widget (multiline)
                self.sequence_text.config(state=tk.NORMAL)
                self.sequence_text.delete(1.0, tk.END)
                # Wrap text nicely at word boundaries
                words = condensed_seq.split()
                line_length = 0
                max_line_length = 80
                for word in words:
                    if line_length + len(word) + 1 > max_line_length and line_length > 0:
                        self.sequence_text.insert(tk.END, "\n")
                        line_length = 0
                    if line_length > 0:
                        self.sequence_text.insert(tk.END, " ")
                        line_length += 1
                    self.sequence_text.insert(tk.END, word)
                    line_length += len(word)
                self.sequence_text.config(state=tk.DISABLED)

                # Shorter version for status bar
                if len(condensed_seq) > 100:
                    condensed_preview = condensed_seq[:100] + "..."
                else:
                    condensed_preview = condensed_seq
                status_text = f"Found {num_segments} segments (interval: {beat_interval:.3f}s)"
            else:
                status_text = f"Found {num_segments} segments with beat interval {beat_interval:.3f}s"
                self.sequence_text.config(state=tk.NORMAL)
                self.sequence_text.delete(1.0, tk.END)
                self.sequence_text.insert(tk.END, "No sequence available")
                self.sequence_text.config(state=tk.DISABLED)

            self.status_label.config(text=status_text)
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Beat Pattern Analyzer - Analyze rhythmic patterns in WAV files")
    parser.add_argument('files', nargs='*', help='WAV file(s) to analyze')
    args = parser.parse_args()

    # Validate file paths if provided
    initial_files = None
    if args.files:
        import os
        valid_files = [f for f in args.files if os.path.isfile(f) and f.lower().endswith('.wav')]
        if valid_files:
            initial_files = valid_files
        elif args.files:
            print(f"Warning: No valid WAV files found in provided arguments")

    root = tk.Tk()
    app = BeatAnalyzerGUI(root, initial_files=initial_files)
    root.mainloop()


if __name__ == "__main__":
    main()
