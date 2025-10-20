import librosa
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
import sys
import argparse


class BeatSegment:
    """Represents a segment of audio between two beats"""
    def __init__(self, audio_data, start_sample, end_sample, sample_rate, index):
        self.audio_data = audio_data
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.sample_rate = sample_rate
        self.index = index
        self.features = None
        self.cluster_id = -1  # Default to -1 (noise)

    def extract_features(self):
        """Extract MFCC features for similarity comparison"""
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
        self.features = np.mean(mfcc, axis=1)
        return self.features


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
    """Analyzes audio files for beat patterns and repetitions"""

    def __init__(self):
        self.segments = []
        self.candidates = []
        self.audio_data = None
        self.sample_rate = None
        self.energy_function = None

    def analyze_file(self, file_path):
        """Analyze a WAV file for energy-based loop detection"""
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

        # Find silence centers
        print("Finding silence centers...")
        silence_centers = self._find_silence_centers()
        print(f"Found {len(silence_centers)} silence centers")

        # Find best matching pairs
        print("Analyzing segment pairs...")
        self._find_candidates_from_silence_centers(silence_centers)

        # Create segments from candidates for playback
        self.segments = []
        for idx, candidate in enumerate(self.candidates):
            t1, t2 = candidate.first_occurrence_index, candidate.first_occurrence_index + candidate.repetition_count
            segment_audio = self.audio_data[t1:t2]
            segment = BeatSegment(segment_audio, t1, t2, sr, idx)
            segment.cluster_id = idx  # Each candidate is its own cluster
            self.segments.append(segment)

        return len(self.segments), len(self.candidates)

    def _find_silence_centers(self):
        """Find centers of minimal energy phases"""
        avg_energy = np.mean(self.energy_function)
        threshold = 0.05 * avg_energy

        # Find intervals where energy is below threshold
        below_threshold = self.energy_function < threshold

        # Find continuous intervals
        silence_intervals = []
        in_silence = False
        start_idx = 0

        for i in range(len(below_threshold)):
            if below_threshold[i] and not in_silence:
                start_idx = i
                in_silence = True
            elif not below_threshold[i] and in_silence:
                silence_intervals.append((start_idx, i - 1))
                in_silence = False

        # Handle case where file ends in silence
        if in_silence:
            silence_intervals.append((start_idx, len(below_threshold) - 1))

        # Get center of each interval
        silence_centers = []
        for start, end in silence_intervals:
            center = (start + end) // 2
            silence_centers.append(center)

        return silence_centers

    def _find_candidates_from_silence_centers(self, silence_centers):
        """Find best matching pairs of silence centers"""
        if len(silence_centers) < 2:
            self.candidates = []
            return

        # Limit number of silence centers to check for performance
        # Take evenly spaced centers if there are too many
        max_centers = 50
        if len(silence_centers) > max_centers:
            step = len(silence_centers) // max_centers
            silence_centers = silence_centers[::step]

        # Calculate deviation for all pairs using vectorized operations
        pair_deviations = []

        for i in range(len(silence_centers)):
            for j in range(i + 1, min(i + 20, len(silence_centers))):  # Limit pairs per center
                t1 = silence_centers[i]
                t2 = silence_centers[j]
                segment_length = t2 - t1

                # Skip very short or very long segments
                if segment_length < 1000 or t2 + segment_length >= len(self.energy_function):
                    continue

                # Use vectorized numpy operations for speed
                segment1 = self.energy_function[t1:t2]
                segment2_start = t1 + segment_length
                segment2_end = t2 + segment_length

                if segment2_end <= len(self.energy_function):
                    segment2 = self.energy_function[segment2_start:segment2_end]

                    # Vectorized calculation of average absolute deviation
                    avg_deviation = np.mean(np.abs(segment1 - segment2))
                    pair_deviations.append((avg_deviation, t1, t2))

        # Sort by deviation (lower is better - more similar)
        pair_deviations.sort(key=lambda x: x[0])

        # Take top 30 best matches
        top_pairs = pair_deviations[:30]

        # Create candidates from top pairs
        self.candidates = []
        for idx, (deviation, t1, t2) in enumerate(top_pairs):
            segment_length = t2 - t1
            # Create a dummy list with one segment for compatibility
            segments = [None]  # Will use actual audio in segments list
            candidate = CandidatePattern(segments, t1)
            candidate.repetition_count = segment_length  # Store segment length here
            candidate.deviation = deviation
            self.candidates.append(candidate)

        print(f"Found {len(self.candidates)} candidate patterns")


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
        self.root.geometry("800x600")

        self.analyzer = AudioAnalyzer()
        self.player = AudioPlayer()
        self.current_files = []
        self.show_beats_mode = False
        self.current_candidate = None
        self.queued_candidate = None

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

        # Candidates list frame
        list_frame = ttk.Frame(self.root, padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(list_frame, text="Candidate Patterns:").pack(anchor=tk.W)

        # Create treeview for candidates
        columns = ("Index", "Position", "Deviation", "Repetitions", "Duration")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended")

        self.tree.heading("Index", text="Index")
        self.tree.heading("Position", text="Position")
        self.tree.heading("Deviation", text="Deviation")
        self.tree.heading("Repetitions", text="Reps")
        self.tree.heading("Duration", text="Duration (s)")

        self.tree.column("Index", width=60)
        self.tree.column("Position", width=100)
        self.tree.column("Deviation", width=100)
        self.tree.column("Repetitions", width=60)
        self.tree.column("Duration", width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind single click to queue, double-click to play immediately
        self.tree.bind("<ButtonRelease-1>", self._on_candidate_single_click)
        self.tree.bind("<Double-Button-1>", self._on_candidate_double_click)

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

    def _on_candidate_single_click(self, event):
        """Handle single click on candidate - queue it to play after current loop"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        idx = int(item["values"][0])

        # If already playing, queue this one to play after current loop completes
        if self.player.playing:
            segment = self.analyzer.segments[idx]
            self.player.queue_next(segment)
            self.queued_candidate = idx
            self.playback_label.config(text=f"Playing candidate {self.current_candidate} (Queued: {idx})")
        else:
            self._start_looping_candidate(idx)

    def _on_candidate_double_click(self, event):
        """Handle double-click on candidate - stop current and play immediately"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        idx = int(item["values"][0])

        # Stop current playback and start new one immediately
        self.player.stop()
        self._start_looping_candidate(idx)

    def _start_looping_candidate(self, idx):
        """Start looping a specific candidate"""
        if idx < len(self.analyzer.segments):
            self.current_candidate = idx
            self.queued_candidate = None
            segment = self.analyzer.segments[idx]

            # Pass callback to handle queued segments
            def on_complete(next_seg):
                # Find the index of the queued segment
                for i, seg in enumerate(self.analyzer.segments):
                    if seg == next_seg:
                        self.root.after(0, lambda: self._start_looping_candidate(i))
                        break

            self.player.play_loop(segment, callback=on_complete)
            self.playback_label.config(text=f"Playing candidate {idx} in loop")

    def _stop_playback(self):
        """Stop playback"""
        self.player.stop()
        self.current_candidate = None
        self.queued_candidate = None
        self.playback_label.config(text="Not playing")

    def _show_beats(self):
        """Show all beat segments"""
        if not self.analyzer.segments:
            self.status_label.config(text="No analysis data! Run Analyze first.")
            return

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Populate treeview with all beat segments
        for idx, segment in enumerate(self.analyzer.segments):
            duration = len(segment.audio_data) / segment.sample_rate
            cluster_id = segment.cluster_id if hasattr(segment, 'cluster_id') else -1
            cluster_label = str(cluster_id) if cluster_id != -1 else "noise"

            self.tree.insert(
                "",
                tk.END,
                values=(idx, f"Beat {idx}", cluster_label, 1, f"{duration:.2f}")
            )

        self.show_beats_mode = True
        self.status_label.config(text=f"Showing all {len(self.analyzer.segments)} beat segments")

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
            num_segments, num_candidates = self.analyzer.analyze_file(file_path)

            # Populate treeview with candidates
            for idx, candidate in enumerate(self.analyzer.candidates):
                position = candidate.first_occurrence_index
                t1 = position
                t2 = position + candidate.repetition_count
                duration = (t2 - t1) / self.analyzer.sample_rate
                deviation = candidate.deviation if hasattr(candidate, 'deviation') else 0
                deviation_str = f"{deviation:.2e}"  # Scientific notation for small numbers

                self.tree.insert(
                    "",
                    tk.END,
                    values=(idx, f"Sample {position}", deviation_str, 2, f"{duration:.2f}")
                )

            self.show_beats_mode = False
            self.status_label.config(
                text=f"Found {num_candidates} candidates from {num_segments} segments"
            )
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Analysis error: {e}")


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
