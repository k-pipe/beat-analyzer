# Beat Pattern Analyzer

An application for analyzing rhythmic patterns in WAV music files.

## Features

- **Beat Detection**: Automatically detects beats in music files
- **Segmentation**: Cuts music into segments from beat to beat
- **Pattern Recognition**: Identifies repeating consecutive patterns
- **Interactive Playback**: Click to play patterns in seamless loops
- **Multi-file Support**: Analyze multiple WAV files

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Run the application:
```bash
python beat_analyzer.py
```

2. Click "Select Files" to choose one or more WAV files
3. Analysis starts automatically when files are selected
4. View candidate patterns sorted by deviation (best matches first)
5. Click a candidate to start it looping
6. While one is playing:
   - Single-click another candidate to queue it (plays when current loop finishes)
   - Double-click another candidate to switch to it immediately
7. Click "Stop" to stop playback and clear the queue

### Command-Line Usage

You can also specify WAV files directly from the command line:

```bash
python beat_analyzer.py your_music.wav
```

Or multiple files:

```bash
python beat_analyzer.py song1.wav song2.wav
```

When files are provided via command line, analysis starts automatically when the GUI opens.

## How It Works

1. **Beat Detection**: Uses librosa's beat tracking algorithm to find the rhythm
2. **Segmentation**: Splits audio between consecutive beats
3. **Feature Extraction**: Extracts MFCC (Mel-frequency cepstral coefficients) features
4. **Clustering**: Uses DBSCAN to find similar segments
5. **Pattern Identification**: Identifies consecutive repetitions as candidates
6. **Seamless Looping**: Plays patterns continuously without gaps

## Requirements

- Python 3.8+
- librosa
- numpy
- scikit-learn
- sounddevice
- soundfile
- scipy
- tkinter (usually included with Python)

