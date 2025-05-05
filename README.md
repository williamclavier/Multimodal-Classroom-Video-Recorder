# Smart Multimodal Classroom Video Recorder
A smart multimodal classroom video recording system that
automatically composes multiple content streams—camera feeds, slides, and whiteboard—
based on real-time cues like gestures and spoken references. By leveraging
computer vision, automatic speech recognition (ASR), and content analysis, it can dynamically
switch between sources to create a more engaging, context-aware
lecture recording. The goal is to overcome the limitations of static cameras and
provide a richer, more immersive experience for both live and recorded viewers.

## Features
- Automatic switching between slide and professor views based on content analysis
- Corner overlay mode to show both feeds simultaneously
- Pose estimation for gesture detection (pointing and writing)
- Real-time pose visualization with skeleton tracking
- Debug mode with comprehensive visualization overlays
- Standalone pose estimation for fine-tuning
- High-quality video output with configurable settings
- Confidence analysis reports for model performance evaluation

## Installation

### Prerequisites
1. Python 3.11 (Only tested with 3.11)
2. FFmpeg installed on your system
3. Tesseract OCR installed on your system

### Installing Dependencies

#### System Requirements
First, install the required system tools:

```bash
# Install FFmpeg
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg

# Install Tesseract OCR
# on Ubuntu or Debian
sudo apt install tesseract-ocr

# on Arch Linux
sudo pacman -S tesseract

# on MacOS using Homebrew
brew install tesseract

# on Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

#### Python Dependencies
We recommend using a virtual environment to install the required Python dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py slide_video.mp4 professor_video.mp4 --output-dir output
```

### Advanced Options
- `--output-dir`: Directory to save outputs (required)
- `--ocr-results`: Path to pre-computed OCR results
- `--pose-results`: Path to pre-computed pose results
- `--transcription`: Path to pre-computed transcription
- `--analysis`: Path to pre-computed content analysis
- `--load-only`: Only load pre-computed results (skip processing)
- `--skip-video`: Skip video creation after processing
- `--pose-only`: Only run pose estimation and exit
- `--debug`: Enable debug mode (faster processing, lower resolution, visual overlays)
- `--quality`: Set output quality ('high', 'medium', 'low')
- `--report`: Generate confidence analysis report

### Examples

#### Debug Mode
```bash
python main.py slide_video.mp4 professor_video.mp4 --debug
```
This will create a lower resolution video with comprehensive debug overlays showing:
- Pose skeleton visualization with keypoints
- Bounding box around the professor
- Gesture detection status (pointing/writing)
- Confidence scores
- Motion tracking information
- Decision-making process

#### Pose Estimation Only
```bash
python main.py slide_video.mp4 professor_video.mp4 --pose-only
```
This will run only the pose estimation and save the results to a JSON file for analysis.

#### High Quality Output
```bash
python main.py slide_video.mp4 professor_video.mp4 --quality high
```
This will create a high-quality output video with the best possible settings.

#### Generate Confidence Report
```bash
python main.py slide_video.mp4 professor_video.mp4 --report
```
This will generate a confidence analysis report showing the performance of each model over time.

#### Use Pre-computed Results
```bash
python main.py slide_video.mp4 professor_video.mp4 --load-only --ocr-results ocr.json --pose-results pose.json --transcription trans.json --analysis analysis.json
```
This will use existing analysis results instead of running the full pipeline.

## Output
The program creates several output files in the specified output directory:
- `output.mp4` (or `debug_output.mp4` in debug mode): The final combined video
- `pose/pose_results.json`: Pose estimation data including keypoints and gesture detection
- `ocr/ocr_results.json`: OCR results from slides
- `transcription/transcription.json`: Speech transcription
- `analysis/analysis_results.json`: Content analysis results
- `decisions/decisions.json`: Camera switching decisions with pose data
- `multi_model_analysis.png`: Confidence analysis report (when using --report)