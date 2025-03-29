# Smart Multimodal Classroom Video Recorder
A smart multimodal classroom video recording system that
automatically composes multiple content streams—camera feeds, slides, and whiteboard—
based on real-time cues like gestures and spoken references. By leveraging
computer vision, automatic speech recognition (ASR), and content analysis, it can dynamically
pan, zoom, and switch between sources to create a more engaging, contextaware
lecture recording. The goal is to overcome the limitations of static cameras and
provide a richer, more immersive experience for both live and recorded viewers
## Installation
You need to have Python installed. We recommend making a [virtual environment](https://docs.python.org/3/library/venv.html) to install the required dependencies. To proceed with installing the required dependencies:
```bash
python -m pip install -r requirements.txt
```

This project needs the command-line tool [ffmpeg](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:
```bash
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
```