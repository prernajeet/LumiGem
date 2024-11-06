---
title: LumiGem
emoji: 💻
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---

# LumiGem

This project includes a Python script, `app.py`, that performs data manipulation, visualization, and file saving based on sample data.

## Setup Instructions

Before running `app.py`, make sure to complete the following setup steps.

### 1. System Dependencies

Run these commands to install the necessary system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
sudo apt-get remove -y python3-pyaudio
sudo apt-get update && sudo apt-get install -y portaudio19-dev
```

### 2. Python Dependencies

First, ensure you have an updated `pip` version:

```bash
pip install --upgrade pip
```

Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

This will install the following packages as specified in `requirements.txt`:

- `gradio==4.19.2`
- `google-generativeai==0.8.3`
- `gTTS==2.5.1`
- `sounddevice==0.4.6`
- `scipy==1.12.0`
- `SpeechRecognition==3.10.1`
- `numpy==1.26.4`
- `watchdog==4.0.0`
- `pyaudio==0.2.13`

## Running the Script

After completing the setup, you can run the script with:

```bash
python app.py
```

### `requirements.txt`

This file should match the contents of `updated_requirements (1).txt`:

```plaintext
gradio==4.19.2
google-generativeai 0.8.3
gTTS==2.5.1
sounddevice==0.4.6
scipy==1.12.0
SpeechRecognition==3.10.1
numpy==1.26.4
watchdog==4.0.0
pyaudio==0.2.13
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
