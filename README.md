# Emotion Recognition for Autism Support: A Hackathon Prototype

[cite_start]This project is a real-time emotion recognition system
designed as an assistive technology for individuals with autism spectrum
disorder (ASD)[cite: 3]. It leverages facial emotion analysis to provide
immediate visual feedback, aiming to help users better understand
emotional cues in social interactions.

[cite_start]This implementation represents **Phase 1** of the development
plan, focusing on core functionality[cite: 76, 77].

## Core Features (Phase 1)

*   [cite start]**Real-time Facial Emotion Recognition**: Uses a live webcam
    feed to detect faces and analyze expressions[cite: 77].
*   [cite_start]**Simple Visual Feedback**: Overlays the detected emotion
    and a corresponding emoji onto the video feed[cite: 35, 79].
*   [cite_start]**Real-time Processing Pipeline**: An efficient pipeline
    built for low latency [cite: 78][cite start], with on-screen FPS
    monitoring to track performance[cite: 51].

## Technology Stack

*   [cite start]**Facial Emotion Recognition**: `DeepFace` library, which
    utilizes powerful pre-trained Convolutional Neural Networks (CNNs)[cite:
    25, 27].
*   [cite start]**Real-time Video Processing**: `OpenCV`[cite: 24].
*   **Programming Language**: Python.

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.8+
    *   A webcam connected to your computer.

2.  **Installation**:
    Clone or download the project files into a local directory. Open your
    terminal, navigate to the project directory, and install the required
    dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Execute the main application script from your terminal:

```bash
python app.py
```

Press the 'q' key on your keyboard to close the application.

## Development Roadmap

This prototype establishes the foundation. Future development will follow the guide's roadmap:

*   **Phase 2: Integrate Speech Tone Analysis** and create a more advanced, customizable user
    interface.
*   **Phase 3: Implement multimodal fusion** to combine facial and speech analysis for higher
    accuracy and conduct user experience testing.
    
STEPS TO FOLLOW:
Install Python 3.11.9 with PATH (without uninstalling your current python)

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

py -3.11 -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

#for desktop
run app.py
or
python app.py --mode desktop 

#for web
python app.py --mode web
 