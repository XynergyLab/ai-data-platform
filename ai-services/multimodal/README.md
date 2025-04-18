# Multimodal Processing Services

This service provides comprehensive multimodal processing capabilities for images, audio, and video content using state-of-the-art AI models.

## Components

1. Image Processor
   - Object detection and recognition
   - Image segmentation
   - Scene understanding
   - OCR capabilities
   - Runs on port 8090

2. Audio Processor
   - Speech recognition (Whisper)
   - Audio classification
   - Speaker diarization
   - Runs on port 8091

3. Video Processor
   - Video analysis
   - Action recognition
   - Object tracking
   - Scene segmentation
   - Runs on port 8092

## Configuration

1. Copy `.env.example` to `.env` and adjust settings
2. Download required models:
   ```bash
   # Image models
   mkdir -p models/image
   # Download YOLO, ResNet, etc.

   # Audio models
   mkdir -p models/audio
   # Download Whisper models

   # Video models
   mkdir -p models/video
   # Download video processing models
   ```

## Usage

Start the services:
```bash
podman-compose up -d
```

Example API calls:

1. Image Processing:
```bash
curl -X POST http://localhost:8090/process \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/image.jpg" \
  -F "tasks=[object_detection,scene_understanding]"
```

2. Audio Processing:
```bash
curl -X POST http://localhost:8091/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@/path/to/audio.mp3"
```

3. Video Processing:
```bash
curl -X POST http://localhost:8092/analyze \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/path/to/video.mp4" \
  -F "tasks=[object_tracking,action_recognition]"
```

## Resource Requirements

- Minimum 32GB RAM total
- 8+ CPU cores
- NVIDIA GPU strongly recommended
- 100GB+ storage for models and temporary data

## Integration

- Works with embedding services for feature extraction
- Connects to vector stores for similarity search
- Supports Redis caching for improved performance
- Part of the global service mesh
