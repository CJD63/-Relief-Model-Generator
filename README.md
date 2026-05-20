# 3D Relief Model Generator

Generate 3D STL relief models from images using AI-powered edge detection and customizable parameters.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red.svg)

## Features

- **Single Image Mode**: Generate relief models from individual images
- **Batch Processing**: Process multiple images with the same settings
- **AI Edge Detection**: Enhanced edge detection for crisper relief features
- **Customizable Presets**: Fine-tune model width, relief height, base thickness, and more
- **Filename Patterns**: Flexible naming with placeholders like `{name}`, `{index}`, `{date}`
- **Progress Tracking**: Real-time progress with percentage and ETA for batch processing

## Quick Start

### Option 1: Launch Script (Windows)

Simply double-click `3D Relief Model Generator.bat` to start the application.

### Option 2: Command Line

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
```

Then open your browser at **http://localhost:8501**

## Usage

### Single Image Mode

1. Select the **📤 Single Image** tab
2. Upload an image file or enter a URL
3. Choose a quality preset (Draft/Preview/Standard/High/Extreme) or customize settings
4. Enable **Edge Detection** for crisper features
5. Click **Generate Relief Model**
6. Download the generated STL file

### Batch Processing

1. Select the **📚 Batch Process** tab
2. Upload multiple images (JPG, PNG, WebP supported)
3. Configure settings:
   - **Filename Pattern**: Use placeholders for unique names
     - `{name}` - Original filename
     - `{preset}` - Quality preset name
     - `{index}` - Sequential number (1, 2, 3...)
     - `{date}` - Date in YYYYMMDD format
     - `{timestamp}` - Date and time
     - `{width}` - Model width in mm
     - `{height}` - Model height in mm
   - **Quality Preset**: Draft, Preview, Standard, High, or Extreme
   - **Enable Edge Detection**: For crisper relief features
   - **Edge Strength**: Adjust edge enhancement intensity (when edge detection enabled)
4. Click **🚀 Process All Images**
5. Download all files as a ZIP archive

### Filename Pattern Examples

| Pattern | Example Output |
|---------|---------------|
| `{name}` | photo.stl |
| `{name}_{preset}` | photo_high.stl |
| `{index:03d}_{name}` | 001_photo.stl |
| `{date}_{name}` | 20240115_photo.stl |
| `{name}_{width}x{height}` | photo_100x80.stl |

## Quality Presets

| Preset | Model Width | Relief Height | Base Thickness | Best For |
|--------|------------|---------------|----------------|----------|
| Draft | 50mm | 2mm | 1mm | Quick previews |
| Preview | 60mm | 3mm | 1.5mm | Fast iterations |
| Standard | 80mm | 5mm | 2mm | General use |
| High | 100mm | 8mm | 2mm | Detailed prints |
| Extreme | 120mm | 12mm | 3mm | Maximum quality |

## Settings Reference

### Model Settings
- **Model Width**: Width of the relief model in mm (20-200)
- **Relief Height**: Maximum depth variation in mm (0.5-20)
- **Base Thickness**: Bottom slab thickness in mm (0.5-10)

### Image Processing
- **Gamma**: Brightness curve adjustment (0.5-2.0)
- **Blur Radius**: Smoothing strength (0-10)
- **Hill Removal**: Suppresses small bumps (0-10)
- **Small Feature Threshold**: Ignores details below threshold (0-20)

### Edge Detection
- **Enable Edge Detection**: Toggle edge enhancement
- **Edge Strength**: Edge detection intensity (0.1-3.0)

## Technical Details

### Algorithm Pipeline

1. **Image Loading**: RGB conversion and resize if needed
2. **Grayscale Conversion**: Single channel depth calculation
3. **Edge Enhancement**: Canny edge detection with morphological gradient
4. **Bilateral Filtering**: Noise reduction while preserving edges
5. **Gradient Limiting**: Vectorized smoothing of extreme depth transitions
6. **Mesh Generation**: Triangulated height map with optional base
7. **STL Export**: Binary STL file generation

### Performance

- Vectorized numpy operations for fast processing
- Typical single image: 1-3 seconds
- Batch processing includes progress percentage and ETA

## Troubleshooting

### Port 8501 Already in Use

If you see a warning about port 8501, another Streamlit instance may be running. Open http://localhost:8501 in your browser to use it, or stop the other process.

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Large Images

Images over 1000px are automatically resized for better performance. Adjust the `max_size` parameter in the code if needed.

## License

MIT License

## Acknowledgments

- OpenCV for image processing
- NumPy/STL for 3D mesh operations
- Streamlit for the web interface