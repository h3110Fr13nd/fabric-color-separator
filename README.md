# Color Separator

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful tool for separating and manipulating colors in images. This application uses image quantization to reduce an image to a specified number of colors and then separates each color into individual layers that can be edited.

## Features

- Quantize images to a specified number of colors
- Separate colors into individual layers
- Edit colors directly through an intuitive interface
- Generate different output formats:
  - Color layers with white background
  - Color layers with transparent background
  - Binary masks for each color
- Real-time preview of changes

## Prerequisites

- Python 3.10 or higher (3.12 recommended)
- pip (Python package manager)
- opencv-python
- numpy
- gradio

## Installation

### Option 1: Using pip and venv (recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/color-separator.git
   cd color-separator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

### Option 2: Using pip directly

```bash
pip install git+https://github.com/yourusername/color-separator.git
```

### Option 3: Manual installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/color-separator.git
   cd color-separator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or use: pip install gradio numpy opencv-python
   ```

## Usage

### Running the Web Interface

1. Launch the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:7860
   ```

### Using the Interface

1. **Upload an image**: Click on the upload area to select an image from your computer.

2. **Adjust the number of colors**: Use the slider to set how many colors the image should be quantized to (range: 2-16).

3. **Process the image**: Click the "Process Image" button to analyze and separate the image.

4. **View results**: Navigate through the tabs to see:
   - Quantized image
   - Color layers with white background
   - Color layers with transparent background
   - Binary masks for each color

5. **Edit colors**: You can modify any color by clicking on its color picker. The changes will be reflected in all output formats.

6. **Update after editing**: Click the "Update Colors" button to apply your color changes.

### Command Line Usage

The tool can also be used from the command line for batch processing:

```bash
python -m utils input_image.jpg [number_of_colors]
```

Example:
```bash
python -m utils my_image.jpg 8
```

This will create an `output` directory with the processed images.

## Output Files

When using the command line tool, the following files are generated in the `output` directory:

```
output/
├── [image_name]/
│   ├── [image_name]_quantized.png
│   ├── colors/
│   │   ├── [image_name]_color_0.png
│   │   ├── [image_name]_color_1.png
│   │   └── ...
│   ├── masks/
│   │   ├── [image_name]_mask_0.png
│   │   ├── [image_name]_mask_1.png
│   │   └── ...
│   └── transparent_colors/
│       ├── [image_name]_transparent_color_0.png
│       ├── [image_name]_transparent_color_1.png
│       └── ...
```

## Technical Details

Color Separator uses the K-means clustering algorithm from OpenCV to quantize images into a specified number of colors. The algorithm works by:

1. Treating each pixel as a point in 3D RGB space
2. Clustering these points into the specified number of clusters
3. Replacing each pixel's color with the centroid of its assigned cluster

This results in a quantized image with only the specified number of colors. The tool then separates these colors into individual layers and provides various output formats.

## Troubleshooting

### Common Issues

- **Image doesn't appear after upload**: Ensure the image format is supported (JPG, PNG, etc.)
- **Processing takes too long**: Try reducing the number of colors or using a smaller image
- **Colors don't update after editing**: Make sure to click the "Update Colors" button

### Error Messages

- "Error: File does not exist" - Ensure the file path is correct
- "Error: Unable to read image" - The file might be corrupt or in an unsupported format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
