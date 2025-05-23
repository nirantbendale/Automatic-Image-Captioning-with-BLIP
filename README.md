# Automatic Image Captioning with BLIP

A Python project that generates multiple descriptive captions for any uploaded image using the BLIP (Bootstrapping Language-Image Pre-training) model. This real-world application can be used for content management, accessibility, social media automation, or digital asset organization.

## Features

- **Universal Image Captioning**: Generate captions for any image content (not limited to predefined categories)
- **Multiple Caption Generation**: Get 5 different descriptive captions per image
- **Flexible Input Support**: Works with local files and URLs
- **Model Saving/Loading**: Save trained model locally for offline use
- **Batch Processing**: Process multiple images at once
- **JSON Output**: Structured results with metadata
- **Error Handling**: Comprehensive error handling for production use

## Real-World Use Cases

- **Content Management**: Automatically caption images in digital libraries
- **Accessibility**: Generate alt-text for web images
- **Social Media**: Auto-generate captions for posts
- **E-commerce**: Describe product images automatically
- **Digital Asset Management**: Organize and search image collections

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Install Dependencies

```bash
pip install torch>=1.9.0
pip install transformers>=4.21.0
pip install Pillow>=8.0.0
pip install requests>=2.25.0
pip install numpy>=1.21.0
```

Or install all at once:

```bash
pip install torch transformers Pillow requests numpy
```

## Quick Start

### 1. Basic Usage

```bash
python image_captioner.py --image "path/to/your/image.jpg"
```

### 2. Generate More Captions

```bash
python image_captioner.py --image "image.jpg" --num_captions 7
```

### 3. Use with URL

```bash
python image_captioner.py --image "https://example.com/image.jpg"
```

### 4. Save Model for Offline Use

```bash
python image_captioner.py --image "test.jpg" --save_model --model_save_path "./my_model"
```

### 5. Use Saved Model (Offline)

```bash
python image_captioner.py --image "image.jpg" --use_saved_model "./my_model"
```

## Command Line Arguments

| Argument             | Short | Description                          | Default                                |
|----------------------|-------|--------------------------------------|----------------------------------------|
| `--image`            | `-i`  | Path to image file or URL            | **Required**                           |
| `--num_captions`     | `-n`  | Number of captions to generate       | 5                                      |
| `--model`            | `-m`  | HuggingFace model name               | Salesforce/blip-image-captioning-base  |
| `--output`           | `-o`  | Output JSON file path                | captions_output.json                   |
| `--device`           | `-d`  | Device (cuda/cpu)                    | Auto-detect                            |
| `--save_model`       |       | Save model locally                   | False                                  |
| `--model_save_path`  |       | Path to save model                   | ./saved_blip_model                     |
| `--use_saved_model`  |       | Path to saved model                  | None                                   |

## Examples

### Example 1: Basic Image Captioning

```bash
python image_captioner.py --image "beach.jpg" --num_captions 3
```

**Output:**

1. a person walking on the beach near the ocean  
2. a beautiful sunset over the sandy beach with waves  
3. people enjoying a sunny day at the seaside  

### Example 2: Using Different Model

```bash
python image_captioner.py --image "city.jpg" --model "Salesforce/blip-image-captioning-large"
```

### Example 3: Batch Processing

```python
from image_captioner import batch_process_images

batch_process_images("./input_images/", "./output_results/")
```

## Output Format

The generated captions are saved in JSON format:

```json
{
  "image_path": "example.jpg",
  "timestamp": "2024-01-15T10:30:00",
  "image_size": [800, 600],
  "model_used": "BLIP",
  "captions": [
    {
      "caption": "a dog playing in the park",
      "caption_id": 1
    },
    {
      "caption": "a golden retriever running on grass",
      "caption_id": 2
    }
  ]
}
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)  
- PNG (.png)  
- BMP (.bmp)  
- GIF (.gif)  
- TIFF (.tiff)

## Model Information

This project uses the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce Research:

- **Default Model**: `Salesforce/blip-image-captioning-base`  
- **Alternative**: `Salesforce/blip-image-captioning-large` (better quality, slower)  
- **Framework**: HuggingFace Transformers  
- **Capability**: Generates natural language captions for any image content  

## Advanced Usage

### 1. Save Model During First Use

```bash
python image_captioner.py --image "test.jpg" --save_model
```

### 2. Use Saved Model (No Internet Required)

```bash
python image_captioner.py --image "new_image.jpg" --use_saved_model "./saved_blip_model"
```

### 3. Custom Output Location

```bash
python image_captioner.py --image "photo.jpg" --output "./results/my_captions.json"
```

### 4. Force CPU Usage

```bash
python image_captioner.py --image "image.jpg" --device cpu
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
python image_captioner.py --image "image.jpg" --device cpu
```

#### Internet Connection Issues

- First save the model: `--save_model`
- Then use saved model: `--use_saved_model path`

#### Image Format Not Supported

- Convert image to JPG/PNG format
- Check file path is correct

#### Model Download Fails

- Check internet connection
- Try different model: `--model "Salesforce/blip-image-captioning-base"`

### Error Messages

- `FileNotFoundError`: Check image path exists  
- `CUDA error`: Use `--device cpu`  
- `Connection timeout`: Save model for offline use  

## Project Structure

```
image-captioning/
│
├── image_captioner.py          # Main script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── examples/                   # Example images
│   ├── sample1.jpg
│   └── sample2.jpg
└── saved_models/               # Saved model directory
    └── blip_model/
        ├── config.json
        ├── pytorch_model.bin
        └── model_info.json
```

## Acknowledgments

- **BLIP Model**: Salesforce Research  
- **HuggingFace**: Transformers library  
- **PyTorch**: Deep learning framework  
