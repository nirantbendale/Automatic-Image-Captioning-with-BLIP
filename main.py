"""
Automatic Image Captioning using BLIP Model
Real-world use case: Generate multiple descriptive captions for any uploaded image

This project uses BLIP (Bootstrapping Language-Image Pre-training) which can generate
captions for ANY image, not limited to predefined lists.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os
import argparse
from typing import List, Dict
import json
from datetime import datetime
import random

class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = None):
        """
        Initialize the Image Captioner with BLIP model
        
        Args:
            model_name (str): BLIP model from HuggingFace hub
            device (str): Device to run inference on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP model: {model_name} on {self.device}")
        
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path or URL
        
        Args:
            image_path (str): Path to local image or URL
            
        Returns:
            PIL.Image: Loaded image
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def generate_single_caption(self, image: Image.Image, max_length: int = 50, 
                              num_beams: int = 5, temperature: float = 1.0) -> str:
        """
        Generate a single caption for an image
        
        Args:
            image (PIL.Image): Input image
            max_length (int): Maximum length of generated caption
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for generation
            
        Returns:
            str: Generated caption
        """
        try:
            # Process the image
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True if temperature != 1.0 else False,
                    early_stopping=True
                )
            
            # Decode the generated caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            raise

    def generate_multiple_captions(self, image: Image.Image, num_captions: int = 5) -> List[str]:
        """
        Generate multiple diverse captions for an image
        
        Args:
            image (PIL.Image): Input image
            num_captions (int): Number of captions to generate
            
        Returns:
            List[str]: List of generated captions
        """
        captions = []
        
        # Different generation parameters for diversity
        generation_configs = [
            {"max_length": 30, "num_beams": 3, "temperature": 0.7},
            {"max_length": 50, "num_beams": 5, "temperature": 1.0},
            {"max_length": 40, "num_beams": 4, "temperature": 0.8},
            {"max_length": 60, "num_beams": 6, "temperature": 1.2},
            {"max_length": 35, "num_beams": 3, "temperature": 0.9},
            {"max_length": 45, "num_beams": 5, "temperature": 1.1},
            {"max_length": 55, "num_beams": 4, "temperature": 0.6}
        ]
        
        try:
            for i in range(num_captions):
                # Use different configs to get diverse captions
                config = generation_configs[i % len(generation_configs)]
                caption = self.generate_single_caption(image, **config)
                
                # Avoid duplicate captions
                if caption not in captions:
                    captions.append(caption)
                else:
                    # Try with slightly different parameters if duplicate
                    config["temperature"] += random.uniform(-0.1, 0.1)
                    caption = self.generate_single_caption(image, **config)
                    captions.append(caption)
            
            return captions[:num_captions]
            
        except Exception as e:
            print(f"Error generating multiple captions: {e}")
            raise

    def generate_captions(self, image_path: str, num_captions: int = 5) -> Dict:
        """
        Generate multiple captions for an image
        
        Args:
            image_path (str): Path to the image file or URL
            num_captions (int): Number of captions to generate
            
        Returns:
            Dict: Dictionary containing captions and metadata
        """
        try:
            # Load image
            image = self.load_image(image_path)
            print(f"Image loaded successfully. Size: {image.size}")
            
            # Generate captions
            print("Generating captions...")
            captions = self.generate_multiple_captions(image, num_captions)
            
            # Format results
            result = {
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "image_size": image.size,
                "model_used": "BLIP",
                "captions": [
                    {
                        "caption": caption,
                        "caption_id": i + 1
                    }
                    for i, caption in enumerate(captions)
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating captions: {e}")
            raise

    def save_results(self, results: Dict, output_path: str = "captions_output.json"):
        """
        Save captioning results to JSON file
        
        Args:
            results (Dict): Results dictionary
            output_path (str): Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
            raise

def main():
    """
    Main function to run the image captioning system
    """
    parser = argparse.ArgumentParser(description="Automatic Image Captioning with BLIP")
    parser.add_argument("--image", "-i", required=True, help="Path to image file or URL")
    parser.add_argument("--num_captions", "-n", type=int, default=5, help="Number of captions to generate")
    parser.add_argument("--model", "-m", default="Salesforce/blip-image-captioning-base", help="BLIP model from HuggingFace")
    parser.add_argument("--output", "-o", default="captions_output.json", help="Output file path")
    parser.add_argument("--device", "-d", default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Initialize captioner
        captioner = ImageCaptioner(model_name=args.model, device=args.device)
        
        # Generate captions
        print(f"Generating captions for: {args.image}")
        results = captioner.generate_captions(args.image, args.num_captions)
        
        # Display results
        print("\n" + "="*50)
        print("GENERATED CAPTIONS")
        print("="*50)
        
        for caption_data in results["captions"]:
            print(f"{caption_data['caption_id']}. {caption_data['caption']}")
            print()
        
        # Save results
        captioner.save_results(results, args.output)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1
    
    return 0

def save_trained_model(model_save_path: str = "./saved_blip_model", 
                      model_name: str = "Salesforce/blip-image-captioning-base"):
    """
    Save the trained BLIP model and processor to local directory
    
    Args:
        model_save_path (str): Path where to save the model
        model_name (str): HuggingFace model name to download and save
    """
    try:
        print(f"Downloading and saving model: {model_name}")
        print(f"Save location: {model_save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Load and save processor
        print("Saving processor...")
        processor = BlipProcessor.from_pretrained(model_name)
        processor.save_pretrained(model_save_path)
        
        # Load and save model
        print("Saving model...")
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(model_save_path)
        
        print(f"Model successfully saved to: {model_save_path}")
        print("You can now use the saved model for inference without internet connection!")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "save_path": model_save_path,
            "saved_timestamp": datetime.now().isoformat(),
            "model_type": "BLIP",
            "usage": "Use --use_saved_model flag with this path"
        }
        
        info_path = os.path.join(model_save_path, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to: {info_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def load_saved_model(model_path: str, device: str = None):
    """
    Load a previously saved BLIP model from local directory
    
    Args:
        model_path (str): Path to the saved model directory
        device (str): Device to load model on
        
    Returns:
        ImageCaptioner: Initialized captioner with saved model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found at: {model_path}")
        
        print(f"Loading saved model from: {model_path}")
        
        # Check if model info exists
        info_path = os.path.join(model_path, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                print(f"Model info: {model_info['model_name']} saved on {model_info['saved_timestamp']}")
        
        # Create custom captioner with saved model
        device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model from saved path
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Create captioner instance
        captioner = ImageCaptioner.__new__(ImageCaptioner)
        captioner.device = device
        captioner.processor = processor
        captioner.model = model
        
        print(f"Saved model loaded successfully on {device}")
        return captioner
        
    except Exception as e:
        print(f"Error loading saved model: {e}")
        raise

def main_with_saved_model():
    """
    Main function that supports both online and saved model usage
    """
    parser = argparse.ArgumentParser(description="Automatic Image Captioning with BLIP")
    parser.add_argument("--image", "-i", required=True, help="Path to image file or URL")
    parser.add_argument("--num_captions", "-n", type=int, default=5, help="Number of captions to generate")
    parser.add_argument("--model", "-m", default="Salesforce/blip-image-captioning-base", help="BLIP model from HuggingFace")
    parser.add_argument("--output", "-o", default="captions_output.json", help="Output file path")
    parser.add_argument("--device", "-d", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--save_model", action="store_true", help="Save the model after loading")
    parser.add_argument("--model_save_path", default="./saved_blip_model", help="Path to save the model")
    parser.add_argument("--use_saved_model", help="Path to previously saved model directory")
    
    args = parser.parse_args()
    
    try:
        # Check if user wants to use a saved model
        if args.use_saved_model:
            print("Using saved model...")
            captioner = load_saved_model(args.use_saved_model, args.device)
        else:
            print("Using online model...")
            captioner = ImageCaptioner(model_name=args.model, device=args.device)
            
            # Save model if requested
            if args.save_model:
                print("Saving model for future use...")
                save_trained_model(args.model_save_path, args.model)
        
        # Generate captions
        print(f"Generating captions for: {args.image}")
        results = captioner.generate_captions(args.image, args.num_captions)
        
        # Display results
        print("\n" + "="*50)
        print("GENERATED CAPTIONS")
        print("="*50)
        
        for caption_data in results["captions"]:
            print(f"{caption_data['caption_id']}. {caption_data['caption']}")
            print()
        
        # Save results
        captioner.save_results(results, args.output)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main_with_saved_model())

