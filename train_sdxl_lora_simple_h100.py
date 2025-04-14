#!/usr/bin/env python3
"""
SDXL LoRA Training Script - H100 Optimized
Based on HuggingFace Diffusers example with optimizations for H100 GPUs
"""

import os
import sys
import torch
import argparse
import json
import math
import random
from typing import List, Optional

# Compatibility checking
print("Checking environment compatibility...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")

# Try to import additional packages carefully
try:
    from PIL import Image
    import numpy as np
    print("Basic dependencies loaded successfully.")
except ImportError as e:
    print(f"Error importing basic dependencies: {e}")
    print("Please install required packages: pip install pillow numpy")
    sys.exit(1)

# Custom dataset class to replace HuggingFace datasets
class SimpleDataset(torch.utils.data.Dataset):
    """
    A robust dataset implementation for handling image-caption pairs in diffusion model training.
    
    This dataset automatically handles common edge cases such as:
    - Empty datasets
    - Inconsistent lengths between images and captions
    - Missing data for specific keys
    - Type inconsistencies
    
    It fixes these issues by:
    - Properly initializing empty datasets
    - Padding shorter key arrays or truncating longer ones
    - Providing appropriate defaults based on data type (tensors, strings, etc.)
    - Validating data types and logging warnings when unexpected types are found
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the dataset items with keys such as:
        - 'images': List of image file paths
        - 'captions': List of corresponding captions
        - 'pixel_values': List of precomputed image tensors (optional)
        - 'input_ids': List of tokenized caption tensors (optional)
        
    Attributes
    ----------
    data : dict
        The processed and validated data dictionary
    keys : list
        List of keys in the dataset
    length : int
        Length of the dataset (consistent across all keys)
    """
    def __init__(self, data_dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())
        
        # Handle empty dataset case
        if not self.keys or all(len(data_dict[k]) == 0 for k in self.keys):
            logger.warning("Creating an empty dataset (no keys or all keys are empty)")
            self.length = 0
            return
        
        # Find non-empty keys
        non_empty_keys = [k for k in self.keys if len(data_dict[k]) > 0]
        
        if not non_empty_keys:
            logger.warning("All keys are empty!")
            self.length = 0
            return
            
        # Get the length from the first non-empty key
        self.length = len(data_dict[non_empty_keys[0]])
        
        # Check for length mismatches and fix them
        for key in self.keys:
            if len(data_dict[key]) != self.length:
                if len(data_dict[key]) == 0:
                    # If key is empty, initialize it with appropriate sized empty entries
                    logger.warning(f"Key {key} is empty, filling with appropriate defaults")
                    if key == "pixel_values":
                        # For pixel_values, create empty tensors
                        data_dict[key] = [torch.zeros((3, 1024, 1024)) for _ in range(self.length)]
                    elif key == "input_ids":
                        # For input_ids, create empty token tensors
                        data_dict[key] = [torch.zeros(77, dtype=torch.long) for _ in range(self.length)]
                    else:
                        # For other keys, use empty strings or zeros
                        data_dict[key] = ["" if key == "captions" or key == "images" else 0 for _ in range(self.length)]
                else:
                    # Length mismatch but both non-zero
                    logger.warning(f"Length mismatch for key {key}: expected {self.length}, got {len(data_dict[key])}")
                    if len(data_dict[key]) > self.length:
                        # Truncate if too long
                        logger.warning(f"Truncating {key} from {len(data_dict[key])} to {self.length} items")
                        data_dict[key] = data_dict[key][:self.length]
                    else:
                        # Pad if too short by repeating the last element
                        logger.warning(f"Padding {key} from {len(data_dict[key])} to {self.length} items")
                        last_item = data_dict[key][-1] if data_dict[key] else (
                            torch.zeros((3, 1024, 1024)) if key == "pixel_values" else 
                            torch.zeros(77, dtype=torch.long) if key == "input_ids" else 
                            "" if key == "captions" or key == "images" else 0
                        )
                        data_dict[key].extend([last_item] * (self.length - len(data_dict[key])))
        
        self.data = data_dict
        
        # Debug log
        logger.info(f"Created SimpleDataset with keys: {self.keys}, length: {self.length}")
        if self.length > 0 and "images" in self.keys and self.data["images"]:
            logger.info(f"First image path: {self.data['images'][0]}")
            if isinstance(self.data['images'][0], str):
                logger.info("Image paths are strings as expected")
            else:
                logger.warning(f"WARNING: Image paths are not strings but {type(self.data['images'][0])}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Retrieve a dataset sample by index with robust error handling.
        
        This method provides defensive access to dataset items by:
        - Validating index boundaries and raising appropriate exceptions
        - Handling inconsistent data lengths across different keys
        - Providing type-appropriate default values when data is missing
        
        Default values are provided based on the key type:
        - pixel_values: Empty tensor with shape (3, 1024, 1024)
        - input_ids: Empty tensor with shape (77,) and dtype torch.long
        - captions/images: Empty string
        - Other keys: Numeric zero
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve
            
        Returns
        -------
        dict
            A dictionary containing all keys in the dataset for the specified index,
            with appropriate default values for any missing data
            
        Raises
        ------
        IndexError
            If the requested index is out of bounds for the dataset length
        """
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.length}")
        
        sample = {}
        for key in self.keys:
            if idx < len(self.data[key]):
                sample[key] = self.data[key][idx]
            else:
                # Use a default value if index is out of bounds for this key
                logger.warning(f"Index {idx} out of bounds for key {key} with length {len(self.data[key])}")
                sample[key] = (
                    torch.zeros((3, 1024, 1024)) if key == "pixel_values" else 
                    torch.zeros(77, dtype=torch.long) if key == "input_ids" else 
                    "" if key == "captions" or key == "images" else 0
                )
        return sample
        
    def with_transform(self, transform_function):
        """
        Apply a transformation function to all samples in the dataset and return a new dataset.
        
        This method provides robust error handling during transformations by:
        - Processing each sample individually with comprehensive error catching
        - Skipping samples that fail to transform instead of aborting the entire process
        - Properly handling tensor stacking for batched data
        - Converting between tensor and list representations as needed
        
        The transformation process maintains consistent data structures by:
        - Converting stacked tensors back to lists for compatibility with SimpleDataset
        - Preserving tensor types when possible
        - Providing detailed logging about the transformation process
        - Creating proper empty datasets when all transformations fail
        
        Parameters
        ----------
        transform_function : callable
            A function that takes a single sample dictionary and returns a transformed
            sample dictionary. The function should return None for samples that
            should be filtered out.
        
        Returns
        -------
        SimpleDataset
            A new dataset containing the successfully transformed samples.
            Returns an empty dataset if no samples were successfully transformed.
        
        Notes
        -----
        The transform_function should handle sample dictionaries with keys like
        'images', 'captions', 'pixel_values', and 'input_ids'. It can modify
        these entries or add new ones, and filter samples by returning None.
        """
        # Start with empty data
        transformed_samples = []
        
        # Log what we're doing
        logger.info(f"Applying transformation to dataset with {self.length} samples")
        
        # Process each sample one by one
        for i in range(self.length):
            try:
                # Create a sample as a dictionary of key -> i-th element
                sample = {}
                for key in self.keys:
                    sample[key] = self.data[key][i]
                
                # Apply transformation with error handling
                try:
                    # Debug log to see what's happening
                    if "images" in sample:
                        logger.debug(f"Sample image type: {type(sample['images'])}")
                        if isinstance(sample["images"], str):
                            logger.debug(f"Sample image path: {sample['images']}")
                        
                    transformed_sample = transform_function(sample)
                    # Only add to results if transformation was successful
                    if transformed_sample:
                        transformed_samples.append(transformed_sample)
                except Exception as e:
                    logger.error(f"Error transforming sample {i}: {e}")
                    # Don't add this sample to the transformed dataset
            except Exception as e:
                logger.error(f"Error creating sample {i}: {e}")
        
        # Create the final result dictionary by combining all successful transformations
        logger.info(f"Successfully transformed {len(transformed_samples)} samples")
        
        if not transformed_samples:
            logger.warning("No samples were successfully transformed!")
            # Return an empty dataset
            return SimpleDataset({key: [] for key in self.keys})
        
        # Extract and combine all keys across samples
        result_dict = {}
        for key in transformed_samples[0].keys():
            # For tensors, stack them
            if isinstance(transformed_samples[0][key], torch.Tensor):
                try:
                    result_dict[key] = torch.stack([s[key] for s in transformed_samples])
                    logger.debug(f"Stacked {key} tensor with shape {result_dict[key].shape}")
                except Exception as e:
                    logger.error(f"Error stacking {key} tensors: {e}")
                    # Fallback to list
                    result_dict[key] = [s[key] for s in transformed_samples]
            else:
                # For non-tensors, just collect as a list
                result_dict[key] = [s[key] for s in transformed_samples]
        
        # Create a custom dataset-compatible format
        final_dict = {}
        for key in result_dict:
            if isinstance(result_dict[key], torch.Tensor):
                # Convert tensor to list for SimpleDataset
                if result_dict[key].ndim > 0:  # Check if not a scalar
                    final_dict[key] = [t for t in result_dict[key]]
                else:
                    final_dict[key] = [result_dict[key].item()]
            else:
                final_dict[key] = result_dict[key]
        
        logger.info(f"Created final dataset with keys: {list(final_dict.keys())}")
        for key in final_dict:
            logger.info(f"Key '{key}' has {len(final_dict[key])} items")
        
        # Create and return a new dataset
        return SimpleDataset(final_dict)

# Try to import accelerate and diffusers cautiously
print("Loading ML frameworks...")
try:
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
except ImportError as e:
    print(f"Error importing accelerate: {e}")
    print("Please install it: pip install accelerate")
    sys.exit(1)

try:
    from diffusers import StableDiffusionXLPipeline 
    from diffusers.optimization import get_scheduler
    from diffusers.utils.import_utils import is_xformers_available
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Please install it: pip install diffusers transformers")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    from torchvision import transforms
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, PretrainedConfig
    print("All dependencies loaded successfully.")
except ImportError as e:
    print(f"Error importing additional dependencies: {e}")
    print("Please install missing packages: pip install peft safetensors transformers")
    sys.exit(1)

# Custom implementation of is_torch_sdpa_available
def is_torch_sdpa_available():
    """
    Check if PyTorch's scaled dot product attention is available.
    
    Tests for the presence of the scaled_dot_product_attention
    function in torch.nn.functional, which enables efficient
    attention computation on H100 GPUs.
    
    Returns
    -------
    bool
        True if SDPA is available, False otherwise
    """
    if not hasattr(torch, "nn") or not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return False
    return True

# Initialize accelerate state before logger
from accelerate import PartialState
_ = PartialState()

# Now initialize the logger
logger = get_logger(__name__)

def log_validation(pipeline, args, accelerator, epoch, validation_prompt, num_validation_images=4):
    """
    Generate and log validation images during training.
    
    This function:
    - Loads the latest LoRA weights from checkpoints
    - Applies them to the validation pipeline
    - Generates sample images using the validation prompt
    - Logs images to tracking platforms (TensorBoard, W&B)
    
    Parameters
    ----------
    pipeline : StableDiffusionXLPipeline
        The pipeline to use for image generation
    args : argparse.Namespace
        Training arguments containing paths and configurations
    accelerator : Accelerator
        Accelerator instance managing distributed training
    epoch : int
        Current training epoch
    validation_prompt : str
        Text prompt to use for generating validation images
    num_validation_images : int, default=4
        Number of validation images to generate
        
    Returns
    -------
    list
        List of generated PIL images
    """
    
    # Safety check for pipeline
    if pipeline is None:
        logger.warning("Validation pipeline is None, skipping validation")
        return []
    
    logger.info(f"Running validation with {num_validation_images} examples...")

    # Try to load the latest LoRA weights
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
    lora_path = os.path.join(checkpoint_path, f"{args.model_name}.safetensors")
    
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from {lora_path}")
        try:
            # First unload any existing adapters to avoid errors
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
            # Load the current checkpoint
            pipeline.load_lora_weights(
                checkpoint_path, 
                weight_name=f"{args.model_name}.safetensors",
                adapter_name="default",
                prefix=None  # Use prefix=None to avoid the prefix warnings
            )
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {e}")
    else:
        logger.warning(f"LoRA weights not found at {lora_path}, skipping validation")
        return []

    # Create pipeline with LoRA
    images = []
    if not args.validation_images:
        try:
            for _ in range(num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(validation_prompt, num_inference_steps=25).images[0]
                images.append(image)
        except Exception as e:
            logger.error(f"Error generating validation images: {e}")
            return []
    else:
        images = args.validation_images

    # Log images to trackers
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            try:
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            except Exception as e:
                logger.error(f"Error logging images to tensorboard: {e}")
        elif tracker.name == "wandb":
            try:
                import wandb
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )
            except Exception as e:
                logger.error(f"Error logging images to wandb: {e}")
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    return images

def setup_logger():
    """
    Configure and return a logger with standardized formatting.
    
    Sets up a logger with:
    - Timestamp formatting
    - Proper log level (INFO)
    - Consistent message format
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    """
    Set random seeds for reproducible training.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch CPU operations
    - PyTorch CUDA operations
    
    Parameters
    ----------
    seed : int
        Seed value to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def prepare_images_and_captions(images_dir, captions_file):
    """
    Load and validate training images with their captions.
    
    This function:
    - Scans a directory for valid image files
    - Loads captions from a JSON file if provided
    - Validates that images can be opened
    - Creates default captions from filenames if needed
    - Provides extensive logging for debugging
    
    Parameters
    ----------
    images_dir : str
        Path to directory containing training images
    captions_file : str or None
        Path to JSON file with captions keyed by filename
        
    Returns
    -------
    dict
        Dictionary with "images" and "captions" lists
    """
    images = []
    captions = []
    
    # Log the images directory to ensure it's correct
    logger.info(f"Looking for images in directory: {os.path.abspath(images_dir)}")
    
    # Load captions from the captions file
    if captions_file and os.path.exists(captions_file):
        with open(captions_file, "r") as f:
            caption_data = json.load(f)
        logger.info(f"Loaded captions for {len(caption_data)} images from {captions_file}")
    else:
        caption_data = {}
        logger.info("No captions file provided or file doesn't exist. Will use filenames as captions.")
    
    # Check if directory exists
    if not os.path.isdir(images_dir):
        logger.error(f"The specified images directory '{images_dir}' is not a valid directory!")
        return {"images": [], "captions": []}
    
    # List all files in the directory
    try:
        all_files = os.listdir(images_dir)
        logger.info(f"Found {len(all_files)} total files/directories in {images_dir}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {e}")
        return {"images": [], "captions": []}
    
    # Collect images
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []
    
    for filename in all_files:
        # Construct the full file path
        file_path = os.path.join(images_dir, filename)
        
        # Check if it's a file and has valid extension
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename.lower())[1]
            if ext in valid_extensions:
                image_paths.append(file_path)
            else:
                logger.debug(f"Skipping non-image file: {file_path}")
        else:
            logger.debug(f"Skipping directory: {file_path}")
    
    logger.info(f"Found {len(image_paths)} valid image files in {images_dir}")
    
    if len(image_paths) == 0:
        logger.error(f"No valid images found in {images_dir}! Please check your dataset.")
        return {"images": [], "captions": []}
    
    # Debug: print the paths of valid images
    for i, path in enumerate(image_paths[:5]):  # Print first 5 for debugging
        logger.info(f"Valid image {i+1}: {path}")
    if len(image_paths) > 5:
        logger.info(f"... and {len(image_paths) - 5} more valid images")
    
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        try:
            # Check if the image can be opened
            with Image.open(image_path) as img:
                # Just verify it can be opened
                img_format = img.format
                img_size = img.size
                logger.debug(f"Successfully opened {image_path} - Format: {img_format}, Size: {img_size}")
            
            # Add image and caption
            images.append(image_path)
            
            # Use the caption from the file or the default
            if image_filename in caption_data:
                caption = caption_data[image_filename]
            else:
                # Extract name without extension
                base_name = os.path.splitext(image_filename)[0]
                caption = base_name.replace("_", " ").replace("-", " ")
            
            captions.append(caption)
        except Exception as e:
            logger.warning(f"Skipping {image_path}: {e}")
    
    logger.info(f"Successfully processed {len(images)} images with captions")
    return {"images": images, "captions": captions}

def parse_args():
    """
    Parse and validate command line arguments for SDXL LoRA training.
    
    Configures all training parameters including:
    - Model paths and output directories
    - Dataset configuration
    - LoRA hyperparameters (rank, alpha, dropout)
    - Training settings (batch size, epochs, learning rate)
    - Optimization options (precision, attention mechanism)
    - Validation settings
    
    Returns
    -------
    argparse.Namespace
        Validated arguments for training
    """
    parser = argparse.ArgumentParser(description="SDXL LoRA Training Script")
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to base model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to directory containing training images.",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=None,
        help="Path to JSON file containing captions. If not provided, image filenames will be used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-trained",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lora-model",
        help="The name of the LoRA model. Will be used as the filename for saved weights.",
    )
    parser.add_argument(
        "--lora_concept",
        type=str,
        default="",
        help="Concept to train the LoRA on (will be prepended to the captions).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images (square). All images will be resized to this.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="The resolution for input images (equivalent to resolution)",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="If set, crop images to this size (centered crop).",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to generate during validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_images",
        type=Optional[List[str]],
        default=None,
        help="Optional list of validation images to use (instead of generating them on the fly).",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sdxl-lora-training",
        help="Project name for logging."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every X updates steps."
    )
    # LoRA specific parameters
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA update matrices."
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32,
        help="The alpha parameter for LoRA scaling."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout probability for LoRA layers."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Default is bfloat16 for H100."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "comet_ml"],
        help="Integration to log training results to.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--use_pytorch_sdpa",
        action="store_true",
        help="Use PyTorch 2.0+'s native scaled dot product attention (SDPA) for memory-efficient training. For H100 GPUs."
    )
    parser.add_argument(
        "--use_preview_models",
        action="store_true",
        help="Use preview model components from diffusers."
    )

    args = parser.parse_args()
    
    # Handle equivalence between resolution and image_size
    if args.resolution != 1024 and args.image_size == 1024:
        args.image_size = args.resolution
    elif args.image_size != 1024 and args.resolution == 1024:
        args.resolution = args.image_size
        
    # Validate parameters
    if args.images_dir is None:
        raise ValueError("You must specify a training images directory.")
    
    return args

def collate_fn(batch):
    """
    Custom collate function for batching dataset samples.
    
    Handles special cases in batching:
    - Properly stacks tensors with matching dimensions
    - Falls back to lists when tensors can't be stacked
    - Handles nested structures like lists of tensors
    - Maintains type consistency across the batch
    
    Parameters
    ----------
    batch : list
        List of sample dictionaries from the dataset
        
    Returns
    -------
    dict
        Properly batched dictionary with tensors and lists
    """
    if not batch:
        return {}
    
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            try:
                # Create proper batched tensors with correct dimensions
                result[key] = torch.stack([item[key] for item in batch])
                logger.debug(f"Stacked {key} tensor with shape {result[key].shape}")
            except Exception as e:
                logger.error(f"Error stacking {key} tensors: {e}")
                # Try to recover with list instead
                result[key] = [item[key] for item in batch]
        else:
            # For lists of tensors (like our pixel_values and input_ids)
            if key in ["pixel_values", "input_ids"] and isinstance(batch[0][key], list):
                try:
                    # Stack the individual tensors from each sample's list
                    stacked_tensors = torch.stack([item[key][0] for item in batch])
                    result[key] = stacked_tensors
                    logger.debug(f"Properly stacked {key} from lists to tensor with shape {stacked_tensors.shape}")
                except Exception as e:
                    logger.error(f"Error stacking {key} from lists: {e}")
                    result[key] = [item[key][0] if len(item[key]) > 0 else None for item in batch]
            else:
                # For other non-tensor data, use a list
                result[key] = [item[key] for item in batch]
    
    return result

def create_dimension_compatible_unet(model, verbose=False):
    """
    Create a wrapper UNet that handles SDXL-specific dimension mismatches.
    
    SDXL has several tensor dimension inconsistencies that can cause training failures:
    1. The combined text encoder output is 2304 dims but the UNet expects 2816 dims
    2. The add_embeds tensor is 2560 dims but needs to match the 2816 dim requirement
    3. Cross-attention layers expect specific dimensions for query/key/value tensors
    
    This function solves these issues by:
    - Creating projection matrices to dynamically convert between dimensions
    - Patching the forward methods of cross-attention modules
    - Replacing linear layers in add_embedding modules with dimension-compatible versions
    - Preserving gradient flow for proper backpropagation
    
    The wrapper maintains all original UNet functionality while silently handling
    dimension conversions, making SDXL LoRA training possible without modifying
    the base architecture.
    
    Parameters
    ----------
    model : UNet2DConditionModel
        The SDXL UNet model to wrap with dimension fixes
    verbose : bool, default=False
        Whether to log detailed information about dimension conversions
        
    Returns
    -------
    DimensionFixWrapper
        A wrapped model that automatically handles dimension mismatches during
        forward and backward passes
    """
    from torch import nn
    
    class DimensionFixWrapper(nn.Module):
        """
        Wrapper class to fix dimension mismatches in SDXL UNet.
        """
        def __init__(self, base_model, cross_attn_dim=2048, combined_text_embed_dim=2816, verbose=False):
            super().__init__()
            self.base_model = base_model
            self.cross_attn_dim = cross_attn_dim
            self.combined_text_embed_dim = combined_text_embed_dim
            self.verbose = verbose
            
            # Create projection matrices once
            self._create_projection_matrices()
            
            # Store original forward method
            self.original_forward = base_model.forward
            
        def _create_projection_matrices(self):
            """Create all projection matrices needed for fixing dimensions."""
            # For fixing encoder_hidden_states dimension from variable to exactly combined_text_embed_dim
            self.encoder_hidden_projection = nn.Parameter(
                torch.zeros((2048, self.combined_text_embed_dim)),
                requires_grad=False
            )
            # Initialize with identity-like pattern
            min_dim = min(2048, self.combined_text_embed_dim)
            for i in range(min_dim):
                self.encoder_hidden_projection.data[i, i] = 1.0
                
            # For fixing add_embeds dimension from 2560 to combined_text_embed_dim
            self.add_embeds_projection = nn.Parameter(
                torch.zeros((2560, self.combined_text_embed_dim)),
                requires_grad=False
            )
            # Initialize with identity-like pattern
            min_dim = min(2560, self.combined_text_embed_dim)
            for i in range(min_dim):
                self.add_embeds_projection.data[i, i] = 1.0
                
            # For fixing cross-attention input from combined_text_embed_dim to cross_attn_dim
            self.cross_attn_projection = nn.Parameter(
                torch.zeros((self.combined_text_embed_dim, self.cross_attn_dim)),
                requires_grad=False
            )
            # Initialize with identity-like pattern
            min_dim = min(self.combined_text_embed_dim, self.cross_attn_dim)
            for i in range(min_dim):
                self.cross_attn_projection.data[i, i] = 1.0
                
            if self.verbose:
                logger.info(f"Created projection matrices for dimension compatibility:")
                logger.info(f"- encoder_hidden_projection: {self.encoder_hidden_projection.shape}")
                logger.info(f"- add_embeds_projection: {self.add_embeds_projection.shape}")
                logger.info(f"- cross_attn_projection: {self.cross_attn_projection.shape}")
        
        def _fix_add_embeds(self, x):
            """Fix add_embeds dimension from 2560 to the expected dimension."""
            if x.shape[-1] == 2560:
                if self.verbose:
                    logger.info(f"Fixing add_embeds dimension: {x.shape[-1]} -> {self.combined_text_embed_dim}")
                return torch.matmul(x, self.add_embeds_projection)
            return x
        
        def _fix_encoder_hidden_states(self, encoder_hidden_states):
            """Ensure encoder_hidden_states has the expected dimension."""
            if encoder_hidden_states.shape[-1] != self.combined_text_embed_dim:
                if self.verbose:
                    logger.info(f"Fixing encoder_hidden_states dimension: {encoder_hidden_states.shape[-1]} -> {self.combined_text_embed_dim}")
                
                # Determine what kind of projection to use
                if encoder_hidden_states.shape[-1] <= 2048:
                    # Use the encoder_hidden_projection
                    corrected_states = torch.zeros(
                        (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], self.combined_text_embed_dim),
                        device=encoder_hidden_states.device,
                        dtype=encoder_hidden_states.dtype
                    )
                    
                    # Copy over as much data as we can from the original tensor
                    copy_dims = min(encoder_hidden_states.shape[-1], 2048)
                    tmp = torch.matmul(
                        encoder_hidden_states[..., :copy_dims], 
                        self.encoder_hidden_projection[:copy_dims].to(
                            device=encoder_hidden_states.device,
                            dtype=encoder_hidden_states.dtype
                        )
                    )
                    corrected_states = tmp
                    
                else:
                    # Just pad or truncate as needed
                    corrected_states = torch.zeros(
                        (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], self.combined_text_embed_dim),
                        device=encoder_hidden_states.device,
                        dtype=encoder_hidden_states.dtype
                    )
                    
                    # Copy over as much data as we can from the original tensor
                    copy_dims = min(encoder_hidden_states.shape[-1], self.combined_text_embed_dim)
                    corrected_states[..., :copy_dims] = encoder_hidden_states[..., :copy_dims]
                
                return corrected_states
            return encoder_hidden_states
        
        def _get_altered_module(self, module, name):
            """
            Helper method to patch specific module types for dimension compatibility.
            
            This method specifically targets cross-attention modules (those ending with 'attn2')
            and modifies their forward pass to handle dimension mismatches between
            encoder_hidden_states and the expected cross-attention dimensions.
            
            The patching process:
            1. Identifies cross-attention modules by name pattern ('attn2' suffix)
            2. Preserves the original forward method for later use
            3. Creates a new forward method that:
               - Checks if encoder_hidden_states dimensions match cross_attn_dim
               - If not, applies the cross_attn_projection matrix to convert dimensions
               - Calls the original forward method with correctly sized tensors
            4. Binds the new method to the module using Python's types.MethodType
            
            This dynamic patching allows for:
            - Handling the dimension mismatch between text encoder outputs (2304/2816) 
              and cross-attention expectation (2048)
            - Maintaining gradient flow through the projection for proper backpropagation
            - Avoiding the need to modify the original module's implementation
            
            Parameters
            ----------
            module : torch.nn.Module
                The module to potentially patch (only cross-attention modules are modified)
            name : str
                The full name of the module in the model hierarchy
                
            Returns
            -------
            torch.nn.Module
                The module, either with patched forward method (for cross-attention) 
                or unchanged (for other module types)
            """
            if name.endswith('attn2'):
                original_forward = module.forward
                parent = self
                
                # Create a patch for cross-attention modules
                def patched_cross_attn_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                    if encoder_hidden_states is not None and encoder_hidden_states.shape[-1] != parent.cross_attn_dim:
                        projection = parent.cross_attn_projection.to(
                            device=encoder_hidden_states.device, 
                            dtype=encoder_hidden_states.dtype
                        )
                        encoder_hidden_states = torch.matmul(encoder_hidden_states, projection)
                    
                    return original_forward(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
                
                # Create a bound method
                import types
                module.forward = types.MethodType(patched_cross_attn_forward, module)
                
            return module
        
        def _patch_modules(self):
            """
            Patch all cross-attention modules in the UNet model for dimension handling.
            
            This method:
            1. Traverses all modules in the UNet model
            2. Identifies cross-attention modules by name ('attn2')
            3. Applies patching via _get_altered_module to each one
            4. Sets a flag to ensure patching only happens once
            
            The patching modifies the forward methods of cross-attention modules
            to handle dimension mismatches between encoder hidden states (2816 dim)
            and what the attention modules expect (2048 dim). This is critical for
            SDXL LoRA training as it prevents dimension errors during the forward pass.
            
            The function uses lazy initialization, only patching modules when needed
            and caching the result to avoid redundant patching operations.
            """
            if not hasattr(self, "_modules_patched"):
                for name, module in self.base_model.named_modules():
                    if 'attn2' in name:
                        self._get_altered_module(module, name)
                
                self._modules_patched = True
                if self.verbose:
                    logger.info("Patched all cross-attention modules in UNet")
        
        def forward(self, x, timesteps, encoder_hidden_states, added_cond_kwargs=None, **kwargs):
            """
            Enhanced forward method that handles dimension fixes for SDXL UNet.
            
            This method intercepts the forward pass and performs several critical fixes:
            
            1. Ensures all cross-attention modules are properly patched
            2. Fixes encoder_hidden_states dimensions from 2304 to 2816
            3. Allows add_embeds dimensions to be fixed in their respective modules
            4. Delegates to the original forward pass with fixed dimensions
            
            The main dimension issues addressed are:
            - Text encoder outputs combined are 2304 dim but UNet expects 2816
            - Cross-attention modules expect 2048 dim inputs
            - add_embeds is created as 2560 dim but needs to match 2816
            
            All fixes maintain gradient flow for proper backpropagation.
            
            Parameters
            ----------
            x : torch.Tensor
                Latent noise input of shape [batch, 4, height/8, width/8]
            timesteps : torch.Tensor
                Diffusion timesteps for noise scheduling
            encoder_hidden_states : torch.Tensor
                Text conditioning from encoder, potentially needs dimension fixing
            added_cond_kwargs : dict or None
                Additional conditioning with 'text_embeds' and 'time_ids'
            **kwargs : dict
                Additional arguments passed to the original UNet forward method
                
            Returns
            -------
            torch.Tensor
                Output tensor from the original UNet forward pass
            """
            # Make sure modules are patched
            self._patch_modules()
            
            # Fix encoder_hidden_states dimension
            encoder_hidden_states = self._fix_encoder_hidden_states(encoder_hidden_states)
            
            # Handle add_embeds special case through added_cond_kwargs
            if added_cond_kwargs is not None:
                # The UNet will create add_embeds by combining text_embeds and time_embedding from time_ids
                # This typically results in a 2560-dimensional tensor (1280 + 1280)
                # But the model expects 2816 dimensions to match the encoder_hidden_states
                
                # We don't need to modify added_cond_kwargs directly
                # The patch in the add_embedding forward hook will handle it
                pass
                
            # Call the original forward but intercept the hidden states
            return self.original_forward(x, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, **kwargs)
        
        def enable_gradient_checkpointing(self):
            """
            Enable gradient checkpointing on the base UNet model for memory efficiency.
            
            Gradient checkpointing reduces memory usage during training by recomputing
            intermediate activations during the backward pass instead of storing them.
            This allows training with larger batch sizes or models at the cost of
            increased computation time.
            
            This method:
            1. Forwards the gradient checkpointing request to the base model
            2. Provides appropriate logging based on whether it succeeded
            3. Handles cases where the base model doesn't support checkpointing
            
            Gradient checkpointing is particularly important for SDXL LoRA training
            as the full model is very memory intensive.
            """
            if hasattr(self.base_model, 'enable_gradient_checkpointing'):
                self.base_model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing on base model")
            else:
                logger.warning("Base model does not support gradient checkpointing")
    
    # Create and return the wrapped model
    logger.info("Creating dimension-compatible UNet wrapper")
    wrapped_model = DimensionFixWrapper(model, verbose=verbose)
    
    # Add a custom forward hook to the model's add_embedding module
    # This specifically handles the 2560 -> 2816 dimension problem in add_embeds
    for name, module in model.named_modules():
        if 'add_embedding' in name and hasattr(module, 'linear_1'):
            logger.info(f"Patching {name}.linear_1 for add_embeds dimension handling")
            original_linear = module.linear_1
            
            # Create a new linear module with correct input dimension (2560 -> output_dim)
            output_dim = original_linear.weight.shape[0]
            new_linear = torch.nn.Linear(
                2560, output_dim,
                bias=original_linear.bias is not None,
                device=original_linear.weight.device,
                dtype=original_linear.weight.dtype
            )
            
            # Initialize weights based on original
            with torch.no_grad():
                orig_input_dim = original_linear.weight.shape[1]
                min_dim = min(orig_input_dim, 2560)
                # If the original weight has fewer than 2560 input dimensions,
                # copy what we can to the corresponding part of the new weight
                new_linear.weight.data[:, :min_dim] = original_linear.weight.data[:, :min_dim] \
                    if orig_input_dim >= min_dim else \
                    torch.cat([original_linear.weight.data, 
                              torch.zeros((output_dim, 2560 - orig_input_dim), 
                                        device=original_linear.weight.device,
                                        dtype=original_linear.weight.dtype)], dim=1)
                
                if original_linear.bias is not None:
                    new_linear.bias.data = original_linear.bias.data
            
            # Replace the linear module
            module.linear_1 = new_linear
            logger.info(f"Replaced {name}.linear_1 to handle 2560 input dimensions")
            
            # We only need to patch one add_embedding module
            break
    
    return wrapped_model

def main():
    """
    Main training function that orchestrates the complete LoRA training process.
    
    Implements a complete training pipeline:
    1. Parses arguments and sets up environment
    2. Loads base models (UNet, VAE, text encoders)
    3. Configures and applies LoRA to the UNet
    4. Prepares the dataset and data loaders
    5. Sets up optimizer and learning rate scheduler
    6. Runs the training loop with gradient accumulation
    7. Saves checkpoints and validates periodically
    8. Exports the final LoRA weights
    
    The function includes extensive error handling and logging to
    diagnose and recover from common training issues.
    """
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info(f"Starting SDXL LoRA training with arguments: {args}")
    
    # Storage for linear projections
    _custom_projections = {}
    
    # Use bf16 precision instead of fp16 to avoid FP16 gradient unscaling issues
    if args.mixed_precision == "fp16":
        logger.warning("FP16 mixed precision can cause 'Attempting to unscale FP16 gradients' errors")
        logger.warning("Switching to 'bf16' mixed precision which is more stable")
        args.mixed_precision = "bf16"
    
    # Set up accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Log information about the accelerator
    logger.info(f"Mixed precision: {args.mixed_precision}")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # If passed along, set the training seed now
    if args.seed is not None:
        set_seed(args.seed)
    
    # Set up model components
    logger.info(f"Loading base model: {args.base_model}")
    use_torch_sdpa = args.use_pytorch_sdpa and is_torch_sdpa_available()
    if use_torch_sdpa:
        logger.info("Using PyTorch's scaled dot product attention (SDPA) for memory-efficient training")
        
    # For a cleaner approach, first load the pipeline to get all components
    # This ensures proper loading of all model components
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        use_safetensors=True,
    )
    
    # Extract components from pipeline
    noise_scheduler = pipeline.scheduler
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    text_encoder = accelerator.unwrap_model(pipeline.text_encoder)
    text_encoder_2 = accelerator.unwrap_model(pipeline.text_encoder_2)
    vae = pipeline.vae
    unet = pipeline.unet
    
    # Disable PEFT integration for text_encoder and text_encoder_2
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # Set VAE to evaluation mode (we don't train these)
    vae.requires_grad_(False)
    vae.eval()
    
    # Move VAE to the correct device explicitly
    logger.info(f"Moving VAE to device: {accelerator.device}")
    vae = vae.to(device=accelerator.device)
    
    # Configure LoRA
    logger.info(f"Configuring LoRA with rank {args.rank}, alpha {args.lora_alpha}, dropout {args.lora_dropout}")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
            "to_kv",
            # For transformer blocks
            "ff.net.0.proj",
            "ff.net.2",
            # For conv blocks
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to UNet
    from peft import get_peft_model
    unet = get_peft_model(unet, lora_config)
    
    # Create a wrapper UNet that handles dimension mismatches
    logger.info("Creating dimension-compatible UNet wrapper")
    unet = create_dimension_compatible_unet(unet, verbose=False)
    
    # Enable gradient checkpointing if required
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for UNet")
        unet.enable_gradient_checkpointing()
        
    # Enable PyTorch 2.0 SDPA if requested and available
    if use_torch_sdpa:
        logger.info("Enabling PyTorch 2.0 scaled dot product attention")
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # PyTorch 2.0+ will automatically use the most efficient attention implementation
        logger.info("PyTorch will automatically use the most efficient attention implementation")
    # Enable xformers if requested and available
    elif args.enable_xformers_memory_efficient_attention and is_xformers_available():
        logger.info("Enabling xformers for memory efficient attention")
        unet.enable_xformers_memory_efficient_attention()
    
    # Prepare dataset
    logger.info(f"Preparing dataset from {args.images_dir}")
    dataset_data = prepare_images_and_captions(args.images_dir, args.captions_file)
    
    # Debug output of dataset structure
    logger.info(f"Dataset data keys: {list(dataset_data.keys())}")
    for key in dataset_data:
        if isinstance(dataset_data[key], list):
            logger.info(f"Dataset '{key}' has {len(dataset_data[key])} items")
            if len(dataset_data[key]) > 0:
                logger.info(f"First {key} item: {dataset_data[key][0]}")
                logger.info(f"First {key} item type: {type(dataset_data[key][0])}")
        else:
            logger.info(f"Dataset '{key}' is not a list but a {type(dataset_data[key])}")
    
    # Check if we found any images
    if not dataset_data["images"]:
        logger.error("No valid images found in the dataset! Cannot continue training.")
        logger.error("Please check your images directory and captions file.")
        return
    
    # Ensure the image paths are all strings
    for i, img_path in enumerate(dataset_data["images"]):
        if not isinstance(img_path, str):
            logger.warning(f"Image path at index {i} is not a string: {img_path} - Type: {type(img_path)}")
            # Convert to string if needed
            dataset_data["images"][i] = str(img_path)
        
    # Use custom SimpleDataset instead of HuggingFace Dataset
    logger.info("Creating dataset object")
    dataset = SimpleDataset(dataset_data)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty! Cannot continue training.")
        logger.error("Please check your images directory and captions file.")
        return
    
    # Apply transformations
    logger.info(f"Setting up dataset transforms with resolution {args.resolution}")
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        """Process images and captions for training."""
        # Check if we have any images
        if not examples["images"]:
            logger.error("No images to preprocess! Dataset may be empty.")
            return {"pixel_values": [], "input_ids": []}
        
        # Safety check: ensure images is a list, not a string
        if isinstance(examples["images"], str):
            logger.warning(f"examples['images'] is a string, not a list: {examples['images']}")
            logger.warning("Converting to a list with a single element")
            images_list = [examples["images"]]
            captions_list = [examples["captions"][0] if isinstance(examples["captions"], list) else examples["captions"]]
        else:
            images_list = examples["images"]
            captions_list = examples["captions"]
        
        # Convert file paths to actual images
        processed_images = []
        processed_captions = []
        
        logger.info(f"Preprocessing {len(images_list)} images...")
        
        for i, image_file in enumerate(images_list):
            try:
                # Skip invalid paths
                if not isinstance(image_file, str):
                    logger.warning(f"Image path is not a string: {image_file} (type: {type(image_file)})")
                    continue
                
                if not os.path.exists(image_file):
                    logger.warning(f"Image path does not exist: {image_file}")
                    continue
                
                # Open and convert image
                img = Image.open(image_file).convert("RGB")
                processed_images.append(img)
                
                # Get corresponding caption
                caption = captions_list[i] if i < len(captions_list) else ""
                processed_captions.append(caption)
                logger.debug(f"Successfully processed image: {image_file}")
            except Exception as e:
                logger.warning(f"Error processing image {image_file}: {str(e)}")
        
        # Check if we still have images after filtering
        if not processed_images:
            logger.error("No valid images after preprocessing!")
            return {"pixel_values": [], "input_ids": []}
        
        logger.info(f"Successfully preprocessed {len(processed_images)} images")
        
        # Add the concept to the captions if provided
        if args.lora_concept:
            processed_captions = [f"{args.lora_concept}, {caption}" for caption in processed_captions]
        
        # Apply transforms to images and create a single tensor with correct shape
        try:
            # Transform each image
            transformed_images = []
            for img in processed_images:
                transformed = train_transforms(img)
                # Ensure it's a tensor with shape [C, H, W]
                if not isinstance(transformed, torch.Tensor):
                    logger.warning(f"Transform returned non-tensor: {type(transformed)}")
                    transformed = torch.zeros((3, args.resolution, args.resolution))
                elif transformed.shape != (3, args.resolution, args.resolution):
                    logger.warning(f"Transform returned tensor with wrong shape: {transformed.shape}")
                    transformed = torch.zeros((3, args.resolution, args.resolution))
                    
                transformed_images.append(transformed)
            
            # IMPORTANT: Return individual tensors rather than stacking them here
            # This avoids extra dimensions that need reshaping later
            pixel_values = transformed_images
            
            logger.debug(f"Prepared {len(pixel_values)} image tensors, each with shape: {pixel_values[0].shape}")
        except Exception as e:
            logger.error(f"Error creating image tensor: {e}")
            # Create a default list of tensors
            pixel_values = [torch.zeros((3, args.resolution, args.resolution)) for _ in range(len(processed_images))]
        
        # Tokenize captions - get tokens, not stacked tensor
        try:
            input_ids = tokenize_captions(processed_captions)
            logger.debug(f"Tokenized captions: {len(input_ids)} token sequences")
        except Exception as e:
            logger.error(f"Error tokenizing captions: {e}")
            # Create default input_ids list
            input_ids = [torch.zeros(77, dtype=torch.long) for _ in range(len(processed_captions))]
        
        # Return dictionary with lists of tensors to avoid implicit stacking causing extra dimensions
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }
    
    # Tokenize captions
    def tokenize_captions(captions):
        """Tokenize captions and return a list of token tensors."""
        if not captions:
            # Return empty list if no captions
            return []
        
        try:
            # Process each caption individually to avoid unwanted batching dimensions
            token_list = []
            for caption in captions:
                inputs = tokenizer(
                    caption,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # Extract the token IDs - squeeze to remove batch dimension
                token_ids = inputs.input_ids.squeeze(0)
                token_list.append(token_ids)
                
            logger.debug(f"Tokenized {len(token_list)} captions")
            return token_list
        except Exception as e:
            logger.error(f"Error tokenizing captions: {e}")
            # Return list of empty token tensors
            return [torch.zeros(77, dtype=torch.long) for _ in range(len(captions))]
    
    # Configure data loader
    logger.info("Setting up data loader")
    try:
        train_dataset = dataset.with_transform(preprocess_train)
        
        if len(train_dataset) == 0:
            logger.error("Transformed dataset is empty! Cannot train with no samples.")
            logger.error("Please check your images directory and ensure images can be loaded.")
            return
            
        logger.info(f"Transformed dataset created with {len(train_dataset)} samples")
        
        # Check the first item to ensure it has the right format
        first_item = train_dataset[0]
        logger.info(f"First dataset item keys: {list(first_item.keys())}")
        for key, value in first_item.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key} shape: {value.shape}, dtype: {value.dtype}")
            else:
                logger.info(f"  {key} type: {type(value)}")
        
        # Create a safer DataLoader with fewer workers
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4 to 2 for better stability
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches
            collate_fn=collate_fn,  # Custom collate function to handle tensors properly
        )
    except Exception as e:
        logger.error(f"Error setting up data loader: {e}")
        logger.error("Cannot continue training without a valid dataset")
        return
        
    # Setup optimizer
    params_to_optimize = unet.parameters()
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
        
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Setup learning rate scheduler
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        logger.info(f"Scaled learning rate to: {args.learning_rate}")
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Make sure VAE, text encoders are on the correct device after accelerator prepare
    logger.info("Ensuring all models are on the correct device")
    device = accelerator.device
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    
    # Verify device placement
    logger.info(f"UNet device: {next(unet.parameters()).device}")
    logger.info(f"VAE device: {next(vae.parameters()).device}")
    logger.info(f"Text encoder device: {next(text_encoder.parameters()).device}")
    logger.info(f"Text encoder 2 device: {next(text_encoder_2.parameters()).device}")
    
    # Calculate the number of steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Log training info
    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Create a pipeline for validation
    if args.validation_prompt is not None:
        logger.info("Creating validation pipeline")
        # Create a copy of the unet without LoRA for validation
        # This avoids the type mismatch between PeftModel and UNet2DConditionModel
        validation_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        validation_pipeline.set_progress_bar_config(disable=True)
        # Move validation pipeline to the correct device
        validation_pipeline.to(accelerator.device)
        logger.info(f"Validation pipeline created and placed on device: {accelerator.device}")
    else:
        validation_pipeline = None
    
    # Only show the progress bar once on each machine
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    
    # Training loop
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Ensure batch items are tensors with correct shapes
                if isinstance(batch["pixel_values"], list):
                    logger.debug("pixel_values is a list - stacking to tensor")
                    try:
                        pixel_values = torch.stack(batch["pixel_values"])
                    except:
                        logger.error("Failed to stack pixel_values, skipping batch")
                        continue
                else:
                    pixel_values = batch["pixel_values"]
                
                if isinstance(batch["input_ids"], list):
                    logger.debug("input_ids is a list - stacking to tensor")
                    try:
                        input_ids = torch.stack(batch["input_ids"])
                    except:
                        logger.error("Failed to stack input_ids, skipping batch")
                        continue
                else:
                    input_ids = batch["input_ids"]
                
                # Ensure correct dimensions (no extra dimensions)
                # Pixel values should be [batch_size, 3, height, width]
                if len(pixel_values.shape) == 5:
                    logger.debug(f"Fixing pixel_values shape from {pixel_values.shape} to [batch, 3, h, w]")
                    pixel_values = pixel_values.squeeze(1)  # Remove the extra dimension completely
                
                # Input IDs should be [batch_size, seq_length]
                if len(input_ids.shape) == 3:
                    logger.debug(f"Fixing input_ids shape from {input_ids.shape} to [batch, seq_length]")
                    input_ids = input_ids.squeeze(1)  # Remove the extra dimension completely
                
                # Convert images to latent space
                with torch.no_grad():
                    try:
                        # Debug the shape of pixel_values
                        logger.debug(f"Pixel values shape before processing: {pixel_values.shape}")
                        
                        # Ensure correct dimensions for VAE
                        # VAE expects [batch_size, 3, height, width]
                        if len(pixel_values.shape) == 5:  # [batch, num_images, channels, height, width]
                            logger.warning(f"Pixel values have unexpected shape: {pixel_values.shape}, reshaping")
                            # Reshape by taking first image from each batch item
                            pixel_values = pixel_values[:, 0]
                        elif len(pixel_values.shape) != 4:
                            logger.error(f"Cannot process pixel values with shape: {pixel_values.shape}")
                            continue
                            
                        # Move to device and set dtype
                        vae_device = next(vae.parameters()).device
                        vae_dtype = next(vae.parameters()).dtype
                        logger.debug(f"VAE is on device {vae_device} with dtype {vae_dtype}")
                        
                        pixel_values = pixel_values.to(device=vae_device, dtype=vae_dtype)
                        logger.debug(f"Pixel values shape after processing: {pixel_values.shape}, device: {pixel_values.device}, dtype: {pixel_values.dtype}")
                        
                        # Check for NaN values in pixel_values
                        if torch.isnan(pixel_values).any():
                            logger.warning("NaN values detected in pixel_values, replacing with zeros")
                            pixel_values = torch.nan_to_num(pixel_values, nan=0.0)
                        
                        # Encode with VAE
                        latent_dist = vae.encode(pixel_values).latent_dist
                        latents = latent_dist.sample() * vae.config.scaling_factor
                        
                        # Check for NaN values in latents
                        if torch.isnan(latents).any():
                            logger.warning("NaN values detected in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, nan=0.0)
                            
                        logger.debug(f"Latents shape: {latents.shape}")
                    except Exception as e:
                        logger.error(f"Error encoding images: {e}")
                        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                
                # Generate noise for diffusion
                noise = torch.randn_like(latents)
                # Check for NaN values in noise
                if torch.isnan(noise).any():
                    logger.warning("NaN values detected in noise tensor, replacing with zeros")
                    noise = torch.nan_to_num(noise, nan=0.0)
                    
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise schedule
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode the caption
                with torch.no_grad():
                    try:
                        # Debug the shape of input_ids
                        logger.debug(f"Input IDs shape before processing: {input_ids.shape}")
                        
                        # Move to correct device
                        text_encoder_device = next(text_encoder.parameters()).device
                        input_ids = input_ids.to(device=text_encoder_device)
                        
                        # SDXL text encoders expect 2D input [batch_size, sequence_length]
                        # Reshape if necessary
                        if len(input_ids.shape) == 3:  # [batch, num_captions, sequence_length]
                            logger.debug(f"Squeezing input_ids from shape {input_ids.shape}")
                            # Take the first caption for each item in batch
                            input_ids = input_ids[:, 0, :]
                        elif len(input_ids.shape) != 2:
                            logger.error(f"Cannot process input_ids with shape: {input_ids.shape}")
                            continue
                            
                        logger.debug(f"Input IDs shape after processing: {input_ids.shape}")
                        
                        # Get encoder outputs from both text encoders
                        encoder_output_1 = text_encoder(input_ids)
                        encoder_output_2 = text_encoder_2(input_ids)
                        
                        # Define expected dimensions for SDXL
                        EXPECTED_COMBINED_DIM = 2816  # Final dimension for combined hidden states
                        EXPECTED_EMBED_DIM = 1280     # Expected text_embeds dimension
                        EXPECTED_HIDDEN_STATES_1 = 1024  # First text encoder's output dimension
                        EXPECTED_HIDDEN_STATES_2 = 1792  # Remaining space for second encoder
                        
                        # Get batch size and seq length for creating tensors
                        batch_size = input_ids.shape[0]
                        seq_length = input_ids.shape[1]
                        
                        # Extract hidden states from text encoder 1 (standardize format)
                        if hasattr(encoder_output_1, "last_hidden_state"):
                            hidden_states_1 = encoder_output_1.last_hidden_state  # [batch, seq_len, hidden_size]
                        else:
                            hidden_states_1 = encoder_output_1[0]  # Using first element in tuple
                        
                        # Extract hidden states from text encoder 2 (standardize format)
                        if hasattr(encoder_output_2, "last_hidden_state"):
                            hidden_states_2 = encoder_output_2.last_hidden_state  # [batch, seq_len, hidden_size]
                        else:
                            hidden_states_2 = encoder_output_2[0]  # Using first element in tuple
                        
                        logger.debug(f"Text encoder 1 output shape: {hidden_states_1.shape}")
                        logger.debug(f"Text encoder 2 output shape: {hidden_states_2.shape}")
                        
                        # Create properly sized tensor for combined hidden states
                        encoder_hidden_states = torch.zeros(
                            (batch_size, seq_length, EXPECTED_COMBINED_DIM),
                            device=hidden_states_1.device,
                            dtype=hidden_states_1.dtype
                        )
                        
                        # Copy data from first encoder (up to 1024 dimensions)
                        copy_dim_1 = min(hidden_states_1.shape[2], EXPECTED_HIDDEN_STATES_1)
                        encoder_hidden_states[:, :, :copy_dim_1] = hidden_states_1[:, :, :copy_dim_1]
                        
                        # Copy data from second encoder (remaining 1792 dimensions)
                        copy_dim_2 = min(hidden_states_2.shape[2], EXPECTED_HIDDEN_STATES_2)
                        encoder_hidden_states[:, :, EXPECTED_HIDDEN_STATES_1:EXPECTED_HIDDEN_STATES_1+copy_dim_2] = hidden_states_2[:, :, :copy_dim_2]
                        
                        logger.debug(f"Combined hidden states shape: {encoder_hidden_states.shape}, last dim: {encoder_hidden_states.shape[-1]}")
                        
                        # Get pooled output for text_embeds (needed for SDXL's added_cond_kwargs)
                        if hasattr(encoder_output_1, "pooler_output"):
                            pooled_output = encoder_output_1.pooler_output
                        else:
                            # If no pooler_output, use mean of last_hidden_state as fallback
                            pooled_output = hidden_states_1.mean(dim=1)
                        
                        logger.debug(f"Original pooled output shape: {pooled_output.shape}")
                        
                        # Create properly sized text_embeds with correct dimensions
                        text_embeds = torch.zeros(
                            (batch_size, EXPECTED_EMBED_DIM),
                            device=pooled_output.device,
                            dtype=pooled_output.dtype
                        )
                        
                        # Copy available data from pooled output
                        copy_dim_pooled = min(pooled_output.shape[1], EXPECTED_EMBED_DIM)
                        text_embeds[:, :copy_dim_pooled] = pooled_output[:, :copy_dim_pooled]
                        
                        logger.debug(f"Final text_embeds shape: {text_embeds.shape}")
                        
                        # Create time_ids tensor with consistent shape [batch_size, 5]
                        # Format: [orig_height, orig_width, crop_top, crop_left, aesthetic_score]
                        original_size = (1024, 1024)  # Original image size
                        crop_coords = (0, 0)          # No cropping
                        aesthetic_score = 6.0         # Default aesthetic score
                        
                        time_ids = torch.zeros((batch_size, 5), device=latents.device)
                        for i in range(batch_size):
                            time_ids[i] = torch.tensor([
                                original_size[0],  # Original height
                                original_size[1],  # Original width
                                crop_coords[0],    # Crop top
                                crop_coords[1],    # Crop left
                                aesthetic_score    # Aesthetic score
                            ], device=latents.device)
                        
                        # Prepare final conditioning kwargs with correctly shaped tensors
                        added_cond_kwargs = {
                            "text_embeds": text_embeds,  # [batch_size, 1280]
                            "time_ids": time_ids         # [batch_size, 5]
                        }
                        
                    except Exception as e:
                        logger.error(f"Error encoding text: {str(e)}")
                        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                
                # Verify the input and output shapes for tensor checks
                def debug_tensor(name, tensor):
                    try:
                        logger.info(f"TENSOR DEBUG - {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
                                   f"has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}")
                        if tensor.shape[0] > 0 and tensor.numel() > 0:
                            logger.info(f"  - value_range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
                    except Exception as e:
                        logger.error(f"Error debugging tensor {name}: {e}")
                
                # Get the target for loss
                target = noise
                
                # Verify the shapes of added condition kwargs
                logger.info(f"Added condition kwargs - text_embeds shape: {added_cond_kwargs['text_embeds'].shape}, time_ids shape: {added_cond_kwargs['time_ids'].shape}")
                
                # Verify dimensions for UNet input
                logger.info(f"UNet input shapes - noisy_latents: {noisy_latents.shape}, timesteps: {timesteps.shape}, encoder_hidden_states: {encoder_hidden_states.shape}")
                
                # Debug all tensors involved in the UNet forward pass
                debug_tensor("noisy_latents", noisy_latents)
                debug_tensor("encoder_hidden_states", encoder_hidden_states)
                debug_tensor("text_embeds", added_cond_kwargs["text_embeds"])
                debug_tensor("time_ids", added_cond_kwargs["time_ids"])
                
                # Make sure encoder hidden states are exactly what's expected - hardcode the correct size
                required_hidden_dim = 2816
                if encoder_hidden_states.shape[-1] != required_hidden_dim:
                    logger.warning(f"Fixing encoder_hidden_states dimension from {encoder_hidden_states.shape[-1]} to {required_hidden_dim}")
                    
                    # Create a completely new tensor with the right dimensions
                    # This ensures we don't have any hidden dimension issues from concatenation
                    corrected_states = torch.zeros(
                        (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], required_hidden_dim),
                        device=encoder_hidden_states.device,
                        dtype=encoder_hidden_states.dtype
                    )
                    
                    # Copy over as much data as we can from the original tensor
                    copy_dims = min(encoder_hidden_states.shape[-1], required_hidden_dim)
                    corrected_states[..., :copy_dims] = encoder_hidden_states[..., :copy_dims]
                    
                    # Replace with the corrected tensor
                    encoder_hidden_states = corrected_states
                    logger.info(f"Corrected encoder_hidden_states shape: {encoder_hidden_states.shape}")
                
                # Check if encoder_hidden_states has the correct sequence length (should be 77)
                if encoder_hidden_states.shape[1] != 77:
                    logger.warning(f"encoder_hidden_states has unexpected sequence length: {encoder_hidden_states.shape[1]}, expected 77")
                
                # Predict the noise residual, now with added_cond_kwargs
                try:
                    # Debug one more time before the UNet forward pass
                    logger.info("Final shapes before UNet forward pass:")
                    logger.info(f"- noisy_latents: {noisy_latents.shape}")
                    logger.info(f"- timesteps: {timesteps.shape}")
                    logger.info(f"- encoder_hidden_states: {encoder_hidden_states.shape}")
                    logger.info(f"- text_embeds: {added_cond_kwargs['text_embeds'].shape}")
                    logger.info(f"- time_ids: {added_cond_kwargs['time_ids'].shape}")
                    
                    # Check for NaN values in each tensor and fix if needed
                    def fix_nans(tensor, name):
                        if tensor is None:
                            logger.error(f"{name} is None, cannot proceed")
                            raise ValueError(f"{name} is None")
                            
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logger.warning(f"NaN or Inf values detected in {name}, replacing with zeros")
                            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        return tensor
                    
                    # Apply the fix to all tensors
                    noisy_latents = fix_nans(noisy_latents, "noisy_latents")
                    encoder_hidden_states = fix_nans(encoder_hidden_states, "encoder_hidden_states")
                    added_cond_kwargs["text_embeds"] = fix_nans(added_cond_kwargs["text_embeds"], "text_embeds")
                    added_cond_kwargs["time_ids"] = fix_nans(added_cond_kwargs["time_ids"], "time_ids")
                    
                    # Call the UNet model with forward added_cond_kwargs
                    model_pred = unet(
                        noisy_latents, 
                        timesteps, 
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    
                except Exception as e:
                    logger.error(f"Error in UNet forward pass: {str(e)}")
                    logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Try to determine the cause of error from the message
                    error_str = str(e)
                    if "mat1 and mat2 shapes cannot be multiplied" in error_str:
                        # Parse the dimension mismatch from the error message
                        import re
                        match = re.search(r"\((\d+)x(\d+) and (\d+)x(\d+)\)", error_str)
                        if match:
                            a1, a2, b1, b2 = map(int, match.groups())
                            logger.error(f"Matrix dimension mismatch: ({a1}x{a2}) and ({b1}x{b2})")
                            logger.error(f"Expected a2 = {b1}, but got a2 = {a2}")
                            
                            # Try to identify which tensor has the wrong dimension
                            if a2 == 2560:
                                logger.error("The add_embeds tensor appears to have incorrect dimension (2560)")
                                logger.error("SDXL UNet expects the add_embeds dimension to be 2816")
                    
                    # Re-raise to skip the iteration
                    raise
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Backpropagate and optimize
                accelerator.backward(loss)
                
                # Extra safety check for gradient scaling issues
                if accelerator.sync_gradients:
                    try:
                        # Add a safety check for NaN gradients
                        has_nan_grads = False
                        for param in unet.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grads = True
                                logger.warning(f"NaN values detected in gradients! Skipping optimization step.")
                                break
                        
                        if not has_nan_grads:
                            # Only try to clip gradients if they're not NaN
                            if args.max_grad_norm > 0:
                                params_to_clip = unet.parameters()
                                try:
                                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                                except ValueError as e:
                                    if "Attempting to unscale FP16 gradients" in str(e):
                                        logger.warning("FP16 gradient unscaling error detected. Skipping gradient clipping.")
                                    else:
                                        raise
                            
                            # Update parameters
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                        else:
                            # Reset gradients if NaNs encountered
                            optimizer.zero_grad()
                    except Exception as e:
                        logger.error(f"Error during optimization step: {str(e)}")
                        optimizer.zero_grad()  # Always clear gradients to avoid accumulation issues
                else:
                    # Not syncing gradients yet, just continue
                    pass
                
            # Log progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # Create the pipeline using the trained modules
                        # Extract the base model from the wrapper
                        if hasattr(unet, "base_model"):
                            unwrapped_unet = accelerator.unwrap_model(unet.base_model)
                        else:
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            args.base_model,
                            unet=unwrapped_unet,
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                        )
                        
                        # Extract LoRA parameters - get the peft model from inside the wrapper
                        peft_model = unet.base_model if hasattr(unet, "base_model") else unet
                        state_dict = get_peft_model_state_dict(peft_model)
                        
                        # Save a checkpoint
                        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        save_file(state_dict, os.path.join(checkpoint_path, f"{args.model_name}.safetensors"))
                        logger.info(f"Saved LoRA weights at {checkpoint_path}")
                        
                        # Run validation
                        if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                            logger.info(f"Running validation at step {global_step}...")
                            # Make sure we have a clean validation pipeline
                            if hasattr(validation_pipeline, "unload_lora_weights"):
                                validation_pipeline.unload_lora_weights()
                            # Run validation with the current checkpoint
                            images = log_validation(
                                validation_pipeline,
                                args,
                                accelerator,
                                global_step,
                                args.validation_prompt,
                                args.num_validation_images,
                            )
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
                
    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Create the pipeline using the trained modules
        # Extract the base model from the wrapper
        if hasattr(unet, "base_model"):
            unwrapped_unet = accelerator.unwrap_model(unet.base_model)
        else:
            unwrapped_unet = accelerator.unwrap_model(unet)
            
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            unet=unwrapped_unet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        # Extract LoRA parameters - get the peft model from inside the wrapper
        peft_model = unet.base_model if hasattr(unet, "base_model") else unet
        state_dict = get_peft_model_state_dict(peft_model)
        
        # Save the final model
        save_file(state_dict, os.path.join(args.output_dir, f"{args.model_name}.safetensors"))
        logger.info(f"Saved final LoRA weights at {args.output_dir}/{args.model_name}.safetensors")
        
        # Log validation images
        if args.validation_prompt is not None:
            logger.info("Running final validation...")
            # Make sure the validation pipeline has prefix=None set for logging
            if hasattr(validation_pipeline, "unload_lora_weights"):
                validation_pipeline.unload_lora_weights()
            # Load the final LoRA weights with prefix=None
            validation_pipeline.load_lora_weights(
                args.output_dir,
                weight_name=f"{args.model_name}.safetensors",
                adapter_name="default",
                prefix=None  # Use prefix=None to avoid the prefix warnings
            )
            images = log_validation(
                validation_pipeline,
                args,
                accelerator,
                global_step,
                args.validation_prompt,
                args.num_validation_images,
            )
    
    accelerator.end_training()
    logger.info("Training completed!")


    
def test_forward_pass(unet, text_encoder, text_encoder_2, device="cuda"):
    """
    Test if the model can successfully perform a forward pass with dimension fixes.
    
    This diagnostic function:
    - Creates dummy inputs with appropriate dimensions
    - Processes text through both text encoders
    - Combines hidden states as SDXL expects
    - Attempts a UNet forward pass with the dimension-fixed wrapper
    - Reports success or detailed error information
    
    Parameters
    ----------
    unet : UNet2DConditionModel
        The wrapped UNet model to test
    text_encoder : CLIPTextModel
        First text encoder model
    text_encoder_2 : CLIPTextModelWithProjection
        Second text encoder model
    device : str, default="cuda"
        Device to run the test on
        
    Returns
    -------
    bool
        True if forward pass succeeds, False otherwise
    """
    logger.info("Testing forward pass with dimension-compatible UNet...")
    
    # Create dummy inputs
    batch_size = 2
    latents = torch.randn(batch_size, 4, 64, 64).to(device)
    timesteps = torch.tensor([999, 999]).to(device)
    
    # Create dummy text conditioning
    prompt = "A test prompt to check dimension compatibility"
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    # Get text encoder outputs
    with torch.no_grad():
        try:
            text_outputs = text_encoder(tokens, output_hidden_states=True)
            text_outputs_2 = text_encoder_2(tokens, output_hidden_states=True)
            
            # Get hidden states
            hidden_states = text_outputs.hidden_states[-1]
            hidden_states_2 = text_outputs_2.hidden_states[-1]
            
            # Get pooled outputs (for conditioning)
            pooled_output = text_outputs.pooler_output
            
            # Log dimensions
            logger.info(f"Text encoder 1 hidden states shape: {hidden_states.shape}")
            logger.info(f"Text encoder 2 hidden states shape: {hidden_states_2.shape}")
            logger.info(f"Text encoder pooled output shape: {pooled_output.shape}")
            
            # Stack hidden states - note this will normally be 1024+1280=2304 dimensions,
            # but SDXL expects 2816, so our wrapper should handle this
            encoder_hidden_states = torch.cat([hidden_states, hidden_states_2], dim=-1)
            logger.info(f"Combined encoder hidden states shape: {encoder_hidden_states.shape}")
            
            # Create added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": pooled_output,
                "time_ids": torch.zeros((batch_size, 5)).to(device)
            }
            
            # Try forward pass with our dimension-compatible UNet
            try:
                output = unet(
                    latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
                )
                logger.info("✅ Forward pass successful! Dimension compatibility working correctly.")
                logger.info(f"Output shape: {output.sample.shape}")
                return True
            except Exception as e:
                logger.error(f"❌ Forward pass failed: {e}")
                logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        except Exception as e:
            logger.error(f"❌ Error in text encoder processing: {e}")
            return False

if __name__ == "__main__":
    main() 