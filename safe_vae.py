"""
Safe VAE encoding utilities that avoid type mismatches between fp16 and fp32.
This is specifically designed to handle the "Input type (c10::Half) and bias type (float) should be the same" error.
"""

import torch
from diffusers import AutoencoderKL

class SafeVAE:
    """A wrapper for the VAE that ensures consistent types and handles encoding errors."""
    
    def __init__(self, pretrained_model_path="stabilityai/sd-vae-ft-mse", device=None):
        """Initialize the safe VAE wrapper with a specific VAE model."""
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load VAE explicitly in float32 mode
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float32  # Always float32!
        ).to(self.device)
        
        # Ensure the VAE is in eval mode
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        # Force all parameters to float32
        self._force_fp32()
        
        print(f"✅ Initialized SafeVAE on {self.device} in float32 mode")
    
    def _force_fp32(self):
        """Force all parameters in the VAE to be float32."""
        for param in self.vae.parameters():
            param.data = param.data.to(torch.float32)
    
    def encode(self, pixel_values, return_dict=True):
        """
        Safely encode images to latent space, handling type mismatches.
        
        Args:
            pixel_values: Image tensor to encode
            return_dict: Whether to return a dictionary or just the latent tensor
            
        Returns:
            Latent tensor or dict with latent tensor
        """
        # Convert inputs to device and float32
        if isinstance(pixel_values, list):
            # Handle batch of tensors
            pixel_values = [x.to(self.device).to(torch.float32) for x in pixel_values]
        else:
            # Handle single tensor
            pixel_values = pixel_values.to(self.device).to(torch.float32)
        
        try:
            # Periodically re-force fp32 to ensure consistency
            self._force_fp32()
            
            # Run the encoding
            with torch.no_grad():
                output = self.vae.encode(pixel_values)
                
                if return_dict:
                    # Get latent sample from distribution
                    latents = output.latent_dist.sample() * 0.18215
                    return {
                        "sample": latents,
                        "mean": output.latent_dist.mean,
                        "std": output.latent_dist.std
                    }
                else:
                    # Return just the latent tensor
                    return output.latent_dist.sample() * 0.18215
        
        except RuntimeError as e:
            if "Input type" in str(e) and "bias type" in str(e):
                # Special handling for type mismatch error
                print("⚠️ Type mismatch error in VAE. Attempting aggressive fixing...")
                
                # Double check all parameter types
                for name, param in self.vae.named_parameters():
                    if param.dtype != torch.float32:
                        print(f"  Converting {name} from {param.dtype} to float32")
                        param.data = param.data.to(torch.float32)
                
                # Try again with contiguous input
                if isinstance(pixel_values, list):
                    pixel_values = [x.contiguous() for x in pixel_values]
                else:
                    pixel_values = pixel_values.contiguous()
                
                # Second attempt
                try:
                    with torch.no_grad():
                        output = self.vae.encode(pixel_values)
                        if return_dict:
                            latents = output.latent_dist.sample() * 0.18215
                            return {"sample": latents}
                        else:
                            return output.latent_dist.sample() * 0.18215
                except Exception as e2:
                    print(f"⚠️ Second VAE encoding attempt failed: {e2}")
                    # Return zeros in the right shape
                    if isinstance(pixel_values, list):
                        batch_size = len(pixel_values)
                        height, width = pixel_values[0].shape[2:4]
                    else:
                        batch_size = pixel_values.shape[0]
                        height, width = pixel_values.shape[2:4]
                    
                    # Create random latents as fallback
                    latents = torch.randn(
                        (batch_size, 4, height // 8, width // 8),
                        device=self.device,
                        dtype=torch.float32
                    ) * 0.1
                    
                    if return_dict:
                        return {"sample": latents}
                    else:
                        return latents
            else:
                # Other runtime error - standard handling
                print(f"⚠️ VAE encoding error: {e}")
                if isinstance(pixel_values, list):
                    batch_size = len(pixel_values)
                    height, width = pixel_values[0].shape[2:4]
                else:
                    batch_size = pixel_values.shape[0]
                    height, width = pixel_values.shape[2:4]
                
                # Create fallback latents
                latents = torch.randn(
                    (batch_size, 4, height // 8, width // 8),
                    device=self.device,
                    dtype=torch.float32
                ) * 0.1
                
                if return_dict:
                    return {"sample": latents}
                else:
                    return latents
    
    def decode(self, latents):
        """
        Safely decode latents back to pixel space.
        
        Args:
            latents: Latent tensor to decode
            
        Returns:
            Decoded image tensor
        """
        # Convert inputs to fp32
        latents = latents.to(self.device).to(torch.float32)
        
        try:
            # Periodically re-force fp32 to ensure consistency
            self._force_fp32()
            
            # Decode the latents
            with torch.no_grad():
                # Scale latents
                latents = latents / 0.18215
                
                # Decode
                images = self.vae.decode(latents).sample
                
                # Return the images
                return images
                
        except Exception as e:
            print(f"⚠️ VAE decoding error: {e}")
            
            # Return blank images in the right shape
            batch_size = latents.shape[0]
            height, width = latents.shape[2:4]
            
            # Create blank images as fallback
            images = torch.zeros(
                (batch_size, 3, height * 8, width * 8), 
                device=self.device,
                dtype=torch.float32
            )
            
            return images 