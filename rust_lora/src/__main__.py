import argparse
import torch
from .lora import inject_lora_into_model

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA injection tool")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pth, .pt)")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA file (.safetensors, .pt)")
    parser.add_argument("--layers", type=str, required=True, 
                      help="Comma-separated list of layer names to apply LoRA to")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA scaling factor")
    parser.add_argument("--output", type=str, required=True, help="Path to save combined model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"📂 Loading model from {args.model}")
    model = torch.load(args.model)
    
    layer_names = [name.strip() for name in args.layers.split(",")]
    print(f"🔍 Target layers: {', '.join(layer_names)}")
    
    inject_lora_into_model(model, args.lora, layer_names, args.alpha)
    
    print(f"💾 Saving merged model to {args.output}")
    torch.save(model, args.output)
    print("✅ Done!")

if __name__ == "__main__":
    main() 