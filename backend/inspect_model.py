import torch

# Load and inspect the .pth file
print("Loading cattle_model.pth...")
checkpoint = torch.load('ml_models/cattle_model.pth', map_location='cpu')

print(f"\nType: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
    for key in list(checkpoint.keys())[:10]:  # Show first 10 keys
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor of shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
else:
    print(f"\nCheckpoint is not a dict, it's: {type(checkpoint)}")
    if hasattr(checkpoint, 'state_dict'):
        print("It has state_dict method")

print("\nFile inspection complete!")
