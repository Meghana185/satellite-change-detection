import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from PIL import Image

import config

def get_broad_category(eurosat_class_name):
    """
    Maps a specific EuroSAT class name (e.g., 'Forest') to our 4 broader categories.
    """
    for category, classes in config.CLASS_MAPPING.items():
        if eurosat_class_name in classes:
            return category
    return "Barren"  # Fallback if unknown

def load_trained_model(model_path=config.MODEL_SAVE_PATH):
    """
    Loads the fine-tuned MobileNetV2 model from disk.
    Returns the model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-initialize the model architecture
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def classify_tiles(image_array, model, tile_size=config.TILE_SIZE):
    """
    Splits an image array into a grid of tiles, classifies each tile,
    and maps the classification to our 4 broader categories.
    
    Args:
        image_array: numpy array (H, W, C)
        model: PyTorch trained model
        tile_size: int, size of square tiles
        
    Returns:
        category_map: 2D numpy array of strings containing the broad category for each tile
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use the same transform as during training (normalization)
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    h, w, _ = image_array.shape
    
    # Calculate grid dimensions (ignoring remainder pixels for simplicity in MVP)
    num_tiles_y = h // tile_size
    num_tiles_x = w // tile_size
    
    # Create empty map to hold our string categories
    category_map = np.empty((num_tiles_y, num_tiles_x), dtype=object)
    
    # We could batch this for speed, but for an MVP, a simple loop is easier to explain in a viva.
    with torch.no_grad():
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Extract tile
                tile = image_array[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size, :]
                
                # Convert numpy array back to PIL for torchvision transform
                tile_pil = Image.fromarray(tile)
                
                # Transform and add batch dimension
                input_tensor = transform(tile_pil).unsqueeze(0).to(device)
                
                # Predict
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                
                # Get EuroSAT class name
                eurosat_class = config.EUROSAT_CLASSES[predicted_idx.item()]
                
                # Map to broad category
                broad_category = get_broad_category(eurosat_class)
                
                category_map[y, x] = broad_category
                
    return category_map

def generate_color_map_image(category_map, tile_size=config.TILE_SIZE):
    """
    Creates a visual representation (RGB image) of the category map using standard colors.
    
    Args:
        category_map: 2D numpy array of strings
        
    Returns:
        numpy array representing the color-coded land cover map
    """
    num_tiles_y, num_tiles_x = category_map.shape
    
    # Initialize empty image
    out_img = np.zeros((num_tiles_y * tile_size, num_tiles_x * tile_size, 3), dtype=np.uint8)
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            category = category_map[y, x]
            color = config.CATEGORY_COLORS.get(category, (0, 0, 0))
            
            # Fill the tile with the corresponding color
            out_img[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = color
            
    return out_img
