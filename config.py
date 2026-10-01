# config.py
# Configuration for the Deep Learning-Based Satellite Image Change Detection MVP

# Model settings
MODEL_SAVE_PATH = "models/landcover_model.pth"
TILE_SIZE = 64  # Input images will be split into 64x64 grids
NUM_EPOCHS = 5  # For transfer learning
BATCH_SIZE = 32

# Map the 10 EuroSAT classes (0-9) to our 4 simpler categories.
# The original EuroSAT classes are typically ordered alphabetically in torchvision:
# 0: AnnualCrop, 1: Forest, 2: HerbaceousVegetation, 3: Highway,
# 4: Industrial, 5: Pasture, 6: PermanentCrop, 7: Residential,
# 8: River, 9: SeaLake

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# Mapping the 10 original classes to our 4 broad target categories.
CLASS_MAPPING = {
    "Vegetation": ["Forest", "HerbaceousVegetation", "PermanentCrop", "Pasture"],
    "Water": ["River", "SeaLake"],
    "Built-up": ["Residential", "Industrial", "Highway"],
    "Barren": ["AnnualCrop"]
}

# Assign standard colors (R, G, B) to our categories for visualizing maps.
CATEGORY_COLORS = {
    "Vegetation": (34, 139, 34),    # Green
    "Water": (30, 144, 255),        # Blue
    "Built-up": (128, 128, 128),    # Grey/Reddish could be used. Using Grey.
    "Barren": (218, 165, 32)        # Goldenrod / Yellow-Brown
}

# Environmental Risk Scoring logic
# We add points to the "risk score" based on detrimental environmental changes.
# Max score logic will be simple: 0-100 normalized.
RISK_WEIGHTS = {
    "vegetation_decrease": 30, # +30 points if total vegetation % decreases
    "water_decrease": 30,      # +30 points if total water % decreases
    "builtup_increase": 20,    # +20 points if total built-up % increases
    "barren_increase": 20      # +20 points if barren land % increases
}
