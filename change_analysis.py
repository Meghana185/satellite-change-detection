import numpy as np
import config

def compute_area_percentages(category_map):
    """
    Computes the percentage area for each category in a given map.
    
    Args:
        category_map: 2D numpy array of strings
        
    Returns:
        dict: mapping from category string to percentage (float)
    """
    total_tiles = category_map.size
    unique, counts = np.unique(category_map, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    percentages = {}
    for cat in config.CATEGORY_COLORS.keys():
        count = counts_dict.get(cat, 0)
        percentages[cat] = (count / total_tiles) * 100
        
    return percentages

def analyze_changes(map_before, map_after):
    """
    Compares two category maps to find changes.
    
    Args:
        map_before: 2D numpy array of strings (Year 1)
        map_after: 2D numpy array of strings (Year 2)
        
    Returns:
        dict containing:
            - 'percentages_before': dict
            - 'percentages_after': dict
            - 'change_in_pp': dict (percentage point change per category)
            - 'changed_tiles_mask': 2D boolean array (True where categories differ)
    """
    # Ensure maps are the same shape
    assert map_before.shape == map_after.shape, "Maps must have the same dimensions for comparison."
    
    # Calculate percentages
    perc_before = compute_area_percentages(map_before)
    perc_after = compute_area_percentages(map_after)
    
    # Calculate change in percentage points
    change_pp = {}
    for cat in config.CATEGORY_COLORS.keys():
        change_pp[cat] = perc_after[cat] - perc_before[cat]
        
    # Create mask of changed tiles (where the category string differs)
    changed_tiles_mask = (map_before != map_after)
    
    return {
        'percentages_before': perc_before,
        'percentages_after': perc_after,
        'change_in_pp': change_pp,
        'changed_tiles_mask': changed_tiles_mask
    }

def expand_mask_to_image_size(mask_2d, tile_size=config.TILE_SIZE):
    """
    Expands a 2D tile mask to the full pixel resolution.
    
    Args:
        mask_2d: boolean numpy array (num_tiles_y, num_tiles_x)
        
    Returns:
        boolean numpy array (num_tiles_y * tile_size, num_tiles_x * tile_size)
    """
    # Repeat elements along Y and X axes to match pixel dimensions
    expanded = np.repeat(mask_2d, tile_size, axis=0)
    expanded = np.repeat(expanded, tile_size, axis=1)
    return expanded

def calculate_environmental_risk(change_pp):
    """
    Calculates a simple, explainable rule-based environmental risk score.
    Uses weights defined in config.py.
    
    Args:
        change_pp: dict of percentage point changes per category.
        
    Returns:
        tuple: (score (0-100), risk_level_string, explanation_string)
    """
    score = 0
    reasons = []
    
    # Check Vegetation (decreases increase risk)
    veg_change = change_pp.get("Vegetation", 0)
    if veg_change < 0:
        # Add points proportional to the decrease? Or fixed? 
        # For simplicity, we just add the fixed weight if there's *any* significant decrease (>1%)
        # But to be more dynamic, let's just use the weight directly if it's negative
        score += config.RISK_WEIGHTS["vegetation_decrease"]
        reasons.append(f"Vegetation decreased by {abs(veg_change):.1f}%")
        
    # Check Water (decreases increase risk)
    water_change = change_pp.get("Water", 0)
    if water_change < 0:
        score += config.RISK_WEIGHTS["water_decrease"]
        reasons.append(f"Water decreased by {abs(water_change):.1f}%")
        
    # Check Built-up (increases increase risk)
    built_change = change_pp.get("Built-up", 0)
    if built_change > 0:
        score += config.RISK_WEIGHTS["builtup_increase"]
        reasons.append(f"Built-up area increased by {built_change:.1f}%")
        
    # Check Barren (increases increase risk)
    barren_change = change_pp.get("Barren", 0)
    if barren_change > 0:
        score += config.RISK_WEIGHTS["barren_increase"]
        reasons.append(f"Barren land increased by {barren_change:.1f}%")
        
    # Normalize score to 0-100 (Max possible is the sum of all weights)
    max_possible_score = sum(config.RISK_WEIGHTS.values())
    if max_possible_score > 0:
        normalized_score = (score / max_possible_score) * 100
    else:
        normalized_score = 0
        
    # Determine Risk Label
    if normalized_score < 30:
        label = "Low"
    elif normalized_score <= 60:
        label = "Moderate"
    else:
        label = "High"
        
    if not reasons:
        explanation = "No significant detrimental environmental changes detected."
    else:
        explanation = "Risk factors identified: " + ", ".join(reasons) + "."
        
    return normalized_score, label, explanation
