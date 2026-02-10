"""
BCS (Body Condition Score) Integration
Provides estimated body condition scores based on images
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

ML_MODELS_DIR = Path(__file__).parent


def estimate_bcs(image_path: str) -> Optional[Dict]:
    """
    Estimate Body Condition Score from image
    
    Args:
        image_path: Path to the image (typically rear or side view)
        
    Returns:
        Dictionary with BCS score or None if processing fails
    """
    if not os.path.exists(image_path):
        print(f"BCS: Image not found: {image_path}")
        return None
    
    try:
        # TODO: Replace with actual BCS model when available
        # For now, return a placeholder structure
        # BCS model would analyze:
        # - Rib visibility
        # - Hip bone prominence
        # - Tail head depression
        # - Back fill/loin muscling
        
        # Placeholder score (1-9 scale, where 1=emaciated, 9=obese, 5=ideal)
        # In production, this would come from actual model inference
        bcs_score = 5  # Moderate/ideal condition (placeholder)
        
        result = {
            'bcs_score': bcs_score,
            'condition': _get_condition_label(bcs_score),
            'source': 'placeholder',  # Will be 'model' when real model is integrated
            'message': 'Using placeholder BCS - integration ready for actual model'
        }
        
        print(f"✓ BCS estimated: {bcs_score} ({result['condition']})")
        return result
        
    except Exception as e:
        print(f"BCS estimation error: {e}")
        return None


def _get_condition_label(score: int) -> str:
    """
    Map BCS score to condition label
    
    Args:
        score: BCS score (1-9)
        
    Returns:
        Condition label string
    """
    if score <= 3:
        return "Thin"
    elif score == 4:
        return "Borderline"
    elif score == 5:
        return "Moderate"
    elif score == 6:
        return "Good"
    elif score == 7:
        return "Fat"
    elif score >= 8:
        return "Obese"
    else:
        return "Unknown"


def process_bcs(image_path: str) -> Optional[Dict]:
    """
    Process BCS from image (main entry point)
    
    Args:
        image_path: Path to the image
        
    Returns:
        Formatted BCS data or None
    """
    result = estimate_bcs(image_path)
    
    if not result:
        return None
    
    # Format for integration with AI service
    return {
        'score': result['bcs_score'],
        'trait': 'Body condition score',
        'condition': result['condition'],
        'source': result['source'],
        'message': result.get('message')
    }


if __name__ == "__main__":
    # Test the integration
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing BCS estimation with: {test_image}")
        result = process_bcs(test_image)
        if result:
            print(json.dumps(result, indent=2))
    else:
        print("Usage: python bcs_integration.py <image_path>")
