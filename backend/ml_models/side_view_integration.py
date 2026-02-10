"""
Side View Model Integration
Wrapper to run side view analysis and extract traits
"""
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ML_MODELS_DIR = Path(__file__).parent
SIDE_VIEW_SCRIPT = ML_MODELS_DIR / "side_view.py"
SIDE_VIEW_MODEL = ML_MODELS_DIR / "side_view_model_v2.pt"


def process_side_view(image_path: str) -> Optional[Dict]:
    """
    Process a side view image using the side view model
    
    Args:
        image_path: Path to the side view image
        
    Returns:
        Dictionary with traits, scores, and measurements
        Returns None if processing fails
    """
    if not os.path.exists(image_path):
        print(f"Side view image not found: {image_path}")
        return None
    
    if not SIDE_VIEW_SCRIPT.exists():
        print(f"Side view script not found: {SIDE_VIEW_SCRIPT}")
        return None
        
    if not SIDE_VIEW_MODEL.exists():
        print(f"Side view model not found: {SIDE_VIEW_MODEL}")
        return None
    
    try:
        # The script expects image to be named one of: input.jpg, cow.jpg, cow_side.jpg, etc.
        # We'll create a temporary symlink or copy with expected name
        script_dir = SIDE_VIEW_SCRIPT.parent
        temp_input = script_dir / "input.jpg"
        
        # Create a copy (safer than symlink on Windows)
        import shutil
        shutil.copy2(image_path, temp_input)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(SIDE_VIEW_SCRIPT)],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Clean up temporary file
        if temp_input.exists():
            temp_input.unlink()
        
        if result.returncode != 0:
            print(f"Side view script error: {result.stderr}")
            return None
        
        # Parse JSON output
        output_data = json.loads(result.stdout)
        print(f"✓ Side view model processed successfully")
        print(f"  Extracted {len(output_data.get('traits', []))} traits")
        
        return output_data
        
    except subprocess.TimeoutExpired:
        print("Side view processing timed out")
        if temp_input.exists():
            temp_input.unlink()
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse side view output: {e}")
        if temp_input.exists():
            temp_input.unlink()
        return None
    except Exception as e:
        print(f"Side view processing error: {e}")
        if temp_input.exists():
            temp_input.unlink()
        return None


def extract_side_traits(side_data: Dict) -> List[Dict]:
    """
    Extract and format traits from side view model output
    
    Args:
        side_data: Raw output from side view model
        
    Returns:
        List of trait dictionaries with name, score, and measurement
    """
    if not side_data or 'traits' not in side_data:
        return []
    
    formatted_traits = []
    
    for trait in side_data['traits']:
        trait_name = trait.get('trait', 'Unknown')
        score = trait.get('score')
        
        # Get measurement value (prefer cm, fallback to px or degrees)
        measurement = trait.get('value_cm')
        if measurement is None:
            measurement = trait.get('value_deg')
        if measurement is None:
            measurement = trait.get('value_px')
        
        formatted_traits.append({
            'trait': trait_name,
            'score': score,
            'measurement': measurement
        })
    
    return formatted_traits


if __name__ == "__main__":
    # Test the integration
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing side view model with: {test_image}")
        result = process_side_view(test_image)
        if result:
            print(json.dumps(result, indent=2))
            print("\nFormatted traits:")
            traits = extract_side_traits(result)
            for t in traits:
                print(f"  - {t['trait']}: score={t['score']}, measurement={t['measurement']}")
    else:
        print("Usage: python side_view_integration.py <image_path>")
