"""
AI Service for Cattle Classification
Generates trait scores based on uploaded images
Uses side view model for detailed side view analysis
"""
from app.models.trait_definitions import TRAIT_DEFINITIONS, get_all_traits_flat
from typing import List, Dict
import random
from datetime import datetime

class AIService:
    """
    AI Service for cattle trait evaluation
    """
    
    def __init__(self):
        """Initialize service"""
        self.traits = get_all_traits_flat()
        print("✓ AI Service initialized for trait evaluation")
    
    async def classify_animal(self, image_paths: List[str], animal_info: Dict) -> Dict:
        """
        Generate trait scores for cattle classification
        
        Args:
            image_paths: 5 image paths in order [rear, side, top, udder, side_udder]
            animal_info: Animal details dict
        
        Returns:
            Complete classification results with trait scores
        """
        print(f"✓ Processing classification for {len(image_paths)} images")
        
        # Process rear view image for BCS if available
        bcs_data = None
        if len(image_paths) >= 1:  # rear view is first image
            rear_view_path = image_paths[0]
            print(f"Processing rear view image for BCS: {rear_view_path}")
            bcs_data = self._process_bcs(rear_view_path)
        
        # Process side view image with model if available
        side_view_traits = None
        if len(image_paths) >= 2:  # side view is second image
            side_view_path = image_paths[1]
            print(f"Processing side view image with model: {side_view_path}")
            side_view_traits = self._process_side_view(side_view_path)
        
        # Generate base results
        results = self._generate_mock_results(animal_info)
        
        # Merge BCS data if available
        if bcs_data:
            results = self._merge_bcs_data(results, bcs_data)
        
        # Merge side view model results if available
        if side_view_traits:
            results = self._merge_side_view_traits(results, side_view_traits)
        
        return results
    
    def _process_side_view(self, image_path: str) -> Dict:
        """
        Process side view image with the side view model
        
        Args:
            image_path: Path to side view image
            
        Returns:
            Dictionary with processed traits or None if processing fails
        """
        try:
            from ml_models.side_view_integration import process_side_view, extract_side_traits
            
            # Process with model
            raw_data = process_side_view(image_path)
            if not raw_data:
                print("  ⚠ Side view model processing failed, using generated data")
                return None
            
            # Extract formatted traits
            traits = extract_side_traits(raw_data)
            print(f"  ✓ Extracted {len(traits)} side view traits from model")
            
            return {
                'traits': traits,
                'meta': raw_data.get('meta', {})
            }
            
        except Exception as e:
            print(f"  ⚠ Side view model error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_bcs(self, image_path: str) -> Dict:
        """
        Process BCS (Body Condition Score) from rear view image
        
        Args:
            image_path: Path to rear view image
            
        Returns:
            Dictionary with BCS data or None if processing fails
        """
        try:
            from ml_models.bcs_integration import process_bcs
            
            # Process BCS
            bcs_result = process_bcs(image_path)
            if not bcs_result:
                print("  ⚠ BCS processing failed, using generated data")
                return None
            
            print(f"  ✓ BCS Score: {bcs_result['score']} ({bcs_result['condition']})")
            
            return bcs_result
            
        except Exception as e:
            print(f"  ⚠ BCS processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _merge_bcs_data(self, base_results: Dict, bcs_data: Dict) -> Dict:
        """
        Merge BCS data into results
        
        Args:
            base_results: Generated results from mock data
            bcs_data: BCS data from processing
            
        Returns:
            Merged results with BCS data
        """
        if not bcs_data:
            return base_results
        
        # Update General section with BCS
        official_format = base_results.get('officialFormat', {})
        sections = official_format.get('sections', {})
        
        if 'General' in sections:
            for i, trait in enumerate(sections['General']):
                if trait['trait'] == 'Body condition score':
                    # Update with BCS model data
                    sections['General'][i]['score'] = bcs_data['score']
                    sections['General'][i]['measurement'] = bcs_data['score']  # BCS uses same value
                    print(f"  ✓ Updated Body condition score: {bcs_data['score']} ({bcs_data['condition']})")
                    break
        
        # Recalculate overall scores
        all_scores = []
        for section_traits in sections.values():
            all_scores.extend([t['score'] for t in section_traits if t['score'] is not None])
        
        if all_scores:
            overall = round(sum(all_scores) / len(all_scores), 1)
            base_results['overallScore'] = overall
            
            # Update grade
            if overall >= 7.5:
                base_results['grade'] = "Excellent"
            elif overall >= 6.0:
                base_results['grade'] = "Good"
            elif overall >= 4.0:
                base_results['grade'] = "Fair"
            else:
                base_results['grade'] = "Poor"
            
            # Update category scores
            base_results['categoryScores'] = {
                cat: round(sum(t['score'] for t in traits if t['score'] is not None) / 
                          len([t for t in traits if t['score'] is not None]), 1)
                if [t for t in traits if t['score'] is not None] else 0
                for cat, traits in sections.items()
            }
        
        # Add BCS metadata
        base_results['bcsData'] = {
            'score': bcs_data['score'],
            'condition': bcs_data['condition'],
            'source': bcs_data['source']
        }
        
        return base_results
    
    def _merge_side_view_traits(self, base_results: Dict, side_view_data: Dict) -> Dict:
        """
        Merge side view model traits with base results
        
        Args:
            base_results: Generated results from mock data
            side_view_data: Processed traits from side view model
            
        Returns:
            Merged results with side view traits replacing mock data where available
        """
        if not side_view_data or 'traits' not in side_view_data:
            return base_results
        
        # Map side view trait names to our official trait names
        trait_mapping = {
            'Body Length': 'Body Length',
            'Stature': 'Stature',
            'Heart Girth (circumference)': 'Heart Girth',
            'Body Depth': 'Body Depth'
        }
        
        # Update traits in Body Capacity section with model data
        official_format = base_results.get('officialFormat', {})
        sections = official_format.get('sections', {})
        
        if 'Body Capacity' in sections:
            for model_trait in side_view_data['traits']:
                model_trait_name = model_trait['trait']
                
                # Find matching official trait
                official_name = trait_mapping.get(model_trait_name)
                if not official_name:
                    continue
                
                # Update the trait in Body Capacity section
                for i, trait in enumerate(sections['Body Capacity']):
                    if trait['trait'] == official_name:
                        # Update with model data
                        if model_trait['score'] is not None:
                            sections['Body Capacity'][i]['score'] = model_trait['score']
                        if model_trait['measurement'] is not None:
                            sections['Body Capacity'][i]['measurement'] = model_trait['measurement']
                        print(f"  ✓ Updated {official_name} from side view model: score={model_trait['score']}, measurement={model_trait['measurement']}")
        
        # Recalculate overall scores after merging
        all_scores = []
        for section_traits in sections.values():
            all_scores.extend([t['score'] for t in section_traits if t['score'] is not None])
        
        if all_scores:
            overall = round(sum(all_scores) / len(all_scores), 1)
            base_results['overallScore'] = overall
            
            # Update grade based on new overall score
            if overall >= 7.5:
                base_results['grade'] = "Excellent"
            elif overall >= 6.0:
                base_results['grade'] = "Good"
            elif overall >= 4.0:
                base_results['grade'] = "Fair"
            else:
                base_results['grade'] = "Poor"
            
            # Update category scores
            base_results['categoryScores'] = {
                cat: round(sum(t['score'] for t in traits if t['score'] is not None) / 
                          len([t for t in traits if t['score'] is not None]), 1)
                if [t for t in traits if t['score'] is not None] else 0
                for cat, traits in sections.items()
            }
        
        # Add metadata about model usage
        if 'meta' in side_view_data:
            base_results['sideViewModelMeta'] = side_view_data['meta']
        
        return base_results

    def _calculate_milk_yield(self, traits: List[Dict], animal_info: Dict) -> Dict:
        """
        Calculate daily milk yield prediction from body measurements
        """
        measurements = {trait['trait']: trait.get('measurement', 0) for trait in traits}
        scores = {trait['trait']: trait['score'] for trait in traits}
        
        # Body measurements
        stature = measurements.get('Stature', 135)
        body_length = measurements.get('Body Length', 155)
        body_depth = measurements.get('Body Depth', 70)
        heart_girth = measurements.get('Heart Girth', 190)
        
        # Body capacity factor
        body_capacity = (stature * body_length * body_depth * heart_girth) / 1000000
        body_factor = 1 + (body_capacity - 140) / 100
        
        # Udder quality factor
        udder_traits = ['Fore Udder Attachment', 'Rear Udder Height', 'Central Ligament', 'Udder Depth']
        udder_scores = [scores.get(t, 5) for t in udder_traits]
        udder_factor = sum(udder_scores) / (len(udder_scores) * 9)
        
        # Udder measurements
        rear_udder_width = measurements.get('Rear udder width', 25)
        udder_width_factor = 1 + (rear_udder_width - 25) / 50
        
        # Body condition
        bcs = scores.get('Body condition score', 5)
        bcs_factor = 0.9 + (bcs - 5) * 0.02
        
        # Lactation factor
        lactation = animal_info.get('lactationNumber', 1)
        lactation_factor = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.0}.get(lactation, 0.90)
        
        # Breed base yield
        breed = animal_info.get('breed', '').lower()
        if any(b in breed for b in ['holstein', 'hf', 'friesian']):
            breed_base = 25
        elif 'jersey' in breed:
            breed_base = 20
        elif any(b in breed for b in ['gir', 'sahiwal', 'red sindhi']):
            breed_base = 12
        elif any(b in breed for b in ['murrah', 'mehsana', 'surti']):
            breed_base = 10
        else:
            breed_base = 15
        
        # Calculate yield
        daily_yield = breed_base * body_factor * udder_factor * udder_width_factor * bcs_factor * lactation_factor
        daily_yield = round(daily_yield, 1)
        
        return {
            "dailyYield": daily_yield,
            "minYield": round(daily_yield * 0.85, 1),
            "maxYield": round(daily_yield * 1.15, 1),
            "unit": "liters/day",
            "lactationYield": round(daily_yield * 305, 0),
            "lactationUnit": "liters/305 days",
            "confidence": 85
        }
    
    def _generate_mock_results(self, animal_info: Dict) -> Dict:
        """Generate realistic mock data following official format"""
        
        sections = {}
        
        for category, traits in TRAIT_DEFINITIONS.items():
            section_traits = []
            for trait in traits:
                score = random.randint(5, 8)
                measurement = None
                
                # Add realistic measurements
                if trait['measurement_unit'] == 'cm':
                    if 'Stature' in trait['name']:
                        measurement = random.randint(130, 145)
                    elif 'Girth' in trait['name']:
                        measurement = random.randint(180, 200)
                    elif 'Length' in trait['name']:
                        measurement = random.randint(145, 165)
                    elif 'Depth' in trait['name']:
                        measurement = random.randint(65, 80)
                    elif 'Width' in trait['name']:
                        measurement = random.randint(20, 50)
                    elif 'Height' in trait['name']:
                        measurement = random.randint(15, 25)
                    elif 'Teat' in trait['name']:
                        measurement = round(random.uniform(2, 8), 1)
                    else:
                        measurement = random.randint(10, 40)
                elif trait['measurement_unit'] == 'degrees':
                    measurement = random.randint(25, 50)
                
                section_traits.append({
                    "trait": trait['name'],
                    "score": score,
                    "measurement": measurement
                })
            
            sections[category] = section_traits
        
        # Calculate overall
        all_scores = []
        all_traits_list = []
        for traits in sections.values():
            all_scores.extend([t['score'] for t in traits])
            all_traits_list.extend(traits)
        
        overall = round(sum(all_scores) / len(all_scores), 1)
        
        # Calculate milk yield prediction
        milk_yield = self._calculate_milk_yield(all_traits_list, animal_info)
        
        return {
            "animalInfo": animal_info,
            "officialFormat": {
                "villageName": animal_info.get('village', ''),
                "farmerName": animal_info.get('farmerName', ''),
                "animalTagNo": animal_info['tagNumber'],
                "dateOfBirth": animal_info['dateOfBirth'],
                "lactationNo": animal_info['lactationNumber'],
                "dateOfCalving": animal_info['dateOfCalving'],
                "classificationDate": datetime.now().strftime("%Y-%m-%d"),
                "classifiedBy": "AI System (Mock Data)",
                "sections": sections
            },
            "categoryScores": {
                cat: round(sum(t['score'] for t in traits) / len(traits), 1)
                for cat, traits in sections.items()
            },
            "overallScore": overall,
            "grade": "Good" if overall >= 6 else "Fair",
            "totalTraits": 20,
            "confidenceLevel": 88,
            "milkYieldPrediction": milk_yield
        }

# Singleton
ai_service = AIService()
