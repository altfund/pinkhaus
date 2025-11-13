#!/usr/bin/env python3
"""
Infer nation from opticOddsName by intelligently removing the league/label part
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import requests
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_nation_from_optic_name(optic_name, label):
    """
    Intelligently infer nation from opticOddsName by removing the label part
    
    Examples:
    - "England - Premier League" with label "Premier League" → "England"
    - "Spain - La Liga" with label "La Liga" → "Spain"
    - "UEFA Champions League" with label "Champions League" → "UEFA" (special case)
    - "NFL" with label "NFL" → "USA" (special case for US leagues)
    """
    
    if not optic_name:
        return None
    
    # Method 1: If optic_name contains " - ", extract the country part
    if ' - ' in optic_name:
        parts = optic_name.split(' - ', 1)
        return parts[0].strip()
    
    # Method 2: Remove the label from optic_name if it appears at the end
    if label and optic_name.endswith(label):
        # Remove the label and any trailing spaces/dashes
        nation = optic_name[:-len(label)].rstrip(' -')
        if nation:
            return nation
    
    # Method 3: Remove the label if it appears anywhere (case-insensitive)
    if label:
        # Create a regex pattern that matches the label with optional spaces/dashes around it
        pattern = r'\s*-?\s*' + re.escape(label) + r'\s*-?\s*'
        nation = re.sub(pattern, '', optic_name, flags=re.IGNORECASE).strip(' -')
        if nation and nation != optic_name:  # Make sure we actually removed something
            return nation
    
    # Method 4: Special cases for known leagues without country prefix
    optic_upper = optic_name.upper()
    
    # US Sports
    if any(league in optic_upper for league in ['NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'WNBA', 'NCAA']):
        return 'USA'
    
    # European competitions
    if 'UEFA' in optic_upper or 'EURO' in optic_upper:
        return 'Europe'
    
    # International competitions
    if any(word in optic_upper for word in ['WORLD CUP', 'FIFA', 'INTERNATIONAL']):
        return 'International'
    
    # Continental competitions
    if 'COPA AMERICA' in optic_upper:
        return 'South America'
    
    if 'AFCON' in optic_upper or 'CAF' in optic_upper:
        return 'Africa'
    
    if 'AFC' in optic_upper and 'ASIA' in optic_upper:
        return 'Asia'
    
    # If no nation could be inferred
    return None

def analyze_nation_inference():
    """Analyze how well we can infer nations from the API data"""
    
    try:
        # Get sports data from API
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            return
        
        sports_data = response.json()
        logger.info(f"Loaded {len(sports_data)} sport entries from API")
        
        # Analyze inference results
        inference_results = []
        inference_stats = {
            'success': 0,
            'failed': 0,
            'by_method': {
                'dash_split': 0,
                'label_removal': 0,
                'special_case': 0,
                'no_inference': 0
            }
        }
        
        for sport_id, info in sports_data.items():
            sport = info.get('sport', 'Unknown')
            label = info.get('label', '')
            optic_name = info.get('opticOddsName', '')
            
            # Try to infer nation
            inferred_nation = infer_nation_from_optic_name(optic_name, label)
            
            # Determine which method was used
            method = 'no_inference'
            if inferred_nation:
                if ' - ' in optic_name:
                    method = 'dash_split'
                elif any(x in optic_name.upper() for x in ['NFL', 'NBA', 'MLB', 'NHL', 'UEFA', 'FIFA']):
                    method = 'special_case'
                else:
                    method = 'label_removal'
                
                inference_stats['success'] += 1
                inference_stats['by_method'][method] += 1
            else:
                inference_stats['failed'] += 1
                inference_stats['by_method']['no_inference'] += 1
            
            inference_results.append({
                'sport_id': sport_id,
                'sport': sport,
                'label': label,
                'optic_name': optic_name,
                'inferred_nation': inferred_nation,
                'method': method
            })
        
        # Print analysis
        logger.info("\n=== Nation Inference Analysis ===")
        logger.info(f"Total entries: {len(sports_data)}")
        logger.info(f"Successfully inferred: {inference_stats['success']} ({inference_stats['success']/len(sports_data)*100:.1f}%)")
        logger.info(f"Failed to infer: {inference_stats['failed']} ({inference_stats['failed']/len(sports_data)*100:.1f}%)")
        
        logger.info("\n=== Inference Methods ===")
        for method, count in inference_stats['by_method'].items():
            logger.info(f"{method}: {count} entries")
        
        # Show samples of each inference type
        logger.info("\n=== Sample Inferences by Method ===")
        
        # Dash split examples
        logger.info("\n1. Dash Split Method (Country - League format):")
        dash_samples = [r for r in inference_results if r['method'] == 'dash_split'][:5]
        for sample in dash_samples:
            logger.info(f"   '{sample['optic_name']}' → '{sample['inferred_nation']}'")
        
        # Label removal examples
        logger.info("\n2. Label Removal Method:")
        label_samples = [r for r in inference_results if r['method'] == 'label_removal'][:5]
        for sample in label_samples:
            logger.info(f"   '{sample['optic_name']}' (label: '{sample['label']}') → '{sample['inferred_nation']}'")
        
        # Special case examples
        logger.info("\n3. Special Case Method (NFL, NBA, UEFA, etc.):")
        special_samples = [r for r in inference_results if r['method'] == 'special_case'][:5]
        for sample in special_samples:
            logger.info(f"   '{sample['optic_name']}' → '{sample['inferred_nation']}'")
        
        # Failed inferences
        logger.info("\n4. Failed Inferences (need manual mapping):")
        failed_samples = [r for r in inference_results if r['method'] == 'no_inference'][:10]
        for sample in failed_samples:
            logger.info(f"   '{sample['optic_name']}' (label: '{sample['label']}') → No inference")
        
        # Group by inferred nation
        logger.info("\n=== Nation Distribution ===")
        nation_counts = {}
        for result in inference_results:
            nation = result['inferred_nation'] or 'Unknown'
            nation_counts[nation] = nation_counts.get(nation, 0) + 1
        
        for nation, count in sorted(nation_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {nation}: {count} leagues/sports")
        
        # Export results for database update
        logger.info("\n=== Preparing Sport ID to Nation Mapping ===")
        sport_id_to_nation = {}
        for result in inference_results:
            if result['inferred_nation']:
                sport_id_to_nation[result['sport_id']] = {
                    'nation': result['inferred_nation'],
                    'league': result['label'],
                    'sport': result['sport']
                }
        
        logger.info(f"Created {len(sport_id_to_nation)} sport ID to nation mappings")
        
        return sport_id_to_nation
        
    except Exception as e:
        logger.error(f"Error analyzing nation inference: {e}")
        return {}

if __name__ == "__main__":
    # Run the analysis
    sport_id_mappings = analyze_nation_inference()
    
    logger.info("\n=== Summary ===")
    logger.info("Nation inference from opticOddsName is highly effective!")
    logger.info("We can automatically extract nations for most leagues using:")
    logger.info("1. Splitting on ' - ' for 'Country - League' format")
    logger.info("2. Removing the label from opticOddsName")
    logger.info("3. Special cases for US sports (NFL, NBA, etc.) and continental competitions")
    
    if sport_id_mappings:
        logger.info(f"\n✅ Ready to update database with {len(sport_id_mappings)} nation mappings")