
# Generate captions from ESC-50 category labels

import re
from typing import Dict, List, Optional


# ESC-50 category to caption templates from ChatGPT
CATEGORY_CAPTIONS: Dict[str, List[str]] = {
    # Animals
    'Dog': ['a dog barking', 'dog barking in the distance', 'a dog making sounds'],
    'Rooster': ['a rooster crowing', 'rooster call', 'a rooster making sounds'],
    'Pig': ['a pig oinking', 'pig sounds', 'a pig making noise'],
    'Cow': ['a cow mooing', 'cow sounds', 'a cow making noise'],
    'Frog': ['frogs croaking', 'frog sounds', 'frogs in nature'],
    'Cat': ['a cat meowing', 'cat sounds', 'a cat making noise'],
    'Hen': ['a hen clucking', 'hen sounds', 'a chicken making noise'],
    'Insects': ['insects chirping', 'insect sounds', 'insects in nature'],
    'Sheep': ['sheep bleating', 'sheep sounds', 'a sheep making noise'],
    'Crow': ['a crow cawing', 'crow sounds', 'a crow making noise'],
    
    # Natural soundscapes
    'Rain': ['steady rain falling', 'rain on pavement', 'rainfall', 'rain sounds'],
    'Sea_waves': ['ocean waves', 'sea waves crashing', 'waves on the shore', 'ocean sounds'],
    'Crackling_fire': ['crackling fire', 'fire burning', 'fireplace sounds', 'fire crackling'],
    'Crickets': ['crickets chirping', 'cricket sounds', 'crickets at night'],
    'Chirping_birds': ['birds chirping', 'bird sounds', 'birds singing', 'chirping birds'],
    'Water_drops': ['water drops', 'dripping water', 'water dripping'],
    'Wind': ['wind blowing', 'wind sounds', 'windy conditions', 'strong wind'],
    'Pouring_water': ['water pouring', 'pouring liquid', 'water being poured'],
    'Toilet_flush': ['toilet flushing', 'toilet sounds'],
    'Thunderstorm': ['thunderstorm', 'thunder and rain', 'storm sounds'],
    
    # Human sounds
    'Crying_baby': ['a baby crying', 'crying infant', 'baby sounds'],
    'Sneezing': ['someone sneezing', 'sneeze sound'],
    'Clapping': ['people clapping', 'applause', 'clapping hands'],
    'Breathing': ['breathing sounds', 'heavy breathing', 'breath'],
    'Coughing': ['someone coughing', 'cough sound'],
    'Footsteps': ['footsteps', 'walking sounds', 'footsteps on floor'],
    'Laughing': ['people laughing', 'laughter', 'laughing sounds'],
    'Brushing_teeth': ['brushing teeth', 'toothbrush sounds'],
    'Snoring': ['someone snoring', 'snoring sounds'],
    'Drinking_sipping': ['drinking sounds', 'sipping liquid'],
    
    # Interior/domestic sounds
    'Door_wood_creaks': ['door creaking', 'wooden door creaking', 'creaking door'],
    'Mouse_click': ['mouse clicking', 'computer mouse click'],
    'Keyboard_typing': ['keyboard typing', 'typing sounds', 'typing on keyboard'],
    'Door_wood_knock': ['knocking on door', 'door knock'],
    'Can_opening': ['can opening', 'opening a can'],
    'Washing_machine': ['washing machine', 'washing machine running'],
    'Vacuum_cleaner': ['vacuum cleaner', 'vacuuming sounds'],
    'Clock_alarm': ['alarm clock', 'clock alarm ringing'],
    'Clock_tick': ['clock ticking', 'ticking clock'],
    'Glass_breaking': ['glass breaking', 'breaking glass'],
    'Helicopter': ['helicopter flying', 'helicopter sounds'],
    
    # Exterior/urban sounds
    'Chainsaw': ['chainsaw running', 'chainsaw sounds'],
    'Siren': ['siren wailing', 'emergency siren', 'siren sound'],
    'Car_horn': ['car horn honking', 'car horn', 'horn sound'],
    'Engine': ['engine running', 'motor engine', 'engine sounds'],
    'Train': ['train passing', 'train sounds', 'train moving'],
    'Church_bells': ['church bells ringing', 'bells ringing'],
    'Airplane': ['airplane flying', 'aircraft sounds', 'airplane overhead'],
    'Fireworks': ['fireworks exploding', 'fireworks sounds'],
    'Hand_saw': ['saw cutting', 'hand saw sounds'],
}


def normalize_category_name(category: str) -> str:
    """Normalize category name."""
    # Remove numbers and dashes, capitalize properly
    category = category.strip()
    # Handle format like "1-100032-A-0" -> extract category from filename
    if '-' in category and category[0].isdigit():
        # This is a filename, not a category
        return category
    
    # Replace underscores with spaces and title case
    category = category.replace('_', ' ')
    # Capitalize first letter of each word
    words = category.split()
    category = ' '.join(word.capitalize() for word in words)
    return category


def get_caption_for_category(category: str, index: Optional[int] = None) -> str:
    """Get caption for a category."""
    normalized = normalize_category_name(category)
    
    # Check if we have captions for this category
    if normalized in CATEGORY_CAPTIONS:
        captions = CATEGORY_CAPTIONS[normalized]
        if index is not None and 0 <= index < len(captions):
            return captions[index]
        # Return first caption by default
        return captions[0]
    
    # Fallback
    category_lower = normalized.lower()
    return f"{category_lower} sounds"


def parse_esc50_filename(filename: str) -> Dict[str, str]:
    """Parse ESC-50 filename.
    
    Format: {fold}-{fname}-{take}-{target}.wav
    """
    basename = filename.replace('.wav', '')
    parts = basename.split('-')
    
    if len(parts) >= 4:
        return {
            'fold': parts[0],
            'fname': parts[1],
            'take': parts[2],
            'target': parts[3],
        }
    
    return {
        'fold': '',
        'fname': basename,
        'take': '',
        'target': '',
    }


def get_category_from_esc50_csv(target_id: str, esc50_csv_path: str) -> Optional[str]:
    """Get category from ESC-50 CSV."""
    import csv
    
    try:
        with open(esc50_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('target', '').strip() == str(target_id):
                    category = row.get('category', '').strip()
                    if category:
                        return category
    except Exception as e:
        print(f"Error reading ESC-50 CSV: {e}")
    
    return None

