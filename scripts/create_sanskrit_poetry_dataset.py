# scripts/create_sanskrit_poetry_dataset.py
from datasets import Dataset
import json

def create_sanskrit_poetry_dataset():
    """Create dataset with various Sanskrit meters."""
    data = []
    
    # Popular Sanskrit meters that Chandas can identify
    meters_info = {
        "Śloka": "the most common Sanskrit meter with 8 syllables per quarter",
        "Anuṣṭubh": "the epic meter used in Mahabharata and Ramayana",
        "Vasantatilakā": "the spring meter with 14 syllables per line",
        "Mandākrāntā": "the slow-stepping meter with 17 syllables per line",
        "Śikhariṇī": "the peaked meter with 17 syllables per line",
        "Praharṣiṇī": "the delightful meter with 13 syllables per line",
        "Śārdūlavikrīḍita": "the tiger's play meter with 19 syllables per line"
    }
    
    # Topics for poetry
    topics = {
        "rāma": "Lord Rama, the ideal king and avatar of Vishnu",
        "kṛṣṇa": "Lord Krishna, the divine cowherd and teacher of Bhagavad Gita",
        "śiva": "Lord Shiva, the destroyer and transformer",
        "devī": "The Divine Mother in her various forms",
        "gaṅgā": "The sacred river Ganga",
        "himālaya": "The Himalaya mountains, abode of gods",
        "dharma": "righteous duty and cosmic order",
        "mokṣa": "liberation from the cycle of rebirth",
        "prakṛti": "nature and the manifest world",
        "guru": "the spiritual teacher"
    }
    
    # Create diverse prompts
    for meter, meter_desc in meters_info.items():
        for topic, topic_desc in topics.items():
            # Standard prompt
            prompt = f"Compose a Sanskrit poem about {topic} ({topic_desc}) in {meter} meter ({meter_desc}). Ensure strict adherence to the metrical pattern."
            
            verification_info = {
                "meter_type": meter,
                "topic": topic,
                "meter_description": meter_desc,
                "topic_description": topic_desc
            }
            
            data.append({
                "prompt": prompt,
                "task_type": "sanskrit_poetry",
                "verification_info": json.dumps(verification_info)
            })
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Push to HuggingFace
    dataset.push_to_hub("badhanr/sanskrit-poetry-rl")
    print(f"Created dataset with {len(data)} examples")
    
if __name__ == "__main__":
    create_sanskrit_poetry_dataset()