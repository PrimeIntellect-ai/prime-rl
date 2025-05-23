# scripts/create_sanskrit_poetry_dataset.py
from datasets import Dataset
import json

def create_sanskrit_poetry_dataset():
    """Create dataset with various Sanskrit meters."""
    data = []
    
    # Popular Sanskrit meters that Chandas can identify
    meters_info = {
        # Vedic Meters (Classical Seven)
        "Gāyatrī": "the most sacred Vedic meter with 8 syllables per quarter (3×8=24 total)",
        "Uṣṇik": "the Vedic meter with 7 syllables per quarter (4×7=28 total)",
        "Anuṣṭubh": "the epic meter used in Mahabharata and Ramayana with 8 syllables per quarter",
        "Bṛhatī": "the great Vedic meter with 9 syllables per quarter (4×9=36 total)",
        "Paṅkti": "the five-fold Vedic meter with 8 syllables per quarter (5×8=40 total)",
        "Triṣṭubh": "the triple-step Vedic meter with 11 syllables per quarter, second most common in Rigveda",
        "Jagatī": "the cosmic Vedic meter with 12 syllables per quarter (4×12=48 total)",
        
        # Classical Śloka
        "Śloka": "the most common Sanskrit meter with 8 syllables per quarter, identical to Anuṣṭubh",
        
        # Popular 11-syllable Meters
        "Indravajrā": "the heroic meter with 11 syllables (ta-ta-ja-ga-ga pattern)",
        "Upendravajrā": "the noble meter with 11 syllables (ja-ta-ja-ga-ga pattern)",
        "Upajāti": "the mixed meter alternating Indravajrā and Upendravajrā",
        "Rathoddhatā": "the chariot-raising meter with 11 syllables per quarter",
        
        # 12-syllable Meters  
        "Vaṃśastha": "the bamboo-like meter with 12 syllables (ja-ta-ja-ra pattern)",
        "Toṭaka": "the rapid meter with 12 syllables, every 3rd syllable heavy",
        
        # 13-syllable Meters
        "Praharṣiṇī": "the delightful meter with 13 syllables per line",
        
        # 14-syllable Meters
        "Vasantatilakā": "the spring ornament meter with 14 syllables (ta-bha-ja-ja-ga-ga pattern)",
        
        # 15-syllable Meters
        "Mālinī": "the garlanded meter with 15 syllables per line (na-na-ma-ya-ya pattern)",
        
        # 17-syllable Meters
        "Mandākrāntā": "the slow-stepping meter with 17 syllables, favored by Kālidāsa",
        "Śikhariṇī": "the peaked meter with 17 syllables per line",
        
        # 19-syllable Meters
        "Śārdūlavikrīḍita": "the tiger's play meter with 19 syllables suggesting power and grace",
        
        # 21-syllable Meters
        "Sragdharā": "the garland-bearing meter with 21 syllables for elaborate compositions",
        
        # Additional Classical Meters
        "Bhujaṅgaprayāt": "the serpent's motion meter with flowing rhythm",
        "Svāgatā": "the welcome meter with 11 syllables per quarter",
        "Śālinī": "the modest meter with elegant cadence",
        "Pañcacāmara": "the five-whisk meter with ornate structure",
        
        # Extended Vedic Meters
        "Atijagati": "the beyond-cosmic meter with 13 syllables per quarter",
        "Śakkarī": "the sugar-sweet meter with 14 syllables per quarter", 
        "Atiśakarī": "the beyond-sweet meter with 15 syllables per quarter",
        "Aṣṭi": "the eight-fold meter with 16 syllables per quarter",
        "Atyaṣṭi": "the beyond-eight meter with 17 syllables per quarter",
        "Dhṛti": "the steadfast meter with 18 syllables per quarter",
        "Atidhṛti": "the beyond-steadfast meter with 19 syllables per quarter",
        "Kṛti": "the accomplished meter with 20 syllables per quarter",
        "Prakṛti": "the natural meter with 21 syllables per quarter",
        "Ākṛti": "the formed meter with 22 syllables per quarter",
        "Vikṛti": "the transformed meter with 23 syllables per quarter",
        "Saṅkṛti": "the composed meter with 24 syllables per quarter",
        
        # Quantitative Meters (Mātrāvṛtta)
        "Āryā": "the noble quantitative meter based on morae count, common in Prakrit",
        "Vaitālīya": "the demonic meter derived from measure metres",
        "Aparāntikā": "the western-border meter with sixteen mātrās per quarter",
        
        # Rare and Specialized Meters
        "Virāj": "the shining meter with 10 syllables per quarter",
        "Dvipadā Virāj": "the two-footed shining meter with 2×10 syllables",
        "Hariṇaplutā": "the deer-leap meter with 12 syllables",
        "Sundara": "the beautiful meter for aesthetic compositions",
        "Lalitā": "the graceful meter with delicate rhythm",
        "Citrā": "the variegated meter with complex patterns",
        "Mattebha": "the intoxicated elephant meter with heavy rhythm",
        "Campaka": "the champak flower meter with fragrant cadence",
        "Mallikā": "the jasmine meter with sweet rhythm",
        
        # Technical Categories
        "Samavṛtta": "meters where all four quarters have identical patterns", 
        "Ardhasamavṛtta": "meters where alternate quarters have similar patterns",
        "Viṣamavṛtta": "meters where all four quarters are different",
        "Varṇavṛtta": "syllabo-quantitative verse with fixed light-heavy patterns",
        "Akṣaravṛtta": "syllabic verse with syllable count freedom",
        "Mātrāvṛtta": "quantitative verse based on duration and morae",
        "Daṇḍaka": "extended meters exceeding 26 syllables per line"
    }
    
    # Topics for poetry
    topics = {
        # Original topics
        "rāma": "Lord Rama, the ideal king and avatar of Vishnu",
        "kṛṣṇa": "Lord Krishna, the divine cowherd and teacher of Bhagavad Gita",
        "śiva": "Lord Shiva, the destroyer and transformer",
        "devī": "The Divine Mother in her various forms",
        "gaṅgā": "The sacred river Ganga",
        "himālaya": "The Himalaya mountains, abode of gods",
        "dharma": "righteous duty and cosmic order",
        "mokṣa": "liberation from the cycle of rebirth",
        "prakṛti": "nature and the manifest world",
        "guru": "the spiritual teacher",

        # From Bhagavad Gita
        "arjuna": "the warrior prince and Krishna's disciple on the battlefield of Kurukshetra",
        "karma": "action and the law of cause and effect governing rebirth",
        "bhakti": "devotional worship and loving surrender to the divine",
        "yoga": "spiritual discipline and the path of union with the divine",
        "jñāna": "sacred knowledge and wisdom leading to liberation",
        "saṃsāra": "the cycle of birth, death, and rebirth",
        "guṇas": "the three fundamental qualities of nature: sattva, rajas, and tamas",
        "brahman": "the ultimate reality and absolute truth",
        "ātman": "the eternal self or soul within all beings",
        "saṃnyāsa": "renunciation and the path of spiritual detachment",
        "kurukṣetra": "the sacred battlefield where the cosmic dialogue took place",
        "pāṇḍavas": "the five righteous brothers including Arjuna",
        "kauravas": "the hundred brothers who opposed dharma in the great war",
        "niṣkāma karma": "selfless action performed without attachment to results",
        "sthitaprajña": "the sage of steady wisdom, unmoved by dualities",

        # From Shiva Purana  
        "pārvatī": "the Divine Mother as Shiva's consort and supreme shakti",
        "naṭarāja": "Shiva as the cosmic dancer performing the dance of creation and destruction",
        "liṅga": "the sacred symbol representing Shiva's infinite nature",
        "umā": "the gentle aspect of the goddess as daughter of Himalaya",
        "kailāsa": "Mount Kailash, the sacred abode of Shiva and Parvati",
        "rudra": "the fierce and benevolent form of Shiva as cosmic destroyer",
        "mahādeva": "the Great God, supreme epithet of Shiva",
        "bhasma": "sacred ash symbolizing the transient nature of physical existence",
        "triśūla": "Shiva's trident representing the three gunas and time periods",
        "ḍamaru": "Shiva's drum from which the sacred sounds of creation emerge",
        "gaṅgā-dhara": "Shiva as the bearer of Ganga in his matted locks",
        "ardhanārīśvara": "the half-male, half-female form showing unity of Shiva and Shakti",

        # From Markandeya Purana / Devi Mahatmya
        "durgā": "the warrior goddess who destroys evil and protects dharma",
        "caṇḍī": "the fierce form of the goddess who battles demons",
        "mahiṣāsura": "the buffalo demon slain by Goddess Durga",
        "śakti": "the divine feminine power and creative force of the universe",
        "saptaśatī": "the seven hundred verses glorifying the Divine Mother",
        "cāmuṇḍā": "the fierce goddess who emerges from Durga's forehead",
        "mahākālī": "the great Kali, the all-consuming aspect of time",
        "mahālakṣmī": "the great Lakshmi, the preserving aspect of the goddess",
        "mahāsarasvatī": "the great Saraswati, the creative aspect of the goddess",
        "ambikā": "the Divine Mother addressed as the supreme protector",
        "caṇḍa-muṇḍa": "the demon generals defeated by the goddess",
        "śumbha-niśumbha": "the demon brothers conquered by Devi's supreme power",

        # From Ramayana
        "sītā": "the ideal wife and incarnation of Lakshmi, symbol of purity",
        "hanumān": "the devoted monkey warrior, symbol of selfless service",
        "rāvaṇa": "the ten-headed demon king of Lanka, learned but ego-driven",
        "lakṣmaṇa": "Rama's devoted younger brother who served throughout exile",
        "bharata": "Rama's righteous brother who ruled as regent with Rama's sandals",
        "ayodhyā": "the ideal kingdom ruled by Rama, symbol of righteous governance",
        "laṅkā": "the golden island kingdom of Ravana across the southern ocean",
        "vālmīki": "the sage-poet who composed the first Sanskrit epic",
        "sugrīva": "the monkey king who allied with Rama in the search for Sita",
        "vibhīṣaṇa": "Ravana's righteous brother who chose dharma over family loyalty",
        "jaṭāyu": "the noble eagle who sacrificed his life attempting to save Sita",
        "aśoka vanikā": "the grove of Ashoka trees where Sita was held captive",
        "agniparīkṣā": "the fire ordeal proving Sita's purity and chastity",
        "rāma-rājya": "the ideal golden age of Rama's righteous rule",
        "setubandha": "the great bridge built across the ocean to Lanka",

        # Additional classical Sanskrit concepts
        "tīrtha": "sacred pilgrimage places where the divine is especially accessible",
        "saṅgama": "the sacred confluence of rivers, especially Ganga and Yamuna",
        "vṛndāvana": "Krishna's childhood playground, symbol of divine love",
        "mathurā": "Krishna's birthplace, the sacred city of divine incarnation",
        "dvārakā": "Krishna's golden capital city in his later life",
        "kurma": "the divine turtle incarnation supporting Mount Mandara",
        "garuḍa": "the divine eagle, Vishnu's vehicle and symbol of devotion",
        "nārada": "the celestial sage and devotee, spreader of divine knowledge",
        "vyāsa": "the great sage who compiled the Vedas and authored the Mahabharata",
        "śravaṇa": "the devoted son who carried his blind parents on pilgrimage",
        "prahlāda": "the child devotee whose faith overcame his demon father's hatred"
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