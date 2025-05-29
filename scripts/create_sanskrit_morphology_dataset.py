import os
import random
from typing import List
import itertools

import vidyut
from vidyut.prakriya import Data, Dhatu, Lakara, Prayoga, Purusha, Vacana
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from src.zeroband.inference.genesys.sanskrit_morphology import VerbMorphologySpecification

# --- Constants ---
RANDOM_SEED = 108
DATA_DIR = "vidyut-data"
PRAKRIYA_DIR = os.path.join(DATA_DIR, "prakriya")
HF_REPO_NAME = "saahily/sanskrit-morphology"

# --- Data Utilities ---
def load_dhatupatha() -> List[Dhatu]:
    dhatupatha_path = os.path.join(PRAKRIYA_DIR, "dhatupatha.tsv")
    if not os.path.exists(dhatupatha_path):
        vidyut.download_data(DATA_DIR)

    return [d.dhatu for d in Data(PRAKRIYA_DIR).load_dhatu_entries()]

def generate_examples(num_examples) -> List[VerbMorphologySpecification]:
    # NB: this will run indefinitely if num_examples is greater than the number of unique combinations
    random.seed(RANDOM_SEED)
    dhatus = load_dhatupatha()
    print(f"✅ Loaded {len(dhatus)} dhatus")

    def _generate_unique_specs():
        """Generator that yields unique morphological specifications"""
        seen = set()
        for _ in itertools.count():
            spec = VerbMorphologySpecification(
                dhatu=(dhatu := random.choice(dhatus)).aupadeshika,
                gana=dhatu.gana,
                lakara=random.choice(Lakara.choices()),
                prayoga=random.choice(Prayoga.choices()),
                purusha=random.choice(Purusha.choices()),
                vacana=random.choice(Vacana.choices())
            )
            
            spec_key = tuple(sorted(spec.to_dict().items()))
            if spec_key not in seen:
                seen.add(spec_key)
                yield spec

    examples = list(itertools.islice(_generate_unique_specs(), num_examples))
    print(f"✅ Generated {len(examples)} unique examples")
    return examples

# --- Prompt Builder ---
def create_morphology_prompt(spec: VerbMorphologySpecification) -> str:
    return (
        "Generate the correct Sanskrit surface form for the following morphological specification:\n\n"
        f"Dhātu: {spec.dhatu}\n"
        f"Gaṇa: {spec.gana}\n"
        f"Lakāra: {spec.lakara}\n"
        f"Prayoga: {spec.prayoga}\n"
        f"Puruṣa: {spec.purusha}\n"
        f"Vacana: {spec.vacana}\n\n"
        "Please provide the correct surface form following Pāṇinian grammar rules, and in the format: [[surface_form]]"
    )

# --- Dataset Builder ---
def create_datasets(size: int = 75000, test_size: float = 0.1) -> DatasetDict:
    print(f"\n🔄 Generating {size} Sanskrit morphology examples...")
    examples = generate_examples(size)

    dataset_rows = []
    for spec in examples:
        row = {
            "prompt": create_morphology_prompt(spec),
            "task_type": "sanskrit_morphology",
            "verification_info": spec.to_dict(),
        }
        dataset_rows.append(row)

    dataset = Dataset.from_list(dataset_rows)
    print(f"📊 Created dataset with {len(dataset)} examples")

    datasets = dataset.train_test_split(test_size=test_size, seed=RANDOM_SEED)
    print(f"✂️ Split into train ({len(datasets['train'])}) and test ({len(datasets['test'])}) sets")

    return datasets

# --- Upload Utility ---
def push_datasets_to_hub(datasets: DatasetDict):
    print(f"\n🚀 Uploading dataset to Hugging Face: {HF_REPO_NAME}")
    datasets.push_to_hub(
        HF_REPO_NAME,
        commit_message="Add Sanskrit morphology dataset for tinantas"
    )
    print(f"✅ Upload complete: https://huggingface.co/datasets/{HF_REPO_NAME}")

# --- Main ---
if __name__ == "__main__":
    datasets = create_datasets()
    push_datasets_to_hub(datasets)
