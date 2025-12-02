# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
# ]
# ///
"""
Transform and combine multiple datasets into PrimeIntellect/Intellect-4-data:
1. POLARIS-Project/Polaris-Dataset-53K -> subset 'math'
   - Rename 'problem' column to 'question'
   - Remove 'difficulty' column
2. zai-org/DeepDive (splits: qa_rl, qa_sft) -> subset 'agentic_search'
   - Remove 'id' and 'conversations' columns
   - Combine qa_rl and qa_sft into single 'train' split
3. inclusionAI/ASearcher-train-data (split: ASearcherLRM35k) -> subset 'agentic_search'
   - Keep only 'question' and 'answer' columns (remove 'source', 'id', 'idx')
   - Combined with DeepDive data
   - Add 'tools' column with JSON string for all samples
4. SAA-Lab/LitBench-Train -> subset 'general_judge'
   - Rename: prompt -> question, chosen_story -> chosen, rejected_story -> rejected
   - Remove: chosen_timestamp, rejected_timestamp, chosen_upvotes, rejected_upvotes
5. SAA-Lab/LitBench-Train -> subset 'general'
   - Rename: prompt -> question
   - Keep only 'question' column
- Push privately to Hugging Face Hub
"""
import argparse
import json
from datasets import DatasetDict, load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download


def process_polaris_dataset():
    """Process POLARIS dataset and return DatasetDict for 'math' subset."""
    print("=" * 60)
    print("Processing POLARIS-Project/Polaris-Dataset-53K")
    print("=" * 60)
    
    dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K")
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Rename 'problem' to 'question'
            if "problem" in split_dataset.column_names:
                transformed_split = split_dataset.rename_column("problem", "question")
            else:
                print(f"Warning: 'problem' column not found in split {split_name}")
                transformed_split = split_dataset
            
            # Remove 'difficulty' column if it exists
            if "difficulty" in transformed_split.column_names:
                transformed_split = transformed_split.remove_columns("difficulty")
            
            transformed_dict[split_name] = transformed_split
    else:
        # Single Dataset
        print("Processing single dataset")
        if "problem" in dataset.column_names:
            transformed_dataset = dataset.rename_column("problem", "question")
        else:
            print("Warning: 'problem' column not found")
            transformed_dataset = dataset
        
        # Remove 'difficulty' column if it exists
        if "difficulty" in transformed_dataset.column_names:
            transformed_dataset = transformed_dataset.remove_columns("difficulty")
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: transformed_dataset}
    
    return DatasetDict(transformed_dict)


def process_deepdive_dataset():
    """Process DeepDive dataset and return list of processed datasets."""
    print("=" * 60)
    print("Processing zai-org/DeepDive")
    print("=" * 60)
    
    # Load specific splits
    splits_to_load = ["qa_rl", "qa_sft"]
    processed_datasets = []
    
    for split_name in splits_to_load:
        print(f"Loading split: {split_name}")
        split_dataset = load_dataset("zai-org/DeepDive", split=split_name)
        
        # Remove 'id' and 'conversations' columns if they exist
        columns_to_remove = []
        if "id" in split_dataset.column_names:
            columns_to_remove.append("id")
        if "conversations" in split_dataset.column_names:
            columns_to_remove.append("conversations")
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            split_dataset = split_dataset.remove_columns(columns_to_remove)
        
        processed_datasets.append(split_dataset)
    
    return processed_datasets


def process_asearcher_dataset():
    """Process ASearcher dataset and return list of processed datasets."""
    print("=" * 60)
    print("Processing inclusionAI/ASearcher-train-data")
    print("=" * 60)
    
    print(f"Loading split: ASearcherLRM35k")
    # Load using data_files to bypass schema validation issues
    # The dataset has inconsistent schemas across files, so we load the specific file
    from huggingface_hub import hf_hub_download
    import json
    
    # Download the specific JSONL file
    jsonl_path = hf_hub_download(
        repo_id="inclusionAI/ASearcher-train-data",
        filename="ASearcher-LRM-35k.jsonl",
        repo_type="dataset"
    )
    
    # Load from the JSONL file directly
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    
    # Keep only 'question' and 'answer' columns
    columns_to_keep = ["question", "answer"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    
    if columns_to_remove:
        print(f"Removing columns: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)
    else:
        print(f"Kept columns: {columns_to_keep}")
    
    return [dataset]


def get_tools_json_string():
    """Return the tools JSON as a string."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Google, getting up to 10 results and search metadata",
                "parameters": {
                    "properties": {
                        "query": {
                            "title": "Query",
                            "type": "string"
                        },
                        "num_results": {
                            "default": 10,
                            "title": "Num Results"
                        }
                    },
                    "required": [
                        "query",
                        "num_results"
                    ],
                    "title": "search_args",
                    "type": "object",
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "open",
                "description": "Get the content of webpages given a list of URLs",
                "parameters": {
                    "properties": {
                        "urls": {
                            "items": {
                                "type": "string"
                            },
                            "title": "Urls",
                            "type": "array"
                        }
                    },
                    "required": [
                        "urls"
                    ],
                    "title": "open_args",
                    "type": "object",
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Get the contents of webpages from the previous search results\nCan open multiple results at once",
                "parameters": {
                    "properties": {
                        "result_indices": {
                            "items": {
                                "type": "integer"
                            },
                            "title": "Result Indices",
                            "type": "array"
                        }
                    },
                    "required": [
                        "result_indices"
                    ],
                    "title": "click_args",
                    "type": "object",
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Provide the final answer to the task. Stops execution.",
                "parameters": {
                    "properties": {
                        "final_answer": {
                            "title": "Final Answer",
                            "type": "string"
                        }
                    },
                    "required": [
                        "final_answer"
                    ],
                    "title": "finish_args",
                    "type": "object",
                    "additionalProperties": False
                }
            }
        }
    ]
    return json.dumps(tools)


def process_agentic_search_datasets():
    """Process and combine all agentic_search datasets."""
    all_datasets = []
    
    # Process DeepDive datasets
    deepdive_datasets = process_deepdive_dataset()
    all_datasets.extend(deepdive_datasets)
    
    # Process ASearcher dataset
    asearcher_datasets = process_asearcher_dataset()
    all_datasets.extend(asearcher_datasets)
    
    # Combine all datasets into one
    print(f"\nCombining all agentic_search datasets into single 'train' split")
    print(f"Total datasets to combine: {len(all_datasets)}")
    combined_dataset = concatenate_datasets(all_datasets)
    
    # Add 'tools' column as JSON string to all samples
    tools_json_string = get_tools_json_string()
    num_rows = len(combined_dataset)
    print(f"Adding 'tools' column to {num_rows} samples")
    combined_dataset = combined_dataset.add_column(
        "tools",
        [tools_json_string] * num_rows
    )
    
    return DatasetDict({"train": combined_dataset})


def transform_question_to_context(example):
    """Transform question string to context list format."""
    if "question" in example:
        question_value = example["question"]
        example["context"] = [{"content": question_value, "role": "user"}]
        del example["question"]
    return example


def process_litbench_general_judge():
    """Process LitBench dataset for 'general_judge' subset."""
    print("=" * 60)
    print("Processing SAA-Lab/LitBench-Train for 'general_judge' subset")
    print("=" * 60)
    
    dataset = load_dataset("SAA-Lab/LitBench-Train")
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Rename columns
            if "prompt" in split_dataset.column_names:
                split_dataset = split_dataset.rename_column("prompt", "question")
            if "chosen_story" in split_dataset.column_names:
                split_dataset = split_dataset.rename_column("chosen_story", "chosen")
            if "rejected_story" in split_dataset.column_names:
                split_dataset = split_dataset.rename_column("rejected_story", "rejected")
            
            # Remove unwanted columns
            columns_to_remove = [
                "chosen_timestamp",
                "rejected_timestamp",
                "chosen_upvotes",
                "rejected_upvotes"
            ]
            columns_to_remove = [col for col in columns_to_remove if col in split_dataset.column_names]
            
            if columns_to_remove:
                print(f"Removing columns: {columns_to_remove}")
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            
            # Transform question to context format
            print("Transforming 'question' to 'context' with list format")
            split_dataset = split_dataset.map(transform_question_to_context)
            
            transformed_dict[split_name] = split_dataset
    else:
        # Single Dataset
        print("Processing single dataset")
        if "prompt" in dataset.column_names:
            dataset = dataset.rename_column("prompt", "question")
        if "chosen_story" in dataset.column_names:
            dataset = dataset.rename_column("chosen_story", "chosen")
        if "rejected_story" in dataset.column_names:
            dataset = dataset.rename_column("rejected_story", "rejected")
        
        # Remove unwanted columns
        columns_to_remove = [
            "chosen_timestamp",
            "rejected_timestamp",
            "chosen_upvotes",
            "rejected_upvotes"
        ]
        columns_to_remove = [col for col in columns_to_remove if col in dataset.column_names]
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Transform question to context format
        print("Transforming 'question' to 'context' with list format")
        dataset = dataset.map(transform_question_to_context)
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: dataset}
    
    return DatasetDict(transformed_dict)


def process_litbench_general():
    """Process LitBench dataset for 'general' subset."""
    print("=" * 60)
    print("Processing SAA-Lab/LitBench-Train for 'general' subset")
    print("=" * 60)
    
    dataset = load_dataset("SAA-Lab/LitBench-Train")
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Rename prompt to question
            if "prompt" in split_dataset.column_names:
                split_dataset = split_dataset.rename_column("prompt", "question")
            
            # Keep only 'question' column
            columns_to_remove = [col for col in split_dataset.column_names if col != "question"]
            
            if columns_to_remove:
                print(f"Removing columns: {columns_to_remove}")
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            
            # Transform question to context format
            print("Transforming 'question' to 'context' with list format")
            split_dataset = split_dataset.map(transform_question_to_context)
            
            transformed_dict[split_name] = split_dataset
    else:
        # Single Dataset
        print("Processing single dataset")
        if "prompt" in dataset.column_names:
            dataset = dataset.rename_column("prompt", "question")
        
        # Keep only 'question' column
        columns_to_remove = [col for col in dataset.column_names if col != "question"]
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Transform question to context format
        print("Transforming 'question' to 'context' with list format")
        dataset = dataset.map(transform_question_to_context)
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: dataset}
    
    return DatasetDict(transformed_dict)


def process_helpsteer3_general():
    """Process HelpSteer3 dataset for 'general' subset - just context."""
    print("=" * 60)
    print("Processing nvidia/HelpSteer3 (preference subset) for 'general' subset")
    print("=" * 60)
    
    dataset = load_dataset("nvidia/HelpSteer3", "preference")
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Keep only 'context' column
            columns_to_remove = [col for col in split_dataset.column_names if col != "context"]
            
            if columns_to_remove:
                print(f"Removing columns: {columns_to_remove}")
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            
            transformed_dict[split_name] = split_dataset
    else:
        # Single Dataset
        print("Processing single dataset")
        # Keep only 'context' column
        columns_to_remove = [col for col in dataset.column_names if col != "context"]
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: dataset}
    
    return DatasetDict(transformed_dict)


def process_helpsteer3_general_judge():
    """Process HelpSteer3 dataset for 'general_judge' subset with chosen/rejected logic."""
    print("=" * 60)
    print("Processing nvidia/HelpSteer3 (preference subset) for 'general_judge' subset")
    print("=" * 60)
    
    dataset = load_dataset("nvidia/HelpSteer3", "preference")
    
    def transform_preference(example):
        """Transform preference data to chosen/rejected format."""
        overall_pref = example.get("overall_preference", 0)
        
        # Skip if preference is 0
        if overall_pref == 0:
            return None
        
        # Set chosen and rejected based on preference
        if overall_pref < 0:
            example["chosen"] = example["response1"]
            example["rejected"] = example["response2"]
        else:  # overall_pref > 0
            example["chosen"] = example["response2"]
            example["rejected"] = example["response1"]
        
        # Keep context column (it's already in the dataset)
        # Remove unwanted columns
        columns_to_remove = [
            "domain",
            "language",
            "response1",
            "response2",
            "overall_preference",
            "individual_preference"
        ]
        for col in columns_to_remove:
            if col in example:
                del example[col]
        
        return example
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Filter and transform
            print("Filtering samples with overall_preference != 0 and transforming...")
            # Transform first (needs response1, response2, overall_preference)
            split_dataset = split_dataset.map(
                transform_preference,
                desc="Transforming preferences"
            )
            # Remove None values (samples with preference = 0)
            split_dataset = split_dataset.filter(lambda x: x is not None)
            
            # Now remove unwanted columns (after transform has used them)
            columns_to_remove_after = [
                "domain",
                "language",
                "response1",
                "response2",
                "overall_preference",
                "individual_preference"
            ]
            columns_to_remove_after = [col for col in columns_to_remove_after if col in split_dataset.column_names]
            if columns_to_remove_after:
                print(f"Removing columns: {columns_to_remove_after}")
                split_dataset = split_dataset.remove_columns(columns_to_remove_after)
            
            transformed_dict[split_name] = split_dataset
    else:
        # Single Dataset
        print("Processing single dataset")
        # Transform first (needs response1, response2, overall_preference)
        dataset = dataset.map(
            transform_preference,
            desc="Transforming preferences"
        )
        # Remove None values (samples with preference = 0)
        dataset = dataset.filter(lambda x: x is not None)
        
        # Now remove unwanted columns (after transform has used them)
        columns_to_remove_after = [
            "domain",
            "language",
            "response1",
            "response2",
            "overall_preference",
            "individual_preference"
        ]
        columns_to_remove_after = [col for col in columns_to_remove_after if col in dataset.column_names]
        if columns_to_remove_after:
            print(f"Removing columns: {columns_to_remove_after}")
            dataset = dataset.remove_columns(columns_to_remove_after)
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: dataset}
    
    return DatasetDict(transformed_dict)


def normalize_context_format(example):
    """Normalize context format to remove tool_calls if present."""
    if "prompt" in example:
        prompt_value = example["prompt"]
        # If prompt is a list, normalize it to only have content and role
        if isinstance(prompt_value, list):
            normalized_context = []
            for item in prompt_value:
                if isinstance(item, dict):
                    normalized_item = {
                        "content": item.get("content", ""),
                        "role": item.get("role", "user")
                    }
                    normalized_context.append(normalized_item)
                else:
                    # If it's not a dict, wrap it
                    normalized_context.append({"content": str(item), "role": "user"})
            example["context"] = normalized_context
        else:
            # If prompt is a string, convert to list format
            example["context"] = [{"content": str(prompt_value), "role": "user"}]
        del example["prompt"]
    return example


def process_deepwriting_general():
    """Process deepwriting dataset for 'general' subset - extract prompt as context."""
    print("=" * 60)
    print("Processing PrimeIntellect/INTELLECT-3-SFT-Stage-2 (deepwriting subset) for 'general' subset")
    print("=" * 60)
    
    dataset = load_dataset("PrimeIntellect/INTELLECT-3-SFT-Stage-2", "deepwriting")
    
    # Handle both DatasetDict and Dataset types
    if isinstance(dataset, DatasetDict):
        transformed_dict = {}
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name}")
            # Normalize context format and rename prompt to context
            print("Normalizing context format...")
            split_dataset = split_dataset.map(normalize_context_format)
            
            # Keep only 'context' column
            columns_to_remove = [col for col in split_dataset.column_names if col != "context"]
            
            if columns_to_remove:
                print(f"Removing columns: {columns_to_remove}")
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            
            transformed_dict[split_name] = split_dataset
    else:
        # Single Dataset
        print("Processing single dataset")
        # Normalize context format and rename prompt to context
        print("Normalizing context format...")
        dataset = dataset.map(normalize_context_format)
        
        # Keep only 'context' column
        columns_to_remove = [col for col in dataset.column_names if col != "context"]
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Determine split name
        split_name = "train"  # default
        if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
            splits = dataset.info.splits
            if splits:
                split_name = list(splits.keys())[0]
        
        transformed_dict = {split_name: dataset}
    
    return DatasetDict(transformed_dict)


def remove_validation_splits(dataset_dict):
    """Remove validation splits from a DatasetDict, keeping only train splits."""
    filtered_dict = {}
    for split_name, split_dataset in dataset_dict.items():
        if split_name != "validation":
            filtered_dict[split_name] = split_dataset
        else:
            print(f"Removing validation split from dataset")
    return DatasetDict(filtered_dict)


def shuffle_dataset_dict(dataset_dict, seed=42):
    """Shuffle all splits in a DatasetDict."""
    shuffled_dict = {}
    for split_name, split_dataset in dataset_dict.items():
        print(f"Shuffling {split_name} split ({len(split_dataset)} samples)...")
        shuffled_dict[split_name] = split_dataset.shuffle(seed=seed)
    return DatasetDict(shuffled_dict)


def main():
    parser = argparse.ArgumentParser(description="Transform and upload datasets")
    parser.add_argument(
        "--target-dataset",
        default="PrimeIntellect/Intellect-4-data",
        help="Target dataset name",
    )
    args = parser.parse_args()

    # Process POLARIS dataset for 'math' subset
    math_dataset_dict = process_polaris_dataset()
    
    # Process and combine all agentic_search datasets
    agentic_search_dataset_dict = process_agentic_search_datasets()
    
    # Process LitBench dataset for 'general_judge' subset
    litbench_general_judge = process_litbench_general_judge()
    
    # Process HelpSteer3 dataset for 'general_judge' subset
    helpsteer3_general_judge = process_helpsteer3_general_judge()
    
    # Combine LitBench and HelpSteer3 for general_judge
    print("\nCombining LitBench and HelpSteer3 for 'general_judge' subset")
    general_judge_datasets = []
    for split_name in litbench_general_judge.keys():
        if split_name in helpsteer3_general_judge:
            combined = concatenate_datasets([litbench_general_judge[split_name], helpsteer3_general_judge[split_name]])
            general_judge_datasets.append((split_name, combined))
        else:
            general_judge_datasets.append((split_name, litbench_general_judge[split_name]))
    
    # Add any splits from HelpSteer3 that aren't in LitBench
    for split_name in helpsteer3_general_judge.keys():
        if split_name not in litbench_general_judge:
            general_judge_datasets.append((split_name, helpsteer3_general_judge[split_name]))
    
    general_judge_dataset_dict = DatasetDict(dict(general_judge_datasets))
    
    # Process LitBench dataset for 'general' subset
    litbench_general = process_litbench_general()
    
    # Process HelpSteer3 dataset for 'general' subset
    helpsteer3_general = process_helpsteer3_general()
    
    # Process deepwriting dataset for 'general' subset
    deepwriting_general = process_deepwriting_general()
    
    # Combine LitBench, HelpSteer3, and deepwriting for general
    print("\nCombining LitBench, HelpSteer3, and deepwriting for 'general' subset")
    general_datasets = []
    
    # Get all unique split names
    all_splits = set(litbench_general.keys()) | set(helpsteer3_general.keys()) | set(deepwriting_general.keys())
    
    for split_name in all_splits:
        datasets_to_combine = []
        if split_name in litbench_general:
            datasets_to_combine.append(litbench_general[split_name])
        if split_name in helpsteer3_general:
            datasets_to_combine.append(helpsteer3_general[split_name])
        if split_name in deepwriting_general:
            datasets_to_combine.append(deepwriting_general[split_name])
        
        if len(datasets_to_combine) > 1:
            combined = concatenate_datasets(datasets_to_combine)
            general_datasets.append((split_name, combined))
        elif len(datasets_to_combine) == 1:
            general_datasets.append((split_name, datasets_to_combine[0]))
    
    general_dataset_dict = DatasetDict(dict(general_datasets))
    
    # Remove validation splits from all datasets
    print("\nRemoving validation splits from all datasets")
    math_dataset_dict = remove_validation_splits(math_dataset_dict)
    agentic_search_dataset_dict = remove_validation_splits(agentic_search_dataset_dict)
    general_judge_dataset_dict = remove_validation_splits(general_judge_dataset_dict)
    general_dataset_dict = remove_validation_splits(general_dataset_dict)
    
    # Shuffle all datasets
    print("\n" + "=" * 60)
    print("Shuffling all subsets")
    print("=" * 60)
    math_dataset_dict = shuffle_dataset_dict(math_dataset_dict)
    agentic_search_dataset_dict = shuffle_dataset_dict(agentic_search_dataset_dict)
    general_judge_dataset_dict = shuffle_dataset_dict(general_judge_dataset_dict)
    general_dataset_dict = shuffle_dataset_dict(general_dataset_dict)
    
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print("\nMath subset:")
    print(math_dataset_dict)
    print("\nAgentic Search subset:")
    print(agentic_search_dataset_dict)
    print("\nGeneral Judge subset:")
    print(general_judge_dataset_dict)
    print("\nGeneral subset:")
    print(general_dataset_dict)
    
    print(f"\n{'=' * 60}")
    print(f"Pushing to Hugging Face Hub: {args.target_dataset}")
    print("This will be pushed as a PRIVATE dataset.")
    print("=" * 60)
    
    # Push each subset as a separate configuration
    print("\nPushing 'math' subset...")
    math_dataset_dict.push_to_hub(
        args.target_dataset,
        config_name="math",
        private=True,
    )
    print("✓ Successfully pushed 'math' subset")
    
    print("\nPushing 'agentic_search' subset...")
    agentic_search_dataset_dict.push_to_hub(
        args.target_dataset,
        config_name="agentic_search",
        private=True,
    )
    print("✓ Successfully pushed 'agentic_search' subset")
    
    print("\nPushing 'general_judge' subset...")
    general_judge_dataset_dict.push_to_hub(
        args.target_dataset,
        config_name="general_judge",
        private=True,
    )
    print("✓ Successfully pushed 'general_judge' subset")
    
    print("\nPushing 'general' subset...")
    general_dataset_dict.push_to_hub(
        args.target_dataset,
        config_name="general",
        private=True,
    )
    print("✓ Successfully pushed 'general' subset")
    
    print(f"\n{'=' * 60}")
    print(f"Successfully pushed all datasets to {args.target_dataset} (private)")
    print("Available subsets: 'math', 'agentic_search', 'general_judge', 'general'")
    print("=" * 60)


if __name__ == "__main__":
    main()
