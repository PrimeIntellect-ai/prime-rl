"""Sanskrit dataset loader for prime-rl inference pipeline."""

import random
from typing import Dict, List, Optional, Iterator
from datasets import load_dataset, Dataset
import json
from pathlib import Path
from zeroband.utils.logger import get_logger

logger = get_logger(__name__)


class SanskritQuoteDataset:
    """Dataset loader for Sanskrit quote identification task."""
    
    def __init__(
        self,
        dataset_name: str = "paws/sanskrit-verses-gretil",
        split: str = "train",
        cache_dir: Optional[str] = None,
        local_path: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize Sanskrit quote dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            cache_dir: Cache directory for downloaded dataset
            local_path: Local path to dataset (alternative to HF)
            seed: Random seed for sampling
        """
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        random.seed(seed)
        
        # Load dataset
        if local_path:
            logger.info(f"Loading dataset from local path: {local_path}")
            self._load_local_dataset(local_path)
        else:
            logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            
        logger.info(f"Loaded {len(self.dataset)} Sanskrit quotes")
        
        # Analyze dataset
        self._analyze_dataset()
        
    def _load_local_dataset(self, path: str) -> None:
        """Load dataset from local JSON/JSONL file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        self.dataset = Dataset.from_list(data)
        
    def _analyze_dataset(self) -> None:
        """Analyze dataset statistics."""
        self.genre_counts = {}
        self.author_counts = {}
        self.text_counts = {}
        
        for example in self.dataset:
            # Handle both nested ground_truth format and direct fields
            if 'ground_truth' in example:
                genre = example.get('ground_truth', {}).get('genre', 'unknown')
                author = example.get('ground_truth', {}).get('author', 'unknown')
                text = example.get('ground_truth', {}).get('text', 'unknown')
            else:
                # Direct fields as in your dataset
                genre = example.get('genre', 'unknown')
                author = example.get('author', 'unknown')
                text = example.get('text', 'unknown')
            
            self.genre_counts[genre] = self.genre_counts.get(genre, 0) + 1
            self.author_counts[author] = self.author_counts.get(author, 0) + 1
            self.text_counts[text] = self.text_counts.get(text, 0) + 1
            
        logger.info(f"Dataset statistics:")
        logger.info(f"  Genres: {len(self.genre_counts)}")
        logger.info(f"  Authors: {len(self.author_counts)}")
        logger.info(f"  Texts: {len(self.text_counts)}")
        
        # Show top genres
        top_genres = sorted(self.genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top genres:")
        for genre, count in top_genres:
            logger.info(f"    {genre}: {count}")
            
    def get_random_batch(self, batch_size: int) -> List[Dict]:
        """Get a random batch of examples.
        
        Args:
            batch_size: Number of examples to return
            
        Returns:
            List of examples
        """
        indices = random.sample(range(len(self.dataset)), min(batch_size, len(self.dataset)))
        return [self.dataset[i] for i in indices]
        
    def get_balanced_batch(self, batch_size: int) -> List[Dict]:
        """Get a batch with balanced genre distribution.
        
        Args:
            batch_size: Number of examples to return
            
        Returns:
            List of examples with balanced genres
        """
        # Group examples by genre
        examples_by_genre = {}
        for i, example in enumerate(self.dataset):
            genre = example.get('ground_truth', {}).get('genre', 'unknown')
            if genre not in examples_by_genre:
                examples_by_genre[genre] = []
            examples_by_genre[genre].append(i)
            
        # Sample from each genre
        batch = []
        genres = list(examples_by_genre.keys())
        examples_per_genre = batch_size // len(genres)
        extra = batch_size % len(genres)
        
        for i, genre in enumerate(genres):
            n_from_genre = examples_per_genre + (1 if i < extra else 0)
            genre_indices = examples_by_genre[genre]
            
            if len(genre_indices) >= n_from_genre:
                sampled_indices = random.sample(genre_indices, n_from_genre)
            else:
                # Sample with replacement if needed
                sampled_indices = random.choices(genre_indices, k=n_from_genre)
                
            batch.extend([self.dataset[idx] for idx in sampled_indices])
            
        # Shuffle the batch
        random.shuffle(batch)
        return batch[:batch_size]
        
    def iterate_batches(self, batch_size: int, balanced: bool = False) -> Iterator[List[Dict]]:
        """Iterate over batches of examples.
        
        Args:
            batch_size: Batch size
            balanced: Whether to balance genres in each batch
            
        Yields:
            Batches of examples
        """
        while True:
            if balanced:
                yield self.get_balanced_batch(batch_size)
            else:
                yield self.get_random_batch(batch_size)
                
    def format_prompt(self, example: Dict) -> str:
        """Format example into a prompt for the model.
        
        Args:
            example: Dataset example
            
        Returns:
            Formatted prompt string
        """
        # Use the prompt from the dataset or create one
        if 'prompt' in example:
            return example['prompt']
            
        quote = example.get('quote_text', '')
        prompt = f"""Identify the source of this Sanskrit quote:

"{quote}"

Please provide:
- Genre (e.g., veda, epic, purana, kavya, shastra)
- Author
- Text/Work name
- Chapter/Canto number (if applicable)
- Verse number

Answer in this format:
Genre: [genre], Author: [author], Text: [text], Chapter: [chapter], Verse: [verse]"""
        
        return prompt
        
    def get_ground_truth(self, example: Dict) -> Dict[str, str]:
        """Extract ground truth from example.
        
        Args:
            example: Dataset example
            
        Returns:
            Ground truth dictionary
        """
        if 'ground_truth' in example:
            return example.get('ground_truth', {})
        else:
            # Build ground truth from direct fields
            return {
                'genre': example.get('genre', ''),
                'author': example.get('author', ''),
                'text': example.get('text', ''),
                'chapter': example.get('chapter', ''),
                'verse': example.get('verse', '')
            }


# Convenience function for creating dataset
def create_sanskrit_dataset(
    dataset_name: str = "paws/sanskrit-gretil-quotes",
    split: str = "train",
    cache_dir: Optional[str] = None,
    local_path: Optional[str] = None,
    seed: int = 42
) -> SanskritQuoteDataset:
    """Create Sanskrit quote dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        cache_dir: Cache directory
        local_path: Local dataset path
        seed: Random seed
        
    Returns:
        SanskritQuoteDataset instance
    """
    return SanskritQuoteDataset(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        local_path=local_path,
        seed=seed
    )