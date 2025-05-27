"""Unit tests for Sanskrit data loader."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
from zeroband.inference.sanskrit_data_loader import (
    SanskritQuoteDataset,
    create_sanskrit_dataset
)


class TestSanskritQuoteDataset:
    """Test Sanskrit quote dataset functionality."""
    
    @pytest.fixture
    def mock_dataset_data(self):
        """Create mock dataset data."""
        return [
            {
                "id": "test1",
                "prompt": "Identify this Sanskrit quote...",
                "quote_text": "धर्मक्षेत्रे कुरुक्षेत्रे...",
                "quote_devanagari": "धर्मक्षेत्रे कुरुक्षेत्रे...",
                "ground_truth": {
                    "genre": "epic",
                    "author": "Vyasa",
                    "text": "Bhagavad Gita",
                    "chapter": "1",
                    "verse": "1"
                }
            },
            {
                "id": "test2",
                "prompt": "Identify this Sanskrit quote...",
                "quote_text": "कश्चित्कान्ताविरहगुरुणा...",
                "ground_truth": {
                    "genre": "kavya",
                    "author": "Kalidasa",
                    "text": "Meghaduta",
                    "verse": "1"
                }
            },
            {
                "id": "test3",
                "prompt": "Identify this Sanskrit quote...",
                "quote_text": "ॐ नमः शिवाय...",
                "ground_truth": {
                    "genre": "purana",
                    "author": "Unknown",
                    "text": "Shiva Purana",
                    "chapter": "1",
                    "verse": "1"
                }
            }
        ]
    
    @patch('zeroband.inference.sanskrit_data_loader.load_dataset')
    def test_init_from_huggingface(self, mock_load_dataset, mock_dataset_data):
        """Test initialization from HuggingFace."""
        # Mock the dataset
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset_data)
        
        dataset = SanskritQuoteDataset(
            dataset_name="test/dataset",
            split="train"
        )
        
        assert len(dataset.dataset) == 3
        assert dataset.dataset_name == "test/dataset"
        assert dataset.split == "train"
        
        # Check statistics
        assert dataset.genre_counts == {"epic": 1, "kavya": 1, "purana": 1}
        assert dataset.author_counts == {"Vyasa": 1, "Kalidasa": 1, "Unknown": 1}
        
    def test_get_random_batch(self, mock_dataset_data):
        """Test random batch sampling."""
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list(mock_dataset_data)
            dataset = SanskritQuoteDataset("test/dataset")
            
            # Test batch smaller than dataset
            batch = dataset.get_random_batch(2)
            assert len(batch) == 2
            assert all(ex in mock_dataset_data for ex in batch)
            
            # Test batch larger than dataset
            batch = dataset.get_random_batch(5)
            assert len(batch) == 3  # Should return all available
            
    def test_get_balanced_batch(self, mock_dataset_data):
        """Test balanced batch sampling."""
        # Add more examples to test balancing
        extended_data = mock_dataset_data + [
            {
                "id": "test4",
                "quote_text": "अग्निमीळे पुरोहितं...",
                "ground_truth": {
                    "genre": "veda",
                    "author": "Unknown",
                    "text": "Rigveda",
                    "chapter": "1",
                    "verse": "1"
                }
            },
            {
                "id": "test5",
                "quote_text": "सत्यमेव जयते...",
                "ground_truth": {
                    "genre": "veda",
                    "author": "Unknown",
                    "text": "Mundaka Upanishad",
                    "verse": "3.1.6"
                }
            }
        ]
        
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list(extended_data)
            dataset = SanskritQuoteDataset("test/dataset")
            
            batch = dataset.get_balanced_batch(4)
            assert len(batch) == 4
            
            # Check genre distribution
            genres = [ex['ground_truth']['genre'] for ex in batch]
            # Should have representation from multiple genres
            assert len(set(genres)) > 1
            
    def test_format_prompt_with_existing_prompt(self, mock_dataset_data):
        """Test prompt formatting when prompt exists."""
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list(mock_dataset_data)
            dataset = SanskritQuoteDataset("test/dataset")
            
            example = mock_dataset_data[0]
            prompt = dataset.format_prompt(example)
            assert prompt == example['prompt']
            
    def test_format_prompt_without_existing_prompt(self):
        """Test prompt formatting when prompt doesn't exist."""
        example = {
            "quote_text": "तत्त्वमसि",
            "ground_truth": {
                "genre": "veda",
                "author": "Unknown",
                "text": "Chandogya Upanishad"
            }
        }
        
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list([example])
            dataset = SanskritQuoteDataset("test/dataset")
            
            prompt = dataset.format_prompt(example)
            assert "तत्त्वमसि" in prompt
            assert "Genre" in prompt
            assert "Author" in prompt
            
    def test_get_ground_truth(self, mock_dataset_data):
        """Test ground truth extraction."""
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list(mock_dataset_data)
            dataset = SanskritQuoteDataset("test/dataset")
            
            example = mock_dataset_data[0]
            ground_truth = dataset.get_ground_truth(example)
            
            assert ground_truth == example['ground_truth']
            assert ground_truth['genre'] == 'epic'
            assert ground_truth['author'] == 'Vyasa'
            
    def test_iterate_batches(self, mock_dataset_data):
        """Test batch iteration."""
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list(mock_dataset_data)
            dataset = SanskritQuoteDataset("test/dataset")
            
            # Test a few iterations
            batch_iter = dataset.iterate_batches(batch_size=2)
            
            batch1 = next(batch_iter)
            assert len(batch1) == 2
            
            batch2 = next(batch_iter)
            assert len(batch2) == 2
            
            # Should continue indefinitely
            batch3 = next(batch_iter)
            assert len(batch3) == 2
            
    def test_load_local_json(self, tmp_path, mock_dataset_data):
        """Test loading from local JSON file."""
        # Create temporary JSON file
        json_file = tmp_path / "test_dataset.json"
        import json
        with open(json_file, 'w') as f:
            json.dump(mock_dataset_data, f)
            
        dataset = SanskritQuoteDataset(local_path=str(json_file))
        assert len(dataset.dataset) == 3
        assert dataset.genre_counts == {"epic": 1, "kavya": 1, "purana": 1}
        
    def test_load_local_jsonl(self, tmp_path, mock_dataset_data):
        """Test loading from local JSONL file."""
        # Create temporary JSONL file
        jsonl_file = tmp_path / "test_dataset.jsonl"
        import json
        with open(jsonl_file, 'w') as f:
            for item in mock_dataset_data:
                f.write(json.dumps(item) + '\n')
                
        dataset = SanskritQuoteDataset(local_path=str(jsonl_file))
        assert len(dataset.dataset) == 3
        assert dataset.genre_counts == {"epic": 1, "kavya": 1, "purana": 1}
        
    def test_create_sanskrit_dataset_function(self):
        """Test convenience function."""
        with patch('zeroband.inference.sanskrit_data_loader.load_dataset') as mock_load:
            mock_load.return_value = Dataset.from_list([])
            
            dataset = create_sanskrit_dataset(
                dataset_name="test/dataset",
                split="validation",
                seed=123
            )
            
            assert isinstance(dataset, SanskritQuoteDataset)
            assert dataset.dataset_name == "test/dataset"
            assert dataset.split == "validation"
            assert dataset.seed == 123