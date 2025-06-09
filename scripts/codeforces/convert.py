import json
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

random.seed(42)  # For reproducibility

# USAGE:
# Copy this script to the root of the codeforces repository and run it.
# This creates a folder called genesys_codeforces/ with the combined verifiable prompts and test cases.

def create_verification_info(row) -> str:
    verinfo = json.dumps({
        "language": row['language'],
        "input_mode": row['input_mode'], # stdio or file
        "time_limit": row['time_limit'],
        "memory_limit": row['memory_limit'],
        "tests": json.loads(row['tests']),
    })
    assert isinstance(verinfo, str)
    return verinfo

def filter_tests_by_count(df, min_tests=10, max_tests=100):
    """
    Filter DataFrame by test count and limit the number of tests per row.
    
    Args:
        df: DataFrame with 'tests' column containing JSON strings
        min_tests: Minimum number of tests required (rows with fewer are filtered out)
        max_tests: Maximum number of tests to keep (randomly sample if more)
    
    Returns:
        tuple: (filtered_df, num_filtered_out)
    """
    def process_tests(tests_json):
        try:
            tests = json.loads(tests_json)
            if len(tests) < min_tests:
                return None  # Mark for filtering
            elif len(tests) > max_tests:
                # Randomly sample max_tests
                sampled_tests = random.sample(tests, max_tests)
                return json.dumps(sampled_tests)
            else:
                return tests_json  # Keep as is
        except (json.JSONDecodeError, TypeError):
            return None  # Mark for filtering if JSON is invalid

    # Process all rows
    before_len = len(df)
    df = df.copy()
    df['tests'] = df['tests'].apply(process_tests)

    # Filter out None values (rows that didn't meet criteria)
    df = df[df['tests'].notna()]
    after_len = len(df)

    filtered_count = before_len - after_len
    return df, filtered_count

columns_to_drop = [
    "title",
    "contest_id",
    "contest_name",
    "contest_type",
    "contest_start",
    "contest_start_year",
    "index",
    "description",
    "input_format",
    "output_format",
    "note",
    "examples",
    "editorial",
    "rating",
    "tags",
    "testset_size",
    "official_tests_complete",
    "generated_checker",
    "executable",
    "generated_tests",
    #"prompt",
    "official_tests",
    "interaction_format",
    "time_limit",
    "memory_limit",
    "input_mode",
    "language",
    "tests",
    "combined_prompt",
    "id",
    "aliases",
]

def get_unified_schema(verifiable_dir, generated_tests_dir):
    """Pre-compute unified schema by sampling files"""
    print("Computing unified schema...")
    
    schemas = []
    
    # Sample ALL train files to understand the base schema
    train_files = list(Path(verifiable_dir).glob("train-*.parquet"))
    assert len(train_files) > 0, f"No train files found in {verifiable_dir}"
    
    for train_file in train_files:
        parquet_file = pq.ParquetFile(train_file)
        
        # Get first batch to understand structure
        batch = next(parquet_file.iter_batches(batch_size=100))
        df = batch.to_pandas()

        # Add the verification_info column as string
        df['verification_info'] = ''  # Empty string as default
        df['problem_id'] = ''
        df['task_type'] = ''

        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Reset index to avoid __index_level_0__ columns
        df = df.reset_index(drop=True)
        
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Normalize schema before adding to list
        normalized_fields = []
        for field in table.schema:
            if field.name == 'interaction_format':
                # Handle interaction_format as nullable string instead of null
                normalized_fields.append(pa.field('interaction_format', pa.string(), nullable=True))
            elif field.name == 'verification_info':
                # Ensure verification_info column is always a string
                normalized_fields.append(pa.field('verification_info', pa.string(), nullable=True))
            elif field.name == 'official_tests':
                # Skip official_tests from the final schema since we'll drop it later
                continue
            else:
                # Make all string fields nullable by default
                if pa.types.is_string(field.type):
                    normalized_fields.append(pa.field(field.name, field.type, nullable=True))
                else:
                    normalized_fields.append(field)
        
        # Ensure verification_info column is always included in schema as string
        verification_info_field_exists = any(f.name == 'verification_info' for f in normalized_fields)
        if not verification_info_field_exists:
            normalized_fields.append(pa.field('verification_info', pa.string(), nullable=True))
        
        normalized_schema = pa.schema(normalized_fields)
        schemas.append(normalized_schema)
    
    # Also sample a test cases file to understand that structure
    test_files = list(Path(generated_tests_dir).glob("test_cases_*.parquet"))
    test_files += list(Path(generated_tests_dir).glob("train-*.parquet"))
    assert len(test_files) > 0, f"No test files found in {generated_tests_dir}"
    
    # Unify all schemas (now they should be compatible)
    unified_schema = pa.unify_schemas(schemas)
    print(f"Unified schema computed with {len(train_files)} train files and {len(test_files)} test files.")
    
    return unified_schema

def safe_cast_table(table, target_schema):
    """Safely cast table to target schema, handling null columns gracefully"""
    # Create a mapping of field names to types for easier lookup
    target_field_map = {field.name: field for field in target_schema}
    
    # Process each column from the table
    new_columns = []
    new_names = []
    
    # First, handle all existing columns in the table
    for i, field in enumerate(table.schema):
        column = table.column(i)
        field_name = field.name
        
        if field_name in target_field_map:
            target_field = target_field_map[field_name]
            
            # Special handling for problematic casts
            if pa.types.is_null(target_field.type) and not pa.types.is_null(field.type):
                # If target is null but source isn't, create a null column
                null_array = pa.nulls(len(table), type=pa.string())
                new_columns.append(null_array)
            elif field_name == 'interaction_format':
                # Handle interaction_format specifically - keep as nullable string
                if pa.types.is_string(field.type):
                    new_columns.append(column)
                else:
                    # Cast to string if it's not already
                    new_columns.append(pa.compute.cast(column, pa.string()))
            else:
                # Normal cast
                try:
                    new_columns.append(pa.compute.cast(column, target_field.type))
                except pa.ArrowNotImplementedError:
                    # If cast fails, keep original column
                    print(f"Warning: Could not cast {field_name} from {field.type} to {target_field.type}, keeping original")
                    new_columns.append(column)
            
            new_names.append(field_name)
    
    # Add any missing columns from target schema that weren't in the source table
    table_field_names = [f.name for f in table.schema]
    for target_field in target_schema:
        if target_field.name not in table_field_names:
            # Create null column for missing fields
            if pa.types.is_null(target_field.type):
                null_array = pa.nulls(len(table), type=pa.string())
            else:
                null_array = pa.nulls(len(table), type=target_field.type)
            new_columns.append(null_array)
            new_names.append(target_field.name)
    
    # Reorder columns to match target schema order
    target_order = [f.name for f in target_schema]
    reordered_columns = []
    reordered_names = []
    
    for target_name in target_order:
        if target_name in new_names:
            idx = new_names.index(target_name)
            reordered_columns.append(new_columns[idx])
            reordered_names.append(target_name)
    
    return pa.table(reordered_columns, names=reordered_names)

def combine_verifiable_prompts_with_tests(verifiable_dir, generated_tests_dir, output_dir, batch_size=1000, process_first_only=False, min_tests=10, max_tests=100):
    """Combine verifiable-prompts with test cases in a new 'tests' column as JSON string"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pre-compute unified schema
    unified_schema = get_unified_schema(verifiable_dir, generated_tests_dir)
    
    # Get all train files
    train_files = list(Path(verifiable_dir).glob("train-*.parquet"))
    assert len(train_files) > 0, f"No train files found in {verifiable_dir}"
    
    # Option to process only the first file
    if process_first_only:
        train_files = train_files[:1]
        print(f"Processing only the first file: {train_files[0]}")
    
    # Initialize file counter for output naming
    output_file_counter = 0
    current_output_file = output_path / f"combined_{output_file_counter:05d}.parquet"
    writer = pq.ParquetWriter(current_output_file, unified_schema)
    
    # Track rows written to current file
    rows_per_output_file = 250
    current_file_rows = 0
    
    # Track total filtered counts
    total_filtered_empty = 0
    total_filtered_tests = 0
    
    for train_file in train_files:
        print(f"Processing {train_file}...")
        parquet_file = pq.ParquetFile(train_file)
        
        # Track filtered counts for this file
        file_filtered_tests = 0
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            
            # Filter out rows where None or empty before processing
            before_len = len(df)
            df = df[df['title'].notna()]
            df = df[df['description'].notna()]
            df = df[df['input_format'].notna()]
            df = df[df['output_format'].notna()]
            df = df[df['prompt'].notna()]
            df = df[df['examples'].notna()]
            df = df[df['examples'].apply(lambda x: x is not None and len(x) > 0 if isinstance(x, (list, tuple)) else x is not None)]
            after_len = len(df)

            filtered_empty = before_len - after_len
            total_filtered_empty += filtered_empty
            if filtered_empty > 0:
                print(f" Filtered {filtered_empty} rows with empty or None examples, format, prompt, or description.")
            if after_len == 0:
                print(" No rows remaining after filtering, skipping batch")
                continue

            # Filter out rows that have aliases
            before_alias_len = len(df)
            df = df[df['aliases'].isna()]
            after_alias_len = len(df)
            filtered_alias = before_alias_len - after_alias_len
            if filtered_alias > 0:
                print(f" Filtered {filtered_alias} rows of duplicates.")

            # Group by contest_id to minimize file reads
            for contest_id, group in df.groupby('contest_id'):
                # Load corresponding test cases file
                test_file = Path(generated_tests_dir) / f"test_cases_{contest_id}.parquet"
                
                # Create tests column by combining test cases for each problem_id
                def create_tests_for_problem(row):
                    problem_id = row['id']
                    official_tests_complete = row.get('official_tests_complete', False)
                    official_tests = row.get('official_tests', [])
                    
                    # Handle pandas/numpy arrays and convert to list if needed
                    if hasattr(official_tests, 'tolist'):
                        official_tests = official_tests.tolist()
                    elif official_tests is None:
                        official_tests = []
                    
                    # Convert official_tests to the same format if it exists
                    all_tests = []
                    if official_tests and isinstance(official_tests, list):
                        for test in official_tests:
                            all_tests.append({'input': test['input'].replace("\r\n", "\n"), 'output': test['output'].replace("\r\n", "\n")})
                    
                    # If official tests are complete, use only those
                    if official_tests_complete:
                        return json.dumps(all_tests)
                    
                    if test_file.exists():
                        try:
                            test_df = pd.read_parquet(test_file)
                            problem_tests = test_df[test_df['problem_id'] == problem_id]
                            for _, test_row in problem_tests.iterrows():
                                all_tests.append({'input': test_row['input'].replace("\r\n", "\n"), 'output': test_row['output'].replace("\r\n", "\n")})
                        except Exception as e:
                            print(f"Error reading test file {test_file} for problem {problem_id}: {e}")
                    
                    return json.dumps(all_tests)

                group['tests'] = group.apply(create_tests_for_problem, axis=1, result_type='reduce')
                group, filtered_tests = filter_tests_by_count(group, min_tests, max_tests)
                file_filtered_tests += filtered_tests

                group['problem_id'] = group.apply(lambda x: f"codeforces_{x['contest_id']}_{x['id']}", axis=1, result_type='reduce')
                group['task_type'] = 'codeforces'
                group['verification_info'] = group.apply(create_verification_info, axis=1, result_type='reduce')

                # Now drop official_tests column after processing
                if 'official_tests' in group.columns:
                    group = group.drop('official_tests', axis=1)
                group = group.drop('tests', axis=1)
                group = group[['problem_id', 'task_type', 'prompt', 'verification_info']]
                
                # Write the group
                if len(group) > 0:
                    try:
                        # Reset index to avoid __index_level_0__ columns
                        group = group.reset_index(drop=True)
                        
                        # Convert to Arrow table 
                        table = pa.Table.from_pandas(group, preserve_index=False)
                        
                        # Use safe cast instead of direct cast
                        table = safe_cast_table(table, unified_schema)
                        
                        # Check if we need to start a new output file
                        if current_file_rows + len(table) > rows_per_output_file:
                            writer.close()
                            print(f" Wrote {current_file_rows} rows to {current_output_file}.")
                            output_file_counter += 1
                            current_output_file = output_path / f"combined_{output_file_counter:05d}.parquet"
                            writer = pq.ParquetWriter(current_output_file, unified_schema)
                            current_file_rows = 0
                        
                        writer.write_table(table)
                        current_file_rows += len(table)
                        
                    except Exception as e:
                        print(f"Error writing batch for contest_id {contest_id}: {e}")
                        print(f"Group columns: {group.columns.tolist()}")
                        print(f"Group dtypes: {group.dtypes.to_dict()}")
                        print(f"Expected schema: {unified_schema}")
                        raise e
        
        # Print filtered test count once per file
        total_filtered_tests += file_filtered_tests
        if file_filtered_tests > 0:
            print(f" Filtered {file_filtered_tests} rows with insufficient tests.")
    
    writer.close()
    print(f" Wrote {current_file_rows} rows to {current_output_file}.")
    print(f"Processing completed. Output files written to {output_dir}")
    print(f"Total rows filtered for empty/None fields: {total_filtered_empty}")
    print(f"Total rows filtered for insufficient tests: {total_filtered_tests}")

# Usage
combine_verifiable_prompts_with_tests(
    verifiable_dir="verifiable-prompts/",
    generated_tests_dir="generated_tests/", 
    output_dir="genesys_codeforces/",
    process_first_only=False,
    min_tests=10,
    max_tests=100
)
