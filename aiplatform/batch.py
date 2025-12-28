"""
Batch Processing Module for AIPlatform CLI

This module provides batch processing capabilities for multiple prompts,
files, and operations across different AI models.
"""

import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a single item in batch processing."""
    id: str
    prompt: str
    model: str
    parameters: Dict[str, Any]
    output_file: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    timeout: int = 300
    retry_attempts: int = 3
    delay_between_requests: float = 0.1
    save_progress: bool = True
    progress_file: str = "batch_progress.json"


class BatchProcessor:
    """Handles batch processing of AI operations."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.items: List[BatchItem] = []
        self.results: List[BatchItem] = []
        
    def add_item(self, item: BatchItem):
        """Add item to batch queue."""
        self.items.append(item)
        
    def load_from_file(self, file_path: str) -> List[BatchItem]:
        """Load batch items from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = []
            for item_data in data.get('items', []):
                item = BatchItem(
                    id=item_data['id'],
                    prompt=item_data['prompt'],
                    model=item_data['model'],
                    parameters=item_data.get('parameters', {}),
                    output_file=item_data.get('output_file')
                )
                items.append(item)
                self.add_item(item)
            
            logger.info(f"Loaded {len(items)} items from {file_path}")
            return items
            
        except Exception as e:
            logger.error(f"Error loading batch file: {e}")
            return []
    
    def load_from_directory(self, directory: str, pattern: str = "*.txt") -> List[BatchItem]:
        """Load prompts from files in directory."""
        items = []
        directory_path = Path(directory)
        
        for file_path in directory_path.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    item = BatchItem(
                        id=file_path.stem,
                        prompt=content,
                        model="gigachat3-702b",  # default model
                        parameters={},
                        output_file=f"output/{file_path.stem}_result.txt"
                    )
                    items.append(item)
                    self.add_item(item)
                    
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        logger.info(f"Loaded {len(items)} items from directory {directory}")
        return items
    
    def save_progress(self):
        """Save current progress to file."""
        if not self.config.save_progress:
            return
        
        progress_data = {
            'total_items': len(self.items),
            'completed_items': len([i for i in self.items if i.status == 'completed']),
            'failed_items': len([i for i in self.items if i.status == 'failed']),
            'items': [
                {
                    'id': item.id,
                    'status': item.status,
                    'error': item.error,
                    'start_time': item.start_time,
                    'end_time': item.end_time
                }
                for item in self.items
            ]
        }
        
        try:
            with open(self.config.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def process_item(self, item: BatchItem, processor_func: Callable) -> BatchItem:
        """Process a single batch item."""
        item.status = "processing"
        item.start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Process the item
                result = processor_func(item.prompt, item.model, **item.parameters)
                item.result = result
                item.status = "completed"
                item.end_time = time.time()
                
                # Save result to file if specified
                if item.output_file:
                    self._save_result(item, result)
                
                break
                
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    item.error = str(e)
                    item.status = "failed"
                    item.end_time = time.time()
                    logger.error(f"Failed to process item {item.id}: {e}")
                else:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return item
    
    def _save_result(self, item: BatchItem, result: Any):
        """Save result to output file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(item.output_file), exist_ok=True)
            
            # Save result
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False, indent=2)
            
            with open(item.output_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Error saving result for {item.id}: {e}")
    
    def process_batch(self, processor_func: Callable) -> List[BatchItem]:
        """Process all items in batch."""
        logger.info(f"Starting batch processing of {len(self.items)} items")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_item, item, processor_func): item
                for item in self.items
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout):
                item = future_to_item[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Task failed for item {item.id}: {e}")
                
                # Save progress periodically
                if len(self.results) % 10 == 0:
                    self.save_progress()
                
                # Delay between requests
                time.sleep(self.config.delay_between_requests)
        
        # Final progress save
        self.save_progress()
        
        logger.info(f"Batch processing completed. Success: {len([r for r in self.results if r.status == 'completed'])}, Failed: {len([r for r in self.results if r.status == 'failed'])}")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary."""
        completed = [i for i in self.results if i.status == 'completed']
        failed = [i for i in self.results if i.status == 'failed']
        
        total_time = sum([
            (item.end_time - item.start_time) 
            for item in completed 
            if item.start_time and item.end_time
        ])
        
        return {
            'total_items': len(self.items),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': len(completed) / len(self.items) if self.items else 0,
            'total_time': total_time,
            'average_time_per_item': total_time / len(completed) if completed else 0,
            'failed_items': [{'id': i.id, 'error': i.error} for i in failed]
        }
    
    def export_results(self, output_file: str, format: str = 'json'):
        """Export results to file."""
        try:
            if format == 'json':
                data = {
                    'summary': self.get_summary(),
                    'results': [
                        {
                            'id': item.id,
                            'prompt': item.prompt,
                            'model': item.model,
                            'status': item.status,
                            'result': item.result,
                            'error': item.error,
                            'processing_time': item.end_time - item.start_time if item.start_time and item.end_time else None
                        }
                        for item in self.results
                    ]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                import csv
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Prompt', 'Model', 'Status', 'Result', 'Error', 'Processing Time'])
                    
                    for item in self.results:
                        processing_time = item.end_time - item.start_time if item.start_time and item.end_time else ''
                        result = item.result if isinstance(item.result, str) else json.dumps(item.result) if item.result else ''
                        writer.writerow([
                            item.id,
                            item.prompt,
                            item.model,
                            item.status,
                            result,
                            item.error or '',
                            processing_time
                        ])
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")


def create_batch_config(**kwargs) -> BatchConfig:
    """Create batch configuration from parameters."""
    return BatchConfig(
        max_workers=kwargs.get('max_workers', 4),
        timeout=kwargs.get('timeout', 300),
        retry_attempts=kwargs.get('retry_attempts', 3),
        delay_between_requests=kwargs.get('delay', 0.1),
        save_progress=kwargs.get('save_progress', True),
        progress_file=kwargs.get('progress_file', 'batch_progress.json')
    )


def create_sample_batch_file(file_path: str):
    """Create a sample batch file for testing."""
    sample_data = {
        "items": [
            {
                "id": "item1",
                "prompt": "Explain quantum computing in simple terms",
                "model": "gigachat3-702b",
                "parameters": {"max_tokens": 100},
                "output_file": "output/quantum_explanation.txt"
            },
            {
                "id": "item2", 
                "prompt": "What is machine learning?",
                "model": "gigachat3-702b",
                "parameters": {"max_tokens": 100},
                "output_file": "output/ml_explanation.txt"
            }
        ]
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample batch file created: {file_path}")
