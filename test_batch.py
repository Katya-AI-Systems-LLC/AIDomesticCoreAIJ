#!/usr/bin/env python3
"""
Simple Batch Processing Test

This script demonstrates batch processing functionality without relying on the damaged CLI.
"""

import json
import time
from typing import List, Dict, Any

class SimpleBatchProcessor:
    def __init__(self):
        self.items = []
        self.results = []
    
    def load_from_file(self, file_path: str):
        """Load batch items from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.items = data.get('items', [])
        print(f"Loaded {len(self.items)} items from {file_path}")
    
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        start_time = time.time()
        
        try:
            # Simulate processing
            prompt = item.get('prompt', '')
            model = item.get('model', 'gigachat3-702b')
            params = item.get('parameters', {})
            
            # Simulate AI response
            response = f"Generated response for: {prompt[:50]}... using {model}"
            
            end_time = time.time()
            
            return {
                'id': item.get('id'),
                'status': 'completed',
                'prompt': prompt,
                'model': model,
                'response': response,
                'processing_time': end_time - start_time,
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'id': item.get('id'),
                'status': 'failed',
                'prompt': item.get('prompt', ''),
                'model': item.get('model', ''),
                'response': None,
                'processing_time': end_time - start_time,
                'error': str(e)
            }
    
    def process_batch(self) -> List[Dict[str, Any]]:
        """Process all items in batch."""
        print(f"Processing batch of {len(self.items)} items...")
        
        for i, item in enumerate(self.items, 1):
            print(f"Processing item {i}/{len(self.items)}: {item.get('id')}")
            result = self.process_item(item)
            self.results.append(result)
            
            # Small delay between items
            time.sleep(0.1)
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        completed = [r for r in self.results if r['status'] == 'completed']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        total_time = sum(r['processing_time'] for r in self.results)
        
        return {
            'total_items': len(self.items),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': len(completed) / len(self.items) if self.items else 0,
            'total_time': total_time,
            'average_time_per_item': total_time / len(self.results) if self.results else 0
        }
    
    def save_results(self, output_file: str):
        """Save results to file."""
        output_data = {
            'summary': self.get_summary(),
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")


def main():
    """Main function to test batch processing."""
    print("=== Simple Batch Processing Test ===")
    
    # Create processor
    processor = SimpleBatchProcessor()
    
    # Load batch file
    try:
        processor.load_from_file('test_batch.json')
    except FileNotFoundError:
        print("Error: test_batch.json not found")
        return 1
    
    # Process batch
    results = processor.process_batch()
    
    # Show summary
    summary = processor.get_summary()
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total items: {summary['total_items']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Total time: {summary['total_time']:.2f}s")
    print(f"Average time per item: {summary['average_time_per_item']:.2f}s")
    
    # Show individual results
    print(f"\n=== Individual Results ===")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Response: {result['response']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("-" * 40)
    
    # Save results
    processor.save_results('batch_results.json')
    
    print(f"\n=== Batch Processing Complete ===")
    return 0


if __name__ == "__main__":
    exit(main())
