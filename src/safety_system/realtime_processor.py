"""
Real-time processing for AI safety models.
"""

import asyncio
import queue
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from safety_system.safety_manager import SafetyManager
from core.base_model import SafetyLevel


class RealTimeProcessor:
    """Real-time processor for AI Safety Models."""
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize the real-time processor."""
        self.safety_manager = SafetyManager()
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'avg_processing_time': 0,
            'queue_size': 0
        }
    
    def start_processing(self):
        """Start the real-time processing thread."""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            print("🚀 Real-time processing started")
    
    def stop_processing(self):
        """Stop the real-time processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        print("⏹️  Real-time processing stopped")
    
    def add_text(self, text: str, user_id: str, session_id: str, 
                 priority: str = 'normal', **kwargs) -> bool:
        """Add text to processing queue."""
        try:
            processing_item = {
                'text': text,
                'user_id': user_id,
                'session_id': session_id,
                'priority': priority,
                'timestamp': datetime.now().isoformat(),
                'kwargs': kwargs
            }
            
            self.input_queue.put(processing_item, timeout=1)
            return True
        except queue.Full:
            print("⚠️  Processing queue is full")
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get processed result from output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_processing:
            try:
                # Get next item from queue
                item = self.input_queue.get(timeout=1)
                
                start_time = time.time()
                
                # Process the text
                result = self._process_text(item)
                
                processing_time = time.time() - start_time
                
                # Add processing metadata
                result['processing_metadata'] = {
                    'processing_time': processing_time,
                    'queue_time': (datetime.now() - datetime.fromisoformat(item['timestamp'])).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to output queue
                self.output_queue.put(result)
                
                # Update stats
                self._update_stats(processing_time)
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                self.stats['error_count'] += 1
    
    def _process_text(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single text item."""
        try:
            result = self.safety_manager.analyze(
                text=item['text'],
                user_id=item['user_id'],
                session_id=item['session_id'],
                **item['kwargs']
            )
            
            # Add original request info
            result['request_info'] = {
                'user_id': item['user_id'],
                'session_id': item['session_id'],
                'priority': item['priority'],
                'original_timestamp': item['timestamp']
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'request_info': {
                    'user_id': item['user_id'],
                    'session_id': item['session_id'],
                    'priority': item['priority']
                }
            }
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.stats['processed_count'] += 1
        
        # Update average processing time
        total_time = self.stats['avg_processing_time'] * (self.stats['processed_count'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['processed_count']
        
        self.stats['queue_size'] = self.input_queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def clear_queues(self):
        """Clear input and output queues."""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        print("🧹 Queues cleared")


class StreamingProcessor:
    """Streaming processor for continuous text streams."""
    
    def __init__(self, buffer_size: int = 100):
        """Initialize the streaming processor."""
        self.realtime_processor = RealTimeProcessor()
        self.buffer = []
        self.buffer_size = buffer_size
        self.current_session = None
    
    def start_stream(self, user_id: str, session_id: str):
        """Start a new streaming session."""
        self.current_session = {
            'user_id': user_id,
            'session_id': session_id,
            'start_time': datetime.now()
        }
        self.realtime_processor.start_processing()
    
    def add_text_chunk(self, text_chunk: str):
        """Add a text chunk to the stream."""
        if not self.current_session:
            raise ValueError("No active streaming session")
        
        self.buffer.append(text_chunk)
        
        # Process if buffer is full or chunk ends with sentence
        if (len(self.buffer) >= self.buffer_size or 
            text_chunk.endswith(('.', '!', '?'))):
            
            full_text = ' '.join(self.buffer)
            self.realtime_processor.add_text(
                text=full_text,
                user_id=self.current_session['user_id'],
                session_id=self.current_session['session_id']
            )
            self.buffer = []
    
    def get_stream_results(self) -> List[Dict[str, Any]]:
        """Get all available results from the stream."""
        results = []
        while True:
            result = self.realtime_processor.get_result(timeout=0.1)
            if result is None:
                break
            results.append(result)
        return results
    
    def end_stream(self):
        """End the current streaming session."""
        if self.buffer:
            # Process remaining text
            full_text = ' '.join(self.buffer)
            self.realtime_processor.add_text(
                text=full_text,
                user_id=self.current_session['user_id'],
                session_id=self.current_session['session_id']
            )
            self.buffer = []
        
        self.realtime_processor.stop_processing()
        self.current_session = None


def demo_realtime_processing():
    """Demonstrate real-time processing capabilities."""
    print("🚀 Real-time Processing Demo")
    print("=" * 30)
    
    processor = RealTimeProcessor()
    processor.start_processing()
    
    # Add some test texts
    test_texts = [
        "Hello, how are you today?",
        "I hate you, you stupid idiot!",
        "I want to kill myself.",
        "This is great content for kids!"
    ]
    
    for i, text in enumerate(test_texts):
        processor.add_text(
            text=text,
            user_id=f"demo_user_{i}",
            session_id="demo_session",
            age_group="adult"
        )
    
    # Get results
    print("📊 Processing Results:")
    for i in range(len(test_texts)):
        result = processor.get_result(timeout=5)
        if result:
            if 'error' in result:
                print(f"  {i+1}. Error: {result['error']}")
            else:
                risk = result['overall_assessment']['overall_risk']
                processing_time = result['processing_metadata']['processing_time']
                print(f"  {i+1}. Risk: {risk.upper()} (processed in {processing_time:.3f}s)")
    
    # Show stats
    stats = processor.get_stats()
    print(f"\n📈 Processing Stats:")
    print(f"  Processed: {stats['processed_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Avg Time: {stats['avg_processing_time']:.3f}s")
    
    processor.stop_processing()


if __name__ == "__main__":
    demo_realtime_processing()