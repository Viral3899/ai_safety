#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from core.base_model import BaseModel, SafetyResult, SafetyLevel, ModelConfig
        print("✅ Core base model imports successful")
        
        # Test model imports
        from models.abuse_detector import AbuseDetector
        print("✅ Abuse detector import successful")
        
        from models.escalation_detector import EscalationDetector
        print("✅ Escalation detector import successful")
        
        from models.crisis_detector import CrisisDetector
        print("✅ Crisis detector import successful")
        
        from models.content_filter import ContentFilter, AgeGroup
        print("✅ Content filter import successful")
        
        # Test safety manager
        from safety_system.safety_manager import SafetyManager
        print("✅ Safety manager import successful")
        
        # Test initialization
        config = ModelConfig(model_type='sklearn', threshold=0.5)
        safety_manager = SafetyManager()
        print("✅ Safety manager initialization successful")
        
        # Test basic functionality
        result = safety_manager.analyze(
            text="Hello world",
            user_id="test_user",
            session_id="test_session",
            age_group=AgeGroup.ADULT
        )
        print("✅ Basic analysis successful")
        
        print("\n🎉 All imports and basic functionality working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready to run the demo!")
        print("Run: python run_demo.py")
    else:
        print("\n❌ Fix import issues before running the demo.")
        sys.exit(1)
