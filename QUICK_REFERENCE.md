# 🎬 Quick Reference Card
## Video Recording - AI Safety Models POC

### 🚀 **Quick Start Commands**
```bash
# Start web interface
python demo/web_interface.py

# Test demos
python demo_scenarios.py --quick

# Access web interface
# http://localhost:8080
```

### 📋 **Video Structure (10 minutes)**
1. **Introduction** (30s) - POC overview
2. **Architecture** (2m) - System design
3. **Approach** (2m) - Technical decisions
4. **Demo** (4m) - Live scenarios
5. **Analysis** (1.5m) - Pros/cons, scaling
6. **Conclusion** (30s) - Key takeaways

### 🎯 **Key Demo Scenarios**
1. **Child Safety**: "fuck you" → CRITICAL risk
2. **Crisis**: "I want to kill myself" → CRITICAL risk
3. **Safe**: "Hello, how are you?" → MINIMAL risk
4. **Escalation**: "I hate this!" → MEDIUM risk
5. **Adult Filter**: "violence themes" → MINIMAL risk
6. **Multi-Model**: "fucking idiot die!" → CRITICAL risk

### 🏗️ **Architecture Highlights**
- **4 Core Models**: Abuse, Crisis, Escalation, Content Filter
- **Real-time**: <100ms processing
- **Modular**: Easy to extend
- **Safety-first**: Conservative thresholds
- **Human oversight**: Critical decisions reviewed

### ⚡ **Technical Approach**
- **Hybrid**: Rules + ML for reliability
- **Python**: scikit-learn, FastAPI, transformers
- **Age-aware**: Different thresholds per age group
- **Bias-aware**: Fairness evaluation built-in

### 🎯 **Key Points to Emphasize**
- Safety-critical systems need reliability over accuracy
- Child protection is paramount
- Crisis intervention saves lives
- Human oversight remains essential
- Production-ready and scalable

### 📊 **Performance Metrics**
- **Abuse Detection**: 88% accuracy
- **Crisis Detection**: 92% accuracy
- **Content Filtering**: 88% accuracy
- **Escalation Detection**: 78% accuracy
- **Processing Time**: <100ms average

### 🚨 **Emergency Demo Fixes**
```bash
# If web interface fails
python demo/cli_demo.py

# If models not working
python -c "from src.safety_system.safety_manager import SafetyManager; SafetyManager()"

# If demo scenarios fail
python demo_scenarios.py --quick
```

### 📝 **Script Files**
- **Main Script**: `VIDEO_SCRIPT.md`
- **Demo Code**: `demo_scenarios.py`
- **Recording Guide**: `VIDEO_RECORDING_GUIDE.md`
- **Sample Data**: `sample_demo_data.json`

### 🎥 **Recording Tips**
- Speak clearly and at moderate pace
- Show code and outputs clearly
- Follow the 10-minute time limit
- Maintain professional demeanor
- Test everything before recording

### ✅ **Success Checklist**
- [ ] All 4 models demonstrated
- [ ] Real-time processing shown
- [ ] Technical approach explained
- [ ] Pros/cons discussed
- [ ] Production scaling covered
- [ ] Professional quality maintained
- [ ] 10-minute time limit respected

**You're ready to record an impressive video! 🎬**
