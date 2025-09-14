# Video Recording Guide
## 10-Minute AI Safety Models POC Walkthrough

### 🎬 **Pre-Recording Setup**

#### **Technical Setup**
1. **Screen Resolution**: Set to 1920x1080 for best quality
2. **Browser**: Use Chrome or Firefox with developer tools ready
3. **Terminal**: Have terminal/command prompt open and ready
4. **Applications**: Close unnecessary apps for smooth recording
5. **Audio**: Test microphone and ensure good audio quality

#### **Environment Setup**
1. **Lighting**: Ensure good lighting on your face if showing webcam
2. **Background**: Clean, professional background
3. **Distractions**: Turn off notifications, close other applications
4. **Internet**: Stable internet connection for web demo

#### **Pre-Demo Testing**
```bash
# Test the web interface
python demo/web_interface.py

# Test the demo scenarios
python demo_scenarios.py --quick

# Verify all models are working
python -c "from src.safety_system.safety_manager import SafetyManager; sm = SafetyManager(); print('✅ All models loaded successfully')"
```

---

### 📋 **Recording Checklist**

#### **Before Starting Recording**
- [ ] Web interface running on localhost:8080
- [ ] Demo scenarios tested and working
- [ ] Screen recording software ready
- [ ] Audio levels tested
- [ ] Script reviewed and ready
- [ ] Sample data prepared
- [ ] Browser bookmarked to http://localhost:8080

#### **During Recording**
- [ ] Speak clearly and at moderate pace
- [ ] Show code and outputs clearly on screen
- [ ] Follow the script timing (10 minutes max)
- [ ] Demonstrate each scenario step-by-step
- [ ] Explain technical concepts simply
- [ ] Maintain professional demeanor

#### **After Recording**
- [ ] Review video for clarity and completeness
- [ ] Check audio quality throughout
- [ ] Verify all demos worked correctly
- [ ] Edit if necessary (keep under 10 minutes)
- [ ] Export in high quality (1080p minimum)

---

### 🎯 **Video Structure & Timing**

#### **Segment 1: Introduction (30 seconds)**
- Introduce yourself and the POC
- Brief overview of what you'll demonstrate
- Set expectations for the 10-minute walkthrough

#### **Segment 2: Architecture Overview (2 minutes)**
- Show the system architecture diagram
- Explain the modular design
- Highlight key components and their roles
- Emphasize safety-first approach

#### **Segment 3: Technical Approach (2 minutes)**
- Explain model selection rationale
- Discuss framework choices
- Show code structure and organization
- Highlight design principles

#### **Segment 4: Live Demonstration (4 minutes)**
- **Scenario 1**: Child safety with profanity (45 seconds)
- **Scenario 2**: Crisis intervention detection (45 seconds)
- **Scenario 3**: Safe content handling (30 seconds)
- **Scenario 4**: Escalation detection (30 seconds)
- **Scenario 5**: Age-appropriate filtering (30 seconds)
- **Scenario 6**: Multi-model integration (30 seconds)

#### **Segment 5: Analysis & Scaling (1.5 minutes)**
- Discuss pros and cons of the approach
- Explain ethical considerations
- Outline production scaling improvements
- Highlight key achievements

#### **Segment 6: Conclusion (30 seconds)**
- Summarize key takeaways
- Thank the viewer
- Mention availability of code and documentation

---

### 🖥️ **Screen Recording Setup**

#### **Recommended Software**
- **Windows**: OBS Studio (free), Camtasia, or ScreenFlow
- **Mac**: QuickTime Player (built-in), ScreenFlow, or OBS Studio
- **Linux**: OBS Studio or SimpleScreenRecorder

#### **OBS Studio Settings**
```
Video Settings:
- Base Resolution: 1920x1080
- Output Resolution: 1920x1080
- FPS: 30

Audio Settings:
- Sample Rate: 44.1 kHz
- Channels: Stereo
- Bitrate: 128 kbps

Recording Settings:
- Format: MP4
- Encoder: x264
- Rate Control: CBR
- Bitrate: 2500 kbps
```

#### **Screen Layout**
1. **Main Screen**: Web browser with demo interface
2. **Side Panel**: Terminal/command prompt (optional)
3. **Overlay**: Webcam in corner (optional)

---

### 🎤 **Audio Guidelines**

#### **Microphone Setup**
- Use external microphone if available
- Test audio levels before recording
- Speak 6-12 inches from microphone
- Avoid background noise

#### **Speaking Tips**
- Speak clearly and at moderate pace
- Pause briefly between sections
- Use confident, professional tone
- Avoid filler words (um, uh, like)

#### **Audio Levels**
- Peak levels: -6dB to -3dB
- Average levels: -12dB to -9dB
- Avoid clipping or distortion

---

### 🎬 **Demo Execution Guide**

#### **Web Interface Demo**
1. **Open Browser**: Navigate to http://localhost:8080
2. **Show Interface**: Explain the API documentation
3. **Test Endpoint**: Use the interactive API docs
4. **Show Results**: Explain the JSON response structure

#### **Command Line Demo**
1. **Open Terminal**: Show command line interface
2. **Run Demo Script**: `python demo_scenarios.py --quick`
3. **Explain Output**: Break down the results
4. **Show Code**: Briefly show key code sections

#### **Code Walkthrough**
1. **Project Structure**: Show directory layout
2. **Key Files**: Highlight main components
3. **Model Implementation**: Show one model in detail
4. **Integration**: Show how models work together

---

### 📝 **Script Cues**

#### **Transition Phrases**
- "Now let's move on to..."
- "Next, I'll demonstrate..."
- "This brings us to..."
- "Finally, let's look at..."

#### **Technical Explanations**
- "The system uses a hybrid approach..."
- "This ensures that..."
- "The key benefit is..."
- "This approach allows for..."

#### **Demo Transitions**
- "Let me show you this in action..."
- "Here's how it works..."
- "As you can see..."
- "This demonstrates..."

---

### 🎯 **Key Points to Emphasize**

#### **Technical Excellence**
- Real-time processing capabilities
- Modular, scalable architecture
- Comprehensive safety coverage
- Production-ready implementation

#### **Safety Focus**
- Child protection mechanisms
- Crisis intervention capabilities
- Human oversight integration
- Bias mitigation features

#### **Innovation**
- Hybrid rule-based + ML approach
- Multi-model consensus system
- Age-appropriate filtering
- Context-aware processing

---

### 🚨 **Common Issues & Solutions**

#### **Technical Issues**
- **Web interface not loading**: Check if port 8080 is available
- **Models not responding**: Restart the application
- **Demo scenarios failing**: Run `python demo_scenarios.py --quick` to test

#### **Recording Issues**
- **Audio quality poor**: Check microphone settings and levels
- **Screen resolution issues**: Set to 1920x1080 before recording
- **Performance lag**: Close unnecessary applications

#### **Timing Issues**
- **Running over 10 minutes**: Practice timing, edit if necessary
- **Running too fast**: Add pauses between sections
- **Missing key points**: Use the script as a guide

---

### 📤 **Post-Recording Checklist**

#### **Video Quality Check**
- [ ] Audio is clear throughout
- [ ] Screen content is readable
- [ ] No technical glitches
- [ ] All demos worked correctly
- [ ] Professional presentation

#### **Content Verification**
- [ ] All required topics covered
- [ ] Technical accuracy maintained
- [ ] Clear explanations provided
- [ ] Demo scenarios successful
- [ ] Key points emphasized

#### **Final Steps**
- [ ] Export in high quality (1080p+)
- [ ] Upload to YouTube/Loom
- [ ] Set appropriate title and description
- [ ] Provide link in submission materials
- [ ] Test video playback

---

### 🎥 **Video Title & Description**

#### **Suggested Title**
"AI Safety Models POC - 10-Minute Technical Walkthrough | ML Candidate Submission"

#### **Suggested Description**
```
AI Safety Models Proof of Concept demonstration covering:

🔍 Core Models:
• Abuse Language Detection
• Escalation Pattern Recognition  
• Crisis Intervention
• Content Filtering

⚡ Key Features:
• Real-time processing (<100ms)
• Age-appropriate filtering
• Human oversight integration
• Bias mitigation

🏗️ Architecture:
• Modular, scalable design
• Hybrid rule-based + ML approach
• Production-ready implementation

📊 Performance:
• 88%+ accuracy across models
• Comprehensive safety coverage
• Ethical AI considerations

Code repository and technical report available for review.
```

---

### 🎯 **Success Criteria**

Your video will be successful if it:
- ✅ Stays within 10-minute time limit
- ✅ Demonstrates all 4 core safety models
- ✅ Shows real-time processing capabilities
- ✅ Explains technical approach clearly
- ✅ Highlights safety and ethical considerations
- ✅ Demonstrates production readiness
- ✅ Maintains professional quality throughout

**Remember**: The goal is to showcase your technical skills, system design abilities, and understanding of AI safety challenges in a clear, professional manner.
