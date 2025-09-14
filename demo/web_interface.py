"""
Web Interface for AI Safety Models Demo.

This module provides a simple web interface using FastAPI.
"""

import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from safety_system.safety_manager import SafetyManager
    from models.content_filter import AgeGroup
except ImportError as e:
    print(f"❌ Missing dependencies for web interface: {e}")
    print("Install with: pip install fastapi uvicorn")
    sys.exit(1)


# Initialize FastAPI app
app = FastAPI(title="AI Safety Models Demo", version="1.0.0")

# Initialize safety manager
safety_manager = SafetyManager()


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    user_id: str = "web_user"
    session_id: str = "web_session"
    age_group: str = "adult"


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    overall_risk: str
    intervention_level: str
    max_score: float
    models: Dict[str, Any]
    intervention_recommendations: list
    processing_time: float


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Safety Models Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            textarea { height: 100px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .risk-minimal { background: #d4edda; border: 1px solid #c3e6cb; }
            .risk-low { background: #fff3cd; border: 1px solid #ffeaa7; }
            .risk-medium { background: #f8d7da; border: 1px solid #f5c6cb; }
            .risk-high { background: #f8d7da; border: 1px solid #f5c6cb; }
            .risk-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Safety Models Demo</h1>
            <p>Enter text below to analyze it with our AI Safety Models:</p>
            
            <form id="analysisForm">
                <div class="form-group">
                    <label for="text">Text to Analyze:</label>
                    <textarea id="text" name="text" placeholder="Enter text to analyze..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="ageGroup">Age Group:</label>
                    <select id="ageGroup" name="ageGroup">
                        <option value="adult">Adult</option>
                        <option value="teen">Teen</option>
                        <option value="child">Child</option>
                    </select>
                </div>
                
                <button type="submit">Analyze Text</button>
            </form>
            
            <div id="result" class="result" style="display: none;">
                <h3>Analysis Results</h3>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('analysisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const text = document.getElementById('text').value;
                const ageGroup = document.getElementById('ageGroup').value;
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, age_group: ageGroup })
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error analyzing text: ' + error.message);
                }
            });
            
            function displayResults(result) {
                const resultDiv = document.getElementById('result');
                const contentDiv = document.getElementById('resultContent');
                
                const riskClass = 'risk-' + result.overall_risk;
                resultDiv.className = 'result ' + riskClass;
                resultDiv.style.display = 'block';
                
                let html = `
                    <p><strong>Overall Risk:</strong> ${result.overall_risk.toUpperCase()}</p>
                    <p><strong>Intervention Level:</strong> ${result.intervention_level}</p>
                    <p><strong>Max Score:</strong> ${result.max_score.toFixed(3)}</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                    
                    <h4>Model Results:</h4>
                    <ul>
                `;
                
                for (const [modelName, modelResult] of Object.entries(result.models)) {
                    html += `<li><strong>${modelName.replace('_', ' ')}:</strong> ${modelResult.risk_level} (${modelResult.result.score.toFixed(3)})</li>`;
                }
                
                html += '</ul>';
                
                if (result.intervention_recommendations.length > 0) {
                    html += '<h4>Interventions:</h4><ul>';
                    result.intervention_recommendations.forEach(rec => {
                        html += `<li>${rec.action}</li>`;
                    });
                    html += '</ul>';
                }
                
                contentDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text using AI Safety Models."""
    try:
        start_time = datetime.now()
        
        # Convert age group string to enum
        age_group_map = {
            'child': AgeGroup.CHILD,
            'teen': AgeGroup.TEEN,
            'adult': AgeGroup.ADULT
        }
        age_group = age_group_map.get(request.age_group, AgeGroup.ADULT)
        
        # Perform analysis
        result = safety_manager.analyze(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            age_group=age_group
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TextAnalysisResponse(
            overall_risk=result['overall_assessment']['overall_risk'],
            intervention_level=result['overall_assessment']['intervention_level'],
            max_score=result['overall_assessment']['max_score'],
            models=result['models'],
            intervention_recommendations=result['intervention_recommendations'],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models/status")
async def models_status():
    """Get status of all safety models."""
    try:
        status = safety_manager.get_models_status()
        return {"models": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the web server."""
    import uvicorn
    
    print("🚀 Starting AI Safety Models Web Interface")
    print("=" * 40)
    print("Open your browser to: http://localhost:8080")
    print("API docs available at: http://localhost:8080/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()