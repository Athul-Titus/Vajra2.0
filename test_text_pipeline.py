import asyncio
import os
import json
import sys

# Ensure app is importable
sys.path.append(os.getcwd())

# Force UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from app.services.pipeline import run_pipeline_from_text

async def main():
    transcript = """
User 1: Hello, I am very frustrated with my internet service. It has been down for three days now.
User 2: I am so sorry to hear that. I can help you with that. Can I have your account ID please?
User 1: My ID is 12345. I want to cancel my subscription if this isn't fixed today.
User 2: I understand. Let me check the connection. I see a technician is available this afternoon.
"""
    
    print(f"🚀 Starting Vajra 2.0 Text Analysis Integration Test")
    print("-" * 50)

    try:
        # Run the unified pipeline (Step 2-5)
        response = await run_pipeline_from_text(
            transcript=transcript,
            config_id="telecom_default"
        )

        analysis = response.analysis

        print("\n✅ SUMMARY & SENTIMENT")
        print(f"Summary: {analysis.summary}")
        print(f"Overall Sentiment: {analysis.sentiment.overall}")
        
        print("\n✅ RISK SCORING")
        print(f"Risk Level: {analysis.risk_assessment.level}")
        print(f"Risk Score: {analysis.risk_assessment.score}/10")

        print("\n✅ COMPLIANCE")
        print(f"Violations Found: {len(analysis.compliance_violations)}")

        print("\n✅ AGENT QUALITY")
        print(f"Overall Score: {analysis.agent_quality.overall_score}/10")

        print("\n✨ PIPELINE INTEGRATION VERIFIED FOR TEXT")

    except Exception as e:
        print(f"❌ Text Pipeline Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
