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

from app.services.pipeline import run_pipeline_from_audio
from app.core.config import settings

async def main():
    print(f"Debug: GEMINI_API_KEY loaded: {settings.GEMINI_API_KEY[:4]}...{settings.GEMINI_API_KEY[-4:] if settings.GEMINI_API_KEY else 'NONE'}")
    audio_path = os.path.join("audio_transmition", "athul1.ogg")
    
    if not os.path.exists(audio_path):
        print(f"Error: Could not find audio file at {audio_path}")
        return

    print(f"🚀 Starting Vajra 2.0 Full Integration Test")
    print(f"File: {audio_path}")
    print("-" * 50)

    try:
        # Read audio bytes
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Run the unified pipeline
        # Step 1: Transcription (Gemini)
        # Step 2-5: Analysis (Gemini Fallback since no Groq key)
        result = await run_pipeline_from_audio(
            audio_bytes=audio_bytes,
            audio_mime_type="audio/ogg",
            filename="athul1.ogg"
        )

        transcript = result["transcript"]
        analysis_res = result["analysis"]
        analysis = analysis_res.analysis

        print("\n✅ STEP 1: DIARIZED TRANSCRIPT")
        print(transcript)
        print("-" * 50)

        print("\n✅ STEP 2: SUMMARY & SENTIMENT")
        print(f"Summary: {analysis.summary}")
        print(f"Overall Sentiment: {analysis.sentiment.overall} (Conf: {analysis.sentiment.overall_confidence})")
        
        print("\n✅ STEP 3: RISK SCORING (WEIGHTED MATRIX)")
        print(f"Risk Level: {analysis.risk_assessment.level}")
        print(f"Risk Score: {analysis.risk_assessment.score}/10")
        print(f"Factors: {', '.join(analysis.risk_assessment.factors)}")

        print("\n✅ STEP 4: COMPLIANCE & INTENTS")
        print(f"Intents: {', '.join(analysis.customer_intents)}")
        print(f"Violations Found: {len(analysis.compliance_violations)}")
        for v in analysis.compliance_violations:
            print(f" - [{v.type}] {v.description}")

        print("\n✅ STEP 5: AGENT QUALITY")
        print(f"Overall Score: {analysis.agent_quality.overall_score}/10")
        print(f"Strengths: {', '.join(analysis.agent_quality.strengths)}")

        # Save the full JSON result for review
        with open("full_pipeline_test_result.json", "w") as f:
            json.dump(analysis_res.model_dump(mode='json'), f, indent=2)
        print(f"\n💾 Full analysis saved to full_pipeline_test_result.json")

    except Exception as e:
        print(f"❌ Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
