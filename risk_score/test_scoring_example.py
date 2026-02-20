import json
from risk_scoring import EscalationAnalysis, ScoreBreakdown, get_scoring_prompt_instructions

def analyze_escalation_risk_mock(transcript: str) -> EscalationAnalysis:
    """
    Mocks an LLM taking the transcript and the prompt instructions,
    and returning the parsed EscalationAnalysis Pydantic model.
    """
    print("--- Sending to LLM ---")
    print("PROMPT INSTRUCTIONS:\n" + get_scoring_prompt_instructions())
    print("\nTRANSCRIPT:\n" + transcript)
    print("----------------------\n")
    
    # Mocking the LLM's assessment based on the transcript
    # The transcript has legal threats, churn risk, toxicity, and demands a supervisor.
    
    mock_breakdown = ScoreBreakdown(
        legal_points=40,       # Mentioned lawyer/lawsuit
        churn_points=30,       # Threatened to go to competitor
        toxicity_points=15,    # Severe abuse/shouting
        resolution_points=15   # Demanded supervisor
    )
    
    total_score = (
        mock_breakdown.legal_points + 
        mock_breakdown.churn_points + 
        mock_breakdown.toxicity_points + 
        mock_breakdown.resolution_points
    )
    
    # 0-25: Low Risk, 26-50: Monitor, 51-75: High Risk, 76-100: Critical Escalation
    if total_score <= 25:
        risk_category = "Low Risk"
    elif total_score <= 50:
        risk_category = "Monitor"
    elif total_score <= 75:
        risk_category = "High Risk"
    else:
        risk_category = "Critical Escalation"
        
    mock_analysis = EscalationAnalysis(
        escalation_score=total_score,
        risk_category=risk_category,
        escalation_reason="Customer explicitly threatened a lawsuit and mentioned contacting a lawyer (40 pts). Customer threatened to switch to a competitor (30 pts). Customer was highly toxic and abusive (15 pts). Customer demanded to speak to a supervisor as the issue was unresolved (15 pts). Total score is 100, which falls into the Critical Escalation category.",
        breakdown=mock_breakdown
    )
    
    return mock_analysis

if __name__ == "__main__":
    # Example Input Transcript
    sample_transcript = """
Customer: "I've been dealing with your garbage service for three weeks! If this isn't fixed today, I'm cancelling my account and going to Verizon! In fact, I'm calling my lawyer right now to sue you for breach of contract. Put your supervisor on the phone NOW before I lose my mind!"
Agent: "I'm so sorry to hear that, I'd be happy to help—"
Customer: "NO! Supervisor. Now. And getting my lawyer on the other line."
    """
    
    print(">> RUNNING EXAMPLE...\n")
    
    # Get Output
    result: EscalationAnalysis = analyze_escalation_risk_mock(sample_transcript)
    
    print(">> FINAL JSON OUTPUT FROM Pydantic Model:\n")
    print(result.model_dump_json(indent=2))
