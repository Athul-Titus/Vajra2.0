from pydantic import BaseModel

class ScoreBreakdown(BaseModel):
    """
    Detailed breakdown of the risk score based on specific categories.
    """
    legal_points: int
    churn_points: int
    toxicity_points: int
    resolution_points: int

class EscalationAnalysis(BaseModel):
    """
    Overall escalation analysis including the final score and risk category.
    """
    escalation_score: int
    risk_category: str
    escalation_reason: str
    breakdown: ScoreBreakdown

def get_scoring_prompt_instructions() -> str:
    """
    Returns the strict instructions for the LLM to calculate the escalation score.
    """
    return """Calculate an escalation_score (0-100) using this exact weighted matrix:

Legal Threat: 40 points (Threatens lawsuits, lawyers, or regulators)

Churn Risk: 30 points (Threatens to cancel or mentions competitor)

Toxicity: Up to 15 points (15 for severe abuse, 5 for frustration, 0 for polite)

Resolution Failure: 15 points (Issue unresolved or demands supervisor)

Sum the points to get the final score. Assign a risk_category based on the score (0-25: Low Risk, 26-50: Monitor, 51-75: High Risk, 76-100: Critical Escalation). Provide a detailed escalation_reason explaining the math."""

if __name__ == "__main__":
    import json
    
    print("--- Prompt Instructions ---")
    print(get_scoring_prompt_instructions())
    
    print("\n--- Pydantic Schema Outline ---")
    print(json.dumps(EscalationAnalysis.model_json_schema(), indent=2))
