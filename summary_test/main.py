import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

# Define the exact JSON structure for the insights you need
# Google's API natively understands Pydantic models for structured output
class ConversationInsight(BaseModel):
    summary: str
    primary_intents: List[str]
    key_topics: List[str]

def extract_insights(file_path: str) -> dict:
    """
    Analyzes a conversation file and extracts the summary, 
    customer intents, and key topics discussed.
    
    Args:
        file_path (str): Path to an audio file (.mp3, .wav) or text file (.txt).
        
    Returns:
        dict: A dictionary containing the summary, primary_intents, and key_topics.
    """
    # 1. Initialize the client
    # Configure your Gemini API Key here or via environment variables
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
        
    client = genai.Client(api_key=api_key)
    uploaded_file = None
    
    try:
        # 2. Upload the file to Gemini's API
        # (This supports direct audio/text without needing a separate pipeline)
        uploaded_file = client.files.upload(file=file_path)

        # 3. Construct the prompt
        prompt = """
        You are an expert conversation intelligence AI. Analyze this provided conversation.
        Please provide a concise summary, identify the primary customer intents, 
        and list the key topics or entities discussed.
        """

        # 4. Use the Gemini 2.5 Flash model (newest standard model for multimodal tasks)
        # Ask the model to generate the response matching our exact Pydantic schema
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ConversationInsight,
                temperature=0.2, # Low temperature for more analytical/factual extraction
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return {}

    finally:
        # Cleanup: Delete the file from Gemini servers
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"Cleanup error: {e}")

# --- Example Usage for you to test locally ---
if __name__ == "__main__":
    # Create a dummy text file to test with
    test_file = "test_conversation.txt"
    with open(test_file, "w") as f:
        f.write("Customer: Hi, my internet has been down for 3 hours. I need this fixed immediately, I work from home!\n")
        f.write("Agent: Have you tried restarting your router?\n")
        f.write("Customer: Yes, the yellow light is blinking. Can you send a technician?\n")
    
    print("Extracting insights...")
    result = extract_insights(test_file)
    if result:
        print(json.dumps(result, indent=2))
    
    # Cleanup test file
    if os.path.exists(test_file):
        os.remove(test_file)
