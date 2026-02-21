import os
import time
import google.generativeai as genai

API_KEY = "AIzaSyCkwySDnCXLRMdjr9-2AI1YZOsbthja9nc"
genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-2.5-flash"

def test_transcription(file_path):
    print(f"Uploading {file_path} to Gemini...")
    audio_file = genai.upload_file(path=file_path)
    
    print(f"Uploaded as {audio_file.uri}. Waiting for ACTIVE state...")
    # Poll until file is ACTIVE (important for larger files like MPEG)
    for _ in range(30):
        file_info = genai.get_file(audio_file.name)
        if file_info.state.name == "ACTIVE":
            print("File is ACTIVE. Proceeding...")
            break
        print(f"  File state: {file_info.state.name}. Waiting 2s...")
        time.sleep(2)
    else:
        raise RuntimeError("File did not become ACTIVE in time.")

    
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "Please provide a word-for-word transcript of this audio. "
        "There are multiple speakers in this audio. You must diarize the conversation "
        "by labeling each spoken segment with 'User 1:', 'User 2:', etc. "
        "Do not summarize, add any conversational filler, or describe the audio. "
        "Just write the speaker label followed by their spoken words."
    )
    
    print("Generating content...")
    resp = model.generate_content([audio_file, prompt])
    
    transcript = (resp.text or "").strip()
    
    print("\n--- RESPONSE RESULT ---")
    print(transcript)
    print("-----------------------\n")
    
    # Save to txt file with same name as audio
    base_name = os.path.splitext(file_path)[0]
    out_path = f"{base_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcript saved to {out_path}")
    
    genai.delete_file(audio_file.name)
    print("Cleaned up uploaded file.")

if __name__ == "__main__":
    import sys
    file = sys.argv[1] if len(sys.argv) > 1 else "athul1.ogg"
    test_transcription(file)
