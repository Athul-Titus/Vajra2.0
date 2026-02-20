import json
import logging
import typing
import speech_recognition as sr
from langdetect import detect, LangDetectException
from transformers import pipeline


class MockAudioSentimentEngine:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.recognizer = sr.Recognizer()

        try:
            self.analyzer = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def transcribe_audio_file(self, file_path):
        """Convert audio file to text"""
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio)
            print(f"📝 Transcribed Text: {text}")
            return text

        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return None

    def detect_language(self, text):
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    def map_sentiment(self, label: str) -> tuple[str, float]:
        star_rating = int(label.split()[0])
        normalized_score = float((star_rating - 3) / 2.0)

        categories = {
            1: "Highly Negative",
            2: "Negative",
            3: "Neutral",
            4: "Slightly Positive",
            5: "Positive"
        }

        return categories.get(star_rating, "Neutral"), normalized_score

    def analyze_audio(self, file_path: str) -> dict:
        text = self.transcribe_audio_file(file_path)

        if not text:
            return {"error": "Could not transcribe audio."}

        language = self.detect_language(text)
        
        # Bypass type checking confusion
        _out: typing.Any = self.analyzer(str(text)[:512])  # type: ignore
        result: dict = _out[0] if isinstance(_out, list) else _out

        label = str(result.get("label", "3 stars"))
        conf = float(result.get("score", 0.0))

        category, base_score = self.map_sentiment(label)
        score = float(base_score)

        return {
            "audio_file": file_path,
            "detected_language": language,
            "transcribed_text": text,
            "overall_sentiment": category,
            "sentiment_score": float(f"{score:.2f}"),
            "confidence": float(f"{conf:.2f}"),
            "escalation_risk": bool(score <= -0.5)
        }


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # Fix Windows console encoding issues with emojis
    if sys.stdout.encoding != 'utf-8':
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')

    engine = MockAudioSentimentEngine()

    if len(sys.argv) > 1:
        input_data = sys.argv[1]
    else:
        print("\n--- Sentiment Analysis Engine ---")
        input_data = input("Enter an audio file path (e.g., .wav) OR type text to analyze: ")

    if not input_data.strip():
        print("No input provided. Exiting.")
        sys.exit(0)

    # Check if input looks like an audio file path
    if input_data.lower().endswith('.wav') or (os.path.exists(input_data) and os.path.isfile(input_data)):
        print(f"\nProcessing audio file: {input_data}")
        result = engine.analyze_audio(input_data)
    else:
        # Process as direct text input
        print(f"\nProcessing text input: '{input_data}'")
        language = engine.detect_language(input_data)
        
        # Bypass type checking confusion
        _out: typing.Any = engine.analyzer(str(input_data)[:512])  # type: ignore
        sentiment_result: dict = _out[0] if isinstance(_out, list) else _out
        
        label = str(sentiment_result.get("label", "3 stars"))
        conf = float(sentiment_result.get("score", 0.0))
        
        category, base_score = engine.map_sentiment(label)
        score = float(base_score)

        result = {
            "input_text": input_data,
            "detected_language": language,
            "overall_sentiment": category,
            "sentiment_score": float(f"{score:.2f}"),
            "confidence": float(f"{conf:.2f}"),
            "escalation_risk": bool(score <= -0.5)
        }

    print("\n📊 Sentiment Result:")
    print(json.dumps(result, indent=4))