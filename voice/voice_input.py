import speech_recognition as sr

def transcribe_voice():
    """Captures voice input and returns transcribed text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("🎤 Speak now...")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Listening timed out. Try again."
        except sr.UnknownValueError:
            return "Sorry, could not understand."
        except sr.RequestError:
            return "API unavailable. Try again later."
