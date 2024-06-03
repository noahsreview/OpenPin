import openai
import pyttsx3
import whisper
import time
import sounddevice as sd
import numpy as np

# Initialize the Whisper model
model = whisper.load_model("tiny")

def record_audio(duration=5, sample_rate=16000):
    print("Please start speaking...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)

def transcribe_audio(audio, sample_rate=16000):
    print("Transcribing audio...")
    result = model.transcribe(audio, fp16=False)
    return result["text"]

def send_message_to_ollama(message, context=None):
    openai.api_base = "http://localhost:11434/v1"  # Custom endpoint for Ollama
    openai.api_key = "AAAAC3NzaC1lZDI1NTE5AAAAIPrSxi1np/FDEcJSl6yejdLc/cSqHJ3SItX5+0Ks0szg"  # Custom API key for Ollama

    messages = [
        {"role": "system", "content": "You are a prototype of an AI pin by the name open pin, please try to keep messages short and sweet to avoid too much information coming over at once and to improve processing time. Speak humanly and avoid saying too much about yourself being an AI to improve interactions and make the user experience feel smoother and cleaner. Please try and keep your responses to 30 words or less. Please avoid repeating messages. since message history hasn't yet been implemented if someone asks to broad of a question or try to refer to a previous message let them know message history hasn't been implemented and let them know that message history hasn't been implemented and that it should be implemented soon. If they ask where to get news about open pin or to send feedback tell them to go to the official openpin github"},
        {"role": "user", "content": message}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="llama3",  # Set model to Llama3
            messages=messages,
            stream=True
        )
        return response
    except Exception as e:
        return {"error": "Failed to connect to Ollama API", "exception": str(e)}

def get_voice_input():
    audio = record_audio()
    text = transcribe_audio(audio)
    print(f"Transcription result: {text}")
    return text

def clean_response(response):
    # Remove characters that may not be suitable for TTS
    cleaned_response = response.replace("*", "")
    return cleaned_response

def main():
    print("Welcome to the Ollama Message Sender!")
    context = None
    previous_response = ""  # Store the previous response to avoid duplication

    while True:
        try:
            message = get_voice_input()
            if message:
                print(f"You said: {message}")  # Print the recognized message after the user has stopped talking
                responses = send_message_to_ollama(message, context)

                ollama_responded = True  # Flag to track if Ollama responded successfully
                if isinstance(responses, dict) and "error" in responses:
                    print(f"Error: {responses['error']} (Exception: {responses['exception']})")
                    ollama_responded = False  # Set the flag to False if Ollama failed to respond
                    # Initialize the text-to-speech engine
                    engine = pyttsx3.init()
                    engine.say("Failed to connect to Ollama API")
                    engine.runAndWait()
                    time.sleep(5)
                else:
                    full_message = ""
                    for response in responses:
                        content = response['choices'][0]['delta'].get('content', '')
                        full_message += content

                    cleaned_message = clean_response(full_message)
                    # Print the final output only
                    print(f"\nassistant: {cleaned_message}")

                    # Initialize the text-to-speech engine
                    engine = pyttsx3.init()
                    voices = engine.getProperty('voices')
                    engine.setProperty('voice', voices[1].id)
                    engine.say(cleaned_message)
                    engine.runAndWait()

                    # Update previous response for the next request
                    previous_response = cleaned_message

                    # Update context for the next request
                    context = full_message
        except Exception as e:
            print(f"An error occurred: {e}")
            input("Press Enter to exit...")
            break

if __name__ == "__main__":
    main()
