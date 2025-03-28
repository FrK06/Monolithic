from src.web_rag_system import WebRAGSystem
import os
from dotenv import load_dotenv
import logging

# Suppress unnecessary logging
logging.basicConfig(level=logging.WARNING)

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    assert openai_api_key, "OpenAI API key not found in .env"

    rag_system = WebRAGSystem(openai_api_key)
    
    print("=== Web RAG System Demo ===\n")
    session_id = "demo-session"
    current_mode = "explore"

    while True:
        print(f"\nCurrent mode: {current_mode}")
        query = input("Enter your question (type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            print("\nExiting Demo...")
            break

        if query.lower().startswith("mode:"):
            new_mode = query.split(":", 1)[1].strip()
            current_mode = new_mode if new_mode in ["explore", "setup"] else current_mode
            continue

        try:
            # Retry 3 times with exponential backoff
            retries = 3
            response = None
            for attempt in range(retries):
                try:
                    response = rag_system.get_answer(
                        query=query,
                        thread_id=session_id,
                        mode=current_mode
                    )
                    print(f"\nResponse: {response}")
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    print(f"Retry {attempt + 1}/{retries}: {e}")
        except Exception as e:
            print(f"\nRequest failed: {str(e)}")

if __name__ == "__main__":
    main()


'''# Multimodal Web RAG System Demo
# ===============================
# This notebook demonstrates the multimodal capabilities of the enhanced Web RAG System

import os
import base64
import requests
from IPython.display import display, Image, Audio, HTML
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API URL - change this to match your deployment
API_URL = "http://localhost:8000"

# Helper functions for displaying images and audio
def display_image(image_data):
    """Display an image from base64 data or URL"""
    if image_data.startswith('http'):
        return display(Image(url=image_data))
    elif 'base64' in image_data:
        # Already has the data URI prefix
        img_data = image_data.split(',')[1]
        return display(Image(data=base64.b64decode(img_data)))
    else:
        # Raw base64 string
        return display(Image(data=base64.b64decode(image_data)))

def display_audio(audio_data):
    """Display audio from base64 data"""
    if 'base64' in audio_data:
        # Already has the data URI prefix
        audio_b64 = audio_data.split(',')[1]
        return display(Audio(data=base64.b64decode(audio_b64), autoplay=False))
    else:
        # Raw base64 string
        return display(Audio(data=base64.b64decode(audio_data), autoplay=False))

# Check API health
def check_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"API Status: {health_data['status']}")
            print(f"Version: {health_data['version']}")
            print(f"LLM Model: {health_data['llm_model']}")
            print(f"Speech Enabled: {health_data['speech_enabled']}")
            print(f"Image Enabled: {health_data['image_enabled']}")
            return True
        else:
            print(f"API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return False

# Chat with the system
def chat(message, thread_id=None, attached_images=None):
    """Send a chat message to the system"""
    try:
        data = {
            "content": message,
            "mode": "explore",
            "attached_images": attached_images or []
        }
        
        if thread_id:
            data["thread_id"] = thread_id
            
        response = requests.post(f"{API_URL}/api/chat/", json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"Assistant: {result['message']}")
        print(f"Tools used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
        
        return result["thread_id"]
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return None

# Generate an image
def generate_image(prompt, size="1024x1024", style="vivid", quality="standard"):
    """Generate an image using DALL-E"""
    try:
        data = {
            "prompt": prompt,
            "size": size,
            "style": style,
            "quality": quality
        }
        
        response = requests.post(f"{API_URL}/api/generate-image/", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["status"] == "success":
            print("Image generated successfully!")
            display_image(result["image"])
            return result["image"]
        else:
            print(f"Error: {result.get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

# Analyze an image
def analyze_image(image_data):
    """Analyze an image using GPT-4 Vision"""
    try:
        data = {
            "image": image_data
        }
        
        response = requests.post(f"{API_URL}/api/analyze-image/", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["status"] == "success":
            print("Image Analysis:")
            print(result["analysis"])
            return result["analysis"]
        else:
            print(f"Error: {result.get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None

# Process an image
def process_image(image_data, operation):
    """Process an image with operations like resize, crop, or grayscale"""
    try:
        data = {
            "image": image_data,
            "operation": operation
        }
        
        response = requests.post(f"{API_URL}/api/process-image/", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["status"] == "success":
            print(f"Image processed with {operation} successfully!")
            display_image(result["image"])
            return result["image"]
        else:
            print(f"Error: {result.get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Text to speech
def text_to_speech(text, voice="alloy"):
    """Convert text to speech"""
    try:
        data = {
            "text": text,
            "voice": voice
        }
        
        response = requests.post(f"{API_URL}/api/text-to-speech/", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["status"] == "success":
            print("Text converted to speech successfully!")
            display_audio(result["audio"])
            return result["audio"]
        else:
            print(f"Error: {result.get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

# Check if API is healthy
health_status = check_health()
if not health_status:
    print("API is not healthy. Please check your deployment and try again.")
else:
    print("API is healthy and ready for multimodal interactions!")

# Demo examples (uncomment to run):

# Example 1: Basic conversation
# thread_id = chat("What can you help me with today?")

# Example 2: Generate an image
# image_data = generate_image("A futuristic city with flying cars and tall skyscrapers")

# Example 3: Analyze an image (use image_data from previous step)
# if image_data:
#     analysis = analyze_image(image_data)

# Example 4: Process an image (make it grayscale)
# if image_data:
#     processed_image = process_image(image_data, "grayscale")

# Example 5: Chat with an attached image
# if image_data:
#     thread_id = chat("What do you see in this image?", attached_images=[image_data])

# Example 6: Text to speech
# audio_data = text_to_speech("This is a demonstration of the text to speech capability.")

# Example 7: Multimodal conversation
# if image_data:
#     thread_id = chat("Create a story about what you see in this image.", attached_images=[image_data])
#     if thread_id:
#         # Continue the conversation
#         thread_id = chat("Now create an image that represents the next chapter of the story.", thread_id=thread_id)'''
