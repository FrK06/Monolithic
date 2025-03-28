# src/tools/rag_tools.py

from typing import List, Dict, Optional
from langchain.tools import tool
from src.tools.web_scraper import WebScraper
from src.tools.web_searcher import WebSearcher
from src.tools.twilio import TwilioService
from src.tools.speech_tools import SpeechTools
from src.tools.image_tools import ImageTools  # Import the image tools
from urllib.parse import urlparse

class RAGTools:
    def __init__(self):
        self.scraper = WebScraper()
        self.searcher = WebSearcher()
        self.twilio = TwilioService()
        self.speech = SpeechTools()
        self.image = ImageTools()  # Initialize ImageTools
        
    def get_tools(self):
        @tool
        def search_web(query: str) -> list:
            """Search the web using Google and provide comprehensive results"""
            results = self.searcher.search(query, max_results=10)
            
            # Always return at least 5 results if available
            if len(results) < 5:
                # Expand search with a broader query
                broader_query = ' '.join(query.split()[:3]) + " recent news"
                additional_results = self.searcher.search(broader_query)
                results.extend([r for r in additional_results if r['link'] not in [x['link'] for x in results]])
                
            # Format results for better readability and context
            formatted_results = []
            for r in results[:5]:  # Limit to 5 results
                source = urlparse(r['link']).netloc
                # Clean up the title and snippet
                title = r['title'].replace(' | ', ' - ').replace(' - ', ' - ')
                snippet = r['snippet'].replace('...', '').strip()
                formatted_result = {
                    "link": r['link'],
                    "title": title,
                    "snippet": snippet,
                    "source": source
                }
                formatted_results.append(formatted_result)
                
            return formatted_results
            
        @tool
        def scrape_webpage(url: str) -> str:
            """Extract content from a webpage"""
            return self.scraper.scrape_url(url)
        
        @tool
        def send_sms(recipient: str, message: str) -> str:
            """Send an SMS message via Twilio"""
            return self.twilio.send_sms(recipient, message)
        
        @tool
        def make_call(recipient: str, message: str = None) -> str:
            """
            Initiate a voice call to the user's phone number with an optional message.
            
            Args:
                recipient: The phone number to call (with or without + prefix)
                message: The message to speak when the call connects
                
            Returns:
                A string indicating success or failure of the call
            """
            return self.twilio.make_call(recipient, message)
        
        @tool
        def speak_text(text: str) -> str:
            """Convert text to speech and return audio URL or base64 data"""
            audio_data, error = self.speech.text_to_speech(text)
            if error:
                return f"Error generating speech: {error}"
            return "Speech generated successfully"
        
        @tool
        def generate_image(prompt: str) -> str:
            """
            Generate an image based on a text description using DALL-E.
            
            Args:
                prompt: Detailed description of the image to generate
                
            Returns:
                Base64 encoded image data or error message
            """
            try:
                # Call the image generation directly without the full conversation context
                # This prevents token limit errors
                image_data, error = self.image.generate_image(prompt)
                
                if error:
                    return f"Error generating image: {error}"
                
                # Return the image with Markdown image syntax for direct rendering
                return f"![Generated Image of {prompt}]({image_data})"
            except Exception as e:
                return f"Error generating image: {str(e)}"
        
        @tool
        def analyze_image(image_reference: str) -> str:
            """
            Analyze an image and describe its contents using GPT-4 Vision.
            
            Args:
                image_reference: URL, base64 data, or image reference token
                
            Returns:
                Detailed description of the image contents
            """
            # Handle different types of image references
            if image_reference.startswith("http"):
                # Download image from URL
                image_bytes, error = self.image.download_image(image_reference)
                if error:
                    return f"Error downloading image: {error}"
                analysis, error = self.image.analyze_image(image_bytes)
            else:
                # Assume it's base64 or a reference token
                analysis, error = self.image.analyze_image(image_reference)
            
            if error:
                return f"Error analyzing image: {error}"
            return analysis
        
        @tool
        def process_image(image_reference: str, operation: str) -> str:
            """
            Process an image with operations like resize, crop, or convert to grayscale.
            
            Args:
                image_reference: URL, base64 data, or image reference token
                operation: Operation to perform (grayscale, resize_WxH, crop_L,T,R,B, thumbnail)
                
            Returns:
                Base64 encoded processed image or error message
            """
            # Handle different types of image references
            if image_reference.startswith("http"):
                # Download image from URL
                image_bytes, error = self.image.download_image(image_reference)
                if error:
                    return f"Error downloading image: {error}"
                processed_image, error = self.image.process_image(image_bytes, operation)
            else:
                # Assume it's base64 or a reference token
                processed_image, error = self.image.process_image(image_reference, operation)
            
            if error:
                return f"Error processing image: {error}"
            return f"Image processed successfully. Use the following token to reference this image: [IMAGE:{processed_image[:30]}...]"
        
        # Return all tools
        return [
            search_web, 
            scrape_webpage, 
            send_sms, 
            make_call, 
            speak_text, 
            generate_image, 
            analyze_image, 
            process_image
        ]
