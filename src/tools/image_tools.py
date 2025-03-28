# src/tools/image_tools.py

import os
import base64
import tempfile
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import logging
import io
import requests
import re

# Import OpenAI for DALL-E and GPT-4 Vision
from openai import OpenAI

# Import PIL for image processing
from PIL import Image

logger = logging.getLogger(__name__)

class ImageTools:
    def __init__(self):
        """Initialize image tools with OpenAI client."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.image_size = "1024x1024"  # Default size
        self.image_quality = "standard"  # Options: standard, hd
        self.image_style = "vivid"  # Options: vivid, natural
        
    def generate_image(self, prompt: str) -> Tuple[str, Optional[str]]:
        try:
            # Create a fresh client for this request only
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Generate image with minimal context
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=self.image_size,
                quality=self.image_quality,
                style=self.image_style,
                response_format="b64_json",
                n=1
            )
            
            image_b64 = response.data[0].b64_json
            return f"data:image/png;base64,{image_b64}", None
        except Exception as e:
            logger.error(f"Error in generate_image: {str(e)}")
            return "", f"Image generation error: {str(e)}"
    
    def analyze_image(self, image_data: Union[str, bytes]) -> Tuple[str, Optional[str]]:
        """
        Analyze an image using GPT-4 Vision with enhanced table formatting.
        
        Args:
            image_data: Either base64 encoded image or raw image bytes
            
        Returns:
            Tuple with analysis text and any error message
        """
        try:
            # Handle image data
            if isinstance(image_data, bytes):
                # Convert bytes to base64
                b64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                # Already a base64 string, remove data URI if present
                b64_image = image_data
                if "base64," in b64_image:
                    b64_image = b64_image.split("base64,")[1]
            
            # Check if the image likely contains a table
            # If it does, use specialized table extraction, otherwise use general analysis
            if self._likely_contains_table(image_data):
                logger.info("Image likely contains a table, using specialized extraction")
                return self.extract_table_from_image(image_data)
            
            # Enhanced system prompt with specific table formatting instructions
            system_prompt = """You are a helpful assistant that analyzes images. 
            Describe the image in detail including objects, people, text, and other relevant information.
            
            IMPORTANT FORMATTING INSTRUCTIONS:
            
            1. When you detect tables in images, ALWAYS format your response using proper markdown table syntax:
               
               ```
               | Column1 | Column2 | Column3 |
               |---------|---------|---------|
               | Data1   | Data2   | Data3   |
               ```
               
            2. NEVER present table data as a single line with pipe separators.
            
            3. Always use alignment and spacing for readability.
            
            4. For spreadsheets or financial data, ensure all numbers are properly aligned and formatted.
            
            5. Always explain what the table represents after presenting it.
            """
            
            # Use GPT-4 Vision to analyze the image
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze this image in detail. If it contains a table, format it properly as a markdown table."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }}
                    ]}
                ],
                max_tokens=1500
            )
            
            # Get the analysis
            analysis = response.choices[0].message.content
            
            # Post-process to ensure proper table formatting
            analysis = self.format_table_response(analysis)
            
            return analysis, None
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return "", f"Image analysis error: {str(e)}"
    
    def _likely_contains_table(self, image_data: Union[str, bytes]) -> bool:
        """
        Determines if an image likely contains a table by doing a quick analysis.
        This is a heuristic method to decide whether to use specialized table extraction.
        
        Args:
            image_data: Either base64 encoded image or raw image bytes
            
        Returns:
            Boolean indicating if the image likely contains a table
        """
        try:
            # Handle image data
            if isinstance(image_data, bytes):
                # Convert bytes to base64
                b64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                # Already a base64 string, remove data URI if present
                b64_image = image_data
                if "base64," in b64_image:
                    b64_image = b64_image.split("base64,")[1]
            
            # Quick check using GPT-4 Vision
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a table detection assistant. Your only job is to determine if an image contains a table or grid of data. Respond with only 'YES' if the image contains a table or data grid, 'NO' if it does not."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Does this image contain a table or grid of data? Answer only YES or NO."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }}
                    ]}
                ],
                max_tokens=10
            )
            
            # Check if response indicates a table
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
            
        except Exception as e:
            logger.error(f"Error in _likely_contains_table: {str(e)}")
            # Default to False if detection fails
            return False
    
    def format_table_response(self, text: str) -> str:
        """
        Detect potential table data in text responses and convert to proper markdown tables.
        
        Args:
            text: The text response from image analysis
            
        Returns:
            Formatted text with proper markdown tables
        """
        lines = text.split('\n')
        formatted_lines = []
        table_mode = False
        header_processed = False
        table_started = False
        
        for i, line in enumerate(lines):
            # Skip existing markdown code blocks
            if line.strip().startswith('```') and not table_started:
                table_started = True
                formatted_lines.append(line)
                continue
            elif line.strip().startswith('```') and table_started:
                table_started = False
                formatted_lines.append(line)
                continue
            elif table_started:
                formatted_lines.append(line)
                continue
                
            # Check if this line might be part of a table (has multiple pipe characters)
            pipe_count = line.count('|')
            
            # Detect table header rows by looking for multiple pipes
            if pipe_count > 2 and not table_mode:
                table_mode = True
                header_processed = False
                
                # Clean the line and format as a proper header
                cells = [cell.strip() for cell in re.split(r'\s*\|\s*', line) if cell.strip()]
                formatted_lines.append(f"| {' | '.join(cells)} |")
            
            # Add separator row after header
            elif table_mode and not header_processed:
                # Count cells in the header
                header_cells = len(formatted_lines[-1].split('|')) - 2
                if header_cells > 0:
                    formatted_lines.append('|' + '|'.join(['---' for _ in range(header_cells)]) + '|')
                    header_processed = True
                
                # Process current line if it has pipe characters
                if pipe_count > 2:
                    cells = [cell.strip() for cell in re.split(r'\s*\|\s*', line) if cell.strip()]
                    formatted_lines.append(f"| {' | '.join(cells)} |")
                elif line.strip():
                    # This is not a table row
                    table_mode = False
                    formatted_lines.append(line)
            
            # Continue processing table rows
            elif table_mode and pipe_count > 2:
                cells = [cell.strip() for cell in re.split(r'\s*\|\s*', line) if cell.strip()]
                formatted_lines.append(f"| {' | '.join(cells)} |")
            
            # Empty line inside table - keep it
            elif table_mode and not line.strip():
                formatted_lines.append(line)
                
            # End of table
            elif table_mode:
                table_mode = False
                formatted_lines.append(line)
                
            # Regular text line
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def process_image(self, image_data: Union[str, bytes], operation: str) -> Tuple[str, Optional[str]]:
        """
        Process an image using various operations.
        
        Args:
            image_data: Either base64 encoded image or raw image bytes
            operation: Operation to perform (resize, crop, grayscale, etc.)
            
        Returns:
            Tuple with processed image base64 and any error message
        """
        try:
            # Convert image data to PIL Image
            if isinstance(image_data, str):
                # Handle base64 string
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Handle raw bytes
                image = Image.open(io.BytesIO(image_data))
            
            # Perform requested operation
            if operation == "grayscale":
                processed_image = image.convert("L")
            elif operation.startswith("resize_"):
                # Format should be "resize_WIDTHxHEIGHT"
                try:
                    dimensions = operation.split("_")[1]
                    width, height = map(int, dimensions.split("x"))
                    processed_image = image.resize((width, height))
                except (IndexError, ValueError):
                    return "", "Invalid resize format. Use 'resize_WIDTHxHEIGHT'."
            elif operation.startswith("crop_"):
                # Format should be "crop_LEFT,TOP,RIGHT,BOTTOM"
                try:
                    coords = operation.split("_")[1]
                    left, top, right, bottom = map(int, coords.split(","))
                    processed_image = image.crop((left, top, right, bottom))
                except (IndexError, ValueError):
                    return "", "Invalid crop format. Use 'crop_LEFT,TOP,RIGHT,BOTTOM'."
            elif operation == "thumbnail":
                processed_image = image.copy()
                processed_image.thumbnail((256, 256))
            else:
                return "", f"Unsupported operation: {operation}"
            
            # Convert processed image to base64
            buffer = io.BytesIO()
            processed_image.save(buffer, format="PNG")
            b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Return the processed image
            return f"data:image/png;base64,{b64_image}", None
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return "", f"Image processing error: {str(e)}"
    
    def download_image(self, url: str) -> Tuple[bytes, Optional[str]]:
        """
        Download an image from a URL.
        
        Args:
            url: URL of the image to download
            
        Returns:
            Tuple with image bytes and any error message
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if content type is an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return b"", f"URL does not point to an image. Content-Type: {content_type}"
            
            # Return the image data
            return response.content, None
            
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return b"", f"Image download error: {str(e)}"
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """
        Encode image data to base64 string for frontend.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Base64 encoded image string with appropriate data URI prefix
        """
        try:
            b64_encoded = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_encoded}"
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise ValueError(f"Failed to encode image data: {str(e)}")
    
    def decode_base64_image(self, base64_image: str) -> bytes:
        """
        Decode base64 image data.
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Decoded image bytes
        """
        try:
            # Remove potential prefix
            if 'base64,' in base64_image:
                base64_image = base64_image.split('base64,')[1]
            
            # Decode base64 string to bytes
            return base64.b64decode(base64_image)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise ValueError(f"Failed to decode image data: {str(e)}")
            
    def extract_table_from_image(self, image_data: Union[str, bytes]) -> Tuple[str, Optional[str]]:
        """
        Specialized method for extracting and formatting tabular data from images.
        
        Args:
            image_data: Either base64 encoded image or raw image bytes
            
        Returns:
            Tuple with formatted table markdown and any error message
        """
        try:
            # Handle image data
            if isinstance(image_data, bytes):
                # Convert bytes to base64
                b64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                # Already a base64 string, remove data URI if present
                b64_image = image_data
                if "base64," in b64_image:
                    b64_image = b64_image.split("base64,")[1]
            
            # Very specific system prompt for table extraction
            system_prompt = """You are a specialized table extraction assistant. Your only job is to:
            
            1. Extract tables from images 
            2. Format them as proper markdown tables
            3. Maintain the exact structure and all data from the original
            
            ALWAYS follow this format for tables:
            
            ```markdown
            | Header1 | Header2 | Header3 |
            |---------|---------|---------|
            | Data1   | Data2   | Data3   |
            | Data4   | Data5   | Data6   |
            ```
            
            IMPORTANT RULES:
            - NEVER respond with explanations before the table
            - Start your response with the markdown table
            - Ensure all columns and rows align properly
            - For number columns, align values and maintain decimal precision
            - After the table, briefly describe what the table represents
            - For financial or numerical tables, ensure all numbers are accurately transcribed
            """
            
            # Use GPT-4 Vision specifically for table extraction
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract and format this table precisely as markdown. Do not add any text before the table itself."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }}
                    ]}
                ],
                max_tokens=1500
            )
            
            # Get the table extraction
            extraction = response.choices[0].message.content
            
            # Ensure it's properly formatted
            extraction = self.format_table_response(extraction)
            
            return extraction, None
            
        except Exception as e:
            logger.error(f"Error in extract_table_from_image: {str(e)}")
            return "", f"Table extraction error: {str(e)}"
