# src/tools/twilio.py

from twilio.rest import Client
import os
import urllib.parse
import re

class TwilioService:
    def __init__(self):
        self.client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        self.sender = os.getenv('TWILIO_PHONE_NUMBER')
        
    def format_phone_number(self, phone_number: str) -> str:
        """
        Format phone numbers to E.164 format for Twilio.
        Handles UK numbers (07xxx) and converts them to +447xxx format.
        """
        # Remove any spaces, dashes, parentheses
        cleaned = re.sub(r'[\s\-\(\)]', '', phone_number)
        
        # Handle UK mobile numbers (07xxx)
        if cleaned.startswith('07') and len(cleaned) == 11:
            return '+44' + cleaned[1:]  # Convert 07xxx to +447xxx
            
        # Handle numbers with international prefix
        if cleaned.startswith('+'):
            return cleaned
            
        # Handle numbers with 00 international prefix
        if cleaned.startswith('00'):
            return '+' + cleaned[2:]
            
        # If number doesn't have a country code, assume it's UK
        if len(cleaned) == 10 or len(cleaned) == 11:
            # Remove leading 0 if present
            if cleaned.startswith('0'):
                cleaned = cleaned[1:]
            return f'+44{cleaned}'
            
        # If all else fails, just add + at the beginning
        return f'+{cleaned}'
        
    def send_sms(self, recipient: str, message: str) -> str:
        try:
            # Format the phone number
            formatted_recipient = self.format_phone_number(recipient)
            
            # Create and send SMS message
            sms = self.client.messages.create(
                body=message,
                from_=self.sender,
                to=formatted_recipient
            )
            # Return success message with SID
            return f"✅ SMS sent to {formatted_recipient} (SID: {sms.sid})"
        except Exception as e:
            # Handle exceptions and return error message
            return f"⚠️ Failed to send SMS: {e}"
        
    def make_call(self, recipient: str, message: str = None) -> str:
        try:
            # Format the phone number
            formatted_recipient = self.format_phone_number(recipient)
            
            # Get TwiML URL from environment variables
            twiml_url = os.getenv('TWIML_URL')
            
            if not twiml_url:
                # Provide clear error for missing URL
                return "⚠️ TwiML URL is not configured in environment variables"
            
            # If message is provided, add it as a query parameter
            if message:
                # Ensure the message is properly URL encoded
                encoded_message = urllib.parse.quote(message)
                
                # Add query parameter to the TwiML URL
                if '?' in twiml_url:
                    twiml_url = f"{twiml_url}&message={encoded_message}"
                else:
                    twiml_url = f"{twiml_url}?message={encoded_message}"
            
            # Initiate phone call
            call = self.client.calls.create(
                url=twiml_url,
                to=formatted_recipient,
                from_=self.sender
            )
            # Return success message with SID
            return f"✅ Call initiated to {formatted_recipient} with message: '{message or 'default message'}' (SID: {call.sid})"
        except Exception as e:
            # Handle exceptions and return error message
            return f"⚠️ Failed to make call: {e}"
