import os
from twilio.rest import Client
from app.core.logger import logger

def send_emergency_sms(to_number, message_body):
    """Send emergency SMS using Twilio."""
    try:
        if not to_number:
            logger.error("Recipient phone number is missing")
            return False

        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_FROM_NUMBER')
        
        if not all([account_sid, auth_token, from_number]):
            logger.error("Twilio credentials (SID, Token, or From Number) missing in environment")
            return False
            
        client = Client(account_sid, auth_token)
        
        # Normalize and ensure to_number has country code
        to_number = str(to_number).strip()
        if not to_number.startswith('+'):
            # Default to India if no country code provided
            to_number = '+91' + to_number.lstrip('0')
            
        logger.info(f"Sending emergency SMS from {from_number} to {to_number}")
        
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )
        
        logger.info(f"Emergency SMS sent successfully. SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Twilio SMS delivery failure: {str(e)}")
        return False
