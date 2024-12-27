# image_handling.py
import asyncio
import base64
import io
import json  # Added json import
import numpy as np
from PIL import Image
from .logger import log_exception, log_message, log_debug


class ImageHandler:
    def __init__(self):
        self.image_ready = asyncio.Event()
        self.image_np = None
        self.websocket = None

    @staticmethod
    def ensure_base64_padding(base64_string):
        """Ensure the base64 string has correct padding."""
        missing_padding = len(base64_string) % 4
        if missing_padding != 0:
            base64_string += '=' * (4 - missing_padding)
        return base64_string

    def set_websocket(self, websocket):
        self.websocket = websocket

    async def send_message_to_unity(self, channel, data):
        if self.websocket:
            try:
                response = json.dumps({"channel": channel, "data": data})
                await self.websocket.send(response)
                log_message(f"Sent message to Unity on channel {channel}")
            except Exception as e:
                log_exception(f"Error sending message to Unity: {str(e)}")

    async def handle_image_streaming(self, msg_obj):
        try:
            # Log the incoming message structure
            log_debug(f"Received message object type: {type(msg_obj)}")
            if isinstance(msg_obj, dict):
                log_debug(f"Message keys: {list(msg_obj.keys())}")
            
            # Validate message object
            if not isinstance(msg_obj, dict) or 'data' not in msg_obj:
                raise ValueError("Invalid message format: missing 'data' field")

            # Get the data
            data = msg_obj['data']
            log_debug(f"Data type: {type(data)}")
            
            # If data is already bytes or numpy array, process directly
            if isinstance(data, (bytes, np.ndarray)):
                image_bytes = data if isinstance(data, bytes) else data.tobytes()
            else:
                # Handle string data (base64)
                if not isinstance(data, str):
                    raise ValueError(f"Invalid data type: {type(data)}")
                
                # Ensure base64 padding
                data = self.ensure_base64_padding(data)

                # Try to decode base64 directly first
                try:
                    image_bytes = base64.b64decode(data)
                except:
                    # If direct decode fails, try to handle data with prefix
                    if ',' in data:
                        _, image_data_base64 = data.split(',', 1)
                        image_data_base64 = self.ensure_base64_padding(image_data_base64)
                        image_bytes = base64.b64decode(image_data_base64)
                    else:
                        raise ValueError("Invalid data format: not valid base64")

            # Process the image bytes
            try:
                image = Image.open(io.BytesIO(image_bytes))
                log_debug(f"Successfully opened image: {image.size} {image.mode}")
            except Exception as e:
                raise ValueError(f"Failed to open image: {str(e)}")

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array
            self.image_np = np.array(image)
            log_debug(f"Converted to numpy array shape: {self.image_np.shape}")

            # Save image
            try:
                image.save('output_image.jpg', 'JPEG', quality=100)
                log_message("Image saved successfully")
            except Exception as e:
                log_debug(f"Failed to save image: {str(e)}")
                # Continue even if save fails

            self.image_ready.set()
            log_message("Image processing completed successfully")

        except ValueError as e:
            log_exception(f"Validation error: {str(e)}")
            await self.send_message_to_unity("Error", f"Image processing error: {str(e)}")
            self.image_ready.set()
        except Exception as e:
            log_exception(f"Unexpected error: {str(e)}")
            await self.send_message_to_unity("Error", "Unexpected error during image processing")
            self.image_ready.set()

    def get_image(self):
        return self.image_np
