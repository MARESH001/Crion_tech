import asyncio
import json
from Message_handler.image_handling import ImageHandler
from Message_handler.routing import ImageProcessor
from .logger import log_message, log_exception, log_debug

class MessageRouter:
    def __init__(self, websocket, ImageHandler, ImageProcessor):
        self.websocket = websocket
        self.image_handler = ImageHandler
        self.image_handler.set_websocket(websocket)
        self.image_processor = ImageProcessor
        self.current_question = None
        self.tasks = set()  # Track active tasks

    async def handle_message(self, message):
        log_debug(f"Received raw message: {message[:200]}...")
        try:
            msg_obj = json.loads(message)
            log_debug(f"Parsed message object: {msg_obj}")

            if "id" in msg_obj and "channel" in msg_obj:
                channel = msg_obj["channel"]
                await self.route_message(channel, msg_obj)
            else:
                log_exception("Invalid message format.")
        except json.JSONDecodeError as e:
            log_exception(f"Error decoding JSON: {e}")

    async def route_message(self, channel, msg_obj):
        log_message(f"Routing message on channel: {channel}")
        try:
            if channel == "Heartbeat":
                log_debug("Processing Heartbeat message")
                await self.send_message_to_unity("Heartbeat", "Ping")
            elif channel == "Question":
                log_debug("Processing Question message")
                # Create task for question handling
                task = asyncio.create_task(self.handle_question(msg_obj))
                task.add_done_callback(self.handle_task_result)
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
            elif channel == "ImageStreaming":
                log_debug("Processing ImageStreaming message")
                await self.image_handler.handle_image_streaming(msg_obj)
                log_debug("Image streaming completed")
            else:
                log_exception(f"Unknown channel: {channel}")
        except Exception as e:
            log_exception(f"Error routing message on channel {channel}", exc_info=e)

    def handle_task_result(self, task):
        try:
            task.result()  # This will raise any exceptions that occurred
        except Exception as e:
            log_exception(f"Task failed with error: {str(e)}")

    async def handle_question(self, msg_obj):
        try:
            self.current_question = json.loads(msg_obj["data"])
            field_name = self.current_question["fieldName"]

            try:
                await asyncio.wait_for(self.image_handler.image_ready.wait(), timeout=60.0)
                self.image_handler.image_ready.clear()
            except asyncio.TimeoutError:
                log_exception(f"Timeout waiting for image to be ready for field: {field_name}")
                await self.send_message_to_unity("Error", "Image processing timeout")
                return

            if self.image_handler.image_np is None:
                log_exception("Image not available in ImageHandler.")
                await self.send_message_to_unity("Error", "No image data available")
                return

            image_processor = ImageProcessor(self.image_handler.image_np)
            result = await image_processor.process_field(field_name)
            
            if result:
                for option in self.current_question["fieldValue"]:
                    if option["optionValue"] == result:
                        option["isActive"] = True
                await self.send_message_to_unity("Result", self.current_question)
            else:
                log_message(f"No result found for field: {field_name}")
                await self.send_message_to_unity("Error", f"No result found for field: {field_name}")
        
        except Exception as e:
            log_exception("Error handling question", exc_info=e)

    async def send_message_to_unity(self, channel, data):
        try:
            response = json.dumps({"channel": channel, "data": data})
            log_debug(f"Sending message to Unity on channel {channel}: {response[:200]}...")
            await self.websocket.send(response)
            log_message(f"Successfully sent message to Unity on channel {channel}")
        except Exception as e:
            log_exception(f"Error sending message to Unity: {str(e)}")