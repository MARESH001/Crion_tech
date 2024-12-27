from models.utils import process_type_of_milk, process_vehicle_number, process_batch_code, process_sku, process_hand_gesture
import asyncio
from .logger import log_message, log_exception

class ImageProcessor:
    def __init__(self, image_np, current_question=None, send_message_to_unity=None):
        self.image_np = image_np
        self.tasks = set()  # Track processing tasks
        self.current_question = current_question  # Assume current_question is passed during initialization
        self.send_message_to_unity = send_message_to_unity  # Function to send data to Unity

    async def process_field(self, field_name):
        try:
            log_message(f"Processing field: {field_name}")

            # Create the processing task based on field type
            if field_name == "TYPE OF MILK":
                result = await asyncio.to_thread(process_type_of_milk, self.image_np)
                if result:  # If a valid result is detected
                    print(f"Result detected: {result}")
                    if self.current_question:  # Check if current_question is defined
                        for option in self.current_question["fieldValue"]:
                            if option["optionValue"] == result:
                                option["isActive"] = True
                                break
                        if self.send_message_to_unity:  # Ensure function is available
                            await self.send_message_to_unity("Result", self.current_question)

            elif field_name == "SKU":
                result = await asyncio.to_thread(process_sku, self.image_np)

            elif field_name == "VEHICLE NUMBER":
                result = await asyncio.to_thread(process_vehicle_number, self.image_np)

            elif field_name == "BATCH CODE":
                result = await asyncio.to_thread(process_batch_code, self.image_np)

            elif field_name == "TYPE OF LEAK":
                # Since process_hand_gesture is already async, we don't need to_thread
                result = await process_hand_gesture(self.image_path)
                if result:  # If a valid result is detected
                    print(f"Result detected: {result}")
                    if self.current_question:  # Check if current_question is defined
                        for option in self.current_question["fieldValue"]:
                            if option["optionValue"] == result:
                                option["isActive"] = True
                                break
                        if self.send_message_to_unity:  # Ensure function is available
                            await self.send_message_to_unity("Result", self.current_question)

            else:
                log_exception(f"Unrecognized field: {field_name}")
                return None

            log_message(f"Field {field_name} processed with result: {result}")
            return result

        except asyncio.TimeoutError:
            log_exception(f"Processing timeout for field: {field_name}")
            return None
        except Exception as e:
            log_exception(f"Error processing field {field_name}", exc_info=e)
            return None
