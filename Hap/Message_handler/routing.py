from modells.utills import process_type_of_milk, process_vehicle_number, process_batch_code, process_sku, process_hand_gesture
import asyncio
from .logger import log_message, log_exception

class ImageProcessor:
    def __init__(self, image_np, current_question=None, send_message_to_unity=None):
        self.image_np = image_np
        self.tasks = set()  # Track processing tasks
        self.current_question = current_question
        self.send_message_to_unity = send_message_to_unity
        self.image_path = 'output_image.jpg'

    async def process_field(self, field_name):
        try:
            log_message(f"Processing field: {field_name}")
            result = None

            # Process based on field type
            if field_name == "TYPE OF MILK":
                # Get the detection result
                result = await asyncio.to_thread(process_type_of_milk, self.image_path)
                log_message(f"Milk detection result: {result}")

                # Handle the result
                if result:
                    log_message(f"Valid result detected: {result}")
                    if self.current_question and "fieldValue" in self.current_question:
                        # Reset all options
                        for option in self.current_question["fieldValue"]:
                            option["isActive"] = False

                        # Find and set the matching option
                        found_match = False
                        for option in self.current_question["fieldValue"]:
                            if option["optionValue"] == result:
                                option["isActive"] = True
                                found_match = True
                                log_message(f"Found matching option: {result}")
                                break

                        # Send result to Unity
                        if found_match and self.send_message_to_unity:
                            log_message("Sending result to Unity...")
                            await self.send_message_to_unity("Result", self.current_question)
                        else:
                            log_message(f"No matching option found for result: {result}")
                    else:
                        log_message("Current question or fieldValue not properly initialized")
                else:
                    log_message("No valid detection result")
                    if self.send_message_to_unity:
                        await self.send_message_to_unity("Error", "No result found for field: TYPE OF MILK")

            elif field_name == "SKU":
                result = await asyncio.to_thread(process_sku, self.image_np)
            elif field_name == "VEHICLE NUMBER":
                result = await asyncio.to_thread(process_vehicle_number, self.image_np)
            elif field_name == "BATCH CODE":
                result = await asyncio.to_thread(process_batch_code, self.image_path)
            elif field_name == "TYPE OF LEAK":
                result = await process_hand_gesture(self.image_path)
	    

            # Handle result for other fields
            if result:
                log_message(f"Result detected: {result}")
                if self.current_question:
                    for option in self.current_question["fieldValue"]:
                        if option["optionValue"] == result:
                            option["isActive"] = True
                            break
                    if self.send_message_to_unity:
                        await self.send_message_to_unity("Result", self.current_question)

            log_message(f"Field {field_name} processed with result: {result}")
            return result

        except asyncio.TimeoutError:
            log_exception(f"Processing timeout for field: {field_name}")
            return None
        except Exception as e:
            log_exception(f"Error processing field {field_name}", exc_info=e)
            return None
