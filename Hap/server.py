import signal
import asyncio
import websockets
from Message_handler.image_handling import ImageHandler
from Message_handler.processing import MessageRouter
from Message_handler.routing import ImageProcessor
from Message_handler.logger import log_message, log_exception
import os
import socket

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

async def main(websocket):
    image_handler = ImageHandler()
    # Create ImageProcessor with None initially
    image_processor = ImageProcessor(None)
    
    # Create message router
    handler = MessageRouter(websocket, image_handler, image_processor)
    
    try:
        async for message in websocket:
            # Update the image_processor's image when new image is received
            if image_handler.image_np is not None:
                image_processor.image_np = image_handler.image_np
            await handler.handle_message(message)
    except websockets.ConnectionClosed:
        log_message("Connection closed.")
    except Exception as e:
        log_exception("Unexpected error in main", exc_info=e)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error determining local IP: {e}")
        local_ip = "127.0.0.1"
    finally:
        s.close()
    return local_ip

async def start_server():
    try:
        local_ip = get_local_ip()
        server = await websockets.serve(main, local_ip, 8000)
        log_message(f"Server started at ws://{local_ip}:8000")
        
        stop = asyncio.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        await stop.wait()
        server.close()
        await server.wait_closed()
    except Exception as e:
        log_exception("Error starting server", exc_info=e)

if __name__ == "__main__":
    asyncio.run(start_server())