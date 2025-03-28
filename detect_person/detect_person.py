import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

# Set environment variables. You can get them from your machine's CONNECT tab
api_key = os.getenv('VIAM_API_KEY') or ''
api_key_id = os.getenv('VIAM_API_KEY_ID') or ''
address = os.getenv('VIAM_ADDRESS') or ''

model_name = "person-detection-model"
camera_name = "logitech-webcam"

async def connect():
    opts = RobotClient.Options.with_api_key(
      api_key=api_key,
      api_key_id=api_key_id
    )
    return await RobotClient.at_address(address, opts)


async def main():
    machine = await connect()
    person_detector = VisionClient.from_robot(machine, "person-detection")
    cam = Camera.from_robot(robot=machine, name="logitech-webcam")

    while True:
        img = await cam.get_image(mime_type="image/jpeg")
        detections = await person_detector.get_detections(img)

        found = False
        for d in detections:
            if d.confidence > 0.85 and d.class_name.lower() == "person":
                print("This is a person!")
                found = True
        if found == False:
            print("There's nobody here!")
    await asyncio.sleep(5)
    await machine.close()

if __name__ == '__main__':
    asyncio.run(main())