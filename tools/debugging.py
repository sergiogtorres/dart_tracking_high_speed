import pyrealsense2 as rs

# Create a context object. This manages the connected devices
ctx = rs.context()

# Get all connected devices
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense device found.")
else:
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)

        print(f"Device: {name}")
        print(f"Serial Number: {serial}")
        print(f"Firmware Version: {firmware}")

        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"Device: {name} | USB Type: {usb_type}")
