import sumolib
import os

common_paths = [
    r"C:\Program Files (x86)\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo",
    r"C:\Sumo",
]

found_path = None
for path in common_paths:
    if os.path.exists(path):
        print(f"Found SUMO at: {path}")
        found_path = path
        break

if found_path:
    os.environ["SUMO_HOME"] = found_path
    bin_path = os.path.join(found_path, "bin")
    os.environ["PATH"] += os.pathsep + bin_path
    try:
        sumo_binary = sumolib.checkBinary('sumo')
        print(f"SUMO binary found at: {sumo_binary}")
    except Exception as e:
        print(f"Error checking binary after setting path: {e}")
else:
    print("SUMO not found in common locations.")

print(f"SUMO_HOME: {os.environ.get('SUMO_HOME', 'Not Set')}")
