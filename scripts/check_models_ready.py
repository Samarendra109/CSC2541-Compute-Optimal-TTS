import argparse
import os
import sys
import time

import requests


def check_models(controller_addr):
    print(f"Checking models at: {controller_addr}")
    try:
        response = requests.post(controller_addr + "/list_models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("Models registered:", models)
            return len(models) >= 2  # Adjust threshold if needed
        return False
    except Exception as e:
        print("Error:", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--controller_port")
    args = parser.parse_args()
    controller_addr = f"http://{args.host}:{args.controller_port}"
    for i in range(120):
        if check_models(controller_addr):
            print("All models ready!")
            sys.exit(0)
        print(f"Waiting for models... {i}/120")
        time.sleep(10)

    print("Timed out waiting for models to initialize")
    sys.exit(1)


if __name__ == "__main__":
    main()
