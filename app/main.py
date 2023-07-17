import uvicorn
import requests
import json
import os
import numpy as np
import io
from PIL import Image
import base64

import time
SERVER_URI = os.getenv("SERVER_URI", 'http://localhost:8001/get_raw_tiles/')


def process_images(server_uri: str):
    while True:
        # TODO: Расширить API на сервере чтобы можно было запрашивать плитки за определенное время или даже по координатам
        payload = json.dumps({
            "map_name": "landsat8",
            "network_name": "RandomForestClassifier",
            "cnt": 10
        })

        # Запрос необработанных плиток
        response = requests.get(server_uri, data=payload)

        try:
            data = response.json()
        except requests.exceptions.RequestException:
            print(response.text)
            time.sleep(1)
            continue

        tiles = data["tiles"]
        ttt = base64.decodebytes( base64.b64decode(tiles[0]["image"] ))
        img = Image.open(io.BytesIO(ttt))
        f_img_array = np.asarray(img)
        t = 7

if __name__ == "__main__":
    process_images(SERVER_URI)
