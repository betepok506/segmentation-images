import uvicorn
import requests
import json
import os
import numpy as np
import io
from PIL import Image
import base64
from app.model import SegmentationRandomForestClassifier

import time

SERVER_URI = os.getenv("SERVER_URI", 'http://localhost:8001/get_raw_tiles/')


def send_request(array_tiles):
    payload = json.dumps({"tiles": array_tiles})

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post("http://localhost:8001/add_tiles/",
                             data=payload,
                             # files={'upload_file': open(os.path.join(path_to_tiles, zoom, y, x), 'rb')},
                             headers=headers)
    try:
        data = response.json()
        print(data)
    except requests.exceptions.RequestException:
        print(response.text)


def process_images(server_uri: str):
    model = SegmentationRandomForestClassifier()
    model.load('./models/output/')

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
        input_images = []
        info_images = []
        for ind in range(len(tiles)):
            ttt = base64.decodebytes(base64.b64decode(tiles[ind]["image"]))
            img = Image.open(io.BytesIO(ttt))
            img_array = np.asarray(img)
            input_images.append(img_array)
            info_images.append({
                "x": tiles[ind]['x'],
                "y": tiles[ind]['y'],
                "z": tiles[ind]['z']
            })

        predict_masks = model.predict(input_images)
        message = []
        for ind in range(len(predict_masks)):
            img = predict_masks[ind]
            img = img.convert("RGBA")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            im_b64 = base64.b64encode(img_byte_arr).decode("utf8")
            message.append({
                "map_name": "fire",
                'x': info_images[ind]['x'],
                'y': info_images[ind]['y'],
                'z': info_images[ind]['z'],
                "image": im_b64,
            })

        send_request(message)
        message.clear()
        print("Done!")


if __name__ == "__main__":
    process_images(SERVER_URI)
