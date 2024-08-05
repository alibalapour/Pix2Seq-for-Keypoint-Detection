import argparse
import datetime
import os
import time

from fastapi import FastAPI, UploadFile, File
import cv2
import uvicorn
import uuid
import yaml

from model import ModelInterface
from utils import load_configs

# create FastAPI app
app = FastAPI()


def get_arguments():
    """
    creates a parser and return arguments

    @return: given arguments
    """
    parser = argparse.ArgumentParser("Corner Detection API")

    parser.add_argument('--result-path',
                        default='../../best models/[dataset V4.2.0] model 7 - valid loss = 1.04/2023-02-15__07-11-20',
                        type=str)
    parser.add_argument('--checkpoint-name', default='best_model.pth', type=str)

    arguments = parser.parse_args()
    return arguments


# get parser
args = get_arguments()

# get and open config
config_path = os.path.join(args.result_path, 'config.yaml')
with open(config_path) as file:
    dict_config = yaml.full_load(file)
configs = load_configs(dict_config)

# build an interface for inference
interface = ModelInterface(configs, model_path=os.path.join(args.result_path, 'checkpoints', args.checkpoint_name))


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    do inference on an image

    @param file: The file we want to do inference on it.
    @return: Returns a response.
    """
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    img_path = f"{file.filename}"
    with open(img_path, "wb") as f:
        f.write(contents)

    img = cv2.imread(img_path)[..., ::-1]

    start_time = time.time()
    img, len_bbox = interface.inference(img)
    response = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'inference time': time.time() - start_time,
        'len_bbox': len_bbox,
    }
    os.remove(img_path)

    return response


if __name__ == '__main__':
    uvicorn.run("app:app", reload=True)
