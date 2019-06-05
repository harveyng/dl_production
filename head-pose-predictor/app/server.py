from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn
# import aiohttp
# import asyncio
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *

import requests
import grequests
# https://drive.google.com/uc?export=download&id=1_V09p5Txc5S3sPR5ig_jV7JEGcmV9g8Z
# Download large file
gdrive_model_id = '1_V09p5Txc5S3sPR5ig_jV7JEGcmV9g8Z'
model_file_name = 'model'
path = Path(__file__).parent
train_model_destination = path/'models'/f'{model_file_name}.pkl'


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# Download DSFD model trained on WIDERFace to ./weights

# model_file_name = 'model'

# # DESTINATION FILE ON YOUR DISK

# # download_file_from_google_drive(trained_model_id, train_model_destination)


# url = 'https://drive.google.com/uc?export=download&id=1_V09p5Txc5S3sPR5ig_jV7JEGcmV9g8Z'
# model_file_id = '1_V09p5Txc5S3sPR5ig_jV7JEGcmV9g8Z'
# model_file_name = 'model'



app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(id, dest):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    if dest.exists():
        return
    print("weight not exist --- start download")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={'id': id}, stream=True) as response:
            data = await response.read()
            token = get_confirm_token(response)

            if token:
                params = {'id': id, 'confirm': token}
                data = session.get(url, params=params, stream=True)

            save_response_content(data, dest)


def setup_learner():
    # await download_file(model_file_id, path/'models'/f'{model_file_name}.pkl')
    if not train_model_destination.exists():
        print("weight not exist --- start download")
        download_file_from_google_drive(
            gdrive_model_id, train_model_destination)	
    learn = load_learner(path/'models/', 'model.pkl')
    return learn


# loop = asyncio.get_event_loop()
learn = setup_learner()
# learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
# loop.close()


@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
		# print(prediction)
		plt.figure()
		plt.imshow(img)
		plt.annotate('25, 50', xy=(25, 50), xycoords='data',
								xytext=(0.5, 0.5), textcoords='figure fraction',
								arrowprops=dict(arrowstyle="->"))
		plt.scatter(25, 50, s=500, c='red', marker='o')
		plt.show()
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8080)
