from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import pandas as pd

from fastai import *
from fastai.vision import *

import requests

## Download large file
# model_id = '1-1O1CsEDSOa0fu--OcDQwsjTRvWFi0xm'
# model_file_name = 'model'
# train_model_destination = path/'models'/f'{model_file_name}.pkl'
# def download_file_from_google_drive(id, destination):
#     def get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith('download_warning'):
#                 return value

#         return None

#     def save_response_content(response, destination):
#         CHUNK_SIZE = 32768

#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk: # filter out keep-alive new chunks
#                     f.write(chunk)

#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)

## Download DSFD model trained on WIDERFace to ./weights

model_file_name = 'model'

# DESTINATION FILE ON YOUR DISK

# download_file_from_google_drive(trained_model_id, train_model_destination)


model_file_url = 'https://drive.google.com/uc?export=download&id=1_V09p5Txc5S3sPR5ig_jV7JEGcmV9g8Z'
model_file_name = 'model'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):

    def get_confirm_token(response):
      for key, value in response.cookies.items():
          if key.startswith('download_warning'):
              return value

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            token = get_confirm_token(response)

            if token:
                params = { 'id' : id, 'confirm' : token }
                data = session.get(url, params = params, stream = True)

            save_response_content(data, dest)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner(path/'models/', 'model.pkl')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

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
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

