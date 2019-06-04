from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

# import cv2
# import torch
# from face_ssd_infer import SSD
# from utils import vis_detections
# from fastai import *
# from fastai.vision import *

# model_file_url = 'https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1'
# model_file_name = 'model'
# classes = ['black', 'grizzly', 'teddys']
# target_size = (800, 800)
# keep_thresh = 0.01
# path = Path(__file__).parent
#
# device = torch.device("cpu")

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    net = SSD("test")
    net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
    net.to(device).eval();

    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

# loop = asyncio.get_event_loop()
# tasks = [asyncio.ensure_future(setup_learner())]
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

    # detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=keep_thresh)
    # vis_detections(img, detections, keep_thresh, show_text=False)
    return JSONResponse
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

