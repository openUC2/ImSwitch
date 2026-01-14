from fastapi import FastAPI
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import numpy as np
import cv2
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import fractions
import cv2
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import HTMLResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from fastapi import HTTPException

from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay


ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.count = 0

    async def recv(self):
        img = np.random.randint(0, 155, (150, 300, 3)).astype('uint8')
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = self.count
        self.count += 1
        new_frame.time_base = fractions.Fraction(1, 1000)
        return new_frame
    
@app.post("/request-offer")
async def request_offer():
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    pc.addTrack(VideoTransformTrack(relay, transform="rotate"))

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.post("/answer")
async def answer(data: RTCSessionDescription):
    try:
        pc = next(iter(pcs))
        answer = RTCSessionDescription(sdp=data.sdp, type=data.type)
        await pc.setRemoteDescription(answer)
        return JSONResponse(status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stop_stream/")
async def stop_stream():
    try:
        pc = next(iter(pcs))
        await pc.close()
        pcs.clear()
        return JSONResponse(status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/move_stage/{direction}")
async def move_stage(direction: str):
    # Your logic to move the microscope's stage
    #...
    print(direction)

    return {"status": "success", "direction": direction}
    
@app.on_event("shutdown")
async def shutdown_event():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

