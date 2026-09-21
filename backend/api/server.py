import threading
import uuid
from pathlib import Path
import imageio.v2 as imageio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from env.wrappers import run_episode
from models.actor_critic import load_policy

BACKEND = Path(__file__).resolve().parent.parent
CHECKPOINTS = BACKEND / "checkpoints"
VIDEOS = BACKEND / "videos"
FRONTEND = BACKEND.parent / "frontend"
VIDEOS.mkdir(exist_ok=True)
PASS_SCORE = 200

app = FastAPI()
loaded_policies = {}
one_time_lock = threading.Lock()

def get_policy(name):
    path = CHECKPOINTS / f"{name}.pth"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Policy '{name}' not found.")
    if name not in loaded_policies:
        loaded_policies[name] = load_policy(path)
    return loaded_policies[name]

def play_and_record(checkpoint, seed, **settings):
    policy = get_policy(checkpoint)
    with one_time_lock:
        result = run_episode(policy, seed, render=True, **settings)

    video_name = f"{uuid.uuid4().hex}.mp4"
    imageio.mimsave(VIDEOS / video_name, result["frames"], fps=25, macro_block_size=1)
    return {
        "name": checkpoint.replace("_", " "), 
        "video_url": f"/videos/{video_name}",
        "score": round(result["score"], 2),
        "steps": result["steps"],
        "distance": round(result["distance"], 2),
        "passed": result["score"] >= PASS_SCORE and not result["fell"],
        "torques": [[round(t, 2) for t in torque] for torque in result["torques"]],
    }

class ShowdownRequest(BaseModel):
    left: str
    right: str
    hardcore: bool = False
    seed: int = 7

class StressRequest(BaseModel):
    checkpoint: str
    gravity: float = Field(1.0, ge=0.5, le=1.5)
    friction: float = Field(2.5, ge=0.2, le=3.0)
    seed: int = 7

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/checkpoints")
def list_checkpoints():
    return sorted(p.stem for p in CHECKPOINTS.glob("*.pth"))

@app.post("/api/showdown")
def showdown(req: ShowdownRequest):
    return [play_and_record(req.left, req.seed, hardcore=req.hardcore),
            play_and_record(req.right, req.seed, hardcore=req.hardcore),]

@app.post("/api/stress")
def stress(req: StressRequest):
    return play_and_record(req.checkpoint, req.seed, gravity=req.gravity, friction=req.friction)

app.mount("/videos", StaticFiles(directory=VIDEOS))
app.mount("/", StaticFiles(directory=FRONTEND, html=True))