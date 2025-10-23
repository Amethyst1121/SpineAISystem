# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
from fracture_pipeline import fracture  # 你自己的逻辑模块
from pydantic import BaseModel
from generate_report import generatereport

# ======================================
# 初始化 FastAPI 应用
# ======================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 全局目录配置
# ======================================
BASE_DIR = "/app/pictures"
os.makedirs(BASE_DIR, exist_ok=True)

# 静态文件挂载
app.mount("/pictures", StaticFiles(directory=BASE_DIR), name="pictures")

class FractureRequest(BaseModel):
    task_id: str
    points: list  # 前端传过来的中心点坐标列表

@app.post("/predict_fracture")
async def predict_fracture(req: FractureRequest):
    task_id = req.task_id
    points = req.points

    task_dir = os.path.join(BASE_DIR, task_id)
    image_path = os.path.join(task_dir, f"{task_id}.png")
    seg_path = os.path.join(task_dir, f"{task_id}_seg1.png")
    fracture_dir = os.path.join(task_dir, "fracture_images")

    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="任务目录中未找到原图PNG")

    if not os.path.exists(seg_path):
        raise HTTPException(status_code=400, detail="任务目录中未找到分割PNG")

    # pipeline_center 返回内存中的点坐标
    probs = fracture(image_path, seg_path, task_dir, points)
    relative_url = f"/pictures/{task_id}/fracture_images"
    image_files = sorted(
        [f for f in os.listdir(fracture_dir) if f.lower().endswith((".png", ".jpg"))]
    )

    return {
            "probs": probs,
            "fracture_url": relative_url,
            "images": image_files
    }

class ReportRequest(BaseModel):
    probs: list

@app.post("/predict_report")
async def predict_report(req: ReportRequest):
    probs = req.probs

    report = generatereport(probs, ai_model_name="FractureAI-v3.0")

    return {
            "report": report,
    }