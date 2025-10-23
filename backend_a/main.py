from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
import uuid, os
import shutil

from picture import process_image, seg_png  # 你写的功能函数
from fastapi.middleware.cors import CORSMiddleware
from inference_localization import pipeline_center
from inference_segmentation import pipeline_seg
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 用于存储所有图片
BASE_DIR = "/app/pictures"
os.makedirs(BASE_DIR, exist_ok=True)

# 挂载静态目录，前端就能直接访问 /pictures/... 下的文件
app.mount("/pictures", StaticFiles(directory=BASE_DIR), name="pictures")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 校验格式
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "mha"]:
        raise HTTPException(status_code=400, detail="仅支持上传 .png 或 .mha 文件")

    # 每次上传新建一个目录
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(BASE_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 打印目录路径
    print(f"Saving file to directory: {task_dir}")

    # 用 task_id 命名文件，而不是原始文件名
    new_filename = f"{task_id}.{ext}"
    file_path = os.path.join(task_dir, new_filename)

    # 打印文件路径
    print(f"File will be saved as: {file_path}")

    # 保存文件
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        print(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")

    # 调用处理函数
    output_path = process_image(file_path, task_dir)

    relative_url = f"/pictures/{task_id}/{task_id}.png"

    return {
        "task_id": task_id,
        "image_url": relative_url
    }

@app.post("/predict_center")
async def predict_center(task_id: str):
    task_dir = os.path.join(BASE_DIR, task_id)
    mha_path = os.path.join(task_dir, f"{task_id}.mha")

    if not os.path.exists(mha_path):
        raise HTTPException(status_code=400, detail="任务目录中未找到 MHA 文件")

    # pipeline_center 返回内存中的点坐标
    points = pipeline_center(mha_path, task_dir)

    return {"points": points}


class SegRequest(BaseModel):
    task_id: str
    points: list  # 前端传过来的中心点坐标列表

@app.post("/predict_seg")
async def predict_seg(req: SegRequest):
    task_id = req.task_id
    points = req.points

    task_dir = os.path.join(BASE_DIR, task_id)
    mha_path = os.path.join(task_dir, f"{task_id}.mha")

    if not os.path.exists(mha_path):
        raise HTTPException(status_code=400, detail="任务目录中未找到 MHA 文件")

    # 获取 segmentation sitk.Image（现在传入 points）
    seg_sitk = pipeline_seg(mha_path, points)

    # 生成 PNG 并返回路径（本地路径）
    seg_png(seg_sitk, task_dir, task_id)

    # ✅ 转成相对 URL，和 /upload 保持一致
    filename = f"{task_id}_seg2.png"
    relative_url = f"/pictures/{task_id}/{filename}"

    return {"seg_url": relative_url}

@app.delete("/reset/{task_id}")
async def reset_task(task_id: str):
    task_dir = os.path.join(BASE_DIR, task_id)

    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail="任务不存在或已被删除")

    try:
        shutil.rmtree(task_dir)
        return {"message": f"任务 {task_id} 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

@app.post("/reset/{task_id}")
async def reset_task_post(task_id: str):
    task_dir = os.path.join(BASE_DIR, task_id)
    if not os.path.exists(task_dir):
        return {"message": "任务目录不存在（可能已被清理）"}
    try:
        shutil.rmtree(task_dir)
        return {"message": f"任务 {task_id} 已删除（POST 方式）"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
