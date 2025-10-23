import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import LandmarkOverlay from "./LandmarkOverlay";
import ReactMarkdown from "react-markdown";

// App.jsx（文件顶部）
const colorTable = [
  "#566eb5",
  "#b4c0e3",
  "#cbaeb3",
  "#dbcc8a",
  "#92c4dd",
  "#a8ae5e",
  "#f0e5b6",
  "#d0cff6",
  "#CE99B3",
  "#a98175",
  "#8c4356",
  "#f47983",
  "#e29c45",
  "#d9b611",
  "#789262",
  "#96ce54",
  "#177cb0",
  "#065279",
  "#a1afc9",
  "#4a4266",
  "#815476",
  "#e4c6d0",
  "#75878a",
  "#4c8dae",
];

const createInitialPoints = () =>
  Array.from({ length: 24 }, (_, i) => ({ id: i + 1, x: null, y: null }));

export default function App() {
  const [imageUrl, setImageUrl] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [points, setPoints] = useState(createInitialPoints());
  const [originalWidth, setOriginalWidth] = useState(0);
  const [originalHeight, setOriginalHeight] = useState(0);
  const [segUrl, setSegUrl] = useState(null);
  const [editingMode, setEditingMode] = useState(false);
  // 修改按钮id
  const [activePointId, setActivePointId] = useState(null);
  const [fractureImages, setFractureImages] = useState(null);
  const [fractureProbs, setFractureProbs] = useState(null);
  const [reportText, setReportText] = useState("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // === 重置按钮逻辑 ===
  const handleReset = () => {
    setPoints(createInitialPoints());
    setSegUrl(null);
    setEditingMode(false);
    setActivePointId(null);
  };

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // ✅ 如果存在旧任务 ID，先让后端删除对应目录
    if (taskId) {
      try {
        await axios.delete(`/api/reset/${taskId}`);
        console.log(`旧任务 ${taskId} 已清理`);
      } catch (err) {
        console.warn("旧任务清理失败（可能已被删除）:", err);
      }
    }

    const ext = file.name.split(".").pop().toLowerCase();

    // ✅ 上传前清空旧任务
    setImageUrl(null);
    setPoints(createInitialPoints());
    setSegUrl(null);
    setEditingMode(false);
    setActivePointId(null);

    if (!["png", "mha"].includes(ext)) {
      alert("仅支持上传 .png 或 .mha 文件");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("/api_a/upload", formData, {
        // const res = await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // 部署时删除
      // const baseURL = "http://127.0.0.1:8000";
      // const fullImageUrl = baseURL + res.data.image_url;

      setImageUrl(res.data.image_url);
      // setImageUrl(fullImageUrl);
      setTaskId(res.data.task_id);

      // 获取原图大小
      const img = new Image();
      img.src = res.data.image_url;
      img.onload = () => {
        setOriginalWidth(img.width);
        setOriginalHeight(img.height);
      };
    } catch (err) {
      alert("上传失败: " + (err.response?.data?.detail || err.message));
    }
  };

  useEffect(() => {
    const cleanup = () => {
      if (taskId) {
        navigator.sendBeacon(`/api_a/reset/${taskId}`);
        // navigator.sendBeacon(`http://127.0.0.1:8000/reset/${taskId}`);
        console.log("页面关闭，已发送清理请求");
      }
    };

    window.addEventListener("beforeunload", cleanup);
    return () => window.removeEventListener("beforeunload", cleanup);
  }, [taskId]);

  const handlePredictCenter = async () => {
    if (!taskId) {
      alert("请先上传图像");
      return;
    }

    try {
      const res = await axios.post(`/api_a/predict_center?task_id=${taskId}`);
      // const res = await axios.post(
      //   `http://127.0.0.1:8000/predict_center?task_id=${taskId}`
      // );
      // 合并到固定数组（以 id 为准，id 从 1..24）
      const initial = createInitialPoints();
      if (Array.isArray(res.data.points)) {
        res.data.points.forEach((p) => {
          if (p && typeof p.id !== "undefined") {
            const id = Number(p.id);
            if (id >= 1 && id <= 24) {
              // 保持 id 字段，允许 x/y 为 null 或数字
              initial[id - 1] = { id, x: p.x ?? null, y: p.y ?? null };
            }
          }
        });
      }
      setPoints(initial);
    } catch (err) {
      console.error("❌ 预测失败:", err.response?.data || err.message);
      alert(
        "预测失败: " +
          (typeof err.response?.data?.detail === "string"
            ? err.response.data.detail
            : JSON.stringify(err.response?.data || err.message))
      );
    }
  };

  const handlePredictSeg = async () => {
    if (!taskId) {
      alert("请先上传图像");
      return;
    }
    if (!points || points.length === 0 || points.every((p) => p.x === null)) {
      alert("请先预测或绘制中心点");
      return;
    }

    setEditingMode(false);

    try {
      const res = await axios.post("/api_a/predict_seg", {
        // const res = await axios.post("http://127.0.0.1:8000/predict_seg", {
        task_id: taskId,
        points: points,
      });
      console.log("✅ 分割结果:", res.data);

      // 部署时删除
      // const baseURL = "http://127.0.0.1:8000";
      // const fullSegUrl = `${baseURL}${res.data.seg_url}?t=${Date.now()}`;

      const fullSegUrl = `${res.data.seg_url}?t=${Date.now()}`;
      setSegUrl(null);
      setTimeout(() => setSegUrl(fullSegUrl), 50);
    } catch (err) {
      console.error("❌ 分割失败:", err.response?.data || err.message);
      alert(
        "分割失败: " +
          (typeof err.response?.data?.detail === "string"
            ? err.response.data.detail
            : JSON.stringify(err.response?.data || err.message))
      );
    }
  };

  const handlePredictFracture = async () => {
    if (!taskId) {
      alert("请先上传图像");
      return;
    }
    if (!segUrl) {
      alert("请先预测分割");
      return;
    }

    try {
      const response = await fetch(`/api_b/predict_fracture`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_id: taskId,
          points: points || [],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP 错误 ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ 骨折预测结果：", data);

      // 保存概率（如果你用得上）
      setFractureProbs(data.probs || []);

      if (!data.images || data.images.length === 0) {
        setFractureImages([]); // 清空图片
        return;
      }

      const imageBase = data.fracture_url;
      const imageList = data.images.map(
        (filename) => `${imageBase}/${filename}?t=${Date.now()}`
      );

      // 更新状态，触发渲染
      setFractureImages(imageList);
    } catch (error) {
      console.error("❌ 预测骨折失败:", error);
      alert("预测骨折时发生错误，请检查控制台。");
    }
  };

  const handleGenerateReport = async () => {
    if (!fractureProbs || fractureProbs.length === 0) {
      alert("请先进行骨折预测");
      return;
    }

    setIsGeneratingReport(true);
    setReportText("报告生成中，请稍候...");

    try {
      const response = await fetch(`/api_b/predict_report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ probs: fractureProbs }),
      });

      if (!response.ok) {
        throw new Error(`HTTP 错误 ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ 报告生成结果：", data);

      setReportText(data.report || "未生成报告内容");
    } catch (error) {
      console.error("❌ 报告生成失败：", error);
      setReportText("报告生成失败，请检查后端服务。");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="app">
      <h1 className="title">智能脊柱影像分析与诊断报告生成系统</h1>

      {/* ✅ main-container 里同时包裹左、右两块 */}
      <div
        className="main-container"
        style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}
      >
        {/* 🟩 左侧面板 */}
        <div className="left-panel" style={{ flex: 1 }}>
          <div className="toolbar">
            {/* 上传与操作按钮 */}
            <label
              htmlFor="upload-input"
              className="upload-box-inline"
              title="上传图像"
            >
              +
            </label>
            <input
              type="file"
              accept=".png,.mha"
              onChange={handleUpload}
              id="upload-input"
              style={{ display: "none" }}
            />

            <button className="btn c1" onClick={handlePredictCenter}>
              预测中心点
            </button>

            <button
              className="btn c2"
              onClick={() => {
                if (!imageUrl) {
                  alert("请先上传图像");
                  return;
                }
                setEditingMode((prev) => !prev);
              }}
            >
              {editingMode ? "退出修改" : "修改中心点"}
            </button>

            <button className="btn c3" onClick={handlePredictSeg}>
              预测分割
            </button>

            <button className="btn c4" onClick={handleReset}>
              重置
            </button>
          </div>

          {/* 图像 + 右侧点按钮 */}
          <div className="image-container">
            <div className="image-box">
              {imageUrl ? (
                <LandmarkOverlay
                  imageUrl={imageUrl}
                  segUrl={segUrl}
                  points={points}
                  setPoints={setPoints}
                  originalWidth={originalWidth}
                  originalHeight={originalHeight}
                  editingMode={editingMode}
                  activePointId={activePointId}
                  setActivePointId={setActivePointId}
                />
              ) : (
                <p className="placeholder">等待上传图像...</p>
              )}
            </div>

            {editingMode && (
              <div className="sidebar">
                {Array.from({ length: 24 }).map((_, i) => {
                  const pointId = i + 1;
                  const color = colorTable[i];
                  return (
                    <button
                      key={pointId}
                      className={`small-btn ${
                        activePointId === pointId ? "active" : ""
                      }`}
                      style={{ "--btn-color": color }}
                      onClick={() => setActivePointId(pointId)}
                    >
                      {pointId <= 7
                        ? `C${pointId}`
                        : pointId <= 19
                        ? `T${pointId - 7}`
                        : pointId <= 24
                        ? `L${pointId - 19}`
                        : pointId}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* 🟦 右侧面板 —— 注意！这一层要和左侧在同一个 main-container 里 */}
        <div
          className="right-panel"
          style={{
            width: "400px",
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
          }}
        >
          {/* 上半部分 */}
          <div
            className="fracture-section"
            style={{
              border: "1px solid #ccc",
              padding: "1rem",
              borderRadius: "8px",
              background: "#f9f9f9",
            }}
          >
            <button class="btn c5" onClick={handlePredictFracture}>
              预测骨折
            </button>
            <div className="result-panel">
              {fractureImages === null ? (
                // 🚫 未预测，不显示任何内容（或显示占位文本也可以）
                <></>
              ) : fractureImages.length === 0 ? (
                // ⚠️ 已预测，但没有骨折
                <p>没有骨折</p>
              ) : (
                // 🖼️ 有预测结果，显示图片
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(5, 1fr)", // 一行 5 张图
                    gap: "8px",
                    marginTop: "10px",
                  }}
                >
                  {fractureImages.map((url, i) => (
                    <img
                      key={i}
                      src={url}
                      alt={`fracture-${i}`}
                      className="fracture-img"
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 下半部分 */}
          <div
            className="report-section"
            style={{
              border: "1px solid #ccc",
              padding: "1rem",
              borderRadius: "8px",
              background: "#f9f9f9",
            }}
          >
            <button
              className="btn c6"
              onClick={handleGenerateReport}
              disabled={isGeneratingReport}
            >
              {isGeneratingReport ? "生成中..." : "生成报告"}
            </button>

            <div className="report-panel" style={{ marginTop: "10px" }}>
              {reportText && <ReactMarkdown>{reportText}</ReactMarkdown>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
