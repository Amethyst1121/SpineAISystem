import React, { useRef, useState, useEffect } from "react";

/* 配色表 */
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

/* 标签映射（保持你原来的） */
const labelMapping = {};
for (let i = 1; i <= 7; i++) labelMapping[i] = `C${i}`;
for (let i = 8; i <= 19; i++) labelMapping[i] = `T${i - 7}`;
for (let i = 20; i <= 24; i++) labelMapping[i] = `L${i - 19}`;

export default function LandmarkOverlay({
  imageUrl,
  segUrl = null,
  points = [],
  setPoints,
  originalWidth = 0,
  originalHeight = 0,
  editingMode = false,
  activePointId,
  setActivePointId,
}) {
  const imgRef = useRef(null);
  const [naturalSize, setNaturalSize] = useState({
    w: originalWidth || 0,
    h: originalHeight || 0,
  });

  // tooltip 状态（保持原样）
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    text: "",
  });

  // === 新增/修改：兼容父控和本地控 activePointId ===
  const [localActivePointId, setLocalActivePointId] = useState(null);
  const activeId =
    typeof activePointId !== "undefined" ? activePointId : localActivePointId;
  const setActiveId =
    typeof setActivePointId === "function"
      ? setActivePointId
      : setLocalActivePointId;

  // 底图加载后记录原始尺寸（保持你的实现）
  const handleImageLoad = (e) => {
    const img = e.currentTarget;
    const nw = img.naturalWidth || originalWidth || 1;
    const nh = img.naturalHeight || originalHeight || 1;
    setNaturalSize({ w: nw, h: nh });
    console.log("🖼 图片 natural size:", nw, nh);
  };

  // === 新增：每次 points 变化，打印出来（调试友好） ===
  useEffect(() => {
    // 注意：React.StrictMode 下开发模式可能会触发两次，这属于 React 行为
    console.log("📌 points changed:", JSON.stringify(points, null, 2));
  }, [points]);

  const handleSvgClick = (e) => {
    if (!editingMode || activeId == null) {
      console.log("❌ 未进入编辑模式或未选择点 id");
      return;
    }

    // === 使用 SVG 内置 API 转换坐标 ===
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

    const clickX = svgP.x;
    const clickY = svgP.y;

    console.log("🖱️ 点击坐标(原始):", clickX, clickY);

    if (
      clickX < 0 ||
      clickY < 0 ||
      clickX > naturalSize.w ||
      clickY > naturalSize.h
    ) {
      console.log("❌ 点击位置超出图像范围");
      return;
    }

    setPoints((prev) => {
      const newPts = [...prev];
      const idx = Number(activeId) - 1;
      newPts[idx] = { id: activeId, x: clickX, y: clickY };
      return newPts;
    });
  };

  // 删除点：把坐标设为 null（保持 id 不变）
  const handleDeletePoint = (pid) => {
    if (!editingMode) return;
    console.log(`🗑️ 删除点 id=${pid}`);
    setPoints((prev) => {
      const newPts = [...prev];
      const idx = Number(pid) - 1;
      newPts[idx] = {
        id: pid,
        x: null,
        y: null,
      };

      console.log("📌 当前 points (after delete):", newPts);
      return newPts;
    });

    setTooltip({ visible: false, x: 0, y: 0, text: "" });
  };

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* 原图 */}
      <img
        ref={imgRef}
        src={imageUrl}
        alt="原图"
        onLoad={handleImageLoad}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          display: "block",
          userSelect: "none",
          pointerEvents: "none",
        }}
        draggable={false}
      />

      {/* 分割图 */}
      {segUrl && (
        <img
          src={segUrl}
          alt="分割图"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            objectFit: "contain",
            pointerEvents: "none",
            opacity: 0.6,
          }}
          draggable={false}
        />
      )}

      {/* === SVG 层 === */}
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${naturalSize.w || 1} ${naturalSize.h || 1}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "auto" }}
        onClick={handleSvgClick}
      >
        {/* 绘制已有点 */}
        {points
          .filter((pt) => pt && pt.x != null && pt.y != null)
          .map((pt, idx) => {
            const pid = Number(pt.id);
            const color = colorTable[(pid - 1) % colorTable.length];
            const cx = pt.x;
            const cy = pt.y;

            return (
              <g key={`${pid}-${idx}`}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={13}
                  fill={color}
                  stroke="#fff"
                  strokeWidth={3}
                  style={{ cursor: "pointer", pointerEvents: "auto" }}
                  onMouseEnter={(e) => {
                    setTooltip({
                      visible: true,
                      x: e.clientX + 10,
                      y: e.clientY + 10,
                      text: `x: ${pt.x.toFixed(1)}, y: ${pt.y.toFixed(1)}`,
                    });
                  }}
                  onMouseMove={(e) => {
                    setTooltip((t) => ({
                      ...t,
                      x: e.clientX + 10,
                      y: e.clientY + 10,
                    }));
                  }}
                  onMouseLeave={() =>
                    setTooltip({ visible: false, x: 0, y: 0, text: "" })
                  }
                  onContextMenu={(e) => {
                    e.preventDefault();
                    handleDeletePoint(pid);
                  }}
                />
                {/* 白色描边文字 */}
                <text
                  x={cx + 220}
                  y={cy}
                  fontSize={60}
                  fontWeight="bold"
                  fill="#ffffff"
                  stroke="#ffffff"
                  strokeWidth={6}
                  textAnchor="start"
                  alignmentBaseline="middle"
                >
                  {labelMapping[pid] || pid}
                </text>
                {/* 彩色文字 */}
                <text
                  x={cx + 220}
                  y={cy}
                  fontSize={60}
                  fontWeight="bold"
                  fill={color}
                  textAnchor="start"
                  alignmentBaseline="middle"
                >
                  {labelMapping[pid] || pid}
                </text>
              </g>
            );
          })}
      </svg>

      {/* tooltip */}
      {tooltip.visible && (
        <div
          style={{
            position: "fixed",
            top: tooltip.y,
            left: tooltip.x,
            background: "#fff",
            border: "1px solid #ccc",
            padding: "6px 10px",
            borderRadius: "6px",
            boxShadow: "0 2px 6px rgba(0,0,0,0.15)",
            fontSize: "18px",
            pointerEvents: "none",
            zIndex: 2000,
            whiteSpace: "nowrap",
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}
