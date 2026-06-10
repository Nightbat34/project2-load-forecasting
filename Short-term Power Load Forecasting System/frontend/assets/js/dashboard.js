(function () {
  "use strict";

  const data = window.DASHBOARD_DATA;
  const state = {
    forecastTarget: "mean",
    weatherTarget: "日平均负荷",
    validationTarget: "日平均负荷",
    historyWindow: 90,
    historyStart: 0,
    validationWindow: 31,
    validationStart: 0,
  };

  const targetConfig = {
    mean: {
      label: "日平均负荷",
      dailyKey: "load_mean",
      forecastKey: "load_mean",
      color: "#1d5f99",
      className: "line-mean",
    },
    max: {
      label: "日最高负荷",
      dailyKey: "load_max",
      forecastKey: "load_max",
      color: "#bd3a3a",
      className: "line-max",
    },
    min: {
      label: "日最低负荷",
      dailyKey: "load_min",
      forecastKey: "load_min",
      color: "#2f855a",
      className: "line-min",
    },
  };

  const weatherLabels = {
    temp_max: { cn: "日最高气温", en: "temp_max" },
    temp_min: { cn: "日最低气温", en: "temp_min" },
    temp_avg: { cn: "日平均气温", en: "temp_avg" },
    humidity: { cn: "相对湿度", en: "humidity" },
    rainfall: { cn: "日降雨量", en: "rainfall" },
    temp_range: { cn: "昼夜温差", en: "temp_range" },
    hdd: { cn: "采暖度日", en: "hdd" },
    cdd: { cn: "制冷度日", en: "cdd" },
  };

  function qs(selector) {
    return document.querySelector(selector);
  }

  function qsa(selector) {
    return Array.from(document.querySelectorAll(selector));
  }

  function fmt(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return Number(value).toLocaleString("zh-CN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function clear(node) {
    node.innerHTML = "";
  }

  function createSvg(width, height) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("role", "img");
    return svg;
  }

  function elSvg(name, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  }

  function scaled(values, minPixels, maxPixels, minValue, maxValue) {
    const range = maxValue - minValue || 1;
    return values.map((value) => maxPixels - ((value - minValue) / range) * (maxPixels - minPixels));
  }

  function pointsToPath(points) {
    return points
      .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
      .join(" ");
  }

  function niceTicks(minValue, maxValue, desiredCount = 4) {
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return [0, 1];
    if (minValue === maxValue) {
      minValue -= 1;
      maxValue += 1;
    }
    const range = maxValue - minValue;
    const rawStep = range / Math.max(1, desiredCount - 1);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalized = rawStep / magnitude;
    const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
    const step = multiplier * magnitude;
    const niceMin = Math.floor(minValue / step) * step;
    const niceMax = Math.ceil(maxValue / step) * step;
    const ticks = [];
    for (let value = niceMin; value <= niceMax + step * 0.5; value += step) {
      ticks.push(Math.round(value * 1000) / 1000);
      if (ticks.length > 8) break;
    }
    return ticks;
  }

  function showTooltip(evt, html) {
    const tooltip = qs("#tooltip");
    tooltip.innerHTML = html;
    tooltip.style.left = `${evt.clientX}px`;
    tooltip.style.top = `${evt.clientY}px`;
    tooltip.classList.add("visible");
  }

  function hideTooltip() {
    qs("#tooltip").classList.remove("visible");
  }

  function drawGrid(svg, dims, ticks, minValue, maxValue) {
    const { left, right, top, bottom, width } = dims;
    ticks.forEach((tick) => {
      const y = bottom - ((tick - minValue) / (maxValue - minValue || 1)) * (bottom - top);
      svg.appendChild(elSvg("line", { x1: left, y1: y, x2: width - right, y2: y, class: "grid-line" }));
      const text = elSvg("text", { x: left - 10, y: y + 4, "text-anchor": "end", class: "axis-label" });
      text.textContent = fmt(tick, 0);
      svg.appendChild(text);
    });
  }

  function renderLineChart(container, series, options = {}) {
    clear(container);
    const width = options.width || 1040;
    const height = options.height || 330;
    const bottomMargin = options.bottom || 56;
    const dims = {
      left: options.left || 92,
      right: options.right || 34,
      top: options.top || 30,
      bottom: height - bottomMargin,
      width,
      height,
    };
    const svg = createSvg(width, height);
    const allValues = series.flatMap((item) => item.values.map((point) => point.value));
    const rawMin = Math.min(...allValues);
    const rawMax = Math.max(...allValues);
    const padding = Math.max(80, (rawMax - rawMin) * 0.18);
    const ticks = niceTicks(rawMin - padding, rawMax + padding, options.tickCount || 4);
    const minValue = ticks[0];
    const maxValue = ticks[ticks.length - 1];

    drawGrid(svg, dims, ticks, minValue, maxValue);

    if (options.band) {
      const values = options.band.values;
      const xStep = (width - dims.left - dims.right) / Math.max(values.length - 1, 1);
      const upper = values.map((point, index) => ({
        x: dims.left + index * xStep,
        y: scaled([point.high], dims.top, dims.bottom, minValue, maxValue)[0],
      }));
      const lower = values
        .map((point, index) => ({
          x: dims.left + index * xStep,
          y: scaled([point.low], dims.top, dims.bottom, minValue, maxValue)[0],
        }))
        .reverse();
      const bandPath = `${pointsToPath(upper)} L${lower.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" L")} Z`;
      svg.appendChild(elSvg("path", {
        d: bandPath,
        fill: options.band.color || "rgba(29, 95, 153, 0.12)",
        stroke: "none",
      }));
    }

    const pointSets = [];
    series.forEach((item) => {
      const values = item.values;
      const xStep = (width - dims.left - dims.right) / Math.max(values.length - 1, 1);
      const yVals = scaled(values.map((point) => point.value), dims.top, dims.bottom, minValue, maxValue);
      const points = values.map((point, index) => ({
        x: dims.left + index * xStep,
        y: yVals[index],
        raw: point,
      }));
      pointSets.push({ item, points });

      const path = elSvg("path", {
        d: pointsToPath(points),
        fill: "none",
        stroke: item.color,
        "stroke-width": item.width || 2.4,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      });
      svg.appendChild(path);

      const dotEvery = options.dotEvery || Math.max(1, Math.ceil(values.length / 18));
      points.forEach((point, index) => {
        if (index % dotEvery !== 0 && index !== points.length - 1) return;
        const circle = elSvg("circle", {
          cx: point.x,
          cy: point.y,
          r: 3.2,
          fill: item.color,
          "data-date": point.raw.date,
        });
        circle.addEventListener("mousemove", (evt) => {
          showTooltip(
            evt,
            `<strong>${point.raw.date}</strong><br>${item.name}: ${fmt(point.raw.value, 2)} MW`
          );
        });
        circle.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(circle);
      });
    });

    const axis = elSvg("line", {
      x1: dims.left,
      y1: dims.bottom,
      x2: width - dims.right,
      y2: dims.bottom,
      stroke: "#b9c7d3",
    });
    svg.appendChild(axis);

    const xLabels = options.xLabels || [
      { index: 0, anchor: "start" },
      { index: Math.floor((series[0].values.length - 1) / 2), anchor: "middle" },
      { index: series[0].values.length - 1, anchor: "end" },
    ];
    const xStep = (width - dims.left - dims.right) / Math.max(series[0].values.length - 1, 1);
    xLabels.forEach((label) => {
      const point = series[0].values[label.index];
      if (!point) return;
      const x = dims.left + label.index * xStep;
      const text = elSvg("text", { x, y: height - 18, "text-anchor": label.anchor, class: "axis-label" });
      text.textContent = point.date;
      svg.appendChild(text);
    });

    const hoverLayer = elSvg("g");
    const firstValues = series[0].values;
    const hoverStep = (width - dims.left - dims.right) / Math.max(firstValues.length - 1, 1);
    firstValues.forEach((point, index) => {
      const rectX = dims.left + index * hoverStep - hoverStep / 2;
      const hit = elSvg("rect", {
        x: index === 0 ? dims.left : rectX,
        y: dims.top,
        width: index === 0 || index === firstValues.length - 1 ? Math.max(18, hoverStep / 2) : Math.max(18, hoverStep),
        height: dims.bottom - dims.top,
        fill: "transparent",
      });
      hit.addEventListener("mousemove", (evt) => {
        const rows = pointSets
          .map(({ item }) => {
            const value = item.values[index]?.value;
            return value === undefined ? "" : `${item.name}: ${fmt(value, 2)} MW`;
          })
          .filter(Boolean)
          .join("<br>");
        showTooltip(evt, `<strong>${point.date}</strong><br>${rows}`);
      });
      hit.addEventListener("mouseleave", hideTooltip);
      hoverLayer.appendChild(hit);
    });
    svg.appendChild(hoverLayer);

    container.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    series.forEach((item) => {
      const node = document.createElement("span");
      node.className = "legend-item";
      node.style.color = item.color;
      node.innerHTML = `<span class="legend-swatch"></span>${item.name}`;
      legend.appendChild(node);
    });
    container.appendChild(legend);
  }

  function renderBarChart(container, rows, options = {}) {
    clear(container);
    const width = options.width || 940;
    const height = options.height || 360;
    const left = options.left || 230;
    const right = 88;
    const top = 26;
    const bottom = 34;
    const svg = createSvg(width, height);
    const values = rows.map((row) => row.value);
    const min = Math.min(-1, ...values);
    const max = Math.max(1, ...values);
    const zeroX = left + ((0 - min) / (max - min)) * (width - left - right);
    const barH = Math.min(28, (height - top - bottom) / rows.length - 8);

    svg.appendChild(elSvg("line", { x1: zeroX, y1: top, x2: zeroX, y2: height - bottom, stroke: "#9fb0bf" }));

    rows.forEach((row, index) => {
      const y = top + index * ((height - top - bottom) / rows.length) + 5;
      const x = left + ((Math.min(0, row.value) - min) / (max - min)) * (width - left - right);
      const x2 = left + ((Math.max(0, row.value) - min) / (max - min)) * (width - left - right);
      const rect = elSvg("rect", {
        x,
        y,
        width: Math.max(2, x2 - x),
        height: barH,
        rx: 3,
        fill: row.value >= 0 ? "#1d5f99" : "#c56b27",
      });
      rect.addEventListener("mousemove", (evt) => {
        showTooltip(evt, `<strong>${row.name}</strong><br>相关系数: ${fmt(row.value, 4)}`);
      });
      rect.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(rect);

      const label = elSvg("text", { x: left - 14, y: y + barH * 0.45, "text-anchor": "end", class: "axis-label" });
      label.textContent = row.name;
      svg.appendChild(label);

      if (row.subName) {
        const subLabel = elSvg("text", { x: left - 14, y: y + barH * 0.92, "text-anchor": "end", class: "axis-label sub-axis-label" });
        subLabel.textContent = row.subName;
        svg.appendChild(subLabel);
      }

      const value = elSvg("text", {
        x: row.value >= 0 ? x2 + 8 : x - 8,
        y: y + barH * 0.68,
        "text-anchor": row.value >= 0 ? "start" : "end",
        class: "axis-label",
      });
      value.textContent = fmt(row.value, 3);
      svg.appendChild(value);
    });

    container.appendChild(svg);
  }

  function renderKpis() {
    const meta = data.meta;
    const stats = data.daily2014Stats;
    const kpis = [
      { label: "验证集平均 R²", value: meta.meanValidationR2.toFixed(4), note: "Stacking · 2014 下半年" },
      { label: "平均 MAPE", value: `${meta.meanValidationMape.toFixed(3)}%`, note: "三个负荷目标平均" },
      { label: "2014 全年最高负荷", value: fmt(stats.max_load, 0), note: `${stats.max_load_date} · MW` },
      { label: "最终预测期", value: "7 天", note: meta.forecastPeriod },
    ];
    const grid = qs("#kpiGrid");
    clear(grid);
    kpis.forEach((kpi) => {
      const card = document.createElement("article");
      card.className = "kpi-card";
      card.innerHTML = `
        <div class="kpi-label">${kpi.label}</div>
        <div class="kpi-value">${kpi.value}</div>
        <div class="kpi-note">${kpi.note}</div>
      `;
      grid.appendChild(card);
    });
  }

  function renderForecast() {
    const config = targetConfig[state.forecastTarget];
    qs("#forecastChartCaption").textContent = config.label;
    const focusValues = data.forecast.map((row) => ({ date: row.date, value: row[config.forecastKey] }));
    const bandValues = data.forecast.map((row) => ({ date: row.date, low: row.load_min, high: row.load_max }));
    const allLines = [
      {
        name: "日最高负荷",
        color: targetConfig.max.color,
        values: data.forecast.map((row) => ({ date: row.date, value: row.load_max })),
        width: state.forecastTarget === "max" ? 3.2 : 1.8,
      },
      {
        name: "日平均负荷",
        color: targetConfig.mean.color,
        values: data.forecast.map((row) => ({ date: row.date, value: row.load_mean })),
        width: state.forecastTarget === "mean" ? 3.2 : 1.8,
      },
      {
        name: "日最低负荷",
        color: targetConfig.min.color,
        values: data.forecast.map((row) => ({ date: row.date, value: row.load_min })),
        width: state.forecastTarget === "min" ? 3.2 : 1.8,
      },
    ];
    renderLineChart(
      qs("#forecastChart"),
      allLines,
      {
        height: 360,
        dotEvery: 1,
        tickCount: 5,
        left: 98,
        bottom: 58,
        band: { values: bandValues, color: "rgba(29, 95, 153, 0.11)" },
      }
    );
  }

  function renderForecastTable() {
    const tbody = qs("#forecastTable");
    clear(tbody);
    data.forecast.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.date}</td>
        <td>${row.weekday}</td>
        <td>${fmt(row.load_min, 2)}</td>
        <td>${fmt(row.load_mean, 2)}</td>
        <td>${fmt(row.load_max, 2)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderHistory() {
    const stats = data.daily2014Stats;
    qs("#historyStats").textContent = `全年均值 ${fmt(stats.mean_load, 0)} MW · 最高 ${fmt(stats.max_load, 0)} MW · 最低 ${fmt(stats.min_load, 0)} MW`;
    const slider = qs("#historyRange");
    const windowSize = Math.min(state.historyWindow, data.daily2014.length);
    const maxStart = Math.max(0, data.daily2014.length - windowSize);
    state.historyStart = Math.max(0, Math.min(state.historyStart, maxStart));
    if (slider) {
      slider.max = String(maxStart);
      slider.value = String(state.historyStart);
      slider.disabled = maxStart === 0;
    }
    const source = data.daily2014.slice(state.historyStart, state.historyStart + windowSize);
    const first = source[0]?.date || "";
    const last = source[source.length - 1]?.date || "";
    const label = windowSize >= data.daily2014.length ? "全年" : `${windowSize} 天窗口`;
    qs("#historyWindowLabel").textContent = `${label} · ${first} 至 ${last}`;
    renderLineChart(
      qs("#historyChart"),
      [
        {
          name: "日最高负荷",
          color: targetConfig.max.color,
          values: source.map((row) => ({ date: row.date, value: row.load_max })),
        },
        {
          name: "日平均负荷",
          color: targetConfig.mean.color,
          values: source.map((row) => ({ date: row.date, value: row.load_mean })),
        },
        {
          name: "日最低负荷",
          color: targetConfig.min.color,
          values: source.map((row) => ({ date: row.date, value: row.load_min })),
        },
      ],
      { height: 430, dotEvery: Math.max(1, Math.ceil(source.length / 16)), tickCount: 5, left: 98, bottom: 58 }
    );
  }

  function renderPerformanceTable() {
    const tbody = qs("#performanceTable");
    clear(tbody);
    data.stackingPerformance.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.target}</td>
        <td>${fmt(row.RMSE, 2)}</td>
        <td>${fmt(row["MAPE(%)"], 3)}%</td>
        <td>${fmt(row.R2, 4)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderR2Chart() {
    const rows = data.overfittingStacking;
    const container = qs("#r2Chart");
    clear(container);
    const width = 720;
    const height = 260;
    const svg = createSvg(width, height);
    const left = 58;
    const top = 22;
    const bottom = 42;
    const plotH = height - top - bottom;
    const groupW = (width - left - 26) / rows.length;
    const barW = Math.min(42, groupW * 0.26);

    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const y = top + (1 - tick) * plotH;
      svg.appendChild(elSvg("line", { x1: left, y1: y, x2: width - 26, y2: y, class: "grid-line" }));
      const text = elSvg("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "axis-label" });
      text.textContent = tick.toFixed(2);
      svg.appendChild(text);
    });

    rows.forEach((row, index) => {
      const x = left + index * groupW + groupW / 2;
      const trainH = row.R2_train * plotH;
      const valH = row.R2_validation * plotH;
      const trainRect = elSvg("rect", {
        x: x - barW - 3,
        y: top + plotH - trainH,
        width: barW,
        height: trainH,
        fill: "#1d5f99",
        rx: 3,
      });
      const valRect = elSvg("rect", {
        x: x + 3,
        y: top + plotH - valH,
        width: barW,
        height: valH,
        fill: "#c56b27",
        rx: 3,
      });
      [trainRect, valRect].forEach((rect, rectIndex) => {
        rect.addEventListener("mousemove", (evt) => {
          const label = rectIndex === 0 ? "训练集 R²" : "验证集 R²";
          const value = rectIndex === 0 ? row.R2_train : row.R2_validation;
          showTooltip(evt, `<strong>${row.target}</strong><br>${label}: ${fmt(value, 4)}`);
        });
        rect.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(rect);
      });
      const text = elSvg("text", { x, y: height - 16, "text-anchor": "middle", class: "axis-label" });
      text.textContent = row.target.replace("负荷", "");
      svg.appendChild(text);
    });
    container.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML = `
      <span class="legend-item" style="color:#1d5f99"><span class="legend-swatch"></span>训练集 R²</span>
      <span class="legend-item" style="color:#c56b27"><span class="legend-swatch"></span>验证集 R²</span>
    `;
    container.appendChild(legend);
  }

  function renderWeather() {
    const target = state.weatherTarget;
    qs("#weatherCaption").textContent = target;
    const rows = data.weatherCorrelations
      .filter((row) => row.target === target)
      .map((row) => ({
        name: weatherLabels[row.weather_factor]?.cn || row.weather_factor,
        subName: weatherLabels[row.weather_factor]?.en || "",
        value: row.pearson_r,
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    renderBarChart(qs("#weatherChart"), rows, { height: 380, left: 240 });

    const tbody = qs("#regressionTable");
    clear(tbody);
    data.regressionSummary
      .filter((row) => row.target === target && row.term !== "const")
      .forEach((row) => {
        const label = weatherLabels[row.term] || { cn: row.term, en: "" };
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="bilingual-name"><span>${label.cn}</span><small>${label.en}</small></span></td>
          <td>${fmt(row.coef, 3)}</td>
          <td>${Number(row.p_value).toExponential(2)}</td>
        `;
        tbody.appendChild(tr);
      });
  }

  function renderValidation() {
    const allRows = data.validationStacking.filter((row) => row.target === state.validationTarget);
    const slider = qs("#validationRange");
    const windowSize = Math.min(state.validationWindow, allRows.length);
    const maxStart = Math.max(0, allRows.length - windowSize);
    state.validationStart = Math.max(0, Math.min(state.validationStart, maxStart));
    if (slider) {
      slider.max = String(maxStart);
      slider.value = String(state.validationStart);
      slider.disabled = maxStart === 0;
    }
    const rows = allRows.slice(state.validationStart, state.validationStart + windowSize);
    const first = rows[0]?.date || "";
    const last = rows[rows.length - 1]?.date || "";
    const label = windowSize >= allRows.length ? "全部验证期" : `${windowSize} 天窗口`;
    qs("#validationCaption").textContent = `${state.validationTarget} · ${label} · ${first} 至 ${last}`;
    renderLineChart(
      qs("#validationChart"),
      [
        {
          name: "真实值",
          color: "#1d5f99",
          values: rows.map((row) => ({ date: row.date, value: row.actual })),
        },
        {
          name: "预测值",
          color: "#c56b27",
          values: rows.map((row) => ({ date: row.date, value: row.predicted })),
        },
      ],
      { height: 430, dotEvery: Math.max(1, Math.ceil(rows.length / 14)), tickCount: 5, left: 98, bottom: 58 }
    );
  }

  function bindInteractions() {
    qsa(".segment").forEach((button) => {
      button.addEventListener("click", () => {
        qsa(".segment").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        state.forecastTarget = button.dataset.target;
        renderForecast();
      });
    });

    qsa(".validation-segment").forEach((button) => {
      button.addEventListener("click", () => {
        qsa(".validation-segment").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        state.validationTarget = button.dataset.target;
        renderValidation();
      });
    });

    const validationRange = qs("#validationRange");
    if (validationRange) {
      validationRange.addEventListener("input", (evt) => {
        state.validationStart = Number(evt.target.value);
        qsa(".validation-range-button").forEach((item) => item.classList.remove("active"));
        renderValidation();
      });
    }

    qsa(".validation-range-button").forEach((button) => {
      button.addEventListener("click", () => {
        qsa(".validation-range-button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        state.validationWindow = Number(button.dataset.window);
        state.validationStart = 0;
        renderValidation();
      });
    });

    qs("#weatherTargetSelect").addEventListener("change", (evt) => {
      state.weatherTarget = evt.target.value;
      renderWeather();
    });

    const historyRange = qs("#historyRange");
    if (historyRange) {
      historyRange.addEventListener("input", (evt) => {
        state.historyStart = Number(evt.target.value);
        qsa(".range-button").forEach((item) => item.classList.remove("active"));
        renderHistory();
      });
    }

    qsa(".range-button").forEach((button) => {
      button.addEventListener("click", () => {
        qsa(".range-button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        if (button.dataset.window) {
          state.historyWindow = Number(button.dataset.window);
          state.historyStart = 0;
        }
        if (button.dataset.quarter) {
          state.historyWindow = 92;
          state.historyStart = Number(button.dataset.quarter);
        }
        renderHistory();
      });
    });
  }

  function init() {
    if (!data) {
      document.body.innerHTML = "<p>未找到 dashboard_data.js，请检查 assets/data 目录。</p>";
      return;
    }
    renderKpis();
    renderForecast();
    renderForecastTable();
    renderHistory();
    renderPerformanceTable();
    renderR2Chart();
    renderWeather();
    renderValidation();
    bindInteractions();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
