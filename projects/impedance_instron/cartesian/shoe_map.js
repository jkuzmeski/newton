// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

/* Shared shoe compression heat-map renderer. This file is inlined into reports. */
(function(global) {
  "use strict";
  function unpack(block) {
    if (block instanceof Float32Array) return block;
    if (ArrayBuffer.isView(block)) return block instanceof Float32Array ? block : Float32Array.from(block);
    if (block instanceof ArrayBuffer) return new Float32Array(block);
    if (Array.isArray(block)) return Float32Array.from(block);
    if (!block) return new Float32Array(0);
    if (block.f32le) {
      const bytes = Uint8Array.from(atob(block.f32le), c => c.charCodeAt(0));
      const values = new Float32Array(bytes.byteLength / 4);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < values.length; ++i) values[i] = view.getFloat32(i * 4, true);
      return values;
    }
    if (block.data && ArrayBuffer.isView(block.data)) return unpack(block.data);
    return Float32Array.from(block);
  }
  function frameIndex(times, t) {
    if (!times || !times.length) return 0;
    let lo = 0, hi = times.length - 1;
    const value = Number.isFinite(Number(t)) ? Number(t) : 0;
    if (value <= times[0]) return 0;
    if (value >= times[hi]) return hi;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (times[mid] <= value) lo = mid; else hi = mid - 1;
    }
    return lo;
  }
  function color(value) {
    const u = Math.max(0, Math.min(1, Number(value) || 0));
    let rgb;
    if (u < 1 / 3) rgb = [0, u * 3, 1];
    else if (u < 2 / 3) rgb = [u * 3 - 1, 1, 2 - u * 3];
    else rgb = [1, 3 - u * 3, 0];
    return `rgb(${rgb.map(v => Math.round(v * 255)).join(",")})`;
  }
  function resize(canvas, height) {
    const ratio = global.devicePixelRatio || 1;
    const width = Math.max(0, Number(canvas.clientWidth) || 0);
    const h = Math.max(0, Number(height) || 0);
    canvas.width = Math.max(0, Math.round(width * ratio));
    canvas.height = Math.max(0, Math.round(h * ratio));
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, h);
    return [context, width, h];
  }
  function draw(canvas, layer, options) {
    options = options || {};
    const height = Math.max(0, Number(options.height) || 260);
    const [context, width, canvasHeight] = resize(canvas, height);
    const anchors = (layer.anchor_local_m || []).map(p => [
      Number(p[0]) + Number((layer.mount_m || [0, 0])[0] || 0),
      Number(p[1]) + Number((layer.mount_m || [0, 0])[1] || 0),
    ]);
    const spacing = Math.max(0, Number(layer.spacing_m) || 0);
    const count = anchors.length;
    const xs = anchors.map(p => p[0]), ys = anchors.map(p => p[1]);
    let xmin = count ? Math.min(...xs) - spacing : -spacing;
    let xmax = count ? Math.max(...xs) + spacing : spacing;
    let ymin = count ? Math.min(...ys) - spacing : -spacing;
    let ymax = count ? Math.max(...ys) + spacing : spacing;
    if (!(xmax > xmin)) { xmin -= .5; xmax += .5; }
    if (!(ymax > ymin)) { ymin -= .5; ymax += .5; }
    const scale = Math.max(0, Math.min((width - 110) / (xmax - xmin), (canvasHeight - 55) / (ymax - ymin)));
    const ox = width / 2 - (xmin + xmax) * scale / 2;
    const oy = canvasHeight / 2 + (ymin + ymax) * scale / 2;
    const xy = p => [ox + p[0] * scale, oy - p[1] * scale];
    const frame = Math.max(0, Math.min(
      Math.max(0, Math.floor(Number(options.frame) || 0)),
      Math.max(0, (layer.time_s || []).length - 1),
    ));
    const compression = unpack(layer.compression);
    const rest = layer.rest_length_m || [];
    const metric = options.metric === "strain" ? "strain" : "mm";
    const mmMax = Math.max(1e-12, Number(options.mm_max) || 0);
    const valueAt = i => {
      const compressionValue = Number(compression[frame * count + i]) || 0;
      return metric === "strain" ? compressionValue / Math.max(1e-12, Number(rest[i]) || 0) : compressionValue * 1000 / mmMax;
    };
    const projection = {
      scale, ox, oy, frame,
      offsetX: Number((layer.mount_m || [0, 0])[0]) || 0,
      offsetY: Number((layer.mount_m || [0, 0])[1]) || 0,
      xmin, xmax, ymin, ymax,
      spacing_m: spacing,
    };
    if (!width || !canvasHeight || !count) return projection;
    context.font = "11px system-ui";
    const size = Math.max(0, spacing * scale * .82);
    for (let i = 0; i < count; ++i) {
      const point = xy(anchors[i]);
      context.fillStyle = color(valueAt(i));
      if (layer.driven && layer.driven[i]) {
        context.fillRect(point[0] - size / 2, point[1] - size / 2, size, size);
      } else if (size > 0) {
        context.beginPath();
        context.arc(point[0], point[1], Math.max(0, size * .43), 0, 2 * Math.PI);
        context.fill();
      }
    }
    if (options.slice_y_m !== null && options.slice_y_m !== undefined && spacing > 0) {
      const y = Number(options.slice_y_m) || 0;
      const top = xy([xmin, y + spacing / 2]), bottom = xy([xmax, y - spacing / 2]);
      context.strokeStyle = "#233b4a";
      context.lineWidth = 1.5;
      context.setLineDash([5, 3]);
      context.strokeRect(top[0], top[1], bottom[0] - top[0], bottom[1] - top[1]);
      context.setLineDash([]);
    }
    context.fillStyle = "#536774";
    for (let j = 0; j < 5; ++j) {
      const x = xmin + (xmax - xmin) * j / 4;
      context.textAlign = "center";
      context.fillText((x * 1000).toFixed(0), xy([x, ymin])[0], xy([x, ymin])[1] + 15);
    }
    for (let j = 0; j < 3; ++j) {
      const y = ymin + (ymax - ymin) * j / 2;
      context.textAlign = "right";
      context.fillText((y * 1000).toFixed(0), xy([xmin, y])[0] - 7, xy([xmin, y])[1] + 4);
    }
    context.textAlign = "center";
    context.fillText("Intrinsic shoe forward x [mm]", width / 2, canvasHeight - 2);
    context.textAlign = "left";
    context.fillText("Width y [mm]", 10, 15);
    const times = layer.time_s || [];
    const saved = times.length ? Number(times[Math.min(frame, times.length - 1)]).toFixed(3) + " s" : "";
    context.fillText(layer.label || (saved ? "Saved spring frame: " + saved : ""), 10, 30);
    return projection;
  }
  global.NativeShoeMap = {unpack, frameIndex, color, draw};
})(window);
