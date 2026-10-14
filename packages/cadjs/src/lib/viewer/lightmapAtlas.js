/**
 * Lightmap UV atlas builder (renderer-agnostic, no three.js dependency).
 *
 * The radiosity solver needs every surface to own a unique, non-overlapping
 * region of one shared lightmap (attribute `uv1`). CAD meshes arrive as
 * triangle lists grouped into B-rep faces (a face id per triangle from the
 * topology sidecar) and, optionally, with parametric UVs. This module turns
 * them into charts and packs the charts of ALL meshes into one atlas:
 *
 * - A face with usable parametric UVs becomes one chart: its UV rectangle,
 *   sized in the atlas by the face's world size (a per-face Jacobian estimate
 *   so u and v get texels proportional to their world length).
 * - A face without UVs (the GLB render packages carry none) is split by the
 *   dominant axis of each triangle's normal into up to six planar charts (box
 *   projection). Triangles without a face id form one extra face.
 * - Vertices shared by triangles of different charts are duplicated, so the
 *   output may carry more vertices than the input (`remap` maps new → old).
 *   Render packages are de-indexed per triangle, so in practice nothing is
 *   split and `uv1` can be written onto the existing geometry in place.
 *
 * Packing uses a fixed-size atlas: a global texels-per-world-unit scale is
 * searched so the charts fill the atlas, with a texel gutter around each chart
 * (the solver dilates lighting into the gutters so bilinear sampling never
 * bleeds across charts). Charts flagged `fixed` keep their pixel size (used
 * for the floor, which gets a large reserved region). The atlas origin is kept
 * empty (`originGuard`) so geometry without `uv1` reads an empty texel.
 */

/** Face id marking "no topology face" in a per-triangle face id array. */
export const LIGHTMAP_FACE_ID_NONE = 0xffffffff;

const _m = new Float64Array(16);

function transformPoint(out, o, px, py, pz, m) {
  if (m === null) {
    out[o] = px;
    out[o + 1] = py;
    out[o + 2] = pz;
    return;
  }
  out[o] = m[0] * px + m[4] * py + m[8] * pz + m[12];
  out[o + 1] = m[1] * px + m[5] * py + m[9] * pz + m[13];
  out[o + 2] = m[2] * px + m[6] * py + m[10] * pz + m[14];
}

function sequentialIndex(n) {
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  return idx;
}

/**
 * Resolve the face of every triangle from either a per-triangle face id array
 * (`faceIds`, `LIGHTMAP_FACE_ID_NONE` = no face) or a triangle count per face
 * (`trianglesPerFace`). Face ids are remapped to a dense 0..n-1 range.
 */
function resolveFaceOfTriangle(mesh, triCount) {
  const faceOfTri = new Uint32Array(triCount);
  if (mesh.faceIds && mesh.faceIds.length > 0) {
    const dense = new Map();
    let orphan = -1;
    for (let t = 0; t < triCount; t++) {
      const raw = t < mesh.faceIds.length ? mesh.faceIds[t] : LIGHTMAP_FACE_ID_NONE;
      if (raw === LIGHTMAP_FACE_ID_NONE || raw === undefined || raw === null) {
        if (orphan < 0) {
          orphan = dense.size;
          dense.set(-1, orphan);
        }
        faceOfTri[t] = orphan;
        continue;
      }
      let id = dense.get(raw);
      if (id === undefined) {
        id = dense.size;
        dense.set(raw, id);
      }
      faceOfTri[t] = id;
    }
    return { faceOfTri, faceCount: Math.max(dense.size, 1) };
  }
  if (mesh.trianglesPerFace && mesh.trianglesPerFace.length > 0) {
    let t = 0;
    let faceCount = mesh.trianglesPerFace.length;
    for (let f = 0; f < mesh.trianglesPerFace.length; f++) {
      const n = mesh.trianglesPerFace[f];
      for (let k = 0; k < n && t < triCount; k++) faceOfTri[t++] = f;
    }
    // Leftover triangles (malformed input) get their own face.
    if (t < triCount) {
      for (; t < triCount; t++) faceOfTri[t] = faceCount;
      faceCount++;
    }
    return { faceOfTri, faceCount };
  }
  return { faceOfTri, faceCount: 1 };
}

/**
 * Chart one mesh. Pure function: never touches the inputs.
 *
 * @param {object} mesh
 * @param {Float32Array} mesh.positions - Flat xyz positions (object space)
 * @param {Uint32Array|Uint16Array|null} mesh.index - Triangle indices, or null for a triangle soup
 * @param {Float32Array|null} [mesh.uv] - Flat per-vertex parametric uv
 * @param {Uint32Array|null} [mesh.faceIds] - Topology face id per triangle
 * @param {ArrayLike<number>|null} [mesh.trianglesPerFace] - Alternative face grouping
 * @param {ArrayLike<number>|null} [mesh.matrixWorld] - Column-major 4x4 world matrix elements
 * @param {number} [minUvArea] - Faces whose total UV area is below this use planar projection
 */
export function chartMesh(mesh, minUvArea = 1e-10) {
  const nv = mesh.positions.length / 3;
  const index = mesh.index || sequentialIndex(nv);
  const triCount = Math.floor(index.length / 3);

  // World-space positions (doubles) for size estimates.
  let m = null;
  if (mesh.matrixWorld) {
    for (let i = 0; i < 16; i++) _m[i] = mesh.matrixWorld[i];
    m = _m;
  }
  const world = new Float64Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    transformPoint(
      world,
      i * 3,
      mesh.positions[i * 3],
      mesh.positions[i * 3 + 1],
      mesh.positions[i * 3 + 2],
      m
    );
  }

  const { faceOfTri, faceCount } = resolveFaceOfTriangle(mesh, triCount);

  // --- Per-face UV analysis: usable? and the u/v world stretch ---
  const uv = mesh.uv && mesh.uv.length >= nv * 2 ? mesh.uv : null;
  const faceUvOk = new Uint8Array(faceCount);
  const faceSu = new Float64Array(faceCount); // world length per unit u
  const faceSv = new Float64Array(faceCount);
  if (uv) {
    const uvArea = new Float64Array(faceCount);
    const wArea = new Float64Array(faceCount);
    const suAcc = new Float64Array(faceCount);
    const svAcc = new Float64Array(faceCount);
    const bad = new Uint8Array(faceCount);
    for (let t = 0; t < triCount; t++) {
      const f = faceOfTri[t];
      const a = index[t * 3];
      const b = index[t * 3 + 1];
      const c = index[t * 3 + 2];
      const ua = uv[a * 2];
      const va = uv[a * 2 + 1];
      const ub = uv[b * 2];
      const vb = uv[b * 2 + 1];
      const uc = uv[c * 2];
      const vc = uv[c * 2 + 1];
      if (
        !Number.isFinite(ua) || !Number.isFinite(va) ||
        !Number.isFinite(ub) || !Number.isFinite(vb) ||
        !Number.isFinite(uc) || !Number.isFinite(vc)
      ) {
        bad[f] = 1;
        continue;
      }
      const du1 = ub - ua;
      const dv1 = vb - va;
      const du2 = uc - ua;
      const dv2 = vc - va;
      const det = du1 * dv2 - du2 * dv1;
      const e1x = world[b * 3] - world[a * 3];
      const e1y = world[b * 3 + 1] - world[a * 3 + 1];
      const e1z = world[b * 3 + 2] - world[a * 3 + 2];
      const e2x = world[c * 3] - world[a * 3];
      const e2y = world[c * 3 + 1] - world[a * 3 + 1];
      const e2z = world[c * 3 + 2] - world[a * 3 + 2];
      const cx = e1y * e2z - e1z * e2y;
      const cy = e1z * e2x - e1x * e2z;
      const cz = e1x * e2y - e1y * e2x;
      const area3 = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
      const area2 = 0.5 * Math.abs(det);
      uvArea[f] += area2;
      wArea[f] += area3;
      if (area2 > 1e-14 && area3 > 0) {
        // Jacobian J = [e1 e2] * inv([duv1 duv2]); columns = dP/du, dP/dv
        const inv = 1 / det;
        const i00 = dv2 * inv;
        const i01 = -du2 * inv;
        const i10 = -dv1 * inv;
        const i11 = du1 * inv;
        const jux = e1x * i00 + e2x * i10;
        const juy = e1y * i00 + e2y * i10;
        const juz = e1z * i00 + e2z * i10;
        const jvx = e1x * i01 + e2x * i11;
        const jvy = e1y * i01 + e2y * i11;
        const jvz = e1z * i01 + e2z * i11;
        const su = Math.sqrt(jux * jux + juy * juy + juz * juz);
        const sv = Math.sqrt(jvx * jvx + jvy * jvy + jvz * jvz);
        suAcc[f] += su * area3;
        svAcc[f] += sv * area3;
      }
    }
    for (let f = 0; f < faceCount; f++) {
      if (bad[f] || uvArea[f] < minUvArea || wArea[f] <= 0) continue;
      const su = suAcc[f] / wArea[f];
      const sv = svAcc[f] / wArea[f];
      if (!(su > 0) || !(sv > 0) || !Number.isFinite(su) || !Number.isFinite(sv)) continue;
      // Extreme anisotropy means the UVs are unusable for a lightmap.
      const ratio = su / sv;
      if (ratio > 1e3 || ratio < 1e-3) continue;
      faceUvOk[f] = 1;
      faceSu[f] = su;
      faceSv[f] = sv;
    }
  }

  // --- Chart key per triangle: UV faces → face; planar → face*7 + axis ---
  // Chart ids are assigned in first-seen order.
  const chartKeyToId = new Map();
  const chartOfTri = new Uint32Array(triCount);
  const chartAxis = []; // -1 for UV charts, 0..5 for planar
  const chartFace = [];
  for (let t = 0; t < triCount; t++) {
    const f = faceOfTri[t];
    let key;
    let axis = -1;
    if (faceUvOk[f]) {
      key = f * 7 + 6;
    } else {
      const a = index[t * 3];
      const b = index[t * 3 + 1];
      const c = index[t * 3 + 2];
      const e1x = world[b * 3] - world[a * 3];
      const e1y = world[b * 3 + 1] - world[a * 3 + 1];
      const e1z = world[b * 3 + 2] - world[a * 3 + 2];
      const e2x = world[c * 3] - world[a * 3];
      const e2y = world[c * 3 + 1] - world[a * 3 + 1];
      const e2z = world[c * 3 + 2] - world[a * 3 + 2];
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;
      const ax = Math.abs(nx);
      const ay = Math.abs(ny);
      const az = Math.abs(nz);
      if (ax >= ay && ax >= az) axis = nx >= 0 ? 0 : 1;
      else if (ay >= az) axis = ny >= 0 ? 2 : 3;
      else axis = nz >= 0 ? 4 : 5;
      key = f * 7 + axis;
    }
    let id = chartKeyToId.get(key);
    if (id === undefined) {
      id = chartKeyToId.size;
      chartKeyToId.set(key, id);
      chartAxis.push(axis);
      chartFace.push(f);
    }
    chartOfTri[t] = id;
  }
  const chartCount = chartKeyToId.size;

  // --- Vertex splitting: one output vertex per (original vertex, chart) ---
  // The first chart that uses an input vertex keeps the input index, so the
  // output is identity-ordered and, without conflicts, index-compatible with
  // the input geometry. Extra copies for other charts are appended after nv.
  const firstChart = new Int32Array(nv).fill(-1);
  const overflow = new Map();
  const newToOld = [];
  const newChart = [];
  for (let v = 0; v < nv; v++) {
    newToOld.push(v);
    newChart.push(0);
  }
  const newIndex = new Uint32Array(triCount * 3);
  let split = false;
  for (let t = 0; t < triCount; t++) {
    const ch = chartOfTri[t];
    for (let k = 0; k < 3; k++) {
      const old = index[t * 3 + k];
      let nidx;
      if (firstChart[old] === -1) {
        firstChart[old] = ch;
        newChart[old] = ch;
        nidx = old;
      } else if (firstChart[old] === ch) {
        nidx = old;
      } else {
        split = true;
        const key = `${old},${ch}`;
        const existing = overflow.get(key);
        if (existing !== undefined) {
          nidx = existing;
        } else {
          nidx = newToOld.length;
          overflow.set(key, nidx);
          newToOld.push(old);
          newChart.push(ch);
        }
      }
      newIndex[t * 3 + k] = nidx;
    }
  }
  const vertexCount = newToOld.length;
  const chartOfVertex = Uint32Array.from(newChart);

  // --- Chart-local coordinates and world sizes ---
  const local = new Float32Array(vertexCount * 2);
  const raw = new Float64Array(vertexCount * 2);
  const minA = new Float64Array(chartCount).fill(Infinity);
  const maxA = new Float64Array(chartCount).fill(-Infinity);
  const minB = new Float64Array(chartCount).fill(Infinity);
  const maxB = new Float64Array(chartCount).fill(-Infinity);
  const used = new Uint8Array(chartCount);
  for (let v = 0; v < vertexCount; v++) {
    const old = newToOld[v];
    const ch = chartOfVertex[v];
    if (firstChart[old] === -1) continue; // unreferenced vertex
    const axis = chartAxis[ch];
    let a;
    let b;
    if (axis < 0) {
      a = uv[old * 2];
      b = uv[old * 2 + 1];
    } else {
      const wx = world[old * 3];
      const wy = world[old * 3 + 1];
      const wz = world[old * 3 + 2];
      // Project onto the plane perpendicular to the dominant axis, keeping
      // a consistent handedness so charts of one face line up.
      if (axis <= 1) {
        a = axis === 0 ? wy : -wy;
        b = wz;
      } else if (axis <= 3) {
        a = axis === 2 ? wz : -wz;
        b = wx;
      } else {
        a = axis === 4 ? wx : -wx;
        b = wy;
      }
    }
    raw[v * 2] = a;
    raw[v * 2 + 1] = b;
    used[ch] = 1;
    if (a < minA[ch]) minA[ch] = a;
    if (a > maxA[ch]) maxA[ch] = a;
    if (b < minB[ch]) minB[ch] = b;
    if (b > maxB[ch]) maxB[ch] = b;
  }

  const charts = [];
  let uvFaces = 0;
  let planarFaces = 0;
  const uvFaceSeen = new Uint8Array(faceCount);
  for (let ch = 0; ch < chartCount; ch++) {
    const axis = chartAxis[ch];
    const f = chartFace[ch];
    if (axis < 0) {
      if (!uvFaceSeen[f]) {
        uvFaceSeen[f] = 1;
        uvFaces++;
      }
    } else {
      planarFaces++;
    }
    let w = 0;
    let h = 0;
    if (used[ch]) {
      const da = maxA[ch] - minA[ch];
      const db = maxB[ch] - minB[ch];
      if (axis < 0) {
        w = da * faceSu[f];
        h = db * faceSv[f];
      } else {
        w = da;
        h = db;
      }
    }
    charts.push({ w, h, fixed: false, x: 0, y: 0, pw: 0, ph: 0 });
  }

  for (let v = 0; v < vertexCount; v++) {
    const old = newToOld[v];
    if (firstChart[old] === -1) {
      local[v * 2] = 0.5;
      local[v * 2 + 1] = 0.5;
      continue;
    }
    const ch = chartOfVertex[v];
    const da = maxA[ch] - minA[ch];
    const db = maxB[ch] - minB[ch];
    local[v * 2] = da > 0 ? (raw[v * 2] - minA[ch]) / da : 0.5;
    local[v * 2 + 1] = db > 0 ? (raw[v * 2 + 1] - minB[ch]) / db : 0.5;
  }

  return {
    vertexCount,
    index: newIndex,
    remap: split ? Uint32Array.from(newToOld) : null,
    chartOfVertex,
    local,
    charts,
    uvFaces,
    planarFaces
  };
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/**
 * Shelf-free rectangle packer (the potpack algorithm) into a fixed width.
 * Boxes are sorted by height; returns the used height, or Infinity when a
 * box is wider than the atlas.
 */
function packBoxes(boxes, width, guard) {
  // The guard keeps the atlas origin free: a mesh without a `uv1` attribute
  // samples texel (0,0), which must stay empty so the display shader falls
  // back to plain IBL for it.
  const spaces = guard > 0
    ? [
      { x: 0, y: guard, w: width, h: Infinity },
      { x: guard, y: 0, w: width - guard, h: guard }
    ]
    : [{ x: 0, y: 0, w: width, h: Infinity }];
  let height = 0;
  for (const box of boxes) {
    if (box.w > width) return Infinity;
    let placed = false;
    for (let i = spaces.length - 1; i >= 0; i--) {
      const space = spaces[i];
      if (box.w > space.w || box.h > space.h) continue;
      box.x = space.x;
      box.y = space.y;
      height = Math.max(height, box.y + box.h);
      if (box.w === space.w && box.h === space.h) {
        const last = spaces.pop();
        if (i < spaces.length) spaces[i] = last;
      } else if (box.h === space.h) {
        space.x += box.w;
        space.w -= box.w;
      } else if (box.w === space.w) {
        space.y += box.h;
        space.h -= box.h;
      } else {
        spaces.push({
          x: space.x + box.w,
          y: space.y,
          w: space.w - box.w,
          h: box.h
        });
        space.y += box.h;
        space.h -= box.h;
      }
      placed = true;
      break;
    }
    if (!placed) return Infinity;
  }
  return height;
}

/**
 * Pack charts into a `resolution`×`resolution` atlas with `padding` texels of
 * gutter on every side of each chart. Writes `x`, `y`, `pw`, `ph` on the
 * charts. Returns `{ scale, fill }` or null when even the smallest scale does
 * not fit (too many charts).
 */
export function packCharts(charts, resolution, padding = 2, originGuard = 4) {
  if (charts.length === 0) return { scale: 1, fill: 0 };

  let worldArea = 0;
  let fixedArea = 0;
  for (const c of charts) {
    if (c.fixed) fixedArea += (c.w + 2 * padding) * (c.h + 2 * padding);
    else worldArea += Math.max(c.w, 1e-9) * Math.max(c.h, 1e-9);
  }
  const free = Math.max(resolution * resolution - fixedArea, 1);

  const boxes = charts.map((chart) => ({ w: 0, h: 0, x: 0, y: 0, chart }));

  const tryScale = (scale) => {
    for (const box of boxes) {
      const c = box.chart;
      if (c.fixed) {
        box.w = Math.ceil(c.w) + 2 * padding;
        box.h = Math.ceil(c.h) + 2 * padding;
      } else {
        box.w = Math.max(1, Math.ceil(c.w * scale)) + 2 * padding;
        box.h = Math.max(1, Math.ceil(c.h * scale)) + 2 * padding;
      }
    }
    boxes.sort((a, b) => b.h - a.h || b.w - a.w);
    const height = packBoxes(boxes, resolution, originGuard);
    return height <= resolution;
  };

  // Initial guess from the area budget (texel² per world²), then bracket.
  const scale = worldArea > 0 ? Math.sqrt((free * 0.6) / worldArea) : 1;
  let lo = 0;
  let hi = 0;
  if (tryScale(scale)) {
    lo = scale;
    for (let i = 0; i < 12; i++) {
      const next = lo * 1.2;
      if (tryScale(next)) {
        lo = next;
      } else {
        hi = next;
        break;
      }
    }
  } else {
    hi = scale;
    let s = scale;
    let fits = false;
    for (let i = 0; i < 24; i++) {
      s *= 0.75;
      if (tryScale(s)) {
        lo = s;
        fits = true;
        break;
      }
      hi = s;
    }
    if (!fits) return null;
  }
  if (hi > 0) {
    for (let i = 0; i < 8; i++) {
      const mid = (lo + hi) / 2;
      if (tryScale(mid)) lo = mid;
      else hi = mid;
    }
  }
  // Final placement at the best fitting scale.
  tryScale(lo);
  let covered = 0;
  for (const box of boxes) {
    box.chart.x = box.x + padding;
    box.chart.y = box.y + padding;
    box.chart.pw = box.w - 2 * padding;
    box.chart.ph = box.h - 2 * padding;
    covered += box.w * box.h;
  }
  return { scale: lo, fill: covered / (resolution * resolution) };
}

/** Write the atlas-normalized uv1 for a charted mesh after packing. */
export function writeAtlasUv(charted, resolution, out = null) {
  const uv1 = out || new Float32Array(charted.vertexCount * 2);
  const inv = 1 / resolution;
  for (let v = 0; v < charted.vertexCount; v++) {
    const c = charted.charts[charted.chartOfVertex[v]];
    uv1[v * 2] = (c.x + charted.local[v * 2] * c.pw) * inv;
    uv1[v * 2 + 1] = (c.y + charted.local[v * 2 + 1] * c.ph) * inv;
  }
  return uv1;
}
