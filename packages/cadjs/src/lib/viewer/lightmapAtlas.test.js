import assert from "node:assert/strict";
import test from "node:test";

import {
  chartMesh,
  LIGHTMAP_FACE_ID_NONE,
  packCharts,
  writeAtlasUv
} from "./lightmapAtlas.js";

/** Unit cube: 24 vertices (4 per face), 12 triangles, 6 faces, no shared vertices. */
function cube({ withUv = false, faceIds = true } = {}) {
  const positions = [];
  const uv = [];
  const index = [];
  const ids = [];
  const faces = [
    // normal axis, sign, u axis, v axis
    [0, 1, 1, 2],
    [0, -1, 1, 2],
    [1, 1, 2, 0],
    [1, -1, 2, 0],
    [2, 1, 0, 1],
    [2, -1, 0, 1]
  ];
  faces.forEach(([n, s, ua, va], faceIndex) => {
    const base = positions.length / 3;
    for (const [a, b] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
      const p = [0, 0, 0];
      p[n] = s > 0 ? 1 : 0;
      p[ua] = a;
      p[va] = b;
      positions.push(...p);
      uv.push(a * 3, b * 3); // parametric uv with a uniform 3x stretch
    }
    // Wind every face outward so its triangle normal points along ±axis.
    if (s > 0) {
      index.push(base, base + 1, base + 2, base, base + 2, base + 3);
    } else {
      index.push(base, base + 2, base + 1, base, base + 3, base + 2);
    }
    // Sparse topology face rows, as the sidecar hands them out.
    ids.push(100 + faceIndex * 7, 100 + faceIndex * 7);
  });
  return {
    positions: new Float32Array(positions),
    index: new Uint32Array(index),
    uv: withUv ? new Float32Array(uv) : null,
    faceIds: faceIds ? Uint32Array.from(ids) : null,
    matrixWorld: null
  };
}

function assertClose(actual, expected, digits = 5) {
  assert.ok(Math.abs(actual - expected) < 10 ** -digits, `${actual} ≈ ${expected}`);
}

test("faces with parametric UVs become one chart each, no split", () => {
  const charted = chartMesh(cube({ withUv: true }));
  assert.equal(charted.charts.length, 6);
  assert.equal(charted.uvFaces, 6);
  assert.equal(charted.planarFaces, 0);
  assert.equal(charted.remap, null);
  assert.equal(charted.vertexCount, 24);
  // uv stretch of 3 per unit → world size 1x1 for every face
  for (const c of charted.charts) {
    assertClose(c.w, 1);
    assertClose(c.h, 1);
  }
  for (let i = 0; i < charted.local.length; i++) {
    assert.ok(charted.local[i] >= 0 && charted.local[i] <= 1);
  }
});

test("faces without UVs use planar (box) projection per topology face", () => {
  const charted = chartMesh(cube());
  assert.equal(charted.charts.length, 6);
  assert.equal(charted.planarFaces, 6);
  assert.equal(charted.remap, null);
  for (const c of charted.charts) {
    assertClose(c.w, 1);
    assertClose(c.h, 1);
  }
});

test("triangles without a face id share one extra face, split by normal axis", () => {
  const mesh = cube();
  mesh.faceIds.fill(LIGHTMAP_FACE_ID_NONE);
  const charted = chartMesh(mesh);
  // one orphan face × six dominant axes
  assert.equal(charted.charts.length, 6);
  assert.equal(charted.planarFaces, 6);
});

test("trianglesPerFace is accepted as an alternative grouping", () => {
  const mesh = cube({ faceIds: false });
  mesh.trianglesPerFace = [2, 2, 2, 2, 2, 2];
  const charted = chartMesh(mesh);
  assert.equal(charted.charts.length, 6);
});

test("world matrix scales chart sizes", () => {
  const mesh = cube();
  const m = new Array(16).fill(0);
  m[0] = 2;
  m[5] = 2;
  m[10] = 2;
  m[15] = 1;
  mesh.matrixWorld = m;
  const charted = chartMesh(mesh);
  for (const c of charted.charts) {
    assertClose(c.w, 2);
    assertClose(c.h, 2);
  }
});

test("vertices shared across faces are split", () => {
  // Two faces sharing an edge (vertices 1 and 2): a folded strip.
  const positions = new Float32Array([
    0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1
  ]);
  // face A: quad 0-1-2-3 in z=0 plane; face B: quad 1-4-5-2 in x=1 plane
  const index = new Uint32Array([0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2]);
  const charted = chartMesh({
    positions,
    index,
    uv: null,
    faceIds: Uint32Array.from([7, 7, 9, 9]),
    matrixWorld: null
  });
  assert.equal(charted.charts.length, 2);
  assert.notEqual(charted.remap, null);
  // 6 original + 2 duplicated (vertices 1 and 2 used by both faces)
  assert.equal(charted.vertexCount, 8);
  for (let i = 0; i < 6; i++) assert.equal(charted.remap[i], i);
  assert.deepEqual(Array.from(charted.remap.slice(6)).sort(), [1, 2]);
  // no triangle references a vertex from another chart
  for (let t = 0; t < 4; t++) {
    const ch = charted.chartOfVertex[charted.index[t * 3]];
    assert.equal(charted.chartOfVertex[charted.index[t * 3 + 1]], ch);
    assert.equal(charted.chartOfVertex[charted.index[t * 3 + 2]], ch);
  }
});

test("degenerate UVs fall back to planar", () => {
  const mesh = cube({ withUv: true });
  mesh.uv.fill(0.25); // all identical → zero uv area
  const charted = chartMesh(mesh);
  assert.equal(charted.uvFaces, 0);
  assert.equal(charted.planarFaces, 6);
});

test("non-indexed meshes without face info work", () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const charted = chartMesh({ positions, index: null, uv: null, matrixWorld: null });
  assert.equal(charted.charts.length, 1);
  assert.equal(charted.vertexCount, 3);
  assert.deepEqual(Array.from(charted.index), [0, 1, 2]);
});

test("packCharts places charts without overlap inside the atlas", () => {
  const charts = [];
  for (let i = 0; i < 50; i++) {
    charts.push({ w: 1 + (i % 7), h: 1 + (i % 5), fixed: false, x: 0, y: 0, pw: 0, ph: 0 });
  }
  const res = packCharts(charts, 256, 2);
  assert.notEqual(res, null);
  assert.ok(res.scale > 0);
  for (const c of charts) {
    assert.ok(c.pw >= 1 && c.ph >= 1);
    assert.ok(c.x - 2 >= 0 && c.y - 2 >= 0);
    assert.ok(c.x + c.pw + 2 <= 256 && c.y + c.ph + 2 <= 256);
  }
  for (let i = 0; i < charts.length; i++) {
    for (let j = i + 1; j < charts.length; j++) {
      const a = charts[i];
      const b = charts[j];
      const separated =
        a.x + a.pw + 2 <= b.x - 2 ||
        b.x + b.pw + 2 <= a.x - 2 ||
        a.y + a.ph + 2 <= b.y - 2 ||
        b.y + b.ph + 2 <= a.y - 2;
      assert.ok(separated, `charts ${i} and ${j} overlap`);
    }
  }
  assert.ok(res.fill > 0.3);
});

test("fixed charts keep their texel size and the origin stays free", () => {
  const charts = [
    { w: 100, h: 100, fixed: true, x: 0, y: 0, pw: 0, ph: 0 },
    { w: 1, h: 1, fixed: false, x: 0, y: 0, pw: 0, ph: 0 }
  ];
  const res = packCharts(charts, 256, 2, 4);
  assert.notEqual(res, null);
  assert.equal(charts[0].pw, 100);
  assert.equal(charts[0].ph, 100);
  for (const c of charts) {
    assert.ok(c.x - 2 >= 4 || c.y - 2 >= 4, "chart must not cover the atlas origin");
  }
});

test("packCharts returns null when charts cannot fit", () => {
  const charts = [];
  for (let i = 0; i < 2000; i++) {
    charts.push({ w: 1, h: 1, fixed: false, x: 0, y: 0, pw: 0, ph: 0 });
  }
  assert.equal(packCharts(charts, 16, 2), null);
});

test("writeAtlasUv maps local coords into the packed rect", () => {
  const charted = chartMesh(cube({ withUv: true }));
  const res = packCharts(charted.charts, 128, 2);
  assert.notEqual(res, null);
  const uv1 = writeAtlasUv(charted, 128);
  assert.equal(uv1.length, 48);
  for (let v = 0; v < 24; v++) {
    const c = charted.charts[charted.chartOfVertex[v]];
    const u = uv1[v * 2] * 128;
    const w = uv1[v * 2 + 1] * 128;
    assert.ok(u >= c.x - 1e-3 && u <= c.x + c.pw + 1e-3);
    assert.ok(w >= c.y - 1e-3 && w <= c.y + c.ph + 1e-3);
  }
});
