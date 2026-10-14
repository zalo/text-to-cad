/**
 * Unit tests for the radiosity pieces that run without a GPU: the display
 * material patch, the shadow-catching floor, the quality presets and the
 * stage's settings normalization.
 */
import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";

import {
  applyRadiosityPatch,
  hasRadiosityPatch,
  RADIOSITY_DIRECTION_OPTIONS,
  RADIOSITY_QUALITY,
  removeRadiosityPatch
} from "./radiosityLightmap.js";
import { RadiosityFloor } from "./radiosityFloor.js";
import {
  DEFAULT_RADIOSITY_STAGE_SETTINGS,
  normalizeRadiosityStageSettings
} from "./radiosityStage.js";

function uniforms() {
  return { uRadWeight: { value: 1 }, uRadShadow: { value: 0.5 }, uRadIntensity: { value: 1 }, uRadReplaceAmbient: { value: 0 } };
}

function shaderStub() {
  return {
    uniforms: {},
    vertexShader: "#include <common>\n",
    fragmentShader:
      "#include <common>\n#include <lights_fragment_maps>\n#include <aomap_fragment>\n"
  };
}

test("applyRadiosityPatch attaches the lightmap and chains the edge shader hook", () => {
  const material = new THREE.MeshPhysicalMaterial();
  const calls = [];
  material.onBeforeCompile = function edgeShader() {
    calls.push("edges");
  };
  const texture = new THREE.Texture();

  applyRadiosityPatch(material, texture, uniforms());

  assert.equal(hasRadiosityPatch(material), true);
  assert.equal(material.lightMap, texture);
  assert.equal(material.lightMapIntensity, 1);
  // the edge hook's source keys the program; the patch appends its own tag
  assert.match(material.customProgramCacheKey(), /edges[\s\S]*\|radiosity$/);

  const shader = shaderStub();
  material.onBeforeCompile(shader, null);
  assert.deepEqual(calls, ["edges"]);
  assert.ok(shader.uniforms.uRadWeight);
  assert.ok(shader.uniforms.uRadShadow);
  assert.ok(shader.uniforms.uRadIntensity);
  assert.ok(!shader.fragmentShader.includes("#include <lights_fragment_maps>"));
  assert.ok(shader.fragmentShader.includes("tcvRadVisibility"));
  assert.ok(shader.fragmentShader.includes("irradiance *= mix( tcvRadVisibility, 1.0 - uRadWeight, uRadReplaceAmbient )"));
  assert.ok(shader.fragmentShader.includes("#include <aomap_fragment>"));
});

test("applyRadiosityPatch is idempotent and only recompiles on first attach", () => {
  const material = new THREE.MeshPhysicalMaterial();
  const a = new THREE.Texture();
  const b = new THREE.Texture();
  const v0 = material.version;
  applyRadiosityPatch(material, a, uniforms());
  const v1 = material.version;
  assert.ok(v1 > v0);
  const hook = material.onBeforeCompile;
  applyRadiosityPatch(material, b, uniforms());
  assert.equal(material.onBeforeCompile, hook);
  assert.equal(material.lightMap, b);
  assert.equal(material.version, v1, "texture swap must not recompile");
  assert.equal(material.customProgramCacheKey(), "radiosity");
});

test("removeRadiosityPatch restores the previous hooks", () => {
  const material = new THREE.MeshPhysicalMaterial();
  const previous = () => {};
  material.onBeforeCompile = previous;
  applyRadiosityPatch(material, new THREE.Texture(), uniforms());
  removeRadiosityPatch(material);
  assert.equal(hasRadiosityPatch(material), false);
  assert.equal(material.onBeforeCompile, previous);
  assert.equal(material.lightMap, null);
});

test("applyRadiosityPatch ignores non-materials", () => {
  assert.doesNotThrow(() => applyRadiosityPatch(null, new THREE.Texture(), uniforms()));
  assert.doesNotThrow(() => applyRadiosityPatch({}, new THREE.Texture(), uniforms()));
});

test("RadiosityFloor builds a uv1 plane centred under the model", () => {
  const floor = new RadiosityFloor();
  floor.configure(new THREE.Vector3(10, 20, 30), 5, 100);
  const mesh = floor.mesh;
  assert.notEqual(mesh, null);
  assert.equal(mesh.position.z, 5);
  assert.equal(floor.group.visible, true);
  const uv1 = floor.geometry.getAttribute("uv1");
  assert.equal(uv1.count, 16);
  // inner square = 4 × extent, centred on the model
  const pos = floor.geometry.getAttribute("position");
  const xs = new Set();
  for (let i = 0; i < pos.count; i++) xs.add(pos.getX(i));
  assert.ok(xs.has(10 - 200));
  assert.ok(xs.has(10 + 200));
  assert.equal(floor.innerSize, 400);
});

test("RadiosityFloor.setAtlasRect maps the inner square into the atlas, repeatably", () => {
  const floor = new RadiosityFloor();
  floor.configure(new THREE.Vector3(), 0, 10);
  floor.setAtlasRect(100, 200, 300, 300, 1024);
  let uv1 = floor.geometry.getAttribute("uv1");
  const us = new Set();
  const vs = new Set();
  for (let i = 0; i < uv1.count; i++) {
    us.add(uv1.getX(i));
    vs.add(uv1.getY(i));
  }
  assert.deepEqual([...us].sort(), [100 / 1024, 400 / 1024]);
  assert.deepEqual([...vs].sort(), [200 / 1024, 500 / 1024]);
  // A second mapping (quality change → new atlas) must come from the original
  // corner flags, not from the already-mapped attribute.
  floor.setAtlasRect(2, 6, 1020, 1020, 2048);
  uv1 = floor.geometry.getAttribute("uv1");
  const us2 = new Set();
  for (let i = 0; i < uv1.count; i++) us2.add(uv1.getX(i));
  assert.deepEqual([...us2].sort(), [2 / 2048, 1022 / 2048]);
  floor.dispose();
  assert.equal(floor.mesh, null);
});

test("RadiosityFloor albedo follows the stage floor colour", () => {
  const floor = new RadiosityFloor();
  floor.setAlbedo(new THREE.Color(0.1, 0.2, 0.3));
  assert.ok(Math.abs(floor.albedo().g - 0.2) < 1e-6);
  floor.setShadowIntensity(1.7);
  assert.equal(floor._material.uniforms.uShadow.value, 1);
});

test("quality presets use square direction grids and sane sizes", () => {
  for (const preset of Object.values(RADIOSITY_QUALITY)) {
    const side = Math.sqrt(preset.directions);
    assert.ok(Number.isInteger(side));
    assert.equal(preset.lightmapResolution % 2, 0);
    assert.ok(preset.gatherRadius >= 1);
    assert.ok(preset.maxSamples >= 0);
  }
  assert.deepEqual([...RADIOSITY_DIRECTION_OPTIONS], [1, 4, 9, 16]);
});

test("normalizeRadiosityStageSettings clamps and falls back", () => {
  assert.deepEqual(normalizeRadiosityStageSettings(), { ...DEFAULT_RADIOSITY_STAGE_SETTINGS });
  const normalized = normalizeRadiosityStageSettings({
    quality: "ULTRA",
    directions: 5,
    maxSamples: -10,
    shadow: 3,
    bounce: "x",
    envSampling: "uniform"
  });
  assert.equal(normalized.quality, "medium");
  assert.equal(normalized.directions, 16);
  assert.equal(normalized.maxSamples, 0);
  assert.equal(normalized.shadow, 1);
  assert.equal(normalized.bounce, 0);
  assert.equal(normalized.envSampling, "uniform");
});

test("integrateUpIrradiance of a uniform unit sky is π", async () => {
  const { integrateUpIrradiance } = await import("./radiosityLightmap.js");
  const w = 128;
  const h = 64;
  const lum = new Float32Array(w * h).fill(1);
  assert.ok(Math.abs(integrateUpIrradiance(lum, w, h) - Math.PI) < 1e-2);
  // Light only below the horizon contributes nothing to an up-facing surface.
  for (let i = 0; i < h / 2; i++) for (let j = 0; j < w; j++) lum[i * w + j] = 0;
  assert.ok(Math.abs(integrateUpIrradiance(lum, w, h)) < 1e-6);
});
