/**
 * RadiosityStage — connects the viewer's display records to the progressive
 * radiosity solver (`radiosityLightmap.js`).
 *
 * Responsibilities:
 * - Chart every lit display mesh into one shared lightmap atlas
 *   (`lightmapAtlas.js`), writing a `uv1` attribute (in place when the mesh
 *   is de-indexed, which render packages are; through a split copy
 *   otherwise, restored on teardown).
 * - Reserve a large atlas region for the shadow-catching floor and feed its
 *   chart corner to the solver as the unoccluded-ground reference.
 * - Patch the PBR display materials so they read the lightmap
 *   (`applyRadiosityPatch`) and keep their `lightMap` pointing at the
 *   solver's current ping-pong texture.
 * - Detect changes that invalidate the bake every frame: visibility,
 *   transforms and occluder-ness reset the accumulation; added/removed parts
 *   or swapped geometries trigger a rebuild. Colour changes are picked up
 *   progressively (the bounce reads albedo live), so selection highlights do
 *   not restart the bake.
 *
 * The stage is driven from the render loop: `step()` accumulates one bounce
 * per frame and returns true while more frames are needed.
 */

import * as THREE from "three";
import { chartMesh, packCharts, writeAtlasUv } from "./lightmapAtlas.js";
import {
  applyRadiosityPatch,
  createRadiosityDisplayUniforms,
  normalizeRadiosityDirections,
  normalizeRadiosityQuality,
  RadiosityLightmap,
  radiositySupported,
  removeRadiosityPatch
} from "./radiosityLightmap.js";
import { RadiosityFloor } from "./radiosityFloor.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/** Fraction of the atlas side reserved for the floor's chart. */
const FLOOR_ATLAS_FRACTION = 0.5;

/** Chart gutter (texels) on each side. */
const PADDING = 2;

/** Pre-blur (PMREM roughness) applied to the environment for the bake. */
const ENVIRONMENT_BLUR = 0.25;
/**
 * The procedural room's light panels differ per wall (two bright ones on a
 * single side); averaging the room over four rotations about the up axis
 * lights every side of the scene alike while keeping the panels sharp.
 */
const ROOM_ENVIRONMENT_SYMMETRY = 4;
/**
 * The room below the horizon is blacked out for the bake: the ground's
 * contribution to the model comes from the shadow-catching floor's bounce,
 * not from the room's lit lower walls.
 */
const ROOM_BELOW_HORIZON = 0;

const UP = new THREE.Vector3(0, 0, 1);

function luminance(color) {
  return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

function worldVisible(obj) {
  let o = obj;
  while (o) {
    if (!o.visible) return false;
    o = o.parent;
  }
  return true;
}

/** Surfaces that should not block light (glass, mostly transparent). */
function isOccluder(material) {
  if (!material) return false;
  if (material.isMeshPhysicalMaterial && material.transmission > 0.5) return false;
  if (material.transparent && material.opacity < 0.5) return false;
  return true;
}

/** Lit PBR materials get the lightmap; unlit/wireframe display modes do not. */
function isLitMaterial(material) {
  return !!material && (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) === true;
}

/** Diffuse albedo of a display material for the bounce (linear). */
function materialAlbedo(material, out) {
  if (material?.isMeshStandardMaterial) {
    out.copy(material.color).multiplyScalar(1 - material.metalness * 0.9);
    if (material.map) out.multiplyScalar(0.7);
    return out;
  }
  if (material?.color?.isColor) return out.copy(material.color);
  return out.setRGB(0.5, 0.5, 0.5);
}

function materialEmissive(material, out) {
  if (material?.isMeshStandardMaterial && material.emissive) {
    return out.copy(material.emissive).multiplyScalar(material.emissiveIntensity);
  }
  return out.setRGB(0, 0, 0);
}

/** Build a vertex-split copy of `src` following the chart result. */
function buildSplitGeometry(src, charted, uv1) {
  const remap = charted.remap;
  const g = new THREE.BufferGeometry();
  for (const name of Object.keys(src.attributes)) {
    const attr = src.attributes[name];
    if (!attr?.isBufferAttribute) continue;
    const itemSize = attr.itemSize;
    const SrcArray = attr.array.constructor;
    const arr = new SrcArray(charted.vertexCount * itemSize);
    const srcArr = attr.array;
    for (let v = 0; v < charted.vertexCount; v++) {
      const o = remap[v] * itemSize;
      const d = v * itemSize;
      for (let k = 0; k < itemSize; k++) arr[d + k] = srcArr[o + k];
    }
    g.setAttribute(name, new THREE.BufferAttribute(arr, itemSize, attr.normalized));
  }
  g.setAttribute("uv1", new THREE.BufferAttribute(uv1, 2));
  g.setIndex(new THREE.BufferAttribute(charted.index, 1));
  g.userData = { ...src.userData, cadSceneCachedGeometry: false };
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** Order-independent 32-bit hash helpers for change detection. */
function hashInt(h, v) {
  return Math.imul(h ^ (v | 0), 16777619) >>> 0;
}
function hashFloat(h, v) {
  return hashInt(h, Math.round(v * 4096));
}

export const DEFAULT_RADIOSITY_STAGE_SETTINGS = Object.freeze({
  quality: "medium",
  directions: 16,
  maxSamples: 4000,
  shadow: 1,
  // Direct environment light only by default; the Bounce control adds
  // indirect light from the model and floor.
  bounce: 0,
  envSampling: "importance",
  selfBias: 1.5
});

export function normalizeRadiosityStageSettings(settings = {}) {
  const source = settings && typeof settings === "object" ? settings : {};
  const number = (value, fallback, min, max) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? Math.min(Math.max(numeric, min), max) : fallback;
  };
  return {
    quality: normalizeRadiosityQuality(source.quality, DEFAULT_RADIOSITY_STAGE_SETTINGS.quality),
    directions: normalizeRadiosityDirections(source.directions, DEFAULT_RADIOSITY_STAGE_SETTINGS.directions),
    maxSamples: Math.floor(number(source.maxSamples, DEFAULT_RADIOSITY_STAGE_SETTINGS.maxSamples, 0, 100000)),
    shadow: number(source.shadow, DEFAULT_RADIOSITY_STAGE_SETTINGS.shadow, 0, 1),
    bounce: number(source.bounce, DEFAULT_RADIOSITY_STAGE_SETTINGS.bounce, 0, 2),
    envSampling: source.envSampling === "uniform" ? "uniform" : "importance",
    selfBias: number(source.selfBias, DEFAULT_RADIOSITY_STAGE_SETTINGS.selfBias, 0, 8)
  };
}

export class RadiosityStage {
  /** Whether `renderer` can run the solver (WebGL2 + float render targets). */
  static supported(renderer) {
    return radiositySupported(renderer);
  }

  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {object} [settings] - see DEFAULT_RADIOSITY_STAGE_SETTINGS
   */
  constructor(renderer, settings = {}) {
    this._renderer = renderer;
    this._settings = normalizeRadiosityStageSettings(settings);
    this._displayUniforms = createRadiosityDisplayUniforms();
    this._displayUniforms.uRadShadow.value = this._settings.shadow;
    this._solver = null;
    this._floor = new RadiosityFloor();
    this._floorParams = null;
    this._recordsProvider = () => [];
    this._entries = [];
    this._patched = new Set();
    this._needsBuild = true;
    this._built = false;
    this._stateHash = 0;
    this._topologyHash = 0;
    this._info = null;
    this._disposed = false;
    this._environment = { texture: null, rotation: new THREE.Euler(), intensity: 1, symmetry: 1 };
    this._pmremGenerator = null;
    this._roomPmrem = null;
    this._createSolver();
  }

  /** Shadow-catcher group to add to the scene's stage. */
  get group() {
    return this._floor.group;
  }

  get settings() {
    return this._settings;
  }

  /** Atlas/packing diagnostics of the last build (null before the first). */
  get info() {
    return this._info;
  }

  /** Accumulated directions since the last reset. */
  get samples() {
    return this._solver ? this._solver.samples : 0;
  }

  /** 0..1 progress toward the sample budget (1 when unlimited and running). */
  get progress() {
    if (!this._solver) return 0;
    const max = this._solver.maxSamples;
    if (max <= 0) return this._solver.samples > 0 ? 1 : 0;
    return Math.min(1, this._solver.samples / max);
  }

  get converged() {
    return !!this._solver?.converged;
  }

  /** True when the last build produced at least one lit surface. */
  get active() {
    return this._built && this._entries.length > 0;
  }

  /**
   * Apply new settings. Quality and direction changes recreate the solver
   * (new atlas size) and schedule a rebuild; the others apply in place.
   */
  configure(settings = {}) {
    const next = normalizeRadiosityStageSettings({ ...this._settings, ...settings });
    const previous = this._settings;
    this._settings = next;
    this._displayUniforms.uRadShadow.value = next.shadow;
    this._floor.setShadowIntensity(next.shadow);
    if (next.quality !== previous.quality || next.directions !== previous.directions) {
      this._createSolver();
      return;
    }
    const solver = this._solver;
    solver.maxSamples = next.maxSamples;
    if (next.bounce !== previous.bounce) solver.setBounce(next.bounce);
    if (next.envSampling !== previous.envSampling) solver.setEnvSampling(next.envSampling);
    if (next.selfBias !== previous.selfBias) solver.setSelfBias(next.selfBias);
  }

  /** Function returning the live display records (`{ mesh, material, geometry }`). */
  setRecordsProvider(provider) {
    this._recordsProvider = typeof provider === "function" ? provider : () => [];
    this._needsBuild = true;
  }

  /**
   * Configure the ground plane, or null for a floating model (no catcher,
   * no bounce from the ground).
   *
   * @param {object|null} params
   * @param {THREE.Vector3} params.center - World centre of the model bounds
   * @param {number} params.level - Floor height (world Z)
   * @param {number} params.extent - Max extent of the model bounds
   * @param {THREE.Color} [params.albedo] - Floor colour for the bounce (linear)
   */
  setFloor(params) {
    const previous = this._floorParams;
    if (!params) {
      if (previous) {
        this._floorParams = null;
        this._needsBuild = true;
      }
      return;
    }
    const next = {
      center: params.center.clone(),
      level: Number(params.level) || 0,
      extent: Math.max(Number(params.extent) || 0, 1e-6),
      albedo: params.albedo?.isColor ? params.albedo.clone() : new THREE.Color(0.35, 0.35, 0.35)
    };
    const same = previous &&
      previous.center.equals(next.center) &&
      previous.level === next.level &&
      previous.extent === next.extent;
    this._floorParams = next;
    this._floor.setAlbedo(next.albedo);
    if (!same) {
      this._needsBuild = true;
    }
  }

  /**
   * Set the light source: a PMREM environment texture (cube-UV), its
   * rotation and the intensity the display materials scale the lightmap by
   * when they carry no environment map of their own.
   */
  setEnvironment(texture, rotation = null, intensity = 1) {
    const env = this._environment;
    const nextTexture = texture || null;
    const nextIntensity = Math.max(Number(intensity) || 0, 0);
    const nextRotation = rotation?.isEuler ? rotation : null;
    const sameRotation = nextRotation
      ? env.rotation.equals(nextRotation)
      : env.rotation.x === 0 && env.rotation.y === 0 && env.rotation.z === 0;
    if (env.texture === nextTexture && env.intensity === nextIntensity && sameRotation && env.symmetry === 1) {
      return;
    }
    env.texture = nextTexture;
    env.symmetry = 1;
    if (nextRotation) env.rotation.copy(nextRotation);
    else env.rotation.set(0, 0, 0);
    env.intensity = nextIntensity;
    this._displayUniforms.uRadIntensity.value = env.intensity;
    this._displayUniforms.uRadReplaceAmbient.value = 0;
    this._solver.setEnvironment(env.texture, env.rotation, ENVIRONMENT_BLUR, env.symmetry, 1);
  }

  /**
   * Light the bake with three's procedural RoomEnvironment (for themes without
   * an environment map), scaled so an unoccluded up-facing surface receives
   * the irradiance the scene's hemisphere and ambient lights would have given
   * it. The lightmap then carries those lights' occluded, bounced contribution
   * and the display materials fade the plain lights out as it fades in.
   *
   * @param {object} params
   * @param {THREE.Color|string} params.skyColor
   * @param {number} params.hemisphereIntensity - 0 disables the hemisphere term
   * @param {THREE.Color|string} params.ambientColor
   * @param {number} params.ambientIntensity - 0 disables the ambient term
   */
  setRoomEnvironment({ skyColor, hemisphereIntensity = 0, ambientColor, ambientIntensity = 0 } = {}) {
    if (!this._roomPmrem) {
      this._roomPmrem = this._buildRoomPmrem();
    }
    const sky = new THREE.Color(skyColor || 0xffffff);
    const ambient = new THREE.Color(ambientColor || 0xffffff);
    // three's irradiance for an up-facing surface: skyColor·I_hemi + ambient·I_amb.
    const target = luminance(sky) * Math.max(Number(hemisphereIntensity) || 0, 0) +
      luminance(ambient) * Math.max(Number(ambientIntensity) || 0, 0);
    const env = this._environment;
    const texture = this._roomPmrem?.texture || null;
    if (env.texture !== texture || env.symmetry !== ROOM_ENVIRONMENT_SYMMETRY) {
      env.texture = texture;
      env.symmetry = ROOM_ENVIRONMENT_SYMMETRY;
      env.rotation.set(0, 0, 0);
      this._solver.setEnvironment(env.texture, env.rotation, ENVIRONMENT_BLUR, env.symmetry, ROOM_BELOW_HORIZON);
    }
    const upIrradiance = this._solver.environmentUpIrradiance;
    env.intensity = upIrradiance > 1e-6 ? target / upIrradiance : target;
    this._displayUniforms.uRadIntensity.value = env.intensity;
    this._displayUniforms.uRadReplaceAmbient.value = 1;
  }

  /**
   * PMREM of three's RoomEnvironment, rotated so its ceiling faces +Z and
   * re-centred: the room is authored with the origin on its floor, so seen
   * from there every panel is overhead and the lighting is lopsided. Moving
   * the camera to the room's centre puts the wall panels around the horizon.
   */
  _buildRoomPmrem() {
    if (!this._pmremGenerator) {
      this._pmremGenerator = new THREE.PMREMGenerator(this._renderer);
    }
    const room = new RoomEnvironment();
    const shell = room.children.find((child) => child.isMesh && child.material?.side === THREE.BackSide);
    const centre = shell ? shell.position.clone() : new THREE.Vector3();
    const pivot = new THREE.Group();
    for (const child of [...room.children]) pivot.add(child);
    pivot.position.copy(centre).negate();
    room.add(pivot);
    // The room is authored Y-up; the viewer's scenes are Z-up.
    room.rotation.x = Math.PI / 2;
    room.updateMatrixWorld(true);
    let target = null;
    try {
      target = this._pmremGenerator.fromScene(room, 0.04);
    } catch (error) {
      console.warn("radiosity: room environment PMREM failed", error);
    } finally {
      room.traverse((object) => {
        object.geometry?.dispose?.();
        if (Array.isArray(object.material)) object.material.forEach((m) => m.dispose?.());
        else object.material?.dispose?.();
      });
    }
    return target;
  }

  /** Restart the accumulation (e.g. after a context restore). */
  reset() {
    this._solver?.reset();
  }

  /** Force the next step to re-chart the scene. */
  invalidate() {
    this._needsBuild = true;
  }

  /**
   * Advance the bake by one bounce. Call once per rendered frame before
   * `renderer.render`. Returns true while the solution is still refining.
   */
  step() {
    if (this._disposed) return false;
    const records = this._recordsProvider() || [];
    if (this._needsBuild || this._computeTopologyHash(records) !== this._topologyHash) {
      this._build(records);
    }
    if (!this._built || this._entries.length === 0) {
      return false;
    }
    const state = this._computeStateHash();
    if (state !== this._stateHash) {
      this._stateHash = state;
      this._solver.reset();
    }
    const bounced = this._solver.bounce();
    this._syncLightmap();
    return bounced && !this._solver.converged;
  }

  /** Restore geometries and materials; keeps the solver for a later rebuild. */
  teardown() {
    this._restoreGeometries();
    for (const material of this._patched) removeRadiosityPatch(material);
    this._patched.clear();
    this._floor.setLightmap(null, 0);
    this._floor.group.visible = false;
    this._built = false;
    this._needsBuild = true;
    this._info = null;
  }

  dispose() {
    if (this._disposed) return;
    this.teardown();
    this._disposed = true;
    this._solver?.dispose();
    this._solver = null;
    this._floor.dispose();
    this._roomPmrem?.dispose();
    this._roomPmrem = null;
    this._pmremGenerator?.dispose();
    this._pmremGenerator = null;
  }

  // -------------------------------------------------------------------------

  _createSolver() {
    const previous = this._solver;
    const s = this._settings;
    this._solver = new RadiosityLightmap(this._renderer, s.quality, {
      directions: s.directions,
      maxSamples: s.maxSamples,
      selfBias: s.selfBias,
      bounce: s.bounce,
      envSampling: s.envSampling,
      displayUniforms: this._displayUniforms
    });
    this._solver.setEnvironment(
      this._environment.texture,
      this._environment.rotation,
      ENVIRONMENT_BLUR,
      this._environment.symmetry,
      this._environment.symmetry === ROOM_ENVIRONMENT_SYMMETRY ? ROOM_BELOW_HORIZON : 1
    );
    previous?.dispose();
    this._needsBuild = true;
  }

  _build(records) {
    this._restoreGeometries();
    this._needsBuild = false;
    this._built = false;
    this._info = null;
    this._topologyHash = this._computeTopologyHash(records);

    const lit = records.filter((record) => record?.mesh && record.geometry && isLitMaterial(record.material));
    if (!lit.length) {
      this._floor.group.visible = false;
      this._solver.setObjects([], null, UP, null);
      return;
    }
    // Fresh world matrices: the bake mirrors them every bounce, but the
    // charting below sizes charts from them right now.
    let root = lit[0].mesh;
    while (root.parent) root = root.parent;
    root.updateMatrixWorld(true);

    const resolution = this._solver.resolution;
    const charted = [];
    const allCharts = [];
    let uvFaces = 0;
    let planarFaces = 0;
    for (const record of lit) {
      if (!isOccluder(record.material)) continue;
      const geometry = record.mesh.geometry;
      const position = geometry.getAttribute("position");
      if (!position?.isBufferAttribute || position.count === 0) continue;
      const uvAttr = geometry.getAttribute("uv");
      const faceIds = record.mesh.userData?.faceIds;
      let result;
      try {
        result = chartMesh({
          positions: position.array,
          index: geometry.index ? geometry.index.array : null,
          uv: uvAttr?.isBufferAttribute ? uvAttr.array : null,
          faceIds: faceIds instanceof Uint32Array ? faceIds : null,
          trianglesPerFace: geometry.userData?.trianglesPerFace || null,
          matrixWorld: record.mesh.matrixWorld.elements
        });
      } catch (error) {
        console.warn("radiosity: charting failed for a part, skipped", error);
        continue;
      }
      charted.push({ record, result });
      for (const c of result.charts) allCharts.push(c);
      uvFaces += result.uvFaces;
      planarFaces += result.planarFaces;
    }

    let floorChart = null;
    const floorParams = this._floorParams;
    if (floorParams) {
      const side = Math.floor(resolution * FLOOR_ATLAS_FRACTION) - 2 * PADDING;
      floorChart = { w: side, h: side, fixed: true, x: 0, y: 0, pw: 0, ph: 0 };
      allCharts.push(floorChart);
    }

    let pack = packCharts(allCharts, resolution, PADDING);
    if (!pack) pack = packCharts(allCharts, resolution, 1);
    if (!pack) {
      console.warn(`radiosity: ${allCharts.length} charts do not fit a ${resolution}² lightmap; disabled`);
      this._floor.group.visible = false;
      this._solver.setObjects([], null, UP, null);
      return;
    }

    let split = 0;
    const objects = [];
    for (const { record, result } of charted) {
      const mesh = record.mesh;
      const original = mesh.geometry;
      const uv1 = writeAtlasUv(result, resolution);
      let swapped = null;
      let geometry = original;
      if (result.remap) {
        swapped = buildSplitGeometry(original, result, uv1);
        mesh.geometry = swapped;
        geometry = swapped;
        split++;
      } else {
        original.setAttribute("uv1", new THREE.BufferAttribute(uv1, 2));
      }
      const entry = {
        record,
        mesh,
        originalGeometry: original,
        swapped,
        albedo: new THREE.Color(),
        emissive: new THREE.Color()
      };
      this._entries.push(entry);
      objects.push({
        mesh,
        geometry,
        albedo: () => materialAlbedo(mesh.material, entry.albedo),
        emissive: () => materialEmissive(mesh.material, entry.emissive)
      });
    }

    let floorObject = null;
    let floorRefUv = null;
    if (floorParams && floorChart) {
      this._floor.configure(floorParams.center, floorParams.level, floorParams.extent);
      this._floor.setAlbedo(floorParams.albedo);
      this._floor.setShadowIntensity(this._settings.shadow);
      this._floor.setAtlasRect(floorChart.x, floorChart.y, floorChart.pw, floorChart.ph, resolution);
      floorObject = {
        mesh: this._floor.mesh,
        geometry: this._floor.geometry,
        albedo: () => this._floor.albedo(),
        emissive: () => this._floor.emissive()
      };
      // Corner texel of the floor chart: far from the model → unoccluded.
      floorRefUv = new THREE.Vector2((floorChart.x + 0.5) / resolution, (floorChart.y + 0.5) / resolution);
    } else {
      this._floor.group.visible = false;
    }

    this._solver.setObjects(objects, floorObject, UP, floorRefUv);
    for (const record of lit) this._patchMaterial(record.material);
    this._stateHash = this._computeStateHash();
    // Swapped (split) geometries changed the ids the hash reads; take the
    // hash after the swaps so the next step does not rebuild again.
    this._topologyHash = this._computeTopologyHash(records);
    this._built = true;
    this._info = {
      objects: objects.length,
      charts: allCharts.length,
      uvFaces,
      planarFaces,
      split,
      fill: pack.fill,
      texelsPerUnit: pack.scale
    };
  }

  _patchMaterial(material) {
    applyRadiosityPatch(material, this._solver.texture, this._displayUniforms);
    this._patched.add(material);
  }

  /** Point every patched material at the solver's current lightmap. */
  _syncLightmap() {
    const texture = this._solver.texture;
    const weight = this._displayUniforms.uRadWeight.value;
    for (const entry of this._entries) {
      const material = entry.mesh.material;
      if (isLitMaterial(material)) this._patchMaterial(material);
    }
    for (const material of this._patched) {
      if (material.lightMap !== texture) material.lightMap = texture;
    }
    this._floor.setLightmap(texture, weight);
  }

  _restoreGeometries() {
    for (const entry of this._entries) {
      const { mesh, originalGeometry, swapped } = entry;
      if (swapped) {
        if (mesh.geometry === swapped) mesh.geometry = originalGeometry;
        swapped.dispose();
      } else if (originalGeometry.hasAttribute("uv1")) {
        originalGeometry.deleteAttribute("uv1");
      }
    }
    this._entries = [];
  }

  _computeStateHash() {
    let h = 2166136261;
    for (const entry of this._entries) {
      const mesh = entry.mesh;
      h = hashInt(h, worldVisible(mesh) ? 1 : 0);
      const m = mesh.matrixWorld.elements;
      for (let i = 0; i < 16; i++) h = hashFloat(h, m[i]);
      h = hashInt(h, isOccluder(mesh.material) ? 1 : 0);
    }
    if (this._floorParams) {
      h = hashFloat(h, this._floorParams.level);
    }
    return h;
  }

  _computeTopologyHash(records) {
    let h = 2166136261;
    h = hashInt(h, records.length);
    for (const record of records) {
      if (!record?.mesh) continue;
      h = hashInt(h, record.mesh.id);
      h = hashInt(h, record.mesh.geometry?.id ?? 0);
      h = hashInt(h, isLitMaterial(record.material) ? 1 : 0);
    }
    h = hashInt(h, this._floorParams ? 1 : 0);
    return h;
  }
}
