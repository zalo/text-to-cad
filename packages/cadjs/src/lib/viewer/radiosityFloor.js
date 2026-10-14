/**
 * Radiosity shadow catcher — a ground plane lit by the radiosity lightmap.
 *
 * The plane's inner square (`FLOOR_INNER` × scene extent) owns a fixed region
 * of the lightmap atlas; the outer ring extends to the horizon and repeats the
 * inner edge's lighting (its `uv1` is clamped to the chart border), which is
 * unoccluded out there anyway. The plane is invisible except where the
 * lightmap says the ground is occluded: it multiplies whatever is drawn
 * beneath it (the stage floor or the backdrop) by the baked visibility,
 * tinted by the environment colour, so the model gets soft contact shadows
 * and ambient occlusion on the ground. It also feeds the solver as the
 * bounce surface under the model (`albedo()`).
 */

import * as THREE from "three";

/** Side of the lightmapped inner square, in multiples of the scene extent. */
export const RADIOSITY_FLOOR_INNER = 4;
/** Side of the whole plane, in multiples of the scene extent. */
export const RADIOSITY_FLOOR_OUTER = 60;

const FLOOR_VERTEX = /* glsl */ `
  attribute vec2 uv1;
  varying vec2 vUv1;
  varying vec2 vLocal;
  void main() {
    vUv1 = uv1;
    vLocal = position.xy;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  }
`;

const FLOOR_FRAGMENT = /* glsl */ `
  precision highp float;
  uniform sampler2D uLightmap;
  uniform float uWeight;
  uniform float uShadow;
  uniform float uFogStart;
  uniform float uFogEnd;
  uniform vec2 uCenter;
  varying vec2 vUv1;
  varying vec2 vLocal;

  void main() {
    vec4 lm = texture2D( uLightmap, vUv1 );
    float valid = lm.a > 1e-6 ? uWeight : 0.0;
    float lum = dot( lm.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
    float vis = valid > 0.0 ? clamp( lum / lm.a, 0.0, 1.0 ) : 1.0;
    float shadowVis = 1.0 - uShadow * ( 1.0 - vis );
    // Shadow catcher: multiply the background by the (tinted) visibility and
    // emit the shadow as alpha so a transparent backdrop darkens too.
    vec3 tint = valid > 0.0 ? clamp( lm.rgb / max( lum, 1e-6 ), 0.0, 4.0 ) : vec3( 1.0 );
    vec3 mul = mix( vec3( 1.0 ), tint * shadowVis, valid );
    float fog = smoothstep( uFogStart, uFogEnd, length( vLocal - uCenter ) );
    mul = mix( mul, vec3( 1.0 ), fog );
    gl_FragColor = vec4( mul, ( 1.0 - dot( mul, vec3( 0.2126, 0.7152, 0.0722 ) ) ) );
  }
`;

export class RadiosityFloor {
  constructor() {
    /** Group to add to the scene's stage. */
    this.group = new THREE.Group();
    this.group.name = "radiosityFloor";
    this.group.visible = false;

    this._mesh = null;
    this._geometry = null;
    this._albedo = new THREE.Color(0.35, 0.35, 0.35);
    this._emissive = new THREE.Color(0, 0, 0);
    this._innerSize = 1;
    this._level = 0;
    this._center = new THREE.Vector2();
    /** Corner flags (0/1 per vertex) of the inner square, the source for `setAtlasRect`. */
    this._cornerUv = null;

    this._material = new THREE.ShaderMaterial({
      vertexShader: FLOOR_VERTEX,
      fragmentShader: FLOOR_FRAGMENT,
      uniforms: {
        uLightmap: { value: null },
        uWeight: { value: 0 },
        uShadow: { value: 0.5 },
        uFogStart: { value: 1 },
        uFogEnd: { value: 2 },
        uCenter: { value: new THREE.Vector2() }
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      // rgb: multiply the background; alpha: add the shadow amount so a
      // transparent (protected) backdrop darkens the canvas colour.
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.ZeroFactor,
      blendDst: THREE.SrcColorFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneFactor
    });
    this._material.name = "radiosityFloorShadow";
  }

  /** The floor mesh (null before `configure`). */
  get mesh() {
    return this._mesh;
  }

  /** Geometry carrying the floor's `uv1`. */
  get geometry() {
    return this._geometry;
  }

  /** World size of the lightmapped inner square. */
  get innerSize() {
    return this._innerSize;
  }

  /** Diffuse albedo used for the bounce (the stage floor colour, linear). */
  albedo() {
    return this._albedo;
  }

  emissive() {
    return this._emissive;
  }

  /** Colour the floor bounces back into the scene (linear). */
  setAlbedo(color) {
    if (color?.isColor) {
      this._albedo.copy(color);
    }
  }

  /**
   * Build the plane for the given scene bounds (Z up).
   *
   * @param {THREE.Vector3} center - World centre of the model bounds
   * @param {number} level - Floor height (world Z)
   * @param {number} extent - Max extent of the model bounds
   */
  configure(center, level, extent) {
    this._clear();
    const safeExtent = Math.max(Number(extent) || 0, 1e-6);
    const inner = RADIOSITY_FLOOR_INNER * safeExtent;
    const outer = RADIOSITY_FLOOR_OUTER * safeExtent;
    this._innerSize = inner;
    this._level = level;
    this._center.set(center.x, center.y);

    // 4x4 vertex grid: inner square plus a ring to the horizon. The ring's
    // uv1 repeats the inner border so the lightmap extrapolates outward.
    const xs = [-outer, -inner / 2, inner / 2, outer];
    const positions = [];
    const normals = [];
    const uvs = [];
    const uv1 = [];
    for (let j = 0; j < 4; j++) {
      for (let i = 0; i < 4; i++) {
        positions.push(xs[i] + center.x, xs[j] + center.y, 0);
        normals.push(0, 0, 1);
        const u = i <= 1 ? 0 : 1;
        const v = j <= 1 ? 0 : 1;
        uvs.push(u, v);
        uv1.push(u, v);
      }
    }
    const index = [];
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        const a = j * 4 + i;
        const b = a + 1;
        const c = a + 4;
        const d = c + 1;
        index.push(a, b, d, a, d, c);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setAttribute("uv1", new THREE.Float32BufferAttribute(uv1, 2));
    geometry.setIndex(index);
    geometry.computeBoundingSphere();
    this._geometry = geometry;
    this._cornerUv = Float32Array.from(uv1);

    const mesh = new THREE.Mesh(geometry, this._material);
    mesh.name = "radiosityFloorPlane";
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    // Drawn after the stage floor and its key-light shadow plane (-3 … -2) so
    // it multiplies them, and before the model.
    mesh.renderOrder = -1.5;
    mesh.position.set(0, 0, level);
    this._mesh = mesh;
    this.group.add(mesh);
    this.group.visible = true;
    mesh.updateMatrixWorld(true);

    const u = this._material.uniforms;
    u.uFogStart.value = inner * 0.5;
    u.uFogEnd.value = inner * 3;
    u.uCenter.value.copy(this._center);
  }

  /** Map the inner square's uv1 to a packed atlas rect (texels). */
  setAtlasRect(x, y, w, h, resolution) {
    if (!this._geometry || !this._cornerUv) return;
    const attr = this._geometry.getAttribute("uv1");
    // Always map from the original corner flags: the attribute itself holds
    // atlas coordinates after the first call, so it cannot be re-derived.
    for (let i = 0; i < attr.count; i++) {
      const u = this._cornerUv[i * 2];
      const v = this._cornerUv[i * 2 + 1];
      attr.setXY(i, (x + u * w) / resolution, (y + v * h) / resolution);
    }
    attr.needsUpdate = true;
  }

  setLightmap(texture, weight) {
    this._material.uniforms.uLightmap.value = texture;
    this._material.uniforms.uWeight.value = weight;
  }

  /** 0..1 strength of the ground shadow/AO. */
  setShadowIntensity(value) {
    this._material.uniforms.uShadow.value = Math.min(Math.max(Number(value) || 0, 0), 1);
  }

  dispose() {
    this._clear();
    this._material.dispose();
  }

  _clear() {
    if (this._mesh) {
      this.group.remove(this._mesh);
      this._mesh = null;
    }
    if (this._geometry) {
      this._geometry.dispose();
      this._geometry = null;
    }
    this._cornerUv = null;
    this.group.visible = false;
  }
}
