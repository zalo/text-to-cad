/**
 * Progressive radiosity lightmap (WebGL port).
 *
 * A re-implementation of "Approximate Radiosity using Stochastic Depth
 * Buffering" (Thomsen & Nielsen, 2010) after zalo's TSL/WebGPU
 * `ProgressiveLightMapGPU` (three.js branch feat-progressive-radiosity-tsl),
 * adapted to the viewer's WebGLRenderer pipeline and to image-based lighting:
 *
 * - The light source is the environment map (the scene's PMREM). Every
 *   lightmap texel integrates the environment over its hemisphere with
 *   visibility, so shadows are area-light shadows whose penumbrae widen with
 *   occluder distance (contact hardening for free), and the ground receives
 *   ambient occlusion and colour bleeding from the model.
 * - Each bounce rasterizes N global directions at once with an ArrayCamera
 *   into tiles of one "illumination map" (RGB = the surface's current outgoing
 *   radiance, A = its distance along the tile direction) using a random
 *   per-fragment depth so overlapping surfaces shuffle between frames; a
 *   small texel neighbourhood then statistically represents the surfaces
 *   along a ray.
 * - The gather pass runs in lightmap space (vertices placed by `uv1`): for
 *   every texel it looks up the nearest in-hemisphere surface per direction,
 *   falling back to the environment radiance when the ray escapes, and folds
 *   the cosine-weighted sum into the lightmap with a cumulative moving
 *   average, so the solution converges over frames and does not depend on
 *   the viewer camera at all.
 *
 * Lightmap texel layout (linear, float):
 *   rgb = irradiance / π received by the texel (occlusion + bounce, unit
 *         environment intensity)
 *   a   = luminance of the unoccluded irradiance / π for the same normal
 *         (so rgb/a is the per-texel visibility used for specular occlusion,
 *         for occluding the ambient/hemisphere lights and for the user's
 *         shadow-strength control)
 *
 * The display side is a shader patch for MeshPhysicalMaterial that replaces
 * the IBL diffuse irradiance with the lightmap, occludes the ambient and
 * hemisphere lights with the lightmap visibility and applies it as specular
 * occlusion; see `applyRadiosityPatch`.
 */

import * as THREE from "three";

// ---------------------------------------------------------------------------
// Quality presets
// ---------------------------------------------------------------------------

/** Environment luminance table resolution used for importance sampling. */
const ENV_TABLE_W = 128;
const ENV_TABLE_H = 64;

/**
 * Accumulated directions (bounces × directions per bounce) over which the
 * lightmap fades in on the display materials and floor, so the noisy early
 * estimate never pops in; before that the surfaces show plain IBL.
 */
export const RADIOSITY_FADE_IN_DIRECTIONS = 1000;

/** Largest illumination map side we allocate (float RGBA: 4096² = 256 MB). */
const MAX_ILLUMINATION_SIDE = 4096;

/** Direction counts the solver accepts per bounce (square ArrayCamera layouts). */
export const RADIOSITY_DIRECTION_OPTIONS = Object.freeze([1, 4, 9, 16]);

export const RADIOSITY_QUALITY = Object.freeze({
  // Directions per bounce only change how fast the estimate converges; the
  // tile resolution decides how fine the occlusion is (contact AO needs
  // texels smaller than the gaps), so the presets spend their illumination
  // map budget on tile size first.
  low: Object.freeze({
    lightmapResolution: 512,
    directions: 4,
    tileResolution: 384,
    gatherRadius: 1,
    maxSamples: 0
  }),
  medium: Object.freeze({
    lightmapResolution: 1024,
    directions: 9,
    tileResolution: 512,
    gatherRadius: 1,
    maxSamples: 0
  }),
  high: Object.freeze({
    lightmapResolution: 2048,
    directions: 4,
    tileResolution: 1024,
    gatherRadius: 2,
    maxSamples: 0
  })
});

export const RADIOSITY_QUALITY_IDS = Object.freeze(Object.keys(RADIOSITY_QUALITY));

export function normalizeRadiosityQuality(value, fallback = "medium") {
  const normalized = String(value || "").trim().toLowerCase();
  return RADIOSITY_QUALITY_IDS.includes(normalized) ? normalized : fallback;
}

export function normalizeRadiosityDirections(value, fallback = 16) {
  const numeric = Math.floor(Number(value));
  return RADIOSITY_DIRECTION_OPTIONS.includes(numeric) ? numeric : fallback;
}

// ---------------------------------------------------------------------------
// Shared GLSL
// ---------------------------------------------------------------------------

const GLSL_COMMON = /* glsl */ `
  // Integer hash → [0,1); robust for large inputs unlike sin-based hashes.
  float tcvHash( float x ) {
    uint h = floatBitsToUint( x );
    h ^= h >> 16u; h *= 0x7feb352dU; h ^= h >> 15u; h *= 0x846ca68bU; h ^= h >> 16u;
    return float( h ) / 4294967296.0;
  }
  float tcvLuminance( vec3 c ) { return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) ); }
  // Replace NaN/negative texels by zero so one bad texel cannot poison the
  // progressive mean or spread through the gutter dilation.
  vec4 tcvSane( vec4 c ) {
    if ( any( isnan( c ) ) || any( isinf( c ) ) ) return vec4( 0.0 );
    return max( c, vec4( 0.0 ) );
  }
`;

/** Environment radiance along a world direction (unit intensity). */
const GLSL_ENV = /* glsl */ `
  uniform mat3 uEnvRotation;
  uniform float uEnvBlur;
  #ifdef TCV_HAS_ENV
    uniform sampler2D envMap;
    #define ENVMAP_TYPE_CUBE_UV
    #include <cube_uv_reflection_fragment>
    vec3 tcvEnvRadiance( vec3 d ) {
      return textureCubeUV( envMap, uEnvRotation * d, uEnvBlur ).rgb;
    }
  #else
    vec3 tcvEnvRadiance( vec3 d ) { return vec3( 1.0 ); }
  #endif
`;

// --- Illumination pass -------------------------------------------------------

const ILLUM_VERTEX = /* glsl */ `
  attribute vec2 uv1;
  varying vec3 vWorldPos;
  varying vec2 vUv1;
  void main() {
    vec4 wp = modelMatrix * vec4( position, 1.0 );
    vWorldPos = wp.xyz;
    vUv1 = uv1;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const ILLUM_FRAGMENT = /* glsl */ `
  precision highp float;
  uniform sampler2D uPrev;
  uniform vec3 uAlbedo;
  uniform vec3 uEmissive;
  uniform float uFrameSeed;
  uniform float uAlbedoWeight;
  varying vec3 vWorldPos;
  varying vec2 vUv1;
  ${GLSL_COMMON}
  void main() {
    // The tile direction is the sub-camera's forward axis (-Z of its view matrix).
    vec3 d = -vec3( viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2] );
    // Distance along d from the near plane, biased to >= 1 so a cleared texel (0)
    // reads as "no surface".
    float dist = dot( vWorldPos - cameraPosition, d ) + 1.0;
    vec4 prev = tcvSane( texture2D( uPrev, vUv1 ) );
    vec3 outgoing = uAlbedo * prev.rgb * uAlbedoWeight + uEmissive;
    gl_FragColor = vec4( outgoing, dist );
    // Stochastic depth: a stable per-surface rank offset by a per-pixel
    // interleaved-gradient dither that is animated per frame.
    float rank = tcvHash( dot( vWorldPos, d ) * 7.0 );
    float ign = fract( 52.9829189 * fract( 0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y ) );
    float dither = fract( ign + uFrameSeed * 0.61803399 );
    gl_FragDepth = fract( rank + dither );
  }
`;

// --- Gather pass -------------------------------------------------------------

const GATHER_VERTEX = /* glsl */ `
  attribute vec2 uv1;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  varying vec2 vUv1;
  void main() {
    vec4 wp = modelMatrix * vec4( position, 1.0 );
    vWorldPos = wp.xyz;
    vWorldNormal = normalize( mat3( modelMatrix ) * normal );
    vUv1 = uv1;
    gl_Position = vec4( uv1 * 2.0 - 1.0, 0.0, 1.0 );
  }
`;

function gatherFragment(directions, radius) {
  return /* glsl */ `
  precision highp float;
  #define N_DIRS ${directions}
  #define RADIUS ${radius}
  uniform vec3 uDir[N_DIRS];
  uniform vec3 uBasisU[N_DIRS];
  uniform vec3 uBasisV[N_DIRS];
  uniform vec3 uOrigin[N_DIRS];
  uniform vec2 uHalf[N_DIRS];
  uniform vec2 uTileOffset[N_DIRS];
  uniform float uDirWeight[N_DIRS];
  uniform vec2 uTileScale;
  uniform vec2 uIllumTexel;
  uniform sampler2D uIllum;
  uniform sampler2D uPrev;
  uniform float uInvSampleCount;
  uniform float uSelfBias;
  uniform vec3 uUp;
  uniform float uFloorEnabled;
  uniform vec3 uFloorAlbedo;
  uniform vec2 uFloorRefUv;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  varying vec2 vUv1;
  ${GLSL_COMMON}
  ${GLSL_ENV}

  // Nearest in-hemisphere surface seen from p along direction k, or the
  // environment (unit intensity) when the ray escapes. Sets \`found\`.
  vec3 gatherDirection( vec3 p, vec3 n, int k, vec3 lenv, out bool found ) {
    found = false;
    vec3 d = uDir[k];
    vec3 rel = p - uOrigin[k];
    vec2 ndc = vec2( dot( rel, uBasisU[k] ) / uHalf[k].x, dot( rel, uBasisV[k] ) / uHalf[k].y );
    if ( any( greaterThan( abs( ndc ), vec2( 1.0 ) ) ) ) return lenv; // outside the scene bounds → unoccluded
    vec2 base = ndc * 0.5 + 0.5;
    vec2 tileOffset = uTileOffset[k];
    // Snap to the illumination-map texel grid so the texel we read and the
    // position we reconstruct from it refer to the same surface sample.
    vec2 tileUv = ( floor( ( base * uTileScale + tileOffset ) / uIllumTexel ) + 0.5 ) * uIllumTexel;
    vec2 lo = tileOffset + uIllumTexel * 0.5;
    vec2 hi = tileOffset + uTileScale - uIllumTexel * 0.5;
    float pAlong = dot( rel, d );
    float best = 1e20;
    vec3 bestRGB = vec3( 0.0 );
    for ( int oy = -RADIUS; oy <= RADIUS; oy ++ ) {
      for ( int ox = -RADIUS; ox <= RADIUS; ox ++ ) {
        vec2 sUv = clamp( tileUv + uIllumTexel * vec2( float( ox ), float( oy ) ), lo, hi );
        vec4 s = texture2D( uIllum, sUv );
        if ( s.w < 0.5 ) continue; // cleared texel: nothing along this ray
        vec2 local = ( ( sUv - tileOffset ) / uTileScale ) * 2.0 - 1.0;
        vec3 samplePos = uOrigin[k] + uBasisU[k] * ( local.x * uHalf[k].x ) + uBasisV[k] * ( local.y * uHalf[k].y ) + d * ( s.w - 1.0 );
        // Reject the receiver's own surface and near-coplanar self hits.
        if ( dot( samplePos - p, n ) <= uSelfBias ) continue;
        float diff = abs( dot( samplePos - uOrigin[k], d ) - pAlong );
        if ( diff < best ) { best = diff; bestRGB = s.rgb; }
      }
    }
    if ( best < 1e19 ) { found = true; return bestRGB; }
    return lenv;
  }

  void main() {
    vec3 p = vWorldPos;
    // Degenerate (zero) normals would normalize to NaN; treat them as facing up.
    float nLen = length( vWorldNormal );
    vec3 n = nLen > 1e-8 ? vWorldNormal / nLen : uUp;
    vec3 acc = vec3( 0.0 );
    vec3 accOpen = vec3( 0.0 );
    // Radiance of the (unoccluded) floor, used when a ray toward the ground
    // leaves the rasterized area without hitting it.
    vec3 floorRadiance = uFloorAlbedo * texture2D( uPrev, uFloorRefUv ).rgb;
    for ( int k = 0; k < N_DIRS; k ++ ) {
      vec3 d = uDir[k];
      float nd = dot( n, d );
      if ( nd <= 0.0 ) continue;
      vec3 lenv = tcvEnvRadiance( d );
      bool found;
      vec3 l = gatherDirection( p, n, k, lenv, found );
      if ( !found && uFloorEnabled > 0.5 && dot( d, uUp ) < 0.0 ) l = floorRadiance;
      // Monte-Carlo weight 1 / (N · π · pdf_k): 4/N for uniform sphere
      // sampling, the balance-heuristic mixture pdf for importance sampling.
      acc += l * nd * uDirWeight[k];
      accOpen += lenv * nd * uDirWeight[k];
    }
    // E/π = (1/(Nπ)) Σ L_k max(n·d_k, 0) / pdf_k
    vec4 estimate = vec4( acc, tcvLuminance( accOpen ) );
    vec4 prev = tcvSane( texture2D( uPrev, vUv1 ) );
    estimate = tcvSane( estimate );
    // The first sample overwrites (a 0·prev term would keep garbage alive).
    gl_FragColor = uInvSampleCount >= 1.0 ? estimate : mix( prev, estimate, uInvSampleCount );
  }
  `;
}

// --- Seam dilation -----------------------------------------------------------

const DILATE_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4( position.xy, 0.0, 1.0 );
  }
`;

const DILATE_FRAGMENT = /* glsl */ `
  precision highp float;
  uniform sampler2D uPrev;
  uniform vec2 uTexel;
  varying vec2 vUv;
  ${GLSL_COMMON}
  void main() {
    // Every texel becomes the average of its valid neighbours; chart texels
    // are redrawn by the gather afterwards, so only the gutters keep this,
    // and they track the converging solution instead of their first value.
    // Bilinear sampling at a chart border then never bleeds black (or
    // another chart) into the surface.
    vec4 sum = vec4( 0.0 );
    float cnt = 0.0;
    for ( int oy = -1; oy <= 1; oy ++ ) {
      for ( int ox = -1; ox <= 1; ox ++ ) {
        if ( ox == 0 && oy == 0 ) continue;
        vec4 s = tcvSane( texture2D( uPrev, vUv + uTexel * vec2( float( ox ), float( oy ) ) ) );
        if ( s.a > 0.0 ) { sum += s; cnt += 1.0; }
      }
    }
    gl_FragColor = cnt > 0.0 ? sum / cnt : vec4( 0.0 );
  }
`;

// --- Environment luminance table (importance sampling) ----------------------

const ENV_TABLE_FRAGMENT = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  ${GLSL_COMMON}
  ${GLSL_ENV}
  void main() {
    // Equirectangular over world directions, polar axis +Z: row = theta.
    float theta = vUv.y * 3.14159265;
    float phi = vUv.x * 6.28318531;
    vec3 d = vec3( sin( theta ) * cos( phi ), sin( theta ) * sin( phi ), cos( theta ) );
    gl_FragColor = vec4( tcvEnvRadiance( d ), 1.0 );
  }
`;

/**
 * Discrete 2D distribution over an equirectangular luminance table, for
 * drawing world directions proportional to environment radiance.
 */
export class EnvImportanceTable {
  constructor(lum, w, h) {
    this._w = w;
    this._h = h;
    this._dTheta = Math.PI / h;
    this._dPhi = (2 * Math.PI) / w;
    const weights = new Float32Array(w * h);
    const rowCdf = new Float32Array(h);
    const condCdf = new Float32Array(w * h);
    let total = 0;
    for (let i = 0; i < h; i++) {
      const sinT = Math.sin((i + 0.5) * this._dTheta);
      let rowSum = 0;
      for (let j = 0; j < w; j++) {
        const wt = (Math.max(lum[i * w + j], 0) + 1e-4) * sinT;
        weights[i * w + j] = wt;
        rowSum += wt;
        condCdf[i * w + j] = rowSum;
      }
      for (let j = 0; j < w; j++) condCdf[i * w + j] /= rowSum;
      total += rowSum;
      rowCdf[i] = total;
    }
    for (let i = 0; i < h; i++) rowCdf[i] /= total;
    this._weights = weights;
    this._total = total;
    this._rowCdf = rowCdf;
    this._condCdf = condCdf;
  }

  /** Draw a direction for the 2D sample (u1, u2); returns its pdf per solid angle. */
  sample(u1, u2, out) {
    const h = this._h;
    const w = this._w;
    let i = lowerBound(this._rowCdf, 0, h, u1);
    if (i >= h) i = h - 1;
    const r0 = i > 0 ? this._rowCdf[i - 1] : 0;
    const r1 = this._rowCdf[i];
    const xi = r1 > r0 ? Math.min(0.999999, (u1 - r0) / (r1 - r0)) : 0.5;
    const base = i * w;
    let j = lowerBound(this._condCdf, base, base + w, u2) - base;
    if (j >= w) j = w - 1;
    const c0 = j > 0 ? this._condCdf[base + j - 1] : 0;
    const c1 = this._condCdf[base + j];
    const zeta = c1 > c0 ? Math.min(0.999999, (u2 - c0) / (c1 - c0)) : 0.5;
    const theta = (i + xi) * this._dTheta;
    const phi = (j + zeta) * this._dPhi;
    const sinT = Math.sin(theta);
    out.set(sinT * Math.cos(phi), sinT * Math.sin(phi), Math.cos(theta));
    return this._pdfCell(i, j);
  }

  /** Pdf per solid angle of an arbitrary direction. */
  pdf(dir) {
    const theta = Math.acos(Math.max(-1, Math.min(1, dir.z)));
    let phi = Math.atan2(dir.y, dir.x);
    if (phi < 0) phi += 2 * Math.PI;
    const i = Math.min(this._h - 1, Math.floor(theta / this._dTheta));
    const j = Math.min(this._w - 1, Math.floor(phi / this._dPhi));
    return this._pdfCell(i, j);
  }

  _pdfCell(i, j) {
    const sinT = Math.sin((i + 0.5) * this._dTheta);
    return (
      this._weights[i * this._w + j] /
      this._total /
      (Math.max(sinT, 1e-4) * this._dTheta * this._dPhi)
    );
  }
}

/** First index in [lo, hi) whose cdf value is >= u. */
function lowerBound(cdf, lo, hi, u) {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid] < u) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Display-material patch
// ---------------------------------------------------------------------------

const PATCH_MARK = "__cadRadiosity";

/**
 * Uniform objects consumed by every patched display material. three.js runs
 * `onBeforeCompile` only when it compiles a new program, so a material
 * re-patched by a later solver keeps the uniform objects of the first
 * compile; sharing one set keeps fade-in and shadow strength working after a
 * quality or direction change.
 */
export function createRadiosityDisplayUniforms() {
  return {
    /** 0..1 blend between plain IBL irradiance and the lightmap (fade-in). */
    uRadWeight: { value: 0 },
    /** 0..1 shadow/AO strength relative to the unoccluded irradiance. */
    uRadShadow: { value: 1 },
    /** Scale of the lightmap irradiance for materials without an environment map. */
    uRadIntensity: { value: 1 }
  };
}

const LIGHTS_MAPS_PATCH = /* glsl */ `
#if defined( RE_IndirectDiffuse )
  #if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
    iblIrradiance += getIBLIrradiance( geometryNormal );
  #endif
  #ifdef USE_LIGHTMAP
  {
    vec4 lm = texture2D( lightMap, vLightMapUv );
    if ( lm.a > 1e-6 ) {
      float lmLum = dot( lm.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
      // Lift the shadows toward the unoccluded level by (1 - shadow strength).
      vec3 lifted = lm.rgb + vec3( max( lm.a - lmLum, 0.0 ) * ( 1.0 - uRadShadow ) );
      #ifdef USE_ENVMAP
        vec3 lmIrradiance = PI * lifted * envMapIntensity;
      #else
        vec3 lmIrradiance = PI * lifted * uRadIntensity;
      #endif
      iblIrradiance = mix( iblIrradiance, lmIrradiance, uRadWeight );
      tcvRadVisibility = mix( 1.0, 1.0 - uRadShadow * ( 1.0 - clamp( lmLum / lm.a, 0.0, 1.0 ) ), uRadWeight );
      // Ambient and hemisphere lights are unoccluded by construction; scale
      // them by the baked visibility so they do not wash the occlusion out.
      irradiance *= tcvRadVisibility;
    }
  }
  #endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
  #ifdef USE_ANISOTROPY
    radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
  #else
    radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
  #endif
  #ifdef USE_CLEARCOAT
    clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
  #endif
#endif
`;

const AOMAP_PATCH = /* glsl */ `
#include <aomap_fragment>
#if defined( USE_LIGHTMAP ) && defined( USE_ENVMAP ) && defined( STANDARD )
{
  float tcvDotNV = saturate( dot( geometryNormal, geometryViewDir ) );
  float tcvSpecOcc = saturate( pow( tcvDotNV + tcvRadVisibility, exp2( - 16.0 * material.roughness - 1.0 ) ) - 1.0 + tcvRadVisibility );
  reflectedLight.indirectSpecular *= tcvSpecOcc;
  #if defined( USE_CLEARCOAT )
    clearcoatSpecularIndirect *= tcvSpecOcc;
  #endif
  #if defined( USE_SHEEN )
    sheenSpecularIndirect *= tcvRadVisibility;
  #endif
}
#endif
`;

/**
 * Patch a MeshPhysicalMaterial so its diffuse IBL comes from the radiosity
 * lightmap (attribute `uv1`), its ambient/hemisphere light and specular IBL
 * are occluded by the lightmap visibility. Chains any existing
 * `onBeforeCompile` (the CAD surface-edge shader) and keys the program
 * explicitly. Idempotent.
 */
export function applyRadiosityPatch(material, lightmap, uniforms) {
  if (!material || !material.isMaterial) {
    return;
  }
  if (!material[PATCH_MARK]) {
    const prevCompile = material.onBeforeCompile;
    const prevKey = material.customProgramCacheKey;
    material[PATCH_MARK] = {
      onBeforeCompile: prevCompile,
      customProgramCacheKey: prevKey
    };
    material.onBeforeCompile = (shader, renderer) => {
      prevCompile?.call(material, shader, renderer);
      shader.uniforms.uRadWeight = uniforms.uRadWeight;
      shader.uniforms.uRadShadow = uniforms.uRadShadow;
      shader.uniforms.uRadIntensity = uniforms.uRadIntensity;
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          "#include <common>\nuniform float uRadWeight;\nuniform float uRadShadow;\nuniform float uRadIntensity;\nfloat tcvRadVisibility = 1.0;"
        )
        .replace("#include <lights_fragment_maps>", LIGHTS_MAPS_PATCH)
        .replace("#include <aomap_fragment>", AOMAP_PATCH);
    };
    // three's default key is onBeforeCompile.toString(); with the wrapper
    // installed that would be our own source, so key explicitly instead. The
    // previous hook's source still distinguishes edge-shaded materials from
    // plain ones.
    const inherited = prevKey === THREE.Material.prototype.customProgramCacheKey;
    const defaultCompile = prevCompile === THREE.Material.prototype.onBeforeCompile;
    material.customProgramCacheKey = () => {
      if (!prevKey) {
        return "radiosity";
      }
      const base = inherited
        ? (typeof prevCompile === "function" && !defaultCompile ? prevCompile.toString() : "")
        : prevKey.call(material);
      return base ? `${base}|radiosity` : "radiosity";
    };
  }
  if (material.lightMap !== lightmap) {
    // Swapping between the solver's ping-pong textures only changes a uniform;
    // a recompile is needed just when the lightmap is first attached.
    const attach = material.lightMap == null;
    material.lightMap = lightmap;
    material.lightMapIntensity = 1;
    if (attach) material.needsUpdate = true;
  }
}

/** Reverse `applyRadiosityPatch`. */
export function removeRadiosityPatch(material) {
  const rec = material?.[PATCH_MARK];
  if (!rec) return;
  material.onBeforeCompile = rec.onBeforeCompile;
  material.customProgramCacheKey = rec.customProgramCacheKey;
  delete material[PATCH_MARK];
  material.lightMap = null;
  material.needsUpdate = true;
}

/** Whether a material carries the radiosity patch. */
export function hasRadiosityPatch(material) {
  return material?.[PATCH_MARK] != null;
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

const _quat = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _corner = new THREE.Vector3();
const _m4 = new THREE.Matrix4();
const _clearColor = new THREE.Color();

/**
 * Lateral/depth margin around the scene bounds, as a fraction of the max
 * extent. Kept small: the tile resolution across the model is what limits
 * contact occlusion detail. Rays toward the floor that leave the rasterized
 * area fall back to the floor's reference radiance in the gather.
 */
const MARGIN = 0.12;

function isWorldVisible(obj) {
  let o = obj;
  while (o) {
    if (!o.visible) return false;
    o = o.parent;
  }
  return true;
}

/** k-th point of an n-point Fibonacci sphere. */
function fibonacciSphere(k, n) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (k / (n - 1 || 1)) * 2;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const phi = k * golden;
  return new THREE.Vector3(Math.cos(phi) * r, y, Math.sin(phi) * r);
}

/**
 * Low-discrepancy uniform rotation for sample index n: the 3D R3 sequence
 * mapped to a unit quaternion (Shoemake), so successive frames cover SO(3)
 * far more evenly than independent random rotations.
 */
function stratifiedRotation(q, n) {
  const u1 = (0.5 + 0.8191725133961644 * n) % 1;
  const u2 = (0.5 + 0.6710436067037893 * n) % 1;
  const u3 = (0.5 + 0.5497004779019702 * n) % 1;
  const s1 = Math.sqrt(1 - u1);
  const s2 = Math.sqrt(u1);
  const t2 = 2 * Math.PI * u2;
  const t3 = 2 * Math.PI * u3;
  q.set(s1 * Math.sin(t2), s1 * Math.cos(t2), s2 * Math.sin(t3), s2 * Math.cos(t3));
}

/**
 * One surface in the bake:
 *   { mesh, geometry, albedo: () => Color, emissive: () => Color }
 * The geometry must carry a `uv1` attribute; the mesh's world matrix and
 * visibility are mirrored each bounce; albedo/emissive are read each bounce.
 */
export class RadiosityLightmap {
  /**
   * @param {THREE.WebGLRenderer} renderer - WebGL2 renderer with float render targets
   * @param {string} quality - "low" | "medium" | "high"
   * @param {object} [options]
   * @param {number} [options.directions] - Directions per bounce (1, 4, 9 or 16)
   * @param {number} [options.maxSamples] - Accumulated directions before the solver stops (0 = never)
   * @param {number} [options.selfBias] - Self-intersection bias in illumination-map texels
   * @param {number} [options.bounce] - Indirect bounce strength (0 = direct environment light only)
   * @param {string} [options.envSampling] - "uniform" | "importance"
   * @param {object} [options.displayUniforms] - Shared display uniforms (see createRadiosityDisplayUniforms)
   */
  constructor(renderer, quality, options = {}) {
    this._renderer = renderer;
    this.displayUniforms = options.displayUniforms || createRadiosityDisplayUniforms();
    this.displayUniforms.uRadWeight.value = 0;
    this.preset = RADIOSITY_QUALITY[normalizeRadiosityQuality(quality)];
    this.resolution = this.preset.lightmapResolution;
    this._directions = Math.max(1, Math.floor(options.directions ?? this.preset.directions));
    this.maxSamples = Math.max(0, Math.floor(options.maxSamples ?? this.preset.maxSamples));
    this._selfBiasTexels = options.selfBias ?? 1.5;
    this._bounce = options.bounce ?? 1;
    this._importance = (options.envSampling ?? "importance") === "importance";
    this._tilesX = Math.ceil(Math.sqrt(this._directions));
    this._tilesY = Math.ceil(this._directions / this._tilesX);
    // Keep the illumination map within the allocation cap: more directions
    // per bounce trade tile resolution.
    let tile = this.preset.tileResolution;
    while (Math.max(this._tilesX, this._tilesY) * tile > MAX_ILLUMINATION_SIDE && tile > 64) {
      tile = Math.floor(tile / 2);
    }
    this.tileResolution = tile;

    this._scene = new THREE.Scene();
    this._gatherScene = new THREE.Scene();
    this._proxies = [];
    this._gatherProxies = [];
    this._floorProxy = null;
    this._buffer1Active = false;
    this._subCameras = [];
    this._baseDirections = [];
    this._dirData = [];
    this._basisU = [];
    this._basisV = [];
    this._origins = [];
    this._halves = [];
    this._tileOffsets = [];
    this._dirWeights = [];
    this._bounds = new THREE.Box3();
    this._boundsCenter = new THREE.Vector3();
    this._maxExtent = 1;
    this._up = new THREE.Vector3(0, 0, 1);
    this._frame = 0;
    this._envTexture = null;
    this._envRotation = new THREE.Matrix3();
    this._envBlur = 0.1;
    this._envHeight = 0;
    this._disposed = false;
    this._lateral = 1;
    this._needsClear = true;
    this._envTable = null;
    this._envTableRT = null;
    this._gatherCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const mapOpts = {
      type: THREE.FloatType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false
    };
    this._map1 = new THREE.WebGLRenderTarget(this.resolution, this.resolution, mapOpts);
    this._map2 = new THREE.WebGLRenderTarget(this.resolution, this.resolution, mapOpts);
    this._map1.texture.name = "radiosityLightmap1";
    this._map2.texture.name = "radiosityLightmap2";
    // three picks the UV set of a lightMap from the texture's channel; the
    // atlas lives in `uv1`.
    this._map1.texture.channel = 1;
    this._map2.texture.channel = 1;

    this._illumination = new THREE.WebGLRenderTarget(this._tilesX * tile, this._tilesY * tile, {
      type: THREE.FloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: true,
      stencilBuffer: false
    });

    this._rebuildBaseDirections();
    for (let k = 0; k < this._directions; k++) {
      this._dirWeights.push(4 / this._directions);
      this._dirData.push(new THREE.Vector3(0, 0, 1));
      this._basisU.push(new THREE.Vector3(1, 0, 0));
      this._basisV.push(new THREE.Vector3(0, 1, 0));
      this._origins.push(new THREE.Vector3());
      this._halves.push(new THREE.Vector2(1, 1));
      const tx = k % this._tilesX;
      const ty = Math.floor(k / this._tilesX);
      this._tileOffsets.push(new THREE.Vector2(tx / this._tilesX, ty / this._tilesY));
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      cam.viewport = new THREE.Vector4(tx * tile, ty * tile, tile, tile);
      this._subCameras.push(cam);
    }
    // ArrayCamera is meant for perspective sub-cameras but the WebGL renderer
    // only reads each sub-camera's matrices and viewport.
    this._arrayCamera = new THREE.ArrayCamera(this._subCameras);

    this._illumMaterial = new THREE.ShaderMaterial({
      vertexShader: ILLUM_VERTEX,
      fragmentShader: ILLUM_FRAGMENT,
      uniforms: {
        uPrev: { value: null },
        uAlbedo: { value: new THREE.Color(1, 1, 1) },
        uEmissive: { value: new THREE.Color(0, 0, 0) },
        uFrameSeed: { value: 0 },
        uAlbedoWeight: { value: 1 }
      },
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending
    });
    this._illumMaterial.name = "radiosityIllumination";

    this._gatherMaterial = this._createGatherMaterial();

    this._dilateMaterial = new THREE.ShaderMaterial({
      vertexShader: DILATE_VERTEX,
      fragmentShader: DILATE_FRAGMENT,
      uniforms: {
        uPrev: { value: null },
        uTexel: { value: new THREE.Vector2(1 / this.resolution, 1 / this.resolution) }
      },
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending
    });
    this._dilatePlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._dilateMaterial);
    this._dilatePlane.frustumCulled = false;
    this._dilatePlane.renderOrder = -1;
    this._gatherScene.add(this._dilatePlane);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** The lightmap holding the latest solution. */
  get texture() {
    return (this._buffer1Active ? this._map1 : this._map2).texture;
  }

  /** Bounces accumulated since the last reset. */
  get iteration() {
    return this._frame;
  }

  /** Accumulated directions (bounces × directions per bounce). */
  get samples() {
    return this._frame * this._directions;
  }

  /** True once `maxSamples` directions have been accumulated (never when 0). */
  get converged() {
    return this.maxSamples > 0 && this.samples >= this.maxSamples;
  }

  /** Directions rasterized per bounce. */
  get directions() {
    return this._directions;
  }

  /** Self-intersection bias in illumination-map texels. */
  get selfBias() {
    return this._selfBiasTexels;
  }

  /** Set the self-intersection bias (texels) and restart the accumulation. */
  setSelfBias(texels) {
    this._selfBiasTexels = Math.max(0, texels);
    this._applySelfBias();
    this.reset();
  }

  /** Indirect bounce strength (0 = direct environment light only). */
  get bounceStrength() {
    return this._bounce;
  }

  /** Set the bounce strength and restart the accumulation. */
  setBounce(strength) {
    const next = Math.max(0, strength);
    if (next === this._bounce) return;
    this._bounce = next;
    this.reset();
  }

  /** Whether directions are importance-sampled from the environment. */
  get importanceSampling() {
    return this._importance;
  }

  /** Switch between uniform and environment importance sampling. */
  setEnvSampling(mode) {
    const flag = mode === "importance";
    if (flag === this._importance) return;
    this._importance = flag;
    this._rebuildBaseDirections();
    this.reset();
  }

  /**
   * Replace the baked surfaces. `floor` (optional) is a plane whose chart is
   * used as the reference for rays that escape toward the ground.
   */
  setObjects(objects, floor, up, floorRefUv) {
    this._clearProxies();
    this._up.copy(up).normalize();
    for (const source of objects) {
      this._proxies.push(this._makeProxy(source));
    }
    if (floor) {
      this._floorProxy = this._makeProxy(floor);
    }
    const u = this._gatherMaterial.uniforms;
    u.uUp.value.copy(this._up);
    u.uFloorEnabled.value = floor ? 1 : 0;
    if (floorRefUv) u.uFloorRefUv.value.copy(floorRefUv);
    this.reset();
  }

  /**
   * Set the light source: a PMREM environment texture (or null for a uniform
   * white sky), its rotation (the scene's `environmentRotation`) and a
   * pre-blur roughness that softens small bright sources (0 = sharp).
   */
  setEnvironment(texture, rotation, blur) {
    const height = texture?.image?.height ?? 0;
    const hadEnv = this._envTexture !== null;
    this._envTexture = texture || null;
    this._envBlur = blur;
    // three samples the environment with the inverse rotation
    this._envRotation
      .setFromMatrix4(_m4.makeRotationFromEuler(rotation || new THREE.Euler()))
      .transpose();
    if (hadEnv !== (this._envTexture !== null) || height !== this._envHeight) {
      this._envHeight = height;
      this._gatherMaterial.dispose();
      this._gatherMaterial = this._createGatherMaterial();
      for (const m of this._gatherProxies) m.material = this._gatherMaterial;
      const u = this._gatherMaterial.uniforms;
      u.uUp.value.copy(this._up);
      u.uFloorEnabled.value = this._floorProxy ? 1 : 0;
      this._applySelfBias();
    }
    const u = this._gatherMaterial.uniforms;
    u.uEnvRotation.value.copy(this._envRotation);
    u.uEnvBlur.value = blur;
    if (u.envMap) u.envMap.value = this._envTexture;
    this._buildEnvTable();
    this.reset();
  }

  /** Restart the progressive accumulation (scene or lighting changed). */
  reset() {
    this._frame = 0;
    this._needsClear = true;
    this.displayUniforms.uRadWeight.value = 0;
  }

  /**
   * Accumulate one bounce. Returns false when nothing was baked (no objects
   * or already converged).
   */
  bounce() {
    if (this._disposed || this._proxies.length === 0) return false;
    if (this.converged) return false;

    const renderer = this._renderer;
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    const prevClearAlpha = renderer.getClearAlpha();
    renderer.getClearColor(_clearColor);
    const prevShadowAuto = renderer.shadowMap.autoUpdate;
    const prevShadowNeeds = renderer.shadowMap.needsUpdate;
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = false;

    try {
      if (this._needsClear) this._clearLightmaps();
      this._syncProxies();
      if (this._frame === 0) this._computeBounds();
      this._frame++;
      this._updateDirections();

      const readRT = this._buffer1Active ? this._map1 : this._map2;
      const writeRT = this._buffer1Active ? this._map2 : this._map1;

      // --- Illumination pass ---
      const iu = this._illumMaterial.uniforms;
      iu.uPrev.value = readRT.texture;
      iu.uFrameSeed.value = this._frame;
      // The very first bounce has no radiosity yet: surfaces are black apart
      // from emission, so the first sample is pure direct (environment) light.
      iu.uAlbedoWeight.value = this._frame === 1 ? 0 : this._bounce;
      renderer.autoClear = false;
      renderer.setRenderTarget(this._illumination);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, false);
      renderer.render(this._scene, this._arrayCamera);

      // --- Gather pass (lightmap space) ---
      const gu = this._gatherMaterial.uniforms;
      gu.uIllum.value = this._illumination.texture;
      gu.uPrev.value = readRT.texture;
      gu.uInvSampleCount.value = 1 / this._frame;
      this._dilateMaterial.uniforms.uPrev.value = readRT.texture;
      renderer.setRenderTarget(writeRT);
      renderer.render(this._gatherScene, this._gatherCamera);

      this._buffer1Active = !this._buffer1Active;
      // Fade the lightmap in with the accumulated directions (smoothstep), so
      // the noisy early estimate never pops in and the solution is fully
      // visible only once ~RADIOSITY_FADE_IN_DIRECTIONS have been gathered.
      const t = Math.min(1, (this._frame * this._directions) / RADIOSITY_FADE_IN_DIRECTIONS);
      this.displayUniforms.uRadWeight.value = t * t * (3 - 2 * t);
    } finally {
      renderer.setRenderTarget(prevTarget);
      renderer.setClearColor(_clearColor, prevClearAlpha);
      renderer.autoClear = prevAutoClear;
      renderer.shadowMap.autoUpdate = prevShadowAuto;
      renderer.shadowMap.needsUpdate = prevShadowNeeds;
    }
    return true;
  }

  /** Free all GPU resources. */
  dispose() {
    this._disposed = true;
    this._clearProxies();
    this._map1.dispose();
    this._map2.dispose();
    this._illumination.dispose();
    this._envTableRT?.dispose();
    this._illumMaterial.dispose();
    this._gatherMaterial.dispose();
    this._dilateMaterial.dispose();
    this._dilatePlane.geometry.dispose();
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /**
   * Render the (rotated, pre-blurred) environment into a small equirect
   * luminance table and build the sampling distribution from it.
   */
  _buildEnvTable() {
    this._envTable = null;
    if (!this._envTexture || this._envHeight === 0 || this._disposed) return;
    const renderer = this._renderer;
    if (!this._envTableRT) {
      this._envTableRT = new THREE.WebGLRenderTarget(ENV_TABLE_W, ENV_TABLE_H, {
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        generateMipmaps: false,
        depthBuffer: false,
        stencilBuffer: false
      });
    }
    const gu = this._gatherMaterial.uniforms;
    const material = new THREE.ShaderMaterial({
      vertexShader: DILATE_VERTEX,
      fragmentShader: ENV_TABLE_FRAGMENT,
      uniforms: {
        envMap: { value: this._envTexture },
        uEnvRotation: { value: gu.uEnvRotation.value },
        uEnvBlur: { value: this._envBlur }
      },
      defines: { ...this._gatherMaterial.defines },
      depthTest: false,
      depthWrite: false
    });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    plane.frustumCulled = false;
    const scene = new THREE.Scene();
    scene.add(plane);
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    try {
      renderer.autoClear = true;
      renderer.setRenderTarget(this._envTableRT);
      renderer.render(scene, this._gatherCamera);
      const buf = new Float32Array(ENV_TABLE_W * ENV_TABLE_H * 4);
      renderer.readRenderTargetPixels(this._envTableRT, 0, 0, ENV_TABLE_W, ENV_TABLE_H, buf);
      const lum = new Float32Array(ENV_TABLE_W * ENV_TABLE_H);
      for (let i = 0; i < lum.length; i++) {
        lum[i] = 0.2126 * buf[i * 4] + 0.7152 * buf[i * 4 + 1] + 0.0722 * buf[i * 4 + 2];
      }
      this._envTable = new EnvImportanceTable(lum, ENV_TABLE_W, ENV_TABLE_H);
    } catch (error) {
      console.warn("radiosity: environment table failed, uniform sampling", error);
      this._envTable = null;
    } finally {
      renderer.setRenderTarget(prevTarget);
      renderer.autoClear = prevAutoClear;
      plane.geometry.dispose();
      material.dispose();
    }
  }

  /** Number of directions per bounce drawn uniformly (the rest from the env). */
  _uniformCount() {
    if (!this._importance || !this._envTable) return this._directions;
    if (this._directions === 1) return this._frame % 2 === 0 ? 1 : 0;
    return Math.ceil(this._directions / 2);
  }

  /** Fibonacci set for the uniformly drawn directions. */
  _rebuildBaseDirections() {
    const n = this._importance
      ? Math.max(1, Math.ceil(this._directions / 2))
      : this._directions;
    this._baseDirections = [];
    for (let k = 0; k < n; k++) this._baseDirections.push(fibonacciSphere(k, n));
  }

  /**
   * Zero both lightmaps. Render targets start with undefined contents (often
   * NaN on real GPUs); the first bounce samples the previous map for the
   * bounce radiance and the dilation, so garbage would poison every later
   * bounce.
   */
  _clearLightmaps() {
    const renderer = this._renderer;
    renderer.setClearColor(0x000000, 0);
    for (const rt of [this._map1, this._map2]) {
      renderer.setRenderTarget(rt);
      renderer.clear(true, false, false);
    }
    this._needsClear = false;
  }

  _createGatherMaterial() {
    const defines = {};
    if (this._envTexture && this._envHeight > 0) {
      // Same cube-UV layout constants three.js derives for PMREM textures.
      const maxMip = Math.log2(this._envHeight) - 2;
      const texelHeight = 1 / this._envHeight;
      const texelWidth = 1 / (3 * Math.max(Math.pow(2, maxMip), 7 * 16));
      defines.TCV_HAS_ENV = 1;
      defines.CUBEUV_TEXEL_WIDTH = texelWidth;
      defines.CUBEUV_TEXEL_HEIGHT = texelHeight;
      defines.CUBEUV_MAX_MIP = `${maxMip}.0`;
    }
    const tile = this.tileResolution;
    const uniforms = {
      uDir: { value: this._dirData },
      uBasisU: { value: this._basisU },
      uBasisV: { value: this._basisV },
      uOrigin: { value: this._origins },
      uHalf: { value: this._halves },
      uTileOffset: { value: this._tileOffsets },
      uDirWeight: { value: this._dirWeights },
      uTileScale: { value: new THREE.Vector2(1 / this._tilesX, 1 / this._tilesY) },
      uIllumTexel: { value: new THREE.Vector2(1 / (this._tilesX * tile), 1 / (this._tilesY * tile)) },
      uIllum: { value: null },
      uPrev: { value: null },
      uInvSampleCount: { value: 1 },
      uSelfBias: { value: 0.01 },
      uUp: { value: new THREE.Vector3(0, 0, 1) },
      uFloorEnabled: { value: 0 },
      uFloorAlbedo: { value: new THREE.Color(0.3, 0.3, 0.3) },
      uFloorRefUv: { value: new THREE.Vector2(0, 0) },
      uEnvRotation: { value: new THREE.Matrix3() },
      uEnvBlur: { value: this._envBlur }
    };
    if (defines.TCV_HAS_ENV) {
      uniforms.envMap = { value: this._envTexture };
    }
    const material = new THREE.ShaderMaterial({
      vertexShader: GATHER_VERTEX,
      fragmentShader: gatherFragment(this._directions, this.preset.gatherRadius),
      uniforms,
      defines,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending
    });
    material.name = "radiosityGather";
    return material;
  }

  _makeProxy(source) {
    const mesh = new THREE.Mesh(source.geometry, this._illumMaterial);
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    const proxy = {
      source,
      mesh,
      albedo: new THREE.Color(),
      emissive: new THREE.Color()
    };
    // Per-object uniforms through a shared material: set them right before
    // the draw and force the upload.
    mesh.onBeforeRender = () => {
      const u = this._illumMaterial.uniforms;
      u.uAlbedo.value.copy(proxy.albedo);
      u.uEmissive.value.copy(proxy.emissive);
      this._illumMaterial.uniformsNeedUpdate = true;
    };
    this._scene.add(mesh);

    const gather = new THREE.Mesh(source.geometry, this._gatherMaterial);
    gather.frustumCulled = false;
    gather.matrixAutoUpdate = false;
    gather.matrixWorldAutoUpdate = false;
    gather.matrixWorld = mesh.matrixWorld; // shared: synced once per bounce
    gather.userData.proxy = proxy;
    this._gatherScene.add(gather);
    this._gatherProxies.push(gather);
    return proxy;
  }

  _clearProxies() {
    for (const p of this._proxies) this._scene.remove(p.mesh);
    if (this._floorProxy) this._scene.remove(this._floorProxy.mesh);
    for (const g of this._gatherProxies) this._gatherScene.remove(g);
    this._proxies = [];
    this._gatherProxies = [];
    this._floorProxy = null;
  }

  /** Mirror world matrices, visibility and albedo from the display meshes. */
  _syncProxies() {
    const sync = (p) => {
      const src = p.source.mesh;
      p.mesh.matrixWorld.copy(src.matrixWorld);
      p.mesh.visible = isWorldVisible(src);
      p.albedo.copy(p.source.albedo());
      p.emissive.copy(p.source.emissive());
    };
    for (const p of this._proxies) sync(p);
    if (this._floorProxy) {
      sync(this._floorProxy);
      this._gatherMaterial.uniforms.uFloorAlbedo.value.copy(this._floorProxy.albedo);
    }
    for (const g of this._gatherProxies) {
      g.visible = g.userData.proxy.mesh.visible;
    }
  }

  /** World bounds of the visible (non-floor) surfaces. */
  _computeBounds() {
    this._bounds.makeEmpty();
    for (const p of this._proxies) {
      if (!p.mesh.visible) continue;
      const geom = p.source.geometry;
      if (!geom.boundingBox) geom.computeBoundingBox();
      if (!geom.boundingBox) continue;
      for (let i = 0; i < 8; i++) {
        _corner.set(
          i & 1 ? geom.boundingBox.max.x : geom.boundingBox.min.x,
          i & 2 ? geom.boundingBox.max.y : geom.boundingBox.min.y,
          i & 4 ? geom.boundingBox.max.z : geom.boundingBox.min.z
        );
        _corner.applyMatrix4(p.mesh.matrixWorld);
        this._bounds.expandByPoint(_corner);
      }
    }
    if (this._bounds.isEmpty()) {
      this._bounds.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    }
    this._bounds.getCenter(this._boundsCenter);
    this._bounds.getSize(_v);
    this._maxExtent = Math.max(_v.x, _v.y, _v.z, 1e-6);
    this._lateral = this._maxExtent * (1 + 2 * MARGIN);
    this._applySelfBias();
  }

  /** Self-intersection bias in world units ≈ N illumination texels. */
  _applySelfBias() {
    this._gatherMaterial.uniforms.uSelfBias.value =
      (this._lateral / this.tileResolution) * this._selfBiasTexels;
  }

  /**
   * Refresh the global directions (a low-discrepancy rotation of a Fibonacci
   * sphere per frame) and fit each orthographic sub-camera tightly around the
   * scene bounds as seen along its direction.
   */
  _updateDirections() {
    stratifiedRotation(_quat, this._frame);
    const margin = this._maxExtent * MARGIN;
    const b = this._bounds;
    const N = this._directions;
    const table = this._importance ? this._envTable : null;
    const nUniform = this._uniformCount();
    const nImportance = N - nUniform;
    const uniformPdf = 1 / (4 * Math.PI);
    for (let k = 0; k < N; k++) {
      const d = this._dirData[k];
      if (k < nUniform || !table) {
        d.copy(this._baseDirections[k % this._baseDirections.length])
          .applyQuaternion(_quat)
          .normalize();
      } else {
        // Low-discrepancy 2D point (R2 sequence) for this frame and slot.
        const n = this._frame * nImportance + (k - nUniform);
        const u1 = (0.5 + 0.7548776662466927 * n) % 1;
        const u2 = (0.5 + 0.5698402909980532 * n) % 1;
        table.sample(u1, u2, d);
      }
      if (table) {
        // Balance heuristic over the 50/50 mixture of uniform and env pdfs.
        const pdfMix = 0.5 * uniformPdf + 0.5 * table.pdf(d);
        this._dirWeights[k] = 1 / (N * Math.PI * pdfMix);
      } else {
        this._dirWeights[k] = 4 / N;
      }
      const cam = this._subCameras[k];
      cam.up.set(0, 1, 0);
      if (Math.abs(d.y) > 0.99) cam.up.set(0, 0, 1);
      cam.position.copy(this._boundsCenter).addScaledVector(d, -1);
      cam.lookAt(this._boundsCenter);
      cam.updateMatrixWorld();
      const bu = this._basisU[k].setFromMatrixColumn(cam.matrixWorld, 0);
      const bv = this._basisV[k].setFromMatrixColumn(cam.matrixWorld, 1);
      // Project the bounds corners onto the camera basis.
      let minU = Infinity;
      let maxU = -Infinity;
      let minV = Infinity;
      let maxV = -Infinity;
      let minD = Infinity;
      let maxD = -Infinity;
      for (let i = 0; i < 8; i++) {
        _corner
          .set(
            i & 1 ? b.max.x : b.min.x,
            i & 2 ? b.max.y : b.min.y,
            i & 4 ? b.max.z : b.min.z
          )
          .sub(this._boundsCenter);
        const pu = _corner.dot(bu);
        const pv = _corner.dot(bv);
        const pd = _corner.dot(d);
        if (pu < minU) minU = pu;
        if (pu > maxU) maxU = pu;
        if (pv < minV) minV = pv;
        if (pv > maxV) maxV = pv;
        if (pd < minD) minD = pd;
        if (pd > maxD) maxD = pd;
      }
      const halfU = (maxU - minU) / 2 + margin;
      const halfV = (maxV - minV) / 2 + margin;
      const near = minD - margin;
      // Reach well past the far side so the ground under the model is covered.
      const far = maxD + margin + this._maxExtent * 1.5;
      this._halves[k].set(halfU, halfV);
      // Origin: tile centre on the near plane.
      const cu = (minU + maxU) / 2;
      const cv = (minV + maxV) / 2;
      cam.position
        .copy(this._boundsCenter)
        .addScaledVector(bu, cu)
        .addScaledVector(bv, cv)
        .addScaledVector(d, near);
      this._origins[k].copy(cam.position);
      cam.lookAt(_v.copy(cam.position).add(d));
      cam.left = -halfU;
      cam.right = halfU;
      cam.top = halfV;
      cam.bottom = -halfV;
      cam.near = 0;
      cam.far = far - near;
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
      // lookAt may have rolled the basis; re-read it from the final matrix.
      bu.setFromMatrixColumn(cam.matrixWorld, 0);
      bv.setFromMatrixColumn(cam.matrixWorld, 1);
    }
    this._arrayCamera.updateMatrixWorld();
  }
}

/** Whether the renderer can run the solver (WebGL2 + float render targets). */
export function radiositySupported(renderer) {
  try {
    const caps = renderer?.capabilities;
    if (!caps || !caps.isWebGL2) return false;
    return renderer.extensions.has("EXT_color_buffer_float");
  } catch {
    return false;
  }
}
