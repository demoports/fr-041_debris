// WebGL2 execution of the semantic scene/material/IPP jobs emitted by the
// plain-JavaScript WZ3 runtime. The original generated D3D shaders at precalc
// time; this port folds the used behavior into a compact WebGL2 program.
import {
  RenderTarget,
  TextureCache,
  cameraMatrices,
  colorARGB,
  colorRGB,
  createWebGL2,
  linkProgram,
  mat4Inverse,
  uniformLocations,
} from './gl.js';
import { f32, mat4Identity, mat4Mul, mat4SRT } from './core.js';
import { executeEffectJob } from './effects.js';
import {
  LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH,
  LEGACY_ATTENUATION_VOLUME_SIZE,
  LEGACY_NORMALIZER_CUBE_BYTE_LENGTH,
  LEGACY_NORMALIZER_CUBE_SIZE,
  legacyCubeNormalizerFace,
  makeLegacyAttenuationVolume,
  makeLegacyCubeNormalizer,
} from './legacy_lookup.js';

// The released non-KKRIEGER engine used 16 active lights. The player sorts
// candidates by range/distance importance before applying this cap.
const MAX_LIGHTS = 16;
const M11_NORMALIZER_TEXTURE_UNIT = 5;
const M11_ATTENUATION_TEXTURE_UNIT = 6;
const INSTANCE_MATRIX_BYTES = 16 * Float32Array.BYTES_PER_ELEMENT;
const EMPTY_INSTANCE_MATRICES = new Float32Array(0);
// Keep the reusable stream modest even for embedders with unusually large
// transient instance batches. Oversized viewports still render from an exact
// temporary store, which is shrunk again on the next ordinary viewport.
const MAX_RETAINED_INSTANCE_BUFFER_BYTES = 4 * 1024 * 1024;
const MAX_RETAINED_SHADOW_SCRATCH_BYTES = 4 * 1024 * 1024;
const LIGHT_SORT_POST = 0x0f;
const ALPHA_REFERENCE = 128 / 255;
const MBF_ALPHATEST = 0x0001;
const MBF_DOUBLESIDED = 0x0002;
const MBF_INVERTCULL = 0x0004;
const MBF_ZBIASBACK = 0x0008;
const MBF_ZBIASFORE = 0x0010;
const MBF_ZONLY = 0x0020;
const MBF_ZEQUAL = 0x0080;
const MBF_ZWRITE = 0x0100;
const MBF_ZREAD = 0x0200;
const MBF_SHADOWMASK = 0x040000;
const MBF_STENCILTEST = 0x300000;
const LEGACY_DEPTH_BIAS = 1 / 65536;
const DEPTH24_OFFSET_UNITS = LEGACY_DEPTH_BIAS * 0x1000000;
const USAGE_ORDER = Object.freeze({
  base: 0, prelight: 1, ambient: 2, shadow: 3,
  light: 4, postlight: 5, postlight2: 6, other: 7,
});
// Native paint jobs use the address of their GenMaterialPass record as the
// low sort-key field. JavaScript deliberately exposes no object address, so a
// monotonic identity assigned to the pass object is the closest stable
// analogue: it keeps all jobs for one pass together and gives distinct passes
// a deterministic order for the lifetime of the player. Equal identities are
// still resolved by reverse insertion order below, just like the stable native
// radix sort after Engine::AddPaintJob's head insertion.
const materialPassSortIds = new WeakMap();
let nextMaterialPassSortId = 1;
// GenMesh::Init and GenMinMesh::AddCluster do not leave unlinked faces on an
// unlit fallback. They attach GenOverlayManager->DefaultMat, whose sole pass
// is ENGU_LIGHT/MPP_STATIC. This matters visibly for the first greeting text:
// rendering its default cluster as a constant color makes every pixel exceed
// the contemporaneous glare threshold instead of letting the authored light
// direction and attenuation shade it.
const DEFAULT_RENDER_PASS = Object.freeze({
  usage: 'light', program: 'static', renderPass: 0, state: 'default-material-light',
});

// Literal player (sPLAYER=1, sINTRO=0) DefaultMat state from genoverlay.cpp.
// It is represented as a synthetic Material11 only at the WebGL boundary;
// native stores the same sMaterial11 behind GenOverlayManager->DefaultMat.
const DEFAULT_MATERIAL11_PARAMETERS = (() => {
  const parameters = new Array(64).fill(0);
  parameters[0] = 0x0300;       // sMBF_ZON
  parameters[2] = 0x0005;       // sMLF_BUMPX
  parameters.fill(0x0002, 4, 8); // sMTF_MIPMAPS
  parameters[8] = 0x0010;       // LIGHT SET
  parameters[9] = 0x0040;       // COLOR0 MUL
  parameters[28] = 0x00c0c0c0;
  parameters[29] = 0x00404040;
  parameters[30] = 0xffffffff;
  parameters[31] = 0xffffffff;
  parameters[32] = 32;
  // materialView's light-pass dynamic slots normally come from MultiPara.
  // DefaultMat is constructed directly, so mirror its Color0/SpecPower there
  // as well while keeping MultiFlags zero (it has no generated insert).
  parameters[51] = 32;
  parameters[52] = 0x00c0c0c0;
  return Object.freeze(parameters);
})();
const DEFAULT_MATERIAL11 = Object.freeze({
  kind: 'material', system: '1.1',
  parameters: DEFAULT_MATERIAL11_PARAMETERS,
  initialParameters: DEFAULT_MATERIAL11_PARAMETERS,
  textures: Object.freeze([]),
  passes: Object.freeze([DEFAULT_RENDER_PASS]),
});

function materialPassSortIdentity(material, pass) {
  const key = pass && (typeof pass === 'object' || typeof pass === 'function')
    ? pass : material;
  if (!key || (typeof key !== 'object' && typeof key !== 'function')) return 1;
  let identity = materialPassSortIds.get(key);
  if (identity === undefined) {
    identity = nextMaterialPassSortId++;
    materialPassSortIds.set(key, identity);
  }
  return identity;
}

const VERTEX_SOURCE = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in vec4 aColor;
layout(location=4) in vec4 aTangent;
layout(location=5) in vec4 aInstance0;
layout(location=6) in vec4 aInstance1;
layout(location=7) in vec4 aInstance2;
layout(location=8) in vec4 aInstance3;
layout(location=9) in vec2 aUV1;
uniform mat4 uViewProjection;
uniform vec4 uUVTransform;
uniform int uMaterial20;
uniform ivec4 uM20SamplerFlags;
uniform vec4 uM20SamplerScales;
uniform vec4 uM20UVTransform1[2];
uniform vec4 uM20UVTransform2[2];
uniform int uMaterial11;
uniform mat4 uM11View;
uniform mat4 uM11WorldToModel;
uniform mat4 uM11ModelToWorld;
uniform int uM11SpecialFlags;
uniform int uLegacyLightingMode;
uniform int uConditionLegacyBasis;
uniform vec3 uMaterialCameraPosition;
uniform float uSpecularStrength;
uniform int uLightCount;
uniform vec4 uLightPosition[${MAX_LIGHTS}];
uniform vec4 uLightAttenuation[${MAX_LIGHTS}];
uniform ivec4 uM11SamplerFlags;
uniform vec4 uM11SamplerScales;
uniform vec4 uM11UVTransform1[2];
uniform vec4 uM11UVTransform2[2];
out vec3 vWorld;
out vec3 vNormal;
out vec3 vTangent;
out vec3 vLegacyLight;
out vec3 vLegacyHalfway;
out vec3 vLegacyAttenuation;
out vec2 vUV;
out vec2 vM20UV0;
out vec2 vM20UV1;
out vec2 vM20UV2;
out vec2 vM20UV3;
out vec2 vM11UV0;
out vec2 vM11UV1;
out vec2 vM11UV2;
out vec2 vM11UV3;
out vec4 vColor;
out vec4 vClipPosition;

vec2 material20UV(vec2 uv, int samplerIndex) {
  int mode = (uM20SamplerFlags[samplerIndex] >> 12) & 15;
  if (mode == 1) return uv * uM20SamplerScales[samplerIndex];
  vec4 source = vec4(uv, 0.0, 1.0);
  if (mode == 2) return vec2(dot(source, uM20UVTransform1[0]),
    dot(source, uM20UVTransform1[1]));
  if (mode == 3) return vec2(dot(source, uM20UVTransform2[0]),
    dot(source, uM20UVTransform2[1]));
  return uv;
}

vec2 material11UV(int samplerIndex, vec4 world, vec3 worldNormal,
    vec4 modelPosition, vec3 modelNormal) {
  int flags = uM11SamplerFlags[samplerIndex];
  vec2 selectedUV = ((flags >> 4) & 3) == 1 ? aUV1 : aUV;
  vec4 source = vec4(selectedUV, 0.0, 1.0);

  if (samplerIndex == 2) {
    int environmentMode = (uM11SpecialFlags >> 4) & 3;
    vec3 eyeNormal = mat3(uM11View) * worldNormal;
    if (environmentMode == 1) {
      source.xy = eyeNormal.xy * vec2(0.5, -0.5) + 0.5;
    } else if (environmentMode == 2) {
      vec3 eyePosition = (uM11View * world).xyz;
      float eyeSquared = dot(eyePosition, eyePosition);
      vec3 eyeDirection = eyeSquared > 1e-20
        ? eyePosition * inversesqrt(eyeSquared) : vec3(0.0);
      float normalSquared = dot(eyeNormal, eyeNormal);
      float factor = normalSquared > 1e-20
        ? 2.0 * dot(eyeDirection, eyeNormal) / normalSquared : 0.0;
      source.xyz = eyeNormal * factor - eyeDirection;
      source.xy = source.xy * vec2(0.5, -0.5) + 0.5;
    } else if (environmentMode == 3) {
      // Undocumented source mode used by Debris water lighting.
      source.xyz = modelNormal;
    }
  } else if (samplerIndex == 3) {
    int projectionMode = (uM11SpecialFlags >> 8) & 7;
    if (projectionMode == 1) source = modelPosition;
    else if (projectionMode == 2) source = world;
    else if (projectionMode == 3) source = uM11View * world;
  }

  int transformMode = (flags >> 12) & 3;
  if (transformMode == 1) return source.xy * uM11SamplerScales[samplerIndex];
  if (transformMode == 2) return vec2(dot(source, uM11UVTransform1[0]),
    dot(source, uM11UVTransform1[1]));
  if (transformMode == 3) return vec2(dot(source, uM11UVTransform2[0]),
    dot(source, uM11UVTransform2[1]));
  return source.xy;
}

void main() {
  mat4 model = mat4(aInstance0, aInstance1, aInstance2, aInstance3);
  vec4 world = model * vec4(aPosition, 1.0);
  vec3 worldNormal = mat3(model) * aNormal;
  vec4 modelPosition = uM11WorldToModel * world;
  vec3 modelNormal = mat3(uM11WorldToModel) * worldNormal;
  vec3 worldTangent = mat3(model) * aTangent.xyz;
  // material11.vsh CodegenFlags variant 1 repairs the skinned input basis
  // before any model/view transform. This applies to texture/environment
  // phases as well as light phases; static and shader-instance variants keep
  // the authored vectors raw.
  vec3 conditionedInputNormal = aNormal;
  vec3 conditionedInputTangent = aTangent.xyz;
  if (uConditionLegacyBasis != 0) {
    conditionedInputNormal = normalize(conditionedInputNormal);
    conditionedInputTangent = normalize(conditionedInputTangent -
      conditionedInputNormal * dot(conditionedInputNormal,
        conditionedInputTangent));
  }
  vec3 conditionedWorldNormal = mat3(model) * conditionedInputNormal;
  vec3 conditionedWorldTangent = mat3(model) * conditionedInputTangent;
  vec3 material11WorldNormal = uConditionLegacyBasis != 0
    ? conditionedWorldNormal : worldNormal;
  vec3 material11ModelNormal = mat3(uM11WorldToModel) * material11WorldNormal;
  vWorld = world.xyz;
  vNormal = normalize(worldNormal);
  vTangent = normalize(worldTangent);
  // material11.vsh and material20_light.vsh calculate lighting coordinates at
  // vertices. Undo only the paint job's base matrix; shader-instance
  // transforms deliberately remain in position/N/T, matching transformedPos.
  vec3 materialTangent = mat3(uM11WorldToModel) * worldTangent;
  vec3 legacyNormal = modelNormal;
  vec3 legacyTangent = materialTangent;
  // Native's animated-mesh shader variant repairs skinned N/T. Static meshes,
  // CPU instances and shader instances use the authored/transformed vectors
  // directly, including their lengths and lack of orthogonality.
  if (uLegacyLightingMode == 2) {
    if (uConditionLegacyBasis != 0) {
      // Material11 variant 1 performed the repair on its input attributes;
      // transform that repaired basis into the same model space as L/H.
      legacyNormal = material11ModelNormal;
      legacyTangent = mat3(uM11WorldToModel) * conditionedWorldTangent;
    } else {
      legacyNormal = normalize(legacyNormal);
      legacyTangent = normalize(legacyTangent - legacyNormal *
        dot(legacyNormal, legacyTangent));
    }
  }
  vec3 legacyBitangent = cross(legacyNormal, legacyTangent);
  vLegacyLight = vec3(0.0);
  vLegacyHalfway = vec3(0.0);
  vLegacyAttenuation = vec3(0.0);
  bool legacyLightPass = uLegacyLightingMode == 1 || uLegacyLightingMode == 2;
  if (uLegacyLightingMode == 3) {
    // material20_envi.vsh forms B in input space, then transforms raw N/B/T
    // separately. In particular, cross(worldN,worldT) is wrong for authored
    // non-uniform transforms. The otherwise-unused light varyings keep this
    // source order without increasing the varying budget.
    vLegacyLight = worldNormal;
    vLegacyHalfway = mat3(model) * cross(aNormal, aTangent.xyz);
    vLegacyAttenuation = worldTangent;
  } else if (legacyLightPass && uLightCount > 0) {
    vec3 lightVector = uLightPosition[0].xyz -
      modelPosition.xyz * uLightPosition[0].w;
    // The native shaders intentionally leave L unnormalised until after
    // raster interpolation, and project in their unusual N,B,T ordering.
    vLegacyLight = vec3(dot(lightVector, legacyNormal),
      dot(lightVector, legacyBitangent), dot(lightVector, legacyTangent));
    if (uSpecularStrength > 0.0) {
      vec3 eyeVector = normalize(uMaterialCameraPosition - modelPosition.xyz);
      vec3 halfwayVector = normalize(lightVector) + eyeVector;
      vLegacyHalfway = vec3(dot(halfwayVector, legacyNormal),
        dot(halfwayVector, legacyBitangent), dot(halfwayVector, legacyTangent));
    }
    // M20 emits its centred vector directly. material11.vsh instead performs
    // the c5 MAD at each vertex so the 3D sampler receives [0,1] coordinates
    // after raster interpolation, including the source's float ordering.
    vec3 legacyAttenuationVector =
      (modelPosition.xyz - uLightAttenuation[0].xyz) *
      uLightAttenuation[0].w;
    vLegacyAttenuation = uMaterial11 != 0
      ? legacyAttenuationVector * 0.5 + vec3(0.5)
      : legacyAttenuationVector;
  }
  vUV = aUV * uUVTransform.xy + uUVTransform.zw;
  vM20UV0 = uMaterial20 != 0 ? material20UV(aUV, 0) : vUV;
  vM20UV1 = uMaterial20 != 0 ? material20UV(aUV, 1) : vUV;
  vM20UV2 = uMaterial20 != 0 ? material20UV(aUV, 2) : vUV;
  vM20UV3 = uMaterial20 != 0 ? material20UV(aUV, 3) : vUV;
  vM11UV0 = uMaterial11 != 0
    ? material11UV(0, world, material11WorldNormal, modelPosition,
      material11ModelNormal) : vUV;
  vM11UV1 = uMaterial11 != 0
    ? material11UV(1, world, material11WorldNormal, modelPosition,
      material11ModelNormal) : vUV;
  vM11UV2 = uMaterial11 != 0
    ? material11UV(2, world, material11WorldNormal, modelPosition,
      material11ModelNormal) : vUV;
  vM11UV3 = uMaterial11 != 0
    ? material11UV(3, world, material11WorldNormal, modelPosition,
      material11ModelNormal) : vUV;
  vColor = aColor;
  vClipPosition = uViewProjection * world;
  gl_Position = vClipPosition;
}`;

const FRAGMENT_SOURCE = `#version 300 es
precision highp float;
precision highp int;
// GLSL ES defines defaults for sampler2D/samplerCube, but not sampler3D.
// ANGLE's Metal compiler therefore requires this explicit precision.
precision highp sampler3D;
in vec3 vWorld;
in vec3 vNormal;
in vec3 vTangent;
in vec3 vLegacyLight;
in vec3 vLegacyHalfway;
in vec3 vLegacyAttenuation;
in vec2 vUV;
in vec2 vM20UV0;
in vec2 vM20UV1;
in vec2 vM20UV2;
in vec2 vM20UV3;
in vec2 vM11UV0;
in vec2 vM11UV1;
in vec2 vM11UV2;
in vec2 vM11UV3;
in vec4 vColor;
in vec4 vClipPosition;
uniform sampler2D uTexture0;
uniform sampler2D uTexture1;
uniform sampler2D uTexture2;
uniform sampler2D uTexture3;
uniform sampler2D uPrelightTexture;
uniform samplerCube uM11NormalizerCube;
uniform sampler3D uM11AttenuationVolume;
uniform int uTextureMask;
uniform int uMode;
uniform int uVertexColorMode;
uniform int uUsePrelight;
uniform int uMaterial20;
uniform int uMaterial11;
uniform int uM11MultipassLight;
uniform int uM11SpecialFlags;
uniform int uLegacyLightingMode;
uniform int uM20Flags;
uniform int uM20EnvironmentFlags;
uniform int uM20RuntimeEnvironmentFlags;
uniform ivec2 uDetailOps;
uniform int uCombiners[13];
uniform int uAlphaCombiner;
uniform vec4 uMaterialColors[4];
uniform vec4 uBaseColor;
uniform vec3 uAmbient;
uniform vec3 uCameraPosition;
uniform vec3 uMaterialCameraPosition;
uniform mat4 uM11View;
uniform mat4 uM11WorldToModel;
uniform mat4 uM11ModelToWorld;
uniform vec3 uFogColor;
uniform vec2 uFogRange;
uniform int uFogEnabled;
uniform float uAlphaCutoff;
uniform float uSpecularPower;
uniform vec3 uSpecularColor;
uniform float uSpecularStrength;
uniform int uLightCount;
uniform vec4 uLightPosition[${MAX_LIGHTS}];
uniform vec4 uLightAttenuation[${MAX_LIGHTS}];
uniform vec4 uLightColor[${MAX_LIGHTS}];
uniform float uLightSpecular[${MAX_LIGHTS}];
uniform vec4 uM11LightConstant;
out vec4 outColor;

vec4 texOrWhite(sampler2D image, int bit, vec2 uv) {
  return (uTextureMask & bit) != 0 ? texture(image, uv) : vec4(1.0);
}

vec3 bumpVector(vec4 sampleValue) {
  // Q8W8V8U8 arrives directly from an RGBA8_SNORM sampler. Native unsigned
  // bump formats remain their raw 0..1 samples; the released shaders do not
  // apply a conventional *2-1 remap to either representation.
  return sampleValue.xyz;
}

vec3 detailCombine(vec3 current, vec3 detail, int operation) {
  if (operation == 0) return current * detail;
  if (operation == 1) return current * detail * 2.0;
  if (operation == 2) return current * detail * 4.0;
  if (operation == 3) return current + detail;
  if (operation == 4) return current + detail - current * detail;
  return current;
}

// material11.psh implements its PS1.1 specular power with four paired
// multiply instructions rather than a hardware pow. material11.cpp derives
// these two constants from the authored exponent (clamped to 8..60).
float material11Specular(float cosine, float power) {
  float usePower = clamp(power, 8.0, 60.0);
  float inverseScale = 1.0 / 7.336032;
  float scale = inverseScale + (usePower - 8.0) * 0.01565302;
  float bias = inverseScale - scale;
  float value = clamp((cosine * scale + bias) * 4.0, 0.0, 1.0);
  value = value * value * 2.0;
  value = value * value * 2.0;
  return clamp(value * value * 2.0, 0.0, 1.0);
}

vec4 material11Source(int index, vec4 t0, vec4 t1, vec4 t2, vec4 t3) {
  if (index == 1) return uMaterialColors[0];
  if (index == 2) return uMaterialColors[1];
  if (index == 3) return t0;
  if (index == 4) return t1;
  if (index == 7) return uMaterialColors[2];
  if (index == 8) return uMaterialColors[3];
  if (index == 9) return t2;
  if (index == 10) return t3;
  if (index == 11) return vColor;
  return vec4(0.0);
}

float material11Alpha(int source, vec4 t0, vec4 t1, vec4 t2, vec4 t3) {
  if (source == 4) return t0.a;
  if (source == 5) return t1.a;
  if (source == 6) return t2.a;
  if (source == 7) return t3.a;
  if (source == 8) return uMaterialColors[0].a;
  if (source == 9) return uMaterialColors[1].a;
  if (source == 10) return uMaterialColors[2].a;
  if (source == 11) return uMaterialColors[3].a;
  if (source == 12) return 0.5;
  if (source == 13) return 0.0;
  return 1.0;
}

vec4 material11Combine(vec4 t0, vec4 t1, vec4 t2, vec4 t3) {
  vec3 current = vec3(0.0);
  for (int index = 0; index < 13; ++index) {
    int word = uCombiners[index];
    int operation = (word >> 4) & 15;
    if (operation == 0) continue;
    vec3 source = material11Source(index, t0, t1, t2, t3).rgb;
    int alphaCode = word & 15;
    float sourceAlpha = material11Alpha(alphaCode, t0, t1, t2, t3);
    // sMCA_INVERTA/B belong to AlphaCombiner. material11.cpp deliberately
    // does not propagate either bit into a color-combiner stage's AlphaSrc.
    if (alphaCode != 0 && operation != 9) source *= sourceAlpha;
    if (operation == 1) current = source;
    else if (operation == 2) current += source;
    else if (operation == 3) current -= source;
    else if (operation == 4) current *= source;
    else if (operation == 5) current *= source * 2.0;
    else if (operation == 6) current *= source * 4.0;
    else if (operation == 7) current *= source * 8.0;
    else if (operation == 8) current = current * (vec3(1.0) - source) + source;
    else if (operation == 9) current = mix(source, current, sourceAlpha);
    else if (operation == 10) current = source - current;
    else if (operation == 11) current = vec3(dot(current, source));
    else if (operation == 12) current = clamp(current * source, 0.0, 1.0);
  }
  int alphaA = uAlphaCombiner & 15;
  int alphaB = (uAlphaCombiner >> 4) & 15;
  float alpha;
  if (alphaB == 14) {
    // The source's special MIX01 path bypasses AlphaOps (and therefore both
    // inversion bits): +mul r0.w,t0,c0; mad r0.w,t1,c0,r0.
    alpha = (t0.a + t1.a) * uMaterialColors[0].a;
  } else {
    alpha = material11Alpha(alphaA, t0, t1, t2, t3);
    if ((uAlphaCombiner & 256) != 0) alpha = 1.0 - alpha;
    if (alphaB != 0) {
      float second = material11Alpha(alphaB, t0, t1, t2, t3);
      if ((uAlphaCombiner & 512) != 0) second = 1.0 - second;
      alpha *= second;
    }
  }
  return vec4(current, alpha);
}

void main() {
  vec2 textureUV0 = uMaterial11 != 0 ? vM11UV0 : vM20UV0;
  vec2 textureUV1 = uMaterial11 != 0 ? vM11UV1 : vM20UV1;
  vec2 textureUV2 = uMaterial11 != 0 ? vM11UV2 : vM20UV2;
  vec2 textureUV3 = uMaterial11 != 0 ? vM11UV3 : vM20UV3;
  vec4 t0 = texOrWhite(uTexture0, 1, textureUV0);
  vec4 t1 = texOrWhite(uTexture1, 2, textureUV1);
  vec4 t2 = texOrWhite(uTexture2, 4, textureUV2);
  vec4 t3 = texOrWhite(uTexture3, 8, textureUV3);
  vec4 albedo = uBaseColor * t0;
  if (uVertexColorMode == 1) albedo *= vColor;
  else if (uVertexColorMode == 2) albedo.rgb += vColor.rgb;
  else if (uVertexColorMode == 3) albedo = vColor * t0;
  else if (uVertexColorMode == 4) {
    // Generated Material11 BASE: Color0 SET + vertex ADD, with Tex0 used by
    // AlphaCombiner only. Its RGB must not be modulated by the alpha texture.
    albedo = vec4(uBaseColor.rgb + vColor.rgb, uBaseColor.a * t0.a);
  }
  // Legacy light passes receive N,B,T-space L/H from the vertex shader. Their
  // no-bump normal is +N, represented by X in that unusual tangent ordering.
  bool legacyVertexLighting =
    uLegacyLightingMode == 1 || uLegacyLightingMode == 2;
  bool legacyEnvironmentBasis = uLegacyLightingMode == 3;
  vec3 normal = legacyVertexLighting ? vec3(1.0, 0.0, 0.0) : normalize(vNormal);
  if (uMode == 2 && ((uTextureMask & 2) != 0 ||
      (uMaterial20 != 0 && (uTextureMask & 4) != 0))) {
    vec3 mapped = vec3(1.0, 0.0, 0.0);
    int bumpCount = 0;
    if ((uTextureMask & 2) != 0) {
      mapped = bumpVector(t1);
      bumpCount = 1;
    }
    if (uMaterial20 != 0 && (uTextureMask & 4) != 0) {
      vec3 secondMap = bumpVector(t2);
      mapped = (uTextureMask & 2) != 0 ? mapped + secondMap : secondMap;
      bumpCount += 1;
    }
    // material20_light.psh normalizes only a sum of two maps or when Flags[5]
    // requests it. material11.psh uses its filtered bump sample directly.
    // Retain normalization only for non-legacy embedder materials. The
    // released player's DefaultMat is represented as Material11 above.
    bool renormalizeBump = uMaterial20 != 0
      ? bumpCount > 1 || (uM20Flags & 32) != 0
      : uMaterial11 == 0;
    if (renormalizeBump) mapped = normalize(mapped);
    if (legacyVertexLighting) {
      // Native light pixel shaders dot the sampled vector directly against
      // interpolated, cube/NRM-normalized tangent-space L and H.
      normal = mapped;
    } else {
      vec3 tangent = normalize(vTangent - normal * dot(normal, vTangent));
      vec3 bitangent = normalize(cross(normal, tangent));
      normal = normal * mapped.x + bitangent * mapped.y + tangent * mapped.z;
      if (renormalizeBump) normal = normalize(normal);
    }
  }

  vec4 result = albedo;
  if (uMode == 1) {
    result = vec4(0.0, 0.0, 0.0, albedo.a);
  } else if (uMode == 2) {
    vec3 diffuseLighting = vec3(0.0);
    vec3 specularLighting = vec3(0.0);
    float material11SpecularLighting = 0.0;
    float material20AttenuationAlpha = 0.0;
    if (legacyVertexLighting && uLightCount > 0) {
      // Material20 uses NRM after interpolation. Material11 instead samples
      // the native 64x64 cube lookup, including its 8-bit quantization and
      // bilinear face filtering, before PS1.1's _bx2 modifier.
      vec3 lightDirection = uMaterial11 != 0
        ? texture(uM11NormalizerCube, vLegacyLight).rgb * 2.0 - 1.0
        : normalize(vLegacyLight);
      float radiusSquared = clamp(dot(vLegacyAttenuation,
        vLegacyAttenuation), 0.0, 1.0);
      // M11's 32^3 lookup is intentionally asymmetric around texture-space
      // 0.5 by half a texel and quantized to RGBA8. M20 keeps its analytic
      // saturated MAD and exports radiusSquared in alpha.
      float attenuation = uMaterial11 != 0
        ? texture(uM11AttenuationVolume, vLegacyAttenuation).r
        : 1.0 - radiusSquared;
      float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
      float specular = 0.0;
      if (uSpecularStrength > 0.0) {
        vec3 halfwayDirection = uMaterial11 != 0
          ? texture(uM11NormalizerCube, vLegacyHalfway).rgb * 2.0 - 1.0
          : normalize(vLegacyHalfway);
        float cosine = clamp(dot(normal, halfwayDirection), 0.0, 1.0);
        specular = pow(cosine, max(1.0, uSpecularPower));
        if (uM11MultipassLight != 0) {
          specular = material11Specular(cosine, uSpecularPower);
        }
        if (uMaterial20 != 0 && (uM20Flags & 128) != 0) {
          specular *= (uTextureMask & 2) != 0 ? t1.a : t2.a;
        }
        if (uM11MultipassLight != 0 && (uM11SpecialFlags & 128) != 0) specular *= t1.a;
      }
      // sMaterial11 combines Color[0], the selected light color and its
      // amplification on the CPU, then uploads that product to PS1.1 c0.
      // PS1.x constant registers are restricted to [-1,+1].  Keeping the
      // factors separate in GLSL would let amplified values above one survive
      // until the framebuffer blend and substantially over-warm bright lights.
      diffuseLighting = (uMaterial11 != 0
        ? uM11LightConstant.rgb : uLightColor[0].rgb) * attenuation * diffuse;
      specularLighting = uLightColor[0].rgb * attenuation * specular;
      material11SpecularLighting = (uMaterial11 != 0
        ? uM11LightConstant.a : uLightSpecular[0]) * attenuation * specular;
      if (uMaterial20 != 0) material20AttenuationAlpha = radiusSquared;
    } else if (!legacyVertexLighting) {
      // Retain the port's generic world-space material path, including its
      // ability to accumulate several selected lights in one draw.
      vec3 viewDirection = normalize(uCameraPosition - vWorld);
      for (int index = 0; index < ${MAX_LIGHTS}; ++index) {
        if (index >= uLightCount) break;
        vec3 vectorToLight = uLightPosition[index].xyz -
          vWorld * uLightPosition[index].w;
        float distanceToLight = length(vectorToLight);
        vec3 lightDirection = distanceToLight > 1e-5
          ? vectorToLight / distanceToLight : vec3(0.0, 1.0, 0.0);
        float radius = distanceToLight * uLightAttenuation[index].w;
        float attenuation = uLightPosition[index].w == 0.0 ? 1.0 :
          max(0.0, 1.0 - radius * radius);
        float diffuse = max(dot(normal, lightDirection), 0.0);
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(normal, halfwayDirection), 0.0),
          max(1.0, uSpecularPower));
        diffuseLighting += uLightColor[index].rgb * attenuation * diffuse;
        specularLighting += uLightColor[index].rgb * attenuation * specular;
      }
    }
    vec4 surface = albedo;
    if (uUsePrelight != 0) {
      vec2 screenUV = vClipPosition.xy / max(1e-6, vClipPosition.w) * 0.5 + 0.5;
      surface = texture(uPrelightTexture, screenUV) * uBaseColor;
    }
    if (uM11MultipassLight != 0) {
      // Material11's light phase stores diffuse in RGB and specular in alpha.
      // Material11Insert preserves alpha through POSTLIGHT, then adds it back
      // to RGB. Keeping the channels separate is essential: otherwise the
      // multiplicative texture phase incorrectly textures the highlight.
      // The material diffuse and alpha are already part of the native c0
      // constant above; multiplying by surface here would apply them twice.
      result = vec4(diffuseLighting,
        material11SpecularLighting * uSpecularStrength);
    } else {
      vec3 litColor = (uMaterial11 != 0 ? diffuseLighting
        : surface.rgb * diffuseLighting) +
        uSpecularColor * specularLighting * uSpecularStrength;
      // material20_light.psh's attenuation MAD is saturated before the
      // optional final COLOR0 multiplication.
      if (uMaterial20 != 0) litColor = clamp(litColor, 0.0, 1.0);
      result = vec4(litColor,
        uMaterial20 != 0 ? material20AttenuationAlpha : albedo.a);
      // material20_light.psh applies COLOR0 after material, specular and
      // attenuation, so it modulates highlights and radial alpha as well as
      // the diffuse surface. No other pass shares this ordering.
      if (uMaterial20 != 0 && (uM20Flags & 8) != 0) result *= vColor;
    }
  } else if (uMode == 3) {
    result = vec4(albedo.rgb * t2.rgb * 2.0, albedo.a * t2.a);
  } else if (uMode == 4) {
    if (uMaterial11 != 0) {
      // material11.vsh has already produced sphere/reflection/projection UVs
      // for logical Tex2, including its per-slot texture transform.
      vec3 environment = t2.rgb;
      if ((uTextureMask & 1) != 0) environment *= t0.a;
      result = vec4(environment, 1.0);
    } else {
    vec4 environmentMap = (uTextureMask & 2) != 0
      ? texture(uTexture1, vUV) : vec4(1.0);
    vec3 environmentNormal = legacyEnvironmentBasis ? vLegacyLight : normal;
    if (uMaterial20 != 0 && (uM20EnvironmentFlags & 131072) != 0 &&
        (uTextureMask & 2) != 0) {
      vec3 mapped = bumpVector(environmentMap);
      if (legacyEnvironmentBasis) {
        // material20_envi.psh combines the three raw, independently
        // interpolated basis vectors and normalizes only the result.
        environmentNormal = vLegacyLight * mapped.x +
          vLegacyHalfway * mapped.y + vLegacyAttenuation * mapped.z;
      } else {
        vec3 tangent = normalize(vTangent - normal * dot(normal, vTangent));
        vec3 bitangent = normalize(cross(normal, tangent));
        environmentNormal = normalize(normal * mapped.x +
          bitangent * mapped.y + tangent * mapped.z);
      }
    }
    if (legacyEnvironmentBasis && (uM20RuntimeEnvironmentFlags & 65536) == 0) {
      // The non-reflection environment variant uploads ModelSpace*Camera, so
      // its basis is in eye space before the sphere-map projection.
      environmentNormal = mat3(uM11View) * environmentNormal;
    }
    if (legacyEnvironmentBasis) environmentNormal = normalize(environmentNormal);
    vec3 environmentDirection = environmentNormal;
    if (uMaterial20 == 0 || (uM20EnvironmentFlags & 65536) != 0) {
      vec3 eyeVector = legacyEnvironmentBasis
        ? mat3(uM11ModelToWorld) * (uMaterialCameraPosition -
          (uM11WorldToModel * vec4(vWorld, 1.0)).xyz)
        : uCameraPosition - vWorld;
      vec3 eyeDirection = normalize(eyeVector);
      environmentDirection = eyeDirection - 2.0 *
        dot(eyeDirection, environmentNormal) * environmentNormal;
    }
    vec3 normalizedDirection = normalize(environmentDirection);
    vec2 environmentUV = normalizedDirection.xy * vec2(0.5, -0.5) + 0.5;
    vec4 environmentColor = (uTextureMask & 4) != 0
      ? texture(uTexture2, environmentUV) : t2;
    vec3 environment = environmentColor.rgb * uBaseColor.rgb;
    if ((uTextureMask & 1) != 0) environment *= t0.rgb;
    if (uMaterial20 != 0 && (uM20Flags & 128) != 0) environment *= environmentMap.a;
    result = vec4(environment, environmentColor.a * uBaseColor.a);
    }
  } else if (uMode == 5) {
    // Material20's ambient shader writes RGB only; Engine strips ambient
    // alpha before uploading the constant.
    result = vec4(albedo.rgb * uAmbient, 0.0);
  } else if (uMode == 6) {
    // material20_tex.psh's no-main-texture constant is c8=(1,1,1,0), not
    // the ordinary white fallback texture (whose alpha is one).
    result = (uTextureMask & 1) != 0 ? t0 : vec4(1.0, 1.0, 1.0, 0.0);
    if ((uTextureMask & 2) != 0) result.rgb = detailCombine(result.rgb, t1.rgb, uDetailOps.x);
    if ((uTextureMask & 4) != 0) result.rgb = detailCombine(result.rgb, t2.rgb, uDetailOps.y);
  } else if (uMode == 7) {
    result = material11Combine(t0, t1, t2, t3);
  } else {
    result.rgb *= max(uAmbient, vec3(1.0));
    if ((uTextureMask & 4) != 0) result.rgb += t2.rgb * uBaseColor.rgb;
    if ((uTextureMask & 8) != 0) result *= t3 * 2.0;
  }
  if (result.a <= uAlphaCutoff) discard;
  if (uFogEnabled != 0) {
    // material11.vsh emits dot(modelPos,(Model*View).k): camera-space Z.
    // Radial distance produces visibly different fog off the optical axis.
    float eyeZ = (uM11View * vec4(vWorld, 1.0)).z;
    float fog = clamp((eyeZ - uFogRange.x) /
      max(1e-5, uFogRange.y - uFogRange.x), 0.0, 1.0);
    result.rgb = mix(result.rgb, uFogColor, fog);
  }
  outColor = result;
}`;

// material11.vsh represents every shadow vertex twice. w=1 is the source
// vertex and w=0 becomes (vertex-light), a homogeneous point at infinity.
// c4 is the object-space light produced by the engine's deliberately
// transpose-only sMatrix::TransR; apply the subtraction before Model exactly
// as the released shader does. This differs from worldPosition-worldLight for
// authored scaled/non-orthonormal model matrices.
const SHADOW_VERTEX_SOURCE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in float aExtrude;
uniform mat4 uViewProjection;
uniform mat4 uModel;
uniform vec3 uLightPosition;
void main() {
  vec4 modelPosition = vec4(aPosition, 1.0) -
    vec4(uLightPosition, 1.0) * aExtrude;
  // q=1 projects a homogeneous direction (w=0) exactly onto z=w. Nudging it
  // inward would turn the infinite z-fail volume into an arbitrary finite one.
  gl_Position = uViewProjection * (uModel * modelPosition);
}`;

const SHADOW_FRAGMENT_SOURCE = `#version 300 es
precision highp float;
out vec4 outColor;
void main() { outColor = vec4(0.0); }`;

const FULLSCREEN_VERTEX_SOURCE = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUV = position;
  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}`;

const FULLSCREEN_FRAGMENT_SOURCE = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uImage;
uniform sampler2D uMaterialTexture;
uniform int uMode;
uniform vec4 uColor0;
uniform vec4 uColor1;
uniform vec4 uParameters;
uniform vec4 uColorCorrect[7];
uniform vec2 uTexel;
uniform vec4 uUVRect;
out vec4 outColor;

void main() {
  vec2 uv = mix(uUVRect.xy, uUVRect.zw, vUV);
  vec4 source = texture(uImage, uv);
  if (uMode == 1) {
    // effect_glare.psh's bilinear 3x3 approximation of the native
    // (1,4,6,4,1)/16 downsample kernel.
    const float weights[3] = float[3](5.0 / 16.0, 6.0 / 16.0, 5.0 / 16.0);
    source = vec4(0.0);
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 3; x++) {
        vec2 offset = vec2(float(x - 1), float(y - 1)) * 1.2 * uTexel;
        source += texture(uImage, uv + offset) * weights[x] * weights[y];
      }
    }
    // effect_glare.psh seeds accu with c0 as part of its first MAD. c0.rgb
    // is zero, while c0.a is the packed first-tap weight.
    source.a += weights[0] * weights[0];
    source = clamp(source, 0.0, 1.0);
  } else if (uMode == 2) {
    float gray = dot(source.rgb, vec3(1.0 / 3.0));
    vec3 dark = gray * uColorCorrect[5].rgb + source.rgb * uColorCorrect[0].rgb;
    float range = clamp(gray * uColorCorrect[4].y + uColorCorrect[4].x, 0.0, 1.0);
    vec3 light = gray * uColorCorrect[6].rgb + source.rgb * uColorCorrect[1].rgb;
    source.rgb = dark * range + light * (1.0 - range) + uColorCorrect[4].z * range;
    source.rgb = clamp(source.rgb - uColorCorrect[3].rgb, 0.0, 1.0);
    source.rgb *= uColorCorrect[2].rgb;
  } else if (uMode == 3) {
    vec4 material = texture(uMaterialTexture, vUV);
    source *= material * uColor0;
  } else if (uMode == 4) {
    // effect_glare2.psh: subtract the authored threshold, derive its custom
    // gray scalar, then mix the chromatic and grayscale results.
    vec3 color = source.rgb - uColor0.rgb;
    float gray = dot(color, vec3(uParameters.x));
    source.rgb = color * uParameters.y + vec3(gray) * (1.0 - uParameters.y);
    source.a = source.a * uParameters.y + gray * (1.0 - uParameters.y);
  } else if (uMode == 5) {
    // effect_glare.psh reused as one half of each separable 9-tap pass.
    const float weights[9] = float[9](
      1.0 / 25.0, 2.0 / 25.0, 3.0 / 25.0, 4.0 / 25.0, 5.0 / 25.0,
      4.0 / 25.0, 3.0 / 25.0, 2.0 / 25.0, 1.0 / 25.0);
    source = vec4(0.0);
    for (int tap = 0; tap < 9; tap++) {
      source += texture(uImage, uv + float(tap - 4) * uParameters.xy) * weights[tap];
    }
    source.a += weights[0];
    source = clamp(source * uParameters.z, 0.0, 1.0);
  } else if (uMode == 6) {
    // effect_glare3.psh: independently tint glare and the untouched frame,
    // then select additive/soft-add behavior with the authored scalar.
    vec4 glare = texture(uImage, vUV) * uColor0;
    vec4 original = texture(uMaterialTexture, uv) * uColor1;
    source = glare + original - glare * original * uParameters.x;
  } else if (uMode == 7) {
    // Material11Insert's two framebuffer-alpha quads use a constant white
    // source with zero alpha. Their blend factors do the actual clear/add.
    source = vec4(1.0, 1.0, 1.0, 0.0);
  }
  outColor = source;
}`;

function asFloat32(values, expectedLength = 0, fallback = 0) {
  if (values instanceof Float32Array && (!expectedLength || values.length === expectedLength)) return values;
  const result = new Float32Array(expectedLength || values?.length || 0);
  if (values) result.set(Array.from(values).slice(0, result.length));
  if (!values && fallback !== 0) result.fill(fallback);
  return result;
}

function normalizeColors(colors, vertexCount) {
  if (colors instanceof Uint8Array && colors.length === vertexCount * 4) return colors;
  if (colors instanceof Float32Array && colors.length === vertexCount * 4) return colors;
  const result = new Uint8Array(vertexCount * 4);
  if (colors && colors.length === vertexCount) {
    for (let index = 0; index < vertexCount; index++) {
      const color = colors[index] >>> 0, offset = index * 4;
      result[offset] = (color >>> 16) & 255;
      result[offset + 1] = (color >>> 8) & 255;
      result[offset + 2] = color & 255;
      result[offset + 3] = color >>> 24;
    }
  } else {
    result.fill(255);
  }
  return result;
}

function normalizeGeometryBounds(bounds, positions) {
  const sourceMinimum = bounds?.minimum || bounds?.min;
  const sourceMaximum = bounds?.maximum || bounds?.max;
  const validSource = sourceMinimum?.length >= 3 && sourceMaximum?.length >= 3 &&
    [sourceMinimum[0], sourceMinimum[1], sourceMinimum[2],
      sourceMaximum[0], sourceMaximum[1], sourceMaximum[2]].every(Number.isFinite) &&
    sourceMinimum[0] <= sourceMaximum[0] && sourceMinimum[1] <= sourceMaximum[1] &&
    sourceMinimum[2] <= sourceMaximum[2];
  if (validSource) return {
    minimum: new Float32Array([sourceMinimum[0], sourceMinimum[1], sourceMinimum[2]]),
    maximum: new Float32Array([sourceMaximum[0], sourceMaximum[1], sourceMaximum[2]]),
  };

  const minimum = new Float32Array([Infinity, Infinity, Infinity]);
  const maximum = new Float32Array([-Infinity, -Infinity, -Infinity]);
  for (let offset = 0; offset + 2 < positions.length; offset += 3) {
    const x = positions[offset], y = positions[offset + 1], z = positions[offset + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < minimum[0]) minimum[0] = x; if (x > maximum[0]) maximum[0] = x;
    if (y < minimum[1]) minimum[1] = y; if (y > maximum[1]) maximum[1] = y;
    if (z < minimum[2]) minimum[2] = z; if (z > maximum[2]) maximum[2] = z;
  }
  if (!Number.isFinite(minimum[0])) minimum.fill(0), maximum.fill(0);
  return { minimum, maximum };
}

function indexedGeometryBounds(positions, indices, start, count) {
  const minimum = new Float32Array([Infinity, Infinity, Infinity]);
  const maximum = new Float32Array([-Infinity, -Infinity, -Infinity]);
  const end = Math.min(indices.length, Math.max(0, start + count));
  for (let offset = Math.max(0, start); offset < end; offset++) {
    const position = Number(indices[offset]) * 3;
    if (!Number.isSafeInteger(position) || position < 0 || position + 2 >= positions.length) continue;
    const x = positions[position], y = positions[position + 1], z = positions[position + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < minimum[0]) minimum[0] = x; if (x > maximum[0]) maximum[0] = x;
    if (y < minimum[1]) minimum[1] = y; if (y > maximum[1]) maximum[1] = y;
    if (z < minimum[2]) minimum[2] = z; if (z > maximum[2]) maximum[2] = z;
  }
  return Number.isFinite(minimum[0]) ? { minimum, maximum } : null;
}

function normalizePreparedGeometry(mesh, options) {
  const prepared = typeof mesh?.prepare === 'function' ? mesh.prepare(options) : mesh;
  if (!prepared?.positions) throw new TypeError('renderable mesh has no prepared positions');
  const positions = asFloat32(prepared.positions);
  if (positions.length % 3) throw new RangeError('mesh position count is not divisible by three');
  const vertexCount = positions.length / 3;
  const normals = prepared.normals?.length === vertexCount * 3
    ? asFloat32(prepared.normals) : new Float32Array(vertexCount * 3);
  if (!prepared.normals) for (let index = 0; index < vertexCount; index++) normals[index * 3 + 1] = 1;
  const uv0Source = prepared.uv0 || prepared.uvs;
  const uvs = uv0Source?.length === vertexCount * 2
    ? asFloat32(uv0Source) : new Float32Array(vertexCount * 2);
  // MinMesh keeps two authored UV channels. Old Mesh only has the first, for
  // which WZ3's vertex declarations effectively feed the same coordinates to
  // materials that request UV1.
  const uv1 = prepared.uv1?.length === vertexCount * 2
    ? asFloat32(prepared.uv1) : uvs;
  const tangents = prepared.tangents?.length === vertexCount * 4
    ? asFloat32(prepared.tangents) : new Float32Array(vertexCount * 4);
  if (!prepared.tangents) for (let index = 0; index < vertexCount; index++) tangents[index * 4] = 1;
  const colors = normalizeColors(prepared.colors, vertexCount);
  let indices = prepared.indices;
  if (!indices) indices = Uint32Array.from({ length: vertexCount }, (_, index) => index);
  else if (!(indices instanceof Uint16Array) && !(indices instanceof Uint32Array)) indices = Uint32Array.from(indices);
  const groups = prepared.groups?.length ? prepared.groups.map(group => ({
    material: group.material ?? 0,
    materialIndex: group.materialIndex,
    cluster: group.cluster,
    pass: group.pass,
    renderPass: group.renderPass,
    start: Math.max(0, group.start | 0),
    count: Math.max(0, group.count | 0),
  })) : [{ material: 0, start: 0, count: indices.length }];
  let shadowVertexMap = prepared.shadowVertexMap;
  if (!shadowVertexMap || shadowVertexMap.length !== vertexCount) {
    shadowVertexMap = new Uint32Array(vertexCount);
    const oldVertices = mesh?.kind === 'mesh' ? mesh.vertices : null;
    for (let index = 0; index < vertexCount; index++) {
      const first = oldVertices?.[index]?.first;
      shadowVertexMap[index] = first >= 0 && first < vertexCount ? first : index;
    }
  }
  let shadowTriangleMask = prepared.shadowTriangleMask || null;
  if (!shadowTriangleMask && mesh?.kind === 'mesh' && typeof mesh.faceVertices === 'function') {
    const buckets = new Map();
    for (const face of mesh.faces) {
      if (!face.material || face.edge < 0) continue;
      const count = Math.max(0, mesh.faceVertices(face).length - 2);
      let bucket = buckets.get(face.material);
      if (!bucket) buckets.set(face.material, bucket = []);
      for (let index = 0; index < count; index++) bucket.push(face.used ? 1 : 0);
    }
    const values = Array.from(buckets.values()).flat();
    if (values.length === indices.length / 3) shadowTriangleMask = new Uint8Array(values);
  } else if (!shadowTriangleMask && mesh?.kind === 'minmesh') {
    const buckets = Array.from({ length: mesh.clusters?.length || 0 }, () => []);
    for (const face of mesh.faces || []) {
      if (face.cluster <= 0 || face.cluster >= buckets.length || face.count < 3) continue;
      for (let index = 2; index < face.count; index++) buckets[face.cluster].push(face.flags & 1 ? 0 : 1);
    }
    const values = buckets.slice(1).flat();
    if (values.length === indices.length / 3) shadowTriangleMask = new Uint8Array(values);
  }
  // Animated MinMesh positions have already been skinned by prepare(time),
  // while its authored bounds remain in bind-pose space. Recompute those
  // bounds from the prepared positions; static geometry may reuse its native
  // face-derived box (and old Mesh buffers get a box here when none exists).
  const bounds = normalizeGeometryBounds(options?.time === undefined ? prepared.bounds : null, positions);
  // EngMesh prepares a distinct bbox for every material/cluster job (and one
  // complete-mesh bbox for the shared shadow job). Retain the compact group
  // boxes outside enumerable authoring data so planning can reproduce that
  // culling without changing summaries or serialized fixtures. One-group
  // geometry reuses its complete box; animated jobs are deliberately
  // unbounded and avoid an otherwise wasted index traversal every frame.
  for (const group of groups) {
    Object.defineProperty(group, 'bounds', {
      value: groups.length === 1 ? bounds : options?.time === undefined
        ? indexedGeometryBounds(positions, indices, group.start, group.count) : null,
      enumerable: false,
    });
  }
  return {
    source: prepared, positions, normals, uvs, uv1, tangents, colors, indices, groups,
    materials: prepared.materials || mesh?.materials || mesh?.Mtrl || [], vertexCount,
    shadowVertexMap, shadowTriangleMask, bounds,
    // Procedural producers can identify the buffers they changed without
    // forcing every static attribute and the index buffer back across the
    // WebGL boundary. Unknown producers omit this and retain the conservative
    // full-refresh path.
    dynamicAttributes: Array.isArray(prepared.dynamicAttributes)
      ? prepared.dynamicAttributes : null,
  };
}

function geometryTopologyStats(geometry, options = {}) {
  const positions = geometry?.positions || [];
  const normals = geometry?.normals || [];
  const indices = geometry?.indices || [];
  const groups = geometry?.groups || [];
  const triangleCount = Math.floor(indices.length / 3);
  const auditedTriangles = Math.min(triangleCount,
    Math.max(0, options.maxTriangles ?? 20000));
  const quantization = Number(options.quantization) || 1e-5;
  const triangleKeys = new Map();
  let degenerateTriangles = 0, duplicateTriangles = 0;
  let oppositeDuplicateTriangles = 0, sameOrientationDuplicateTriangles = 0;
  let exactDuplicateTriangles = 0, exactOppositeDuplicateTriangles = 0;
  let exactSameOrientationDuplicateTriangles = 0;
  let exactSameGroupIdenticalAttributeTriangles = 0;
  let exactSameGroupAttributeVariantTriangles = 0;
  let exactCrossGroupSameMaterialTriangles = 0, exactCrossMaterialTriangles = 0;
  let exactDegenerateSameOrientationTriangles = 0;
  let normalAlignedWinding = 0, normalOpposedWinding = 0;
  let indeterminateWinding = 0;
  const vertexKey = index => {
    const offset = index * 3;
    return `${Math.round(positions[offset] / quantization)},` +
      `${Math.round(positions[offset + 1] / quantization)},` +
      `${Math.round(positions[offset + 2] / quantization)}`;
  };
  const exactVertexKey = index => {
    const offset = index * 3;
    const clean = value => Object.is(value, -0) ? 0 : value;
    return `${clean(positions[offset])},${clean(positions[offset + 1])},` +
      `${clean(positions[offset + 2])}`;
  };
  const triangleIdentity = (ia, ib, ic, keyForVertex) => {
    const authored = [keyForVertex(ia), keyForVertex(ib), keyForVertex(ic)];
    const canonical = authored.slice().sort();
    const order = authored.map(value => canonical.indexOf(value));
    return {
      key: canonical.join('|'),
      orientation: ((order[0] > order[1]) + (order[0] > order[2]) +
        (order[1] > order[2])) & 1,
    };
  };
  const groupIndexForTriangle = triangle => {
    const cursor = triangle * 3;
    for (let index = 0; index < groups.length; index++) {
      const group = groups[index], start = Math.max(0, group.start | 0);
      if (cursor >= start && cursor < start + Math.max(0, group.count | 0)) return index;
    }
    return -1;
  };
  const groupMaterialIdentity = groupIndex => {
    if (groupIndex < 0) return null;
    const group = groups[groupIndex];
    return materialFromGroup(geometry, group) || materialSlotFromGroup(geometry, group) ||
      group.material || null;
  };
  const valuesEqual = (left, right) => left === right ||
    (Number.isNaN(left) && Number.isNaN(right));
  const cornerAttributesEqual = (leftTriangle, rightTriangle) => {
    const leftIndices = [indices[leftTriangle * 3] | 0, indices[leftTriangle * 3 + 1] | 0,
      indices[leftTriangle * 3 + 2] | 0];
    const rightIndices = [indices[rightTriangle * 3] | 0, indices[rightTriangle * 3 + 1] | 0,
      indices[rightTriangle * 3 + 2] | 0];
    const used = [false, false, false];
    const attributes = [
      [normals, 3], [geometry?.uvs || [], 2], [geometry?.uv1 || [], 2],
      [geometry?.tangents || [], 4], [geometry?.colors || [], 4],
    ];
    for (const leftIndex of leftIndices) {
      const leftPosition = leftIndex * 3;
      let match = -1;
      for (let corner = 0; corner < 3; corner++) {
        if (used[corner]) continue;
        const rightPosition = rightIndices[corner] * 3;
        if (valuesEqual(positions[leftPosition], positions[rightPosition]) &&
            valuesEqual(positions[leftPosition + 1], positions[rightPosition + 1]) &&
            valuesEqual(positions[leftPosition + 2], positions[rightPosition + 2])) {
          match = corner; break;
        }
      }
      if (match < 0) return false;
      used[match] = true;
      const rightIndex = rightIndices[match];
      for (const [values, width] of attributes) {
        if (!values?.length) continue;
        const leftOffset = leftIndex * width, rightOffset = rightIndex * width;
        for (let component = 0; component < width; component++) {
          if (!valuesEqual(values[leftOffset + component], values[rightOffset + component])) return false;
        }
      }
    }
    return true;
  };
  const classifyExactSameOrientation = (triangle, previousTriangles, degenerate) => {
    if (degenerate) { exactDegenerateSameOrientationTriangles++; return; }
    const currentGroup = groupIndexForTriangle(triangle);
    const currentMaterial = groupMaterialIdentity(currentGroup);
    let classification = 3;
    for (const previous of previousTriangles) {
      const previousGroup = groupIndexForTriangle(previous);
      if (previousGroup === currentGroup) {
        if (cornerAttributesEqual(previous, triangle)) { classification = 0; break; }
        classification = Math.min(classification, 1);
      } else if (Object.is(groupMaterialIdentity(previousGroup), currentMaterial)) {
        classification = Math.min(classification, 2);
      }
    }
    if (classification === 0) exactSameGroupIdenticalAttributeTriangles++;
    else if (classification === 1) exactSameGroupAttributeVariantTriangles++;
    else if (classification === 2) exactCrossGroupSameMaterialTriangles++;
    else exactCrossMaterialTriangles++;
  };
  const packTriangle = (triangle, orientationBit) => ((triangle + 1) << 2) | orientationBit;
  const unpackTriangle = packed => (packed >>> 2) - 1;
  for (let triangle = 0; triangle < auditedTriangles; triangle++) {
    const cursor = triangle * 3;
    const ia = indices[cursor] | 0, ib = indices[cursor + 1] | 0, ic = indices[cursor + 2] | 0;
    const a = ia * 3, b = ib * 3, c = ic * 3;
    const abx = positions[b] - positions[a], aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a], acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const areaSquared = nx * nx + ny * ny + nz * nz;
    if (!(areaSquared > 1e-20)) degenerateTriangles++;
    const normalX = (normals[a] || 0) + (normals[b] || 0) + (normals[c] || 0);
    const normalY = (normals[a + 1] || 0) + (normals[b + 1] || 0) + (normals[c + 1] || 0);
    const normalZ = (normals[a + 2] || 0) + (normals[b + 2] || 0) + (normals[c + 2] || 0);
    const alignment = nx * normalX + ny * normalY + nz * normalZ;
    // This is an object-space face/normal consistency check. It deliberately
    // does not call either sign screen-CW/CCW: projection and model transforms
    // determine the screen-space orientation used by WebGL culling.
    if (alignment > 1e-10) normalAlignedWinding++;
    else if (alignment < -1e-10) normalOpposedWinding++;
    else indeterminateWinding++;

    const identity = triangleIdentity(ia, ib, ic, vertexKey);
    const key = identity.key, orientation = identity.orientation;
    const orientationBit = 1 << orientation;
    let triangleState = triangleKeys.get(key);
    if (triangleState !== undefined) {
      if (typeof triangleState === 'number') {
        const firstTriangle = unpackTriangle(triangleState);
        const firstOrientationBit = triangleState & 3;
        const firstCursor = firstTriangle * 3;
        const firstExactIdentity = triangleIdentity(indices[firstCursor] | 0,
          indices[firstCursor + 1] | 0, indices[firstCursor + 2] | 0, exactVertexKey);
        triangleState = { orientations: firstOrientationBit, exact: new Map() };
        triangleState.exact.set(firstExactIdentity.key,
          packTriangle(firstTriangle, 1 << firstExactIdentity.orientation));
        triangleKeys.set(key, triangleState);
      }
      const previousOrientations = triangleState.orientations;
      duplicateTriangles++;
      // Keep both orientations seen for this physical triangle. Comparing
      // only against the first copy misses a real same-winding overlap in a
      // sequence such as CW, CCW, CCW.
      if (previousOrientations & orientationBit) sameOrientationDuplicateTriangles++;
      else oppositeDuplicateTriangles++;
      triangleState.orientations |= orientationBit;

      const exactIdentity = triangleIdentity(ia, ib, ic, exactVertexKey);
      const exactOrientationBit = 1 << exactIdentity.orientation;
      let exactState = triangleState.exact.get(exactIdentity.key);
      if (exactState === undefined) {
        triangleState.exact.set(exactIdentity.key, packTriangle(triangle, exactOrientationBit));
      } else {
        if (typeof exactState === 'number') {
          const firstTriangle = unpackTriangle(exactState), firstOrientationBit = exactState & 3;
          exactState = { orientations: firstOrientationBit, triangles: [[], []] };
          exactState.triangles[firstOrientationBit === 1 ? 0 : 1].push(firstTriangle);
          triangleState.exact.set(exactIdentity.key, exactState);
        }
        exactDuplicateTriangles++;
        if (exactState.orientations & exactOrientationBit) {
          exactSameOrientationDuplicateTriangles++;
          classifyExactSameOrientation(triangle,
            exactState.triangles[exactIdentity.orientation], !(areaSquared > 1e-20));
        } else exactOppositeDuplicateTriangles++;
        exactState.orientations |= exactOrientationBit;
        exactState.triangles[exactIdentity.orientation].push(triangle);
      }
    } else triangleKeys.set(key, packTriangle(triangle, orientationBit));
  }
  return {
    triangleCount, auditedTriangles,
    truncated: auditedTriangles < triangleCount,
    degenerateTriangles, duplicateTriangles, oppositeDuplicateTriangles,
    sameOrientationDuplicateTriangles,
    exactDuplicateTriangles, exactOppositeDuplicateTriangles,
    exactSameOrientationDuplicateTriangles,
    nearOnlyDuplicateTriangles: Math.max(0, duplicateTriangles - exactDuplicateTriangles),
    nearOnlySameOrientationDuplicateTriangles: Math.max(0,
      sameOrientationDuplicateTriangles - exactSameOrientationDuplicateTriangles),
    exactSameGroupIdenticalAttributeTriangles,
    exactSameGroupAttributeVariantTriangles,
    exactCrossGroupSameMaterialTriangles,
    exactCrossMaterialTriangles,
    exactDegenerateSameOrientationTriangles,
    normalAlignedWinding, normalOpposedWinding,
    unexpectedWinding: normalOpposedWinding,
    // Compatibility aliases for callers that used the original ambiguous
    // names. New diagnostics should use the explicit orientation fields.
    alignedWinding: normalAlignedWinding,
    reversedWinding: normalOpposedWinding,
    indeterminateWinding,
  };
}

function meshHasAnimation(mesh) {
  if (!mesh || typeof mesh !== 'object') return false;
  if (typeof mesh.hasAnimation === 'function') return Boolean(mesh.hasAnimation());
  if (typeof mesh.storageSummary === 'function') return (mesh.storageSummary()?.bones || 0) > 0;
  return Boolean(mesh.animation?.matrices?.length);
}

function materialSlotFromGroup(geometry, group) {
  if (group.materialIndex !== undefined) return geometry.materials[group.materialIndex];
  if (group.cluster !== undefined && geometry.materials[group.cluster] !== undefined) {
    return geometry.materials[group.cluster];
  }
  return typeof group.material === 'number' ? geometry.materials[group.material] : group.material;
}

function materialFromGroup(geometry, group) {
  let slot = materialSlotFromGroup(geometry, group);
  if (slot?.material !== undefined) slot = slot.material;
  if (slot?.Material !== undefined) slot = slot.Material;
  return slot?.kind === 'material' ? slot : null;
}

function shadowGroupsForGeometry(geometry) {
  const groups = [];
  for (const group of geometry?.groups || []) {
    const material = materialFromGroup(geometry, group);
    for (const pass of material?.passes || []) {
      if (pass?.usage === 'shadow') groups.push(group);
    }
  }
  return groups;
}

function shadowTopologyKey(groups) {
  return (groups || []).map(group => `${group.start}:${group.count}`).join('|');
}

function preparedGeometryUploadBytes(geometry) {
  if (!geometry) return 0;
  return ['positions', 'normals', 'uvs', 'colors', 'tangents', 'uv1', 'indices']
    .reduce((sum, key) => sum + (geometry[key]?.byteLength || 0), 0);
}

function geometryWarmupEstimateBytes(mesh) {
  if (mesh?._prepared) return preparedGeometryUploadBytes(mesh._prepared);
  const summary = mesh?.storageSummary?.() || {};
  const vertices = Math.max(0, Number(summary.vertices) || 0);
  const faces = Math.max(0, Number(summary.faces) || 0);
  // Prepared legacy geometry carries six vertex streams and a triangle index
  // stream. This deliberately rounds upward for ordinary triangles/quads so
  // the warm-up admission check stays conservative before prepare() exists.
  return Math.ceil(vertices * 64 + faces * 24);
}

function shadowTopologyBytes(topology) {
  return (topology?.positions?.byteLength || 0) + (topology?.faces?.byteLength || 0) +
    (topology?.sourceIndices?.byteLength || 0) +
    (topology?.volumePositions?.byteLength || 0) +
    (topology?.extrusions?.byteLength || 0);
}

function groupRenderPass(geometry, group) {
  const slot = materialSlotFromGroup(geometry, group);
  return (group.pass ?? group.renderPass ?? slot?.pass ?? slot?.Pass ?? 0) | 0;
}

function material20UVTransforms(parameters) {
  const srt1 = new Float32Array(9);
  for (let index = 0; index < 9; index++) {
    const fallback = index < 3 ? 1 : 0;
    srt1[index] = Number.isFinite(parameters?.[24 + index])
      ? parameters[24 + index] : fallback;
  }
  const matrix = mat4SRT(srt1);
  const transform1 = new Float32Array([
    matrix[0], matrix[4], matrix[8], matrix[12],
    matrix[1], matrix[5], matrix[9], matrix[13],
  ]);
  const sx = Number.isFinite(parameters?.[33]) ? parameters[33] : 1;
  const sy = Number.isFinite(parameters?.[34]) ? parameters[34] : 1;
  const angle = Number(parameters?.[35]) || 0;
  const tx = Number(parameters?.[36]) || 0;
  const ty = Number(parameters?.[37]) || 0;
  const sine = Math.sin(angle), cosine = Math.cos(angle);
  const transform2 = new Float32Array([
    sx * cosine, sy * sine, 0, tx,
    -sx * sine, sy * cosine, 0, ty,
  ]);
  return { transform1, transform2 };
}

function material20UV(parameters, sourceIndex, uv) {
  sourceIndex |= 0;
  const lighting = sourceIndex >= 4;
  const slot = lighting ? sourceIndex - 4 : sourceIndex;
  const flags = parameters?.[(lighting ? 16 : 8) + slot] >>> 0;
  const scale = Number(parameters?.[(lighting ? 20 : 12) + slot]);
  const mode = (flags >>> 12) & 15;
  const x = Number(uv?.[0]) || 0, y = Number(uv?.[1]) || 0;
  if (mode === 1) {
    const value = Number.isFinite(scale) ? scale : 1;
    return new Float32Array([x * value, y * value]);
  }
  if (mode === 2 || mode === 3) {
    const rows = mode === 2
      ? material20UVTransforms(parameters).transform1
      : material20UVTransforms(parameters).transform2;
    return new Float32Array([
      x * rows[0] + y * rows[1] + rows[3],
      x * rows[4] + y * rows[5] + rows[7],
    ]);
  }
  return new Float32Array([x, y]);
}

function transformDirection3(matrix, vector, output = new Float32Array(3)) {
  const x = Number(vector?.[0]) || 0, y = Number(vector?.[1]) || 0;
  const z = Number(vector?.[2]) || 0;
  if (!matrix?.length || matrix.length < 16) {
    output[0] = x; output[1] = y; output[2] = z;
  } else {
    output[0] = matrix[0] * x + matrix[4] * y + matrix[8] * z;
    output[1] = matrix[1] * x + matrix[5] * y + matrix[9] * z;
    output[2] = matrix[2] * x + matrix[6] * y + matrix[10] * z;
  }
  return output;
}

// Numerical oracle for material20_envi.vsh:45-68. B is deliberately formed
// from raw input-space N/T before all three basis vectors are transformed.
function material20EnvironmentBumpDirection(normal, tangent, sample,
    modelMatrix = null, viewMatrix = null, reflection = true) {
  const n = [Number(normal?.[0]) || 0, Number(normal?.[1]) || 0,
    Number(normal?.[2]) || 0];
  const t = [Number(tangent?.[0]) || 0, Number(tangent?.[1]) || 0,
    Number(tangent?.[2]) || 0];
  const b = [n[1] * t[2] - n[2] * t[1],
    n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
  const transformedN = transformDirection3(modelMatrix, n);
  const transformedB = transformDirection3(modelMatrix, b);
  const transformedT = transformDirection3(modelMatrix, t);
  const x = Number(sample?.[0]) || 0, y = Number(sample?.[1]) || 0;
  const z = Number(sample?.[2]) || 0;
  const result = new Float32Array([
    transformedN[0] * x + transformedB[0] * y + transformedT[0] * z,
    transformedN[1] * x + transformedB[1] * y + transformedT[1] * z,
    transformedN[2] * x + transformedB[2] * y + transformedT[2] * z,
  ]);
  return !reflection && viewMatrix?.length
    ? transformDirection3(viewMatrix, result) : result;
}

// sMaterial20Envi::Set uploads a TransR camera constant, then the vertex
// shader subtracts model position and applies the model 3x3. This intentionally
// differs from cameraWorld-worldPosition when the authored model has scale.
function material20EnvironmentEye(modelMatrix, modelPosition, cameraWorld) {
  const cameraModel = legacyTransRVector(modelMatrix, cameraWorld, true);
  return transformDirection3(modelMatrix, [
    cameraModel[0] - (Number(modelPosition?.[0]) || 0),
    cameraModel[1] - (Number(modelPosition?.[1]) || 0),
    cameraModel[2] - (Number(modelPosition?.[2]) || 0),
  ]);
}

// material11.vsh CodegenFlags variant 1 normalizes the skinned N before it is
// used either directly (Debris normal-as-UV mode) or transformed to eye space.
function material11EnvironmentNormals(normal, modelMatrix = null,
    conditionBasis = false) {
  let n = [Number(normal?.[0]) || 0, Number(normal?.[1]) || 0,
    Number(normal?.[2]) || 0];
  if (conditionBasis) n = normalize3(n[0], n[1], n[2], [0, 0, 1]);
  return {
    model: Float32Array.from(n),
    world: transformDirection3(modelMatrix, n),
  };
}

function material20EnvironmentDirection(normal, eyeDirection = null, reflection = false) {
  const normalize = value => {
    const x = Number(value?.[0]) || 0, y = Number(value?.[1]) || 0;
    const z = Number(value?.[2]) || 0, length = Math.hypot(x, y, z);
    return length > 1e-20 ? [x / length, y / length, z / length] : [0, 0, 1];
  };
  const n = normalize(normal);
  if (!reflection) return new Float32Array(n);
  const eye = normalize(eyeDirection);
  const projection = 2 * (eye[0] * n[0] + eye[1] * n[1] + eye[2] * n[2]);
  return new Float32Array([
    eye[0] - projection * n[0],
    eye[1] - projection * n[1],
    eye[2] - projection * n[2],
  ]);
}

function material20EnvironmentUV(direction) {
  const x = Number(direction?.[0]) || 0, y = Number(direction?.[1]) || 0;
  const z = Number(direction?.[2]) || 0, length = Math.hypot(x, y, z);
  if (!(length > 1e-20)) return new Float32Array([0.5, 0.5]);
  return new Float32Array([x / length * 0.5 + 0.5, y / length * -0.5 + 0.5]);
}

// material11.cpp uploads InitSRT(SRT1) as two dot-product rows and constructs
// SRT2 explicitly from a radian angle. Keep those layouts literal so every
// logical texture slot can apply its own sMTF transform in the vertex shader.
function material11UVTransforms(srt1Values = [], srt2Values = []) {
  const srt1 = new Float32Array(9);
  for (let index = 0; index < 9; index++) {
    const value = Number(srt1Values[index]);
    srt1[index] = Number.isFinite(value) ? value : 0;
  }
  const matrix = mat4SRT(srt1);
  const transform1 = new Float32Array([
    matrix[0], matrix[4], matrix[8], matrix[12],
    matrix[1], matrix[5], matrix[9], matrix[13],
  ]);
  const sx = Number(srt2Values[0]), sy = Number(srt2Values[1]);
  const angle = Number(srt2Values[2]), tx = Number(srt2Values[3]), ty = Number(srt2Values[4]);
  const sine = Math.sin(Number.isFinite(angle) ? angle : 0);
  const cosine = Math.cos(Number.isFinite(angle) ? angle : 0);
  const scaleX = Number.isFinite(sx) ? sx : 0, scaleY = Number.isFinite(sy) ? sy : 0;
  const transform2 = new Float32Array([
    scaleX * cosine, scaleY * sine, 0, Number.isFinite(tx) ? tx : 0,
    -scaleX * sine, scaleY * cosine, 0, Number.isFinite(ty) ? ty : 0,
  ]);
  return { transform1, transform2 };
}

// Numerical oracle for material11.vsh's texture-coordinate block. Inputs are
// already expressed in the named native spaces, which keeps the source's
// unusual sphere/reflection math independently testable without a GL context.
function material11UV(view, slot, inputs = {}) {
  slot = Math.max(0, Math.min(3, slot | 0));
  const flags = view.slotTextureFlags?.[slot] >>> 0;
  const vector4 = (value, fallback = [0, 0, 0, 1]) => [
    Number(value?.[0] ?? fallback[0]) || 0,
    Number(value?.[1] ?? fallback[1]) || 0,
    Number(value?.[2] ?? fallback[2]) || 0,
    Number(value?.[3] ?? fallback[3]) || 0,
  ];
  const uv = ((flags >>> 4) & 3) === 1 ? inputs.uv1 : inputs.uv0;
  let source = vector4([uv?.[0], uv?.[1], 0, 1]);

  if (slot === 2) {
    const environmentMode = (view.specialFlags >>> 4) & 3;
    const eyeNormal = vector4(inputs.eyeNormal || inputs.normal, [0, 0, 1, 0]);
    if (environmentMode === 1) {
      source[0] = eyeNormal[0] * 0.5 + 0.5;
      source[1] = eyeNormal[1] * -0.5 + 0.5;
    } else if (environmentMode === 2) {
      const eyePosition = vector4(inputs.eyePosition || inputs.position, [0, 0, 1, 1]);
      const eyeLength = Math.hypot(eyePosition[0], eyePosition[1], eyePosition[2]);
      const ex = eyeLength > 1e-20 ? eyePosition[0] / eyeLength : 0;
      const ey = eyeLength > 1e-20 ? eyePosition[1] / eyeLength : 0;
      const ez = eyeLength > 1e-20 ? eyePosition[2] / eyeLength : 0;
      const normalSquared = eyeNormal[0] * eyeNormal[0] + eyeNormal[1] * eyeNormal[1] +
        eyeNormal[2] * eyeNormal[2];
      const factor = normalSquared > 1e-20
        ? 2 * (ex * eyeNormal[0] + ey * eyeNormal[1] + ez * eyeNormal[2]) / normalSquared : 0;
      source[0] = (eyeNormal[0] * factor - ex) * 0.5 + 0.5;
      source[1] = (eyeNormal[1] * factor - ey) * -0.5 + 0.5;
      source[2] = eyeNormal[2] * factor - ez;
    } else if (environmentMode === 3) source = vector4(inputs.normal, [0, 0, 1, 0]);
  } else if (slot === 3) {
    const projectionMode = (view.specialFlags >>> 8) & 7;
    if (projectionMode === 1) source = vector4(inputs.position);
    else if (projectionMode === 2) source = vector4(inputs.worldPosition || inputs.position);
    else if (projectionMode === 3) source = vector4(inputs.eyePosition || inputs.position);
  }

  const transformMode = (flags >>> 12) & 3;
  if (transformMode === 1) {
    const scale = Number(view.slotTextureScales?.[slot]);
    const value = Number.isFinite(scale) ? scale : 0;
    return new Float32Array([source[0] * value, source[1] * value]);
  }
  if (transformMode === 2 || transformMode === 3) {
    const rows = transformMode === 2 ? view.uvTransform1 : view.uvTransform2;
    return new Float32Array([
      source[0] * rows[0] + source[1] * rows[1] + source[2] * rows[2] + source[3] * rows[3],
      source[0] * rows[4] + source[1] * rows[5] + source[2] * rows[6] + source[3] * rows[7],
    ]);
  }
  return new Float32Array([source[0], source[1]]);
}

function materialView(material, pass = null) {
  // Unlinked Mesh/MinMesh cluster 1 is not an ad-hoc neutral shader in the
  // released player: both constructors resolve it to DefaultMat. Ignore an
  // embedder's placeholder pass here and use that material's sole light pass.
  if (!material || material.system === 'default') {
    material = DEFAULT_MATERIAL11;
    pass = DEFAULT_RENDER_PASS;
  }
  const dynamicMaterial = material;
  // Material_Add copies pass records while retaining each pass' original
  // sMaterial. Static/compiled state and texture handles stay with that owner,
  // while Exec_Material_Material deliberately applies the downstream op's
  // dynamic words to every pass in the combined material.
  material = pass?.material || material;
  const parameters = material.parameters || [];
  const usage = pass?.usage || 'other';
  if (material.system === '2.0') {
    // The constructor compiles shader branches, render/sampler state and
    // texture usage from the initial sMaterial20Para. UpdatePara later
    // replaces only the values consumed by Set(). Keep that split explicit:
    // flags affecting generated code stay static, while colors, scales, SRTs
    // and the documented runtime intensity bit remain animated.
    const staticParameters = material.initialParameters || parameters;
    const dynamicParameters = dynamicMaterial?.parameters || parameters;
    const staticFlags = staticParameters[0] >>> 0;
    const dynamicFlags = dynamicParameters[0] >>> 0;
    let baseFlags = 0x0200 | MBF_ZEQUAL;
    // Material20's color prepass only binds Tex[0..2]. Tex[3] is an alpha
    // mask consumed exclusively by the depth fill; treating it as a fourth
    // color detail multiplies most alpha-tested surfaces towards black.
    let textureMap = [0, 1, 2, null];
    if (usage === 'base') {
      baseFlags = 0x0300 | MBF_ZONLY;
      textureMap = [3, null, null, null];
    } else if (usage === 'ambient') {
      baseFlags |= 0x3000;
      textureMap = [null, null, null, null];
    } else if (usage === 'shadow') {
      baseFlags = MBF_ZREAD | MBF_ZONLY | MBF_SHADOWMASK;
      textureMap = [null, null, null, null];
    } else if (usage === 'light') {
      baseFlags |= MBF_STENCILTEST | ((staticFlags & 4) ? 0x5000 : 0x2000);
      // The native light shader samples the captured color prepass in screen
      // space and reserves its material samplers for bump maps 4 and 5.
      textureMap = [null, 4, 5, null];
    } else if (usage === 'postlight') {
      baseFlags |= 0x2000;
      textureMap = [null, 4, 6, null];
    }
    const color = usage === 'light' ? dynamicParameters[1] >>> 0 : 0xffffffff;
    const uvTransforms = material20UVTransforms(dynamicParameters);
    return {
      system: material.system, baseFlags,
      textureFlags: staticParameters.slice(8, 12).map(value => value >>> 0),
      textureScales: dynamicParameters.slice(12, 16).map(value => Number(value) || 0),
      lightFlags: staticParameters.slice(16, 20).map(value => value >>> 0),
      lightScales: dynamicParameters.slice(20, 24).map(value => Number(value) || 0),
      extraTextureFlags: staticParameters.slice(16, 20).map(value => value >>> 0),
      uvTransform1: uvTransforms.transform1, uvTransform2: uvTransforms.transform2,
      color, specular: dynamicParameters[2] >>> 0,
      textures: material.textures || [], textureMap, vertexColorMode: 0,
      usage, state: pass?.state || null,
      specularPower: dynamicParameters[3] || 8,
      specularColor: dynamicParameters[2] >>> 0,
      specularStrength: staticFlags & 1 ? 1 : 0,
      uvScale: 1,
      alphaCutoff: usage === 'base' && material.textures?.[3] ? ALPHA_REFERENCE : 0,
      detailOps: [((staticParameters[9] >>> 16) & 15), ((staticParameters[10] >>> 16) & 15)],
      usePrelight: usage === 'light',
      lightScale: dynamicFlags & 0x40 ? 2 : 1,
      flags: staticFlags,
      runtimeFlags: dynamicFlags,
      environmentFlags: staticParameters[18] >>> 0,
      runtimeEnvironmentFlags: dynamicParameters[18] >>> 0,
    };
  }
  const staticParameters = material.initialParameters || parameters;
  const dynamicParameters = dynamicMaterial?.parameters || parameters;
  const multiFlags = staticParameters[48] >>> 0;
  const dynamicMultiFlags = dynamicParameters[48] >>> 0;
  const isMultipass = Boolean(multiFlags & 0x1e);
  let baseFlags = staticParameters[0] >>> 0;
  let color = dynamicParameters[28] >>> 0;
  let textureMap = [0, 1, 2, 3];
  let uvScale = dynamicParameters[24] || 1;
  let specialFlags = staticParameters[1] >>> 0;
  let alphaCombiner = staticParameters[23] >>> 0;
  let srt1Values = dynamicParameters.slice(34, 43);
  let srt2Values = dynamicParameters.slice(43, 48);
  let alphaCutoff = baseFlags & MBF_ALPHATEST ? ALPHA_REFERENCE : 0;
  let vertexColorMode = 0;
  let generatedBaseAlpha = null;
  if (isMultipass) {
    if (usage === 'base') {
      baseFlags = 0x0300 | 0x020000 | (staticParameters[0] & MBF_DOUBLESIDED);
      color = multiFlags & 2 ? staticParameters[53] >>> 0 : 0x00ffffff;
      textureMap = multiFlags & 0x0400 ? [0, null, null, null] : [null, null, null, null];
      alphaCutoff = multiFlags & 0x0400 ? ALPHA_REFERENCE : 0;
      // GenMaterial's generated base pass starts from a fresh Material11:
      // alpha is ZERO normally and TEX0 for the optional alpha-test path.
      generatedBaseAlpha = multiFlags & 0x0400 ? 'texture0' : 'zero';
      vertexColorMode = 4;
      specialFlags = 0;
      // GenMaterial constructs this pass from a fresh sMaterial11 and only
      // copies TFlags/TScale for its optional alpha texture.
      srt1Values = new Array(9).fill(0);
      srt2Values = new Array(5).fill(0);
    } else if (usage === 'light') {
      baseFlags = MBF_ZREAD | MBF_ZEQUAL | MBF_STENCILTEST |
        (multiFlags & 0x4000 ? 0x5000 : 0x2000) | (staticParameters[0] & MBF_DOUBLESIDED);
      color = dynamicParameters[52] >>> 0;
      if (!(dynamicMultiFlags & 0x0200)) color &= 0x00ffffff;
      textureMap = [null, 4, null, null];
      uvScale = dynamicMultiFlags & 0x1400
        ? dynamicParameters[24] || 1 : dynamicParameters[61] || 1;
      alphaCutoff = 0;
      specialFlags = !(multiFlags & 0x0200) ? 0x40 : (multiFlags & 0x2000 ? 0x80 : 0);
    } else if (usage === 'shadow') {
      baseFlags = MBF_ZREAD | MBF_ZONLY | MBF_SHADOWMASK;
      color = 0;
      textureMap = [null, null, null, null];
      alphaCutoff = 0;
      specialFlags = 0;
      srt1Values = new Array(9).fill(0);
      srt2Values = new Array(5).fill(0);
    } else if (usage === 'postlight') {
      baseFlags = MBF_ZREAD | MBF_ZEQUAL | 0x4000;
      alphaCutoff = 0;
      // The generated texture pass forces HALF so multiplicative blending
      // preserves the destination-alpha specular accumulator.
      alphaCombiner = 0x0c;
      color = dynamicParameters[28] >>> 0;
    } else if (usage === 'postlight2') {
      baseFlags = MBF_ZREAD | MBF_ZEQUAL | (multiFlags & 0x8000 ? 0 : 0x2000);
      color = dynamicParameters[53] >>> 0;
      textureMap = [multiFlags & 0x4000 ? 0 : null, null, 6, null];
      uvScale = dynamicParameters[62] || 1;
      alphaCutoff = 0;
      // The generated environment pass does not inherit the base material's
      // other special bits; bit 8 selects sphere rather than reflection.
      specialFlags = multiFlags & 0x0100 ? 0x10 : 0x20;
      // Unlike LIGHT/POSTLIGHT, Exec only changes Color[2] and TScale[2].
      srt1Values = staticParameters.slice(34, 43);
      srt2Values = staticParameters.slice(43, 48);
    }
  } else if (usage === 'light') color = dynamicParameters[52] >>> 0;
  else if (usage === 'postlight2') color = dynamicParameters[53] >>> 0;
  if (!isMultipass) {
    const vertexOperation = staticParameters[19] & 0x00f0;
    if (vertexOperation === 0x0010) vertexColorMode = 3;
    else if (vertexOperation === 0x0020) vertexColorMode = 2;
    else if (vertexOperation >= 0x0040 && vertexOperation <= 0x0070) vertexColorMode = 1;
  }
  const material11Combiner = (!isMultipass && (staticParameters[2] & 7) === 0) || usage === 'postlight';
  const textureFlags = staticParameters.slice(4, 8).map(value => value >>> 0);
  const extraTextureFlags = staticParameters.slice(56, 60).map(value => value >>> 0);
  const dynamicTextureScales = !isMultipass || usage === 'postlight';
  const textureScales = (dynamicTextureScales
    ? dynamicParameters : staticParameters).slice(24, 28).map(value => {
    value = Number(value); return Number.isFinite(value) ? value : 0;
  });
  const extraTextureScales = staticParameters.slice(60, 64).map(value => {
    value = Number(value); return Number.isFinite(value) ? value : 0;
  });
  if (usage === 'light') {
    extraTextureScales[0] = Number(dynamicParameters[60]) || 0;
  } else if (usage === 'postlight2') {
    extraTextureScales[2] = Number(dynamicParameters[62]) || 0;
  }
  const slotTextureFlags = textureMap.map(sourceIndex => {
    if (sourceIndex === null || sourceIndex === undefined) return 0;
    return sourceIndex < 4 ? textureFlags[sourceIndex] : extraTextureFlags[sourceIndex - 4];
  });
  const slotTextureScales = textureMap.map(sourceIndex => {
    if (sourceIndex === null || sourceIndex === undefined) return 0;
    return sourceIndex < 4 ? textureScales[sourceIndex] : extraTextureScales[sourceIndex - 4];
  });
  const uvTransforms = material11UVTransforms(srt1Values, srt2Values);
  const colors = dynamicParameters.slice(28, 32).map(value => value >>> 0);
  if (isMultipass && usage === 'postlight' && (dynamicMultiFlags & 0x0400)) {
    // Exec_Material_Material forces Color[3] after copying the animated
    // 24..47 block for alpha-tested texture phases.
    colors[3] = 0xff808080;
  }
  return {
    system: material.system || '1.1', baseFlags,
    specialFlags, lightFlags: staticParameters[2] >>> 0,
    textureFlags, extraTextureFlags, textureScales, extraTextureScales,
    slotTextureFlags, slotTextureScales,
    uvTransform1: uvTransforms.transform1, uvTransform2: uvTransforms.transform2,
    color, textures: material.textures || [], textureMap, vertexColorMode,
    usage, state: pass?.state || null,
    specularPower: usage === 'light' ? dynamicParameters[51] || 8 : dynamicParameters[32] || 8,
    specularColor: color,
    specularStrength: isMultipass
      ? (multiFlags & 0x0200 ? 1 : 0)
      : (staticParameters[1] & 0x40 ? 0 : 1),
    uvScale, alphaCutoff, multiFlags, material11Combiner, generatedBaseAlpha,
    combiners: staticParameters.slice(8, 21).map(value => value >>> 0),
    alphaCombiner,
    colors,
  };
}

// CPU mirrors of the two source-defined color combiners. They keep the packed
// flag interpretation testable without a GL context and intentionally cover
// the unlit Material11 and Material20 texture phases used by production.
function evaluateMaterial20Prelight(view, samples = []) {
  const copy = value => Float32Array.from(value || [1, 1, 1, 1]);
  const result = samples[0] ? copy(samples[0]) : new Float32Array([1, 1, 1, 0]);
  for (let index = 1; index <= 2; index++) {
    if (!samples[index]) continue;
    const detail = samples[index], operation = view.detailOps?.[index - 1] ?? 0;
    for (let channel = 0; channel < 3; channel++) {
      const current = result[channel], value = detail[channel];
      if (operation === 0) result[channel] = current * value;
      else if (operation === 1) result[channel] = current * value * 2;
      else if (operation === 2) result[channel] = current * value * 4;
      else if (operation === 3) result[channel] = current + value;
      else if (operation === 4) result[channel] = current + value - current * value;
    }
  }
  return result;
}

function evaluateMaterial11Combiner(view, samples = {}) {
  const textures = samples.textures || [];
  const colors = (view.colors || []).map(value => colorARGB(value >>> 0));
  while (colors.length < 4) colors.push(colorARGB(0xffffffff));
  const white = new Float32Array([1, 1, 1, 1]);
  const texture = index => textures[index] || white;
  const source = index => {
    if (index === 1) return colors[0]; if (index === 2) return colors[1];
    if (index === 3) return texture(0); if (index === 4) return texture(1);
    if (index === 7) return colors[2]; if (index === 8) return colors[3];
    if (index === 9) return texture(2); if (index === 10) return texture(3);
    if (index === 11) return samples.vertex || white;
    return new Float32Array(4);
  };
  const alpha = code => {
    if (code >= 4 && code <= 7) return texture(code - 4)[3];
    if (code >= 8 && code <= 11) return colors[code - 8][3];
    if (code === 12) return 0.5; if (code === 13) return 0;
    return 1;
  };
  const current = new Float32Array(4);
  for (let index = 0; index < 13; index++) {
    const word = view.combiners?.[index] >>> 0;
    const operation = (word >>> 4) & 15;
    if (!operation) continue;
    const input = Float32Array.from(source(index));
    const alphaCode = word & 15;
    let fade = alpha(alphaCode);
    // Color-combiner AlphaSrc never receives the AlphaCombiner inversion bits.
    if (alphaCode && operation !== 9) for (let channel = 0; channel < 3; channel++) input[channel] *= fade;
    for (let channel = 0; channel < 3; channel++) {
      if (operation === 1) current[channel] = input[channel];
      else if (operation === 2) current[channel] += input[channel];
      else if (operation === 3) current[channel] -= input[channel];
      else if (operation === 4) current[channel] *= input[channel];
      else if (operation >= 5 && operation <= 7) current[channel] *= input[channel] * (1 << (operation - 4));
      else if (operation === 8) current[channel] = current[channel] * (1 - input[channel]) + input[channel];
      else if (operation === 9) current[channel] = input[channel] * (1 - fade) + current[channel] * fade;
      else if (operation === 10) current[channel] = input[channel] - current[channel];
      else if (operation === 12) current[channel] = Math.max(0, Math.min(1, current[channel] * input[channel]));
    }
    if (operation === 11) {
      const dot = current[0] * input[0] + current[1] * input[1] + current[2] * input[2];
      current[0] = current[1] = current[2] = dot;
    }
  }
  const alphaWord = view.alphaCombiner >>> 0;
  const secondCode = (alphaWord >>> 4) & 15;
  let outputAlpha;
  if (secondCode === 14) {
    outputAlpha = (texture(0)[3] + texture(1)[3]) * colors[0][3];
  } else {
    outputAlpha = alpha(alphaWord & 15);
    if (alphaWord & 0x100) outputAlpha = 1 - outputAlpha;
    if (secondCode) {
      let second = alpha(secondCode);
      if (alphaWord & 0x200) second = 1 - second;
      outputAlpha *= second;
    }
  }
  current[3] = outputAlpha;
  return current;
}

function tangentBasisNormal(normal, tangent, mapped) {
  const n = normalize3(normal[0], normal[1], normal[2], [0, 0, 1]);
  const projection = n[0] * tangent[0] + n[1] * tangent[1] + n[2] * tangent[2];
  const t = normalize3(tangent[0] - n[0] * projection,
    tangent[1] - n[1] * projection, tangent[2] - n[2] * projection, [1, 0, 0]);
  const b = normalize3(n[1] * t[2] - n[2] * t[1],
    n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0], [0, 1, 0]);
  return normalize3(n[0] * mapped[0] + b[0] * mapped[1] + t[0] * mapped[2],
    n[1] * mapped[0] + b[1] * mapped[1] + t[1] * mapped[2],
    n[2] * mapped[0] + b[2] * mapped[1] + t[2] * mapped[2], n);
}

// Numerical oracle for the radial curve underlying MakeAttenuationVolume and
// Material20's analytic attenuation MAD. The live M11 path samples the exact
// generated 8-bit/voxel lookup above instead of calling this approximation.
function materialLightAttenuation(normalizedDistance) {
  const radius = Number(normalizedDistance);
  return Number.isFinite(radius) ? Math.max(0, 1 - radius * radius) : 0;
}

// sMatrix::TransR in the released engine is deliberately only a transposed
// upper 3x3 plus A^T(-translation), not a general affine inverse. M11 and M20
// both use it for their model-space light and eye constants. In particular,
// authored object scale remains in these vectors, so replacing this with a
// mathematically correct inverse changes both direction and attenuation.
function legacyTransRVector(matrix, vector, point = true,
    output = new Float32Array(3), offset = 0) {
  const dx = Number(vector?.[0] || 0) - (point ? Number(matrix?.[12] || 0) : 0);
  const dy = Number(vector?.[1] || 0) - (point ? Number(matrix?.[13] || 0) : 0);
  const dz = Number(vector?.[2] || 0) - (point ? Number(matrix?.[14] || 0) : 0);
  output[offset] = matrix[0] * dx + matrix[1] * dy + matrix[2] * dz;
  output[offset + 1] = matrix[4] * dx + matrix[5] * dy + matrix[6] * dz;
  output[offset + 2] = matrix[8] * dx + matrix[9] * dy + matrix[10] * dz;
  return output;
}

// Numerical oracle for material11.vsh:87-115,242-280 and
// material20_light.vsh:51-111. Inputs are already in the legacy model space
// uploaded to those shaders. Only skinned variants condition the N/T basis;
// static and instance variants retain the source vectors verbatim.
function legacyVertexLighting(position, normal, tangent, light, attenuation,
    camera, conditionBasis = false) {
  const p = [Number(position?.[0]) || 0, Number(position?.[1]) || 0,
    Number(position?.[2]) || 0];
  let n = [Number(normal?.[0]) || 0, Number(normal?.[1]) || 0,
    Number(normal?.[2]) || 0];
  let t = [Number(tangent?.[0]) || 0, Number(tangent?.[1]) || 0,
    Number(tangent?.[2]) || 0];
  if (conditionBasis) {
    n = normalize3(n[0], n[1], n[2], [0, 0, 1]);
    const projection = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
    t = normalize3(t[0] - n[0] * projection, t[1] - n[1] * projection,
      t[2] - n[2] * projection, [1, 0, 0]);
  }
  const b = [n[1] * t[2] - n[2] * t[1],
    n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
  const lightW = Number(light?.[3]) || 0;
  const l = [(Number(light?.[0]) || 0) - p[0] * lightW,
    (Number(light?.[1]) || 0) - p[1] * lightW,
    (Number(light?.[2]) || 0) - p[2] * lightW];
  const normalizedLight = normalize3(l[0], l[1], l[2], [0, 1, 0]);
  const eye = normalize3((Number(camera?.[0]) || 0) - p[0],
    (Number(camera?.[1]) || 0) - p[1],
    (Number(camera?.[2]) || 0) - p[2], [0, 1, 0]);
  const h = [normalizedLight[0] + eye[0], normalizedLight[1] + eye[1],
    normalizedLight[2] + eye[2]];
  const projectNBT = vector => new Float32Array([
    vector[0] * n[0] + vector[1] * n[1] + vector[2] * n[2],
    vector[0] * b[0] + vector[1] * b[1] + vector[2] * b[2],
    vector[0] * t[0] + vector[1] * t[1] + vector[2] * t[2],
  ]);
  const inverseRange = Number(attenuation?.[3]) || 0;
  return {
    normal: Float32Array.from(n), tangent: Float32Array.from(t),
    bitangent: Float32Array.from(b), light: projectNBT(l), halfway: projectNBT(h),
    attenuation: new Float32Array([
      (p[0] - (Number(attenuation?.[0]) || 0)) * inverseRange,
      (p[1] - (Number(attenuation?.[1]) || 0)) * inverseRange,
      (p[2] - (Number(attenuation?.[2]) || 0)) * inverseRange,
    ]),
  };
}

// Exec_Effect_ColorCorrection uploads seven pixel constants consumed literally
// by effect_colorcorrect.psh. Keeping this packing separate makes the unusual
// dark/light ordering and range-only fake mode explicit and testable.
function colorCorrectionConstants(parameters = []) {
  const result = new Float32Array(7 * 4);
  const threshold0 = Number(parameters[1] ?? 0);
  const threshold1 = Number(parameters[2] ?? 0);
  const denominator = threshold1 - threshold0;
  const thresholdOffset = -threshold0 / denominator;
  const thresholdScale = 1 / denominator;
  if ((parameters[7] >>> 0) & 1) {
    result.set([1, 1, 1, 1], 2 * 4);
    result.set([thresholdOffset, thresholdScale, 1, 0], 4 * 4);
    return result;
  }
  result.set(colorARGB(parameters[3] >>> 0), 0 * 4);
  result.set(colorARGB(parameters[4] >>> 0), 1 * 4);
  const brightness = colorARGB(parameters[5] >>> 0);
  const amplify = Number(parameters[10] ?? 0);
  for (let channel = 0; channel < 4; channel++) brightness[channel] *= amplify;
  result.set(brightness, 2 * 4);
  result.set(colorARGB(parameters[6] >>> 0), 3 * 4);
  result.set([thresholdOffset, thresholdScale, 0, 0], 4 * 4);
  result.set(colorARGB(parameters[8] >>> 0), 5 * 4);
  result.set(colorARGB(parameters[9] >>> 0), 6 * 4);
  return result;
}

function evaluateColorCorrection(parameters, source) {
  const constants = colorCorrectionConstants(parameters);
  const gray = (source[0] + source[1] + source[2]) / 3;
  const range = Math.max(0, Math.min(1, gray * constants[17] + constants[16]));
  const result = new Float32Array(4);
  for (let channel = 0; channel < 3; channel++) {
    const dark = gray * constants[20 + channel] + source[channel] * constants[channel];
    const light = gray * constants[24 + channel] + source[channel] * constants[4 + channel];
    const mixed = dark * range + light * (1 - range) + constants[18] * range;
    result[channel] = saturate(Math.max(0, Math.min(1, mixed - constants[12 + channel])) *
      constants[8 + channel]);
  }
  result[3] = source[3];
  return result;
}

// Exec_Effect_Glare uses one of two fixed render-target levels before its tone
// map and blur passes. This plan is a literal unpacking of geneffectipp.cpp,
// including the otherwise unusual flags=3 fallback to the first level.
function glarePlan(parameters = []) {
  const flags = parameters[1] | 0;
  let downsample;
  if ((flags & 3) === 1) downsample = [false, true];
  else if ((flags & 3) === 2) downsample = [true, true];
  else downsample = [true, false];
  const threshold = colorARGB(parameters[2] >>> 0);
  const grayScale = 1 / ((1 - threshold[0]) + (1 - threshold[1]) + (1 - threshold[2]));
  return {
    flags,
    downsample,
    copyDownsample: Boolean(flags & 4),
    threshold,
    grayScale,
    grayMix: Number(parameters[9] ?? 0),
    stages: [
      { blur: Number(parameters[5] ?? 0), amplify: Number(parameters[7] ?? 0) },
      { blur: Number(parameters[6] ?? 0), amplify: Number(parameters[8] ?? 0) },
    ],
    glareColor: colorARGB(parameters[3] >>> 0),
    originalColor: colorARGB(parameters[4] >>> 0),
    addSmooth: Number(parameters[10] ?? 0),
  };
}

function saturate(value) {
  return Math.max(0, Math.min(1, value));
}

function evaluateGlareTone(parameters, source) {
  const plan = glarePlan(parameters);
  const color = [
    source[0] - plan.threshold[0],
    source[1] - plan.threshold[1],
    source[2] - plan.threshold[2],
  ];
  const gray = (color[0] + color[1] + color[2]) * plan.grayScale;
  const inverseMix = 1 - plan.grayMix;
  return new Float32Array([
    saturate(color[0] * plan.grayMix + gray * inverseMix),
    saturate(color[1] * plan.grayMix + gray * inverseMix),
    saturate(color[2] * plan.grayMix + gray * inverseMix),
    saturate(source[3] * plan.grayMix + gray * inverseMix),
  ]);
}

function evaluateGlareComposite(parameters, glare, original) {
  const plan = glarePlan(parameters);
  const result = new Float32Array(4);
  for (let channel = 0; channel < 4; channel++) {
    const blurred = glare[channel] * plan.glareColor[channel];
    const source = original[channel] * plan.originalColor[channel];
    result[channel] = saturate(blurred + source - blurred * source * plan.addSmooth);
  }
  return result;
}

function evaluateMaterial11Specular(cosine, power) {
  const usePower = Math.max(8, Math.min(60, Number(power) || 0));
  const inverseScale = 1 / 7.336032;
  const scale = inverseScale + (usePower - 8) * 0.01565302;
  const bias = inverseScale - scale;
  let value = Math.max(0, Math.min(1, (Math.max(0, Number(cosine) || 0) * scale + bias) * 4));
  value = value * value * 2;
  value = value * value * 2;
  return Math.max(0, Math.min(1, value * value * 2));
}

// sMaterial11::Set builds pixel-shader c0 in this exact order before the
// Direct3D PS1.1 upload.  Unlike modern GLSL uniforms, PS1.x constant
// registers have a fixed [-1,+1] range, so amplified light colors saturate
// here rather than after attenuation or the diffuse dot product.
function material11LightConstant(materialColor, lightColor, amplify,
    out = new Float32Array(4)) {
  materialColor >>>= 0;
  lightColor >>>= 0;
  const scale = f32(Number(amplify) || 0);
  const shifts = [16, 8, 0, 24];
  for (let channel = 0; channel < 4; channel++) {
    const shift = shifts[channel];
    const material = f32(((materialColor >>> shift) & 255) / 255);
    const light = f32(((lightColor >>> shift) & 255) / 255);
    const value = f32(f32(material * light) * scale);
    out[channel] = Math.max(-1, Math.min(1, value));
  }
  return out;
}

function materialInsertKind(material) {
  if (material?.system === '2.0') return 'material20';
  if (material?.system !== '1.1') return null;
  // Insert topology is fixed during Init, just like the pass list. Runtime
  // animation may update parameters[48], so prefer the captured init value.
  const multiFlags = (material.multiFlags ?? material.parameters?.[48] ?? 0) >>> 0;
  return multiFlags & 0x1e ? 'material11' : null;
}

function materialInsertPlan(items) {
  const result = new Map();
  const priorities = { material11: 0x10, material20: 0x20 };
  for (const item of items || []) {
    if (item?.effectJob) continue;
    const kind = materialInsertKind(item?.material);
    if (!kind) continue;
    const renderPass = item.renderPass | 0;
    const previous = result.get(renderPass);
    if (!previous || priorities[kind] > priorities[previous]) result.set(renderPass, kind);
  }
  return result;
}

function material11InsertAction(insertKind, usage) {
  if (insertKind !== 'material11') return null;
  if (usage === 'base') return 'clear-destination-alpha';
  if (usage === 'postlight') return 'add-destination-alpha';
  return null;
}

function renderUsageChanged(previous, next) {
  if (!previous || !next) return true;
  return previous.renderPass !== next.renderPass || previous.pass?.usage !== next.pass?.usage ||
    previous.light !== next.light;
}

function createMaterialInsertTracker(items) {
  const inserts = materialInsertPlan(items);
  let previous = null;
  const finish = item => item ? material11InsertAction(
    inserts.get(item.renderPass | 0), item.pass?.usage,
  ) : null;
  return {
    transition(item) {
      const action = previous && renderUsageChanged(previous, item) ? finish(previous) : null;
      previous = item;
      return action;
    },
    finish() {
      const action = finish(previous);
      previous = null;
      return action;
    },
  };
}

function renderMode(view) {
  if (view.system === '2.0' && view.usage === 'base') return 1;
  if (view.usage === 'shadow') return 1;
  if (view.material11Combiner) return 7;
  if (view.usage === 'light') return 2;
  if (view.system === '2.0' && view.usage === 'postlight') return 4;
  if (view.system === '2.0' && view.usage === 'prelight') return 6;
  if (view.usage === 'postlight') return 3;
  if (view.usage === 'postlight2') return 4;
  if (view.usage === 'ambient') return 5;
  return 0;
}

const MATERIAL_BLEND_TABLE = Object.freeze([
  null,
  Object.freeze(['src-alpha', 'one-minus-src-alpha', 'add']),
  Object.freeze(['one', 'one', 'add']),
  Object.freeze(['zero', 'src-color', 'add']),
  Object.freeze(['dst-color', 'src-color', 'add']),
  Object.freeze(['one', 'one-minus-src-color', 'add']),
  Object.freeze(['one', 'one', 'reverse-subtract']),
  Object.freeze(['one-minus-dst-color', 'one-minus-src-color', 'add']),
  Object.freeze(['dst-alpha', 'one', 'add']),
  Object.freeze(['src-alpha', 'one', 'add']),
]);
const MATERIAL_STATE_CACHE = Symbol('debrisMaterialState');

function materialState(view) {
  const flags = view.baseFlags >>> 0;
  const cached = view?.[MATERIAL_STATE_CACHE];
  if (cached?.flags === flags) return cached.state;
  const zMode = flags & 0x0300;
  let depthFunc = 'always';
  if (flags & MBF_ZREAD) {
    if (flags & MBF_ZEQUAL) depthFunc = 'equal';
    else if (flags & MBF_SHADOWMASK) depthFunc = 'less';
    else depthFunc = 'lequal';
  }
  const blendMode = (flags & 0xf000) >>> 12;
  const state = Object.freeze({
    depthTest: zMode !== 0,
    depthWrite: Boolean(flags & MBF_ZWRITE),
    depthFunc,
    colorWrite: !(flags & MBF_ZONLY),
    cull: flags & MBF_DOUBLESIDED ? 'none' : (flags & MBF_INVERTCULL ? 'front' : 'back'),
    blend: MATERIAL_BLEND_TABLE[blendMode] || null,
    stencilTest: (flags & 0x700000) === MBF_STENCILTEST,
  });
  if (view && (typeof view === 'object' || typeof view === 'function') && Object.isExtensible(view)) {
    Object.defineProperty(view, MATERIAL_STATE_CACHE, {
      configurable: true, value: { flags, state },
    });
  }
  return state;
}

function compiledMaterialView(material, pass = null) {
  const view = materialView(material, pass);
  materialState(view);
  return view;
}

function glDepthFunction(gl, depthFunc) {
  if (depthFunc === 'equal') return gl.EQUAL;
  if (depthFunc === 'less') return gl.LESS;
  if (depthFunc === 'lequal') return gl.LEQUAL;
  return gl.ALWAYS;
}

function glBlendFactor(gl, factor) {
  if (factor === 'zero') return gl.ZERO;
  if (factor === 'src-alpha') return gl.SRC_ALPHA;
  if (factor === 'one-minus-src-alpha') return gl.ONE_MINUS_SRC_ALPHA;
  if (factor === 'src-color') return gl.SRC_COLOR;
  if (factor === 'one-minus-src-color') return gl.ONE_MINUS_SRC_COLOR;
  if (factor === 'dst-color') return gl.DST_COLOR;
  if (factor === 'one-minus-dst-color') return gl.ONE_MINUS_DST_COLOR;
  if (factor === 'dst-alpha') return gl.DST_ALPHA;
  return gl.ONE;
}

function renderItemLightOrder(item) {
  // Effect sort keys leave the native light field at zero for every usage.
  if (item.effectJob) return 0;
  const usage = item.pass?.usage || 'other';
  if (usage === 'shadow' || usage === 'light') return item.lightIndex ?? 0;
  return (USAGE_ORDER[usage] ?? 99) < USAGE_ORDER.shadow ? 0 : LIGHT_SORT_POST;
}

function composeInstanceMatrices(job) {
  const base = job?.matrix?.length >= 16 ? job.matrix : mat4Identity();
  // A null instance list is an ordinary one-matrix paint job. An explicitly
  // empty list is MPP_INSTANCES with InstanceCount==0, which native PaintJob
  // skips entirely; substituting the base matrix would draw one stray mesh.
  if (job?.instances == null) return new Float32Array(base);
  if (!job.instances.length) return EMPTY_INSTANCE_MATRICES;
  const result = new Float32Array(job.instances.length * 16);
  for (let index = 0; index < job.instances.length; index++) {
    const instance = job.instances[index];
    if (!instance?.length || instance.length < 16) {
      result.fill(NaN, index * 16, index * 16 + 16);
      continue;
    }
    const target = index * 16;
    // Write base*instance straight into the packed upload buffer. Calling the
    // general mat4 helper here used to allocate one temporary typed array per
    // particle every frame.
    for (let column = 0; column < 4; column++) {
      const source = column * 4, output = target + source;
      for (let row = 0; row < 4; row++) {
        result[output + row] = Math.fround(
          base[row] * instance[source] +
          base[4 + row] * instance[source + 1] +
          base[8 + row] * instance[source + 2] +
          base[12 + row] * instance[source + 3],
        );
      }
    }
  }
  return result;
}

function transformGeometryBounds(bounds, matrix, out = null, matrixOffset = 0) {
  const minimum = bounds?.minimum || bounds?.min;
  const maximum = bounds?.maximum || bounds?.max;
  if (!minimum?.length || !maximum?.length || !matrix?.length ||
      matrix.length < matrixOffset + 16 ||
      !Number.isFinite(minimum[0]) || !Number.isFinite(minimum[1]) || !Number.isFinite(minimum[2]) ||
      !Number.isFinite(maximum[0]) || !Number.isFinite(maximum[1]) || !Number.isFinite(maximum[2]) ||
      minimum[0] > maximum[0] || minimum[1] > maximum[1] || minimum[2] > maximum[2]) return null;
  for (let index = 0; index < 16; index++) {
    if (!Number.isFinite(matrix[matrixOffset + index])) return null;
  }
  const m0 = matrix[matrixOffset], m1 = matrix[matrixOffset + 1];
  const m2 = matrix[matrixOffset + 2], m3 = matrix[matrixOffset + 3];
  const m4 = matrix[matrixOffset + 4], m5 = matrix[matrixOffset + 5];
  const m6 = matrix[matrixOffset + 6], m7 = matrix[matrixOffset + 7];
  const m8 = matrix[matrixOffset + 8], m9 = matrix[matrixOffset + 9];
  const m10 = matrix[matrixOffset + 10], m11 = matrix[matrixOffset + 11];
  const m12 = matrix[matrixOffset + 12], m13 = matrix[matrixOffset + 13];
  const m14 = matrix[matrixOffset + 14], m15 = matrix[matrixOffset + 15];
  // Scene matrices are affine. If an embedder supplies a projective model,
  // decline to cull instead of deriving an unsafe Euclidean box.
  if (Math.abs(m3) > 1e-8 || Math.abs(m7) > 1e-8 ||
      Math.abs(m11) > 1e-8 || Math.abs(m15 - 1) > 1e-8) return null;
  const cx = (minimum[0] + maximum[0]) * 0.5;
  const cy = (minimum[1] + maximum[1]) * 0.5;
  const cz = (minimum[2] + maximum[2]) * 0.5;
  const ex = Math.max(0, (maximum[0] - minimum[0]) * 0.5);
  const ey = Math.max(0, (maximum[1] - minimum[1]) * 0.5);
  const ez = Math.max(0, (maximum[2] - minimum[2]) * 0.5);
  const x = m0 * cx + m4 * cy + m8 * cz + m12;
  const y = m1 * cx + m5 * cy + m9 * cz + m13;
  const z = m2 * cx + m6 * cy + m10 * cz + m14;
  const wx = Math.abs(m0) * ex + Math.abs(m4) * ey + Math.abs(m8) * ez;
  const wy = Math.abs(m1) * ex + Math.abs(m5) * ey + Math.abs(m9) * ez;
  const wz = Math.abs(m2) * ex + Math.abs(m6) * ey + Math.abs(m10) * ez;
  out ||= { minimum: new Float64Array(3), maximum: new Float64Array(3) };
  out.minimum[0] = x - wx; out.minimum[1] = y - wy; out.minimum[2] = z - wz;
  out.maximum[0] = x + wx; out.maximum[1] = y + wy; out.maximum[2] = z + wz;
  return out;
}

function meshJobWorldBounds(geometry, job, modelMatrices = null) {
  const matrices = modelMatrices || composeInstanceMatrices(job);
  if (!matrices?.length || matrices.length % 16) return null;
  const minimum = new Float64Array([Infinity, Infinity, Infinity]);
  const maximum = new Float64Array([-Infinity, -Infinity, -Infinity]);
  const transformed = { minimum: new Float64Array(3), maximum: new Float64Array(3) };
  for (let offset = 0; offset < matrices.length; offset += 16) {
    if (!transformGeometryBounds(geometry?.bounds, matrices, transformed, offset)) return null;
    // A single unknown instance means the union is unknown. Keeping its jobs
    // is the only conservative choice.
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], transformed.minimum[axis]);
      maximum[axis] = Math.max(maximum[axis], transformed.maximum[axis]);
    }
  }
  return { minimum, maximum };
}

function boundsIntersectsSphere(bounds, center, radius) {
  const minimum = bounds?.minimum || bounds?.min;
  const maximum = bounds?.maximum || bounds?.max;
  if (!minimum?.length || !maximum?.length || !center?.length || center.length < 3 ||
      !Number.isFinite(radius) || radius < 0 ||
      ![center[0], center[1], center[2]].every(Number.isFinite)) return true;
  let distanceSquared = 0;
  for (let axis = 0; axis < 3; axis++) {
    const delta = center[axis] < minimum[axis] ? center[axis] - minimum[axis]
      : center[axis] > maximum[axis] ? center[axis] - maximum[axis] : 0;
    distanceSquared += delta * delta;
  }
  // Grow the sphere infinitesimally relative to its scale. This only retains
  // borderline work and avoids a floating-point false rejection.
  const conservativeRadius = radius + Math.max(1e-6, Math.abs(radius) * 1e-7);
  return distanceSquared <= conservativeRadius * conservativeRadius;
}

function lightIntersectsWorldBounds(light, bounds) {
  if (light?.kind === 'directional') return true;
  if (!Number.isFinite(light?.range) || light.range < 0) return true;
  return boundsIntersectsSphere(bounds, light.position, light.range);
}

function viewFrustumPlanes(viewProjection) {
  if (!viewProjection?.length || viewProjection.length < 16) return null;
  for (let index = 0; index < 16; index++) {
    if (!Number.isFinite(viewProjection[index])) return null;
  }
  const row0 = [viewProjection[0], viewProjection[4], viewProjection[8], viewProjection[12]];
  const row1 = [viewProjection[1], viewProjection[5], viewProjection[9], viewProjection[13]];
  const row2 = [viewProjection[2], viewProjection[6], viewProjection[10], viewProjection[14]];
  const row3 = [viewProjection[3], viewProjection[7], viewProjection[11], viewProjection[15]];
  const combine = (a, wa, b, wb) => [
    a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb,
    a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb,
  ];
  // The released engine uses the four side planes and the near plane. Its
  // projection has an infinite far plane, so deliberately do not manufacture
  // a sixth one here. The WebGL projection maps native z>=0 clipping to
  // z+w>=0, hence row3+row2 for the near plane.
  const candidates = [
    combine(row3, 1, row0, 1), combine(row3, 1, row0, -1),
    combine(row3, 1, row1, 1), combine(row3, 1, row1, -1),
    combine(row3, 1, row2, 1),
  ];
  const planes = [];
  for (const candidate of candidates) {
    const length = Math.hypot(candidate[0], candidate[1], candidate[2]);
    // A malformed/degenerate plane must only make rejection looser.
    if (!(length > 1e-20) || !Number.isFinite(length)) continue;
    const inverseLength = 1 / length;
    planes.push(new Float64Array([
      candidate[0] * inverseLength, candidate[1] * inverseLength,
      candidate[2] * inverseLength, candidate[3] * inverseLength,
    ]));
  }
  return planes.length ? planes : null;
}

function lightIntersectsViewFrustum(light, planes) {
  // Directional lights affect the whole viewport. Native Debris represents
  // them as a distant point with a practically infinite range; the explicit
  // semantic form can bypass the equivalent sphere test without risk.
  if (light?.kind === 'directional' || !planes?.length) return true;
  const position = light?.position, radius = Number(light?.range);
  if (!position?.length || position.length < 3 || !Number.isFinite(radius) || radius < 0 ||
      !Number.isFinite(position[0]) || !Number.isFinite(position[1]) ||
      !Number.isFinite(position[2])) return true;
  for (const plane of planes) {
    const distance = plane[0] * position[0] + plane[1] * position[1] +
      plane[2] * position[2] + plane[3];
    // Retain tangent and numerically borderline spheres. A false positive only
    // costs one candidate; a false negative would drop visible illumination.
    const tolerance = 1e-7 * (1 + radius + Math.abs(plane[0] * position[0]) +
      Math.abs(plane[1] * position[1]) + Math.abs(plane[2] * position[2]) +
      Math.abs(plane[3]));
    if (distance < -radius - tolerance) return false;
  }
  return true;
}

function shadowViewFrustumPlanes(viewProjection, lightPosition) {
  if (!viewProjection?.length || viewProjection.length < 16 || !lightPosition?.length ||
      lightPosition.length < 3 || !Number.isFinite(lightPosition[0]) ||
      !Number.isFinite(lightPosition[1]) || !Number.isFinite(lightPosition[2])) return null;
  for (let index = 0; index < 16; index++) {
    if (!Number.isFinite(viewProjection[index])) return null;
  }
  const row0 = [viewProjection[0], viewProjection[4], viewProjection[8], viewProjection[12]];
  const row1 = [viewProjection[1], viewProjection[5], viewProjection[9], viewProjection[13]];
  const row2 = [viewProjection[2], viewProjection[6], viewProjection[10], viewProjection[14]];
  const row3 = [viewProjection[3], viewProjection[7], viewProjection[11], viewProjection[15]];
  const combine = (a, wa, b, wb) => new Float64Array([
    a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb,
    a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb,
  ]);
  // This is the native five-plane SVFrustum construction, but uses the full
  // viewport rather than its tighter projected-light rectangle. The broader
  // volume can retain extra casters but cannot reject a shadow the source
  // would have kept. WebGL's near plane is z+w >= 0.
  const planes = [
    combine(row3, 1, row0, 1), combine(row3, 1, row0, -1),
    combine(row3, 1, row1, 1), combine(row3, 1, row1, -1),
    combine(row3, 1, row2, 1),
  ];
  for (const plane of planes) {
    const length = Math.hypot(plane[0], plane[1], plane[2]);
    if (!(length > 1e-20)) continue;
    const distance = plane[0] * lightPosition[0] + plane[1] * lightPosition[1] +
      plane[2] * lightPosition[2] + plane[3];
    if (distance < 0) plane[3] -= distance;
  }
  return planes;
}

// Engine::CalcSphereBounds derives the light's normalized screen rectangle
// before constructing both of its shadow frusta. Keep this separate from the
// ordinary view-frustum test: ZFailVolume deliberately uses the tighter light
// rectangle to decide whether a caster can safely take the cheaper z-pass
// path.
function lightSphereBounds(light, view, zoomX = 1, zoomY = 1) {
  const position = light?.position;
  const radius = Number(light?.range);
  if (!position?.length || position.length < 3 || !view?.length || view.length < 16 ||
      ![position[0], position[1], position[2]].every(Number.isFinite)) return null;
  for (let index = 0; index < 16; index++) if (!Number.isFinite(view[index])) return null;
  zoomX = Number(zoomX); zoomY = Number(zoomY);
  if (!Number.isFinite(zoomX) || !Number.isFinite(zoomY)) return null;

  const cx = view[0] * position[0] + view[4] * position[1] +
    view[8] * position[2] + view[12];
  const cy = view[1] * position[0] + view[5] * position[1] +
    view[9] * position[2] + view[13];
  const cz = view[2] * position[0] + view[6] * position[1] +
    view[10] * position[2] + view[14];
  const rectangle = new Float64Array([-1, -1, 1, 1]);
  // An infinite/unknown radius is the conservative full-screen equivalent.
  if (!Number.isFinite(radius) || radius < 0) return rectangle;
  const components = [cx, cy];
  const zoom = [zoomX, zoomY];
  for (let axis = 0; axis < 2; axis++) {
    const ci = components[axis];
    const distanceSquared = ci * ci + cz * cz;
    let tangent = distanceSquared - radius * radius;
    if (!(tangent > 0)) continue;
    tangent = Math.sqrt(tangent);

    let sine = ci * tangent - cz * radius;
    let cosine = ci * radius + cz * tangent;
    if (cz * distanceSquared > -radius * sine) {
      rectangle[axis] = Math.max(-1, sine / cosine * zoom[axis]);
    }

    sine = cz * radius + ci * tangent;
    cosine = cz * tangent - ci * radius;
    if (cz * distanceSquared > radius * sine) {
      rectangle[axis + 2] = Math.min(1, sine / cosine * zoom[axis]);
    }
  }
  return rectangle;
}

function planeFromPoints(a, b, c) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
  const x = aby * acz - abz * acy;
  const y = abz * acx - abx * acz;
  const z = abx * acy - aby * acx;
  return new Float64Array([x, y, z, -(x * a[0] + y * a[1] + z * a[2])]);
}

// Source counterpart: sFrustum::ZFailVolume. Its planes enclose the region in
// which a caster might cross the near plane and therefore needs Carmack's
// reverse. A bbox outside any plane can use z-pass and omit its caps.
function shadowZFailVolumePlanes(camera, light, matrices = null, rectangle = null) {
  const cameraSpace = camera?.cameraSpace;
  const lightPosition = light?.position;
  if (!cameraSpace?.length || cameraSpace.length < 16 || !lightPosition?.length ||
      lightPosition.length < 3 ||
      ![lightPosition[0], lightPosition[1], lightPosition[2]].every(Number.isFinite)) return null;
  for (let index = 0; index < 16; index++) {
    if (!Number.isFinite(cameraSpace[index])) return null;
  }
  const view = matrices?.view || mat4Inverse(cameraSpace);
  if (!view) return null;
  const near = Math.max(1e-5, Math.abs(Number(camera.nearClip) || 0.125));
  const zoomX = Number(camera.zoomX) || 1;
  const zoomY = Number(camera.zoomY) || 1;
  rectangle ||= lightSphereBounds(light, view, zoomX, zoomY);
  if (!rectangle?.length || rectangle.length < 4 ||
      !Number.isFinite(rectangle[0]) || !Number.isFinite(rectangle[1]) ||
      !Number.isFinite(rectangle[2]) || !Number.isFinite(rectangle[3]) ||
      !(rectangle[0] < rectangle[2]) || !(rectangle[1] < rectangle[3])) return null;

  const xs = near / zoomX, ys = near / zoomY;
  if (!Number.isFinite(xs) || !Number.isFinite(ys)) return null;
  const points = new Array(4);
  for (let index = 0; index < 4; index++) {
    const x = xs * (index & 1 ? rectangle[2] : rectangle[0]);
    const y = ys * (index & 2 ? rectangle[3] : rectangle[1]);
    points[index] = new Float64Array([
      cameraSpace[12] + cameraSpace[0] * x + cameraSpace[4] * y + cameraSpace[8] * near,
      cameraSpace[13] + cameraSpace[1] * x + cameraSpace[5] * y + cameraSpace[9] * near,
      cameraSpace[14] + cameraSpace[2] * x + cameraSpace[6] * y + cameraSpace[10] * near,
    ]);
  }

  const planes = [planeFromPoints(points[0], points[1], points[2])];
  const nearDistance = planes[0][0] * lightPosition[0] +
    planes[0][1] * lightPosition[1] + planes[0][2] * lightPosition[2] + planes[0][3];
  const onEpsilon = 1e-4;
  if (Math.abs(nearDistance) >= onEpsilon) {
    planes.push(
      planeFromPoints(lightPosition, points[1], points[0]),
      planeFromPoints(lightPosition, points[3], points[1]),
      planeFromPoints(lightPosition, points[2], points[3]),
      planeFromPoints(lightPosition, points[0], points[2]),
    );
    if (nearDistance < 0) {
      for (const plane of planes) {
        for (let component = 0; component < 4; component++) plane[component] *= -1;
      }
    }
    const x = cameraSpace[12] - lightPosition[0] + cameraSpace[8] * near;
    const y = cameraSpace[13] - lightPosition[1] + cameraSpace[9] * near;
    const z = cameraSpace[14] - lightPosition[2] + cameraSpace[10] * near;
    planes.push(new Float64Array([
      x, y, z, -(x * lightPosition[0] + y * lightPosition[1] + z * lightPosition[2]),
    ]));
  } else {
    const opposite = new Float64Array(planes[0]);
    for (let component = 0; component < 4; component++) opposite[component] *= -1;
    planes[0][3] += onEpsilon;
    opposite[3] -= onEpsilon;
    planes.push(opposite);
  }
  return planes;
}

function shadowCasterUsesZFail(bounds, planes) {
  const minimum = bounds?.minimum || bounds?.min;
  const maximum = bounds?.maximum || bounds?.max;
  // Animated/instanced jobs have the source's deliberately enormous bbox.
  // Unknown data therefore takes the safe z-fail path as well.
  if (!minimum?.length || !maximum?.length || !planes?.length) return true;
  for (const plane of planes) {
    if (!plane?.length || plane.length < 4 || !Number.isFinite(plane[0]) ||
        !Number.isFinite(plane[1]) || !Number.isFinite(plane[2]) ||
        !Number.isFinite(plane[3])) return true;
    const x = plane[0] > 0 ? maximum[0] : minimum[0];
    const y = plane[1] > 0 ? maximum[1] : minimum[1];
    const z = plane[2] > 0 ? maximum[2] : minimum[2];
    // sCullBBox uses this fixed source epsilon rather than normalized planes.
    if (plane[0] * x + plane[1] * y + plane[2] * z + plane[3] < -1e-6) return false;
  }
  return true;
}

function boundsOutsidePlanes(bounds, planes) {
  const minimum = bounds?.minimum || bounds?.min;
  const maximum = bounds?.maximum || bounds?.max;
  if (!minimum?.length || !maximum?.length || !planes?.length) return false;
  for (const plane of planes) {
    const length = Math.hypot(plane[0], plane[1], plane[2]);
    if (!(length > 1e-20)) continue;
    const x = plane[0] >= 0 ? maximum[0] : minimum[0];
    const y = plane[1] >= 0 ? maximum[1] : minimum[1];
    const z = plane[2] >= 0 ? maximum[2] : minimum[2];
    const distance = plane[0] * x + plane[1] * y + plane[2] * z + plane[3];
    const tolerance = 1e-6 * (1 + Math.abs(plane[0] * x) + Math.abs(plane[1] * y) +
      Math.abs(plane[2] * z) + Math.abs(plane[3]));
    if (distance < -tolerance) {
      return true;
    }
  }
  return false;
}

function shadowCasterMayAffectView(light, bounds, matrices, preparedPlanes = undefined) {
  if (light?.kind === 'directional') return true;
  const planes = preparedPlanes === undefined
    ? shadowViewFrustumPlanes(matrices?.viewProjection, light?.position) : preparedPlanes;
  return !planes || !boundsOutsidePlanes(bounds, planes);
}

function selectActiveLights(viewport, limit = MAX_LIGHTS, preparedFrustumPlanes = undefined) {
  const camera = viewport?.camera?.cameraSpace;
  const cameraPosition = camera?.length >= 15 && Number.isFinite(camera[12]) &&
    Number.isFinite(camera[13]) && Number.isFinite(camera[14])
    ? [camera[12], camera[13], camera[14]] : [0, 0, 0];
  let frustumPlanes = preparedFrustumPlanes;
  if (frustumPlanes === undefined) {
    frustumPlanes = null;
    if (viewport?.camera && camera?.length >= 16) {
      try {
        frustumPlanes = viewFrustumPlanes(cameraMatrices(viewport.camera).viewProjection);
      } catch {
        // Invalid embedder camera data disables this optimization rather than
        // risking a visible-light rejection.
      }
    }
  }
  let candidates = (viewport?.lightJobs || [])
    .filter(light => (light.amplify || 0) >= 1 / 255)
    .filter(light => lightIntersectsViewFrustum(light, frustumPlanes))
    .map((light, order) => {
      const position = light?.position;
      const dx = Number(position?.[0]) - cameraPosition[0];
      const dy = Number(position?.[1]) - cameraPosition[1];
      const dz = Number(position?.[2]) - cameraPosition[2];
      const distance = [dx, dy, dz].every(Number.isFinite) ? Math.hypot(dx, dy, dz) : Infinity;
      const range = Number(light?.range);
      const importance = Number.isFinite(range) ? range / Math.max(distance, 0.1)
        : range === Infinity ? Infinity : -Infinity;
      return { light, order, importance };
    })
    .sort((a, b) => (a.importance === b.importance
      ? a.order - b.order : (a.importance > b.importance ? -1 : 1)));
  // Engine::AddLightJob culls before Exec_IPP_Viewport tests the number of
  // lights and installs its default camera light. The port necessarily culls
  // here, after graph construction, so perform the same fallback at this point.
  // A viewport whose raw list was empty already contains this same object.
  if (candidates.length === 0 && viewport?.defaultLight) {
    candidates = [{ light: viewport.defaultLight, order: 0, importance: Infinity }];
  }
  return candidates
    .slice(0, limit)
    .map(entry => entry.light);
}

// Exec_Misc_Demo executes every active top-level KC_IPP branch in order and
// copies each result to the same master viewport. lastOutput is only the most
// recently executed branch (and may be cleared by a later inactive branch), so
// the ordered frame collection is authoritative whenever it is populated.
function resolveIPPOutputs(output, environment) {
  if (Array.isArray(output)) return output.filter(Boolean);
  const frameOutputs = environment?.frameOutputs;
  if (Array.isArray(frameOutputs) && frameOutputs.length) return frameOutputs.filter(Boolean);
  return output ? [output] : [];
}

function viewportClearFlags(viewport) {
  // All semantic viewport nodes carry flags. Retain the old handcrafted-node
  // default for tools and embedders that predate that field.
  return viewport?.flags === undefined ? 3 : (viewport.flags | 0) & 3;
}

// Match the 1280×640 release-video reference by keeping the authored
// production at 2:1 and fitting it inside the physical screen. Work in
// drawing-buffer pixels so the final copy remains 1:1 even when
// devicePixelRatio is greater than one.
function fitAspectRegion(width, height, aspect = 0) {
  width = Math.max(1, width | 0); height = Math.max(1, height | 0);
  aspect = Number(aspect);
  if (!(aspect > 0) || !Number.isFinite(aspect)) {
    return { x: 0, y: 0, width, height };
  }
  let innerWidth = width;
  let innerHeight = Math.max(1, Math.floor(width / aspect));
  if (innerHeight > height) {
    innerHeight = height;
  }
  // Keep the raster itself at the requested aspect. An odd outer width may
  // therefore leave one extra black column instead of subtly stretching the
  // production across a non-2:1 target.
  innerWidth = Math.max(1, Math.floor(innerHeight * aspect));
  innerWidth = Math.min(width, innerWidth);
  innerHeight = Math.min(height, innerHeight);
  return {
    x: Math.floor((width - innerWidth) / 2),
    y: Math.floor((height - innerHeight) / 2),
    width: innerWidth,
    height: innerHeight,
  };
}

function viewportRegion(viewport, width, height) {
  width = Math.max(1, width | 0); height = Math.max(1, height | 0);
  const crop = viewport?.crop;
  let x0 = 0, y0 = 0, x1 = width, y1 = height;
  if (crop?.length >= 4) {
    const fx0 = Number(crop[0]), fy0 = Number(crop[1]);
    const fx1 = Number(crop[2]), fy1 = Number(crop[3]);
    // The native viewport only accepts the crop when both axes have positive
    // extent. Its window coordinates are top-down; WebGL scissor coordinates
    // are bottom-up.
    if ([fx0, fy0, fx1, fy1].every(Number.isFinite) && fx0 < fx1 && fy0 < fy1) {
      x0 = Math.trunc(width * fx0);
      x1 = Math.trunc(width * fx1);
      const top = Math.trunc(height * fy0), bottom = Math.trunc(height * fy1);
      y0 = height - bottom;
      y1 = height - top;
    }
  }
  x0 = Math.max(0, Math.min(width, x0)); x1 = Math.max(0, Math.min(width, x1));
  y0 = Math.max(0, Math.min(height, y0)); y1 = Math.max(0, Math.min(height, y1));
  if (x1 < x0) x1 = x0;
  if (y1 < y0) y1 = y0;
  return {
    x: x0, y: y0, width: x1 - x0, height: y1 - y0,
    uvRect: new Float32Array([x0 / width, y0 / height, x1 / width, y1 / height]),
  };
}

// sSystem_::SetScissor maps an EngLight::LightRect from normalized D3D
// viewport coordinates into the current integer viewport. D3D window Y runs
// down from the top, while WebGL's scissor origin is at the bottom left. Keep
// the source's truncation at every edge: rounding outward would illuminate a
// one-pixel fringe the original never touched.
function lightScissorRegion(rectangle, viewport) {
  const integer = (value, fallback = 0) => {
    value = Number(value);
    return Number.isFinite(value) ? Math.trunc(value) : fallback;
  };
  const x = integer(viewport?.x);
  const y = integer(viewport?.y);
  const width = Math.max(0, integer(viewport?.width));
  const height = Math.max(0, integer(viewport?.height));
  const full = { x, y, width, height };
  if (!rectangle?.length || rectangle.length < 4) return full;
  const x0 = Number(rectangle[0]), y0 = Number(rectangle[1]);
  const x1 = Number(rectangle[2]), y1 = Number(rectangle[3]);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return full;

  const clampX = value => Math.max(0, Math.min(width, value));
  const clampY = value => Math.max(0, Math.min(height, value));
  const left = clampX(Math.trunc(width * (1 + x0) * 0.5));
  const right = clampX(Math.trunc(width * (1 + x1) * 0.5));
  const top = clampY(Math.trunc(height * (1 - y1) * 0.5));
  const bottom = clampY(Math.trunc(height * (1 - y0) * 0.5));
  return {
    x: x + left,
    y: y + height - bottom,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

// Exec_IPP_Layer2D feeds a TSpace3 quad in normalized overlay coordinates to
// an orthographic material pass. D3D depth 0..1 maps to WebGL clip depth
// -1..1; X/Y and the authored clockwise vertex order otherwise stay intact.
function layerQuadGeometry(layer, positions = new Float32Array(12), uvs = new Float32Array(8)) {
  const screen = layer?.screen || [0, 0, 1, 1];
  const texture = layer?.uv || [0, 0, 1, 1];
  const x0 = Number(screen[0]) * 2 - 1, x1 = Number(screen[2]) * 2 - 1;
  const y0 = 1 - Number(screen[1]) * 2, y1 = 1 - Number(screen[3]) * 2;
  const depth = Number.isFinite(Number(layer?.z)) ? Number(layer.z) * 2 - 1 : -1;
  positions.set([
    x0, y0, depth, x1, y0, depth,
    x1, y1, depth, x0, y1, depth,
  ]);
  uvs.set([
    texture[0], texture[1], texture[2], texture[1],
    texture[2], texture[3], texture[0], texture[3],
  ]);
  return { positions, uvs };
}

function sortRenderItems(items) {
  return items.sort((a, b) =>
    (a.renderPass - b.renderPass) ||
    (renderItemLightOrder(a) - renderItemLightOrder(b)) ||
    ((USAGE_ORDER[a.pass?.usage] ?? 99) - (USAGE_ORDER[b.pass?.usage] ?? 99)) ||
    ((a.sortMaterial ?? 1) - (b.sortMaterial ?? 1)) ||
    ((a.sortOrder ?? -(a.job?.opId ?? 0)) - (b.sortOrder ?? -(b.job?.opId ?? 0))));
}

// AddPaintJob links both native MeshJob and EffectJob lists at their heads.
// BuildPaintJobs consequently visits the completed lists backwards relative
// to scene traversal, while retaining the forward material/pass/light loops
// inside each visited job. Keep that distinction explicit: reversing one
// flattened item stream would incorrectly reverse the inner loops as well.
function forEachNativeHeadInsertedJob(jobs, callback) {
  for (let index = (jobs?.length || 0) - 1; index >= 0; index--) {
    callback(jobs[index], index);
  }
}

function effectRenderItem(job, sortOrder = 0) {
  const effect = job?.effect || {};
  return {
    kind: 'effect', effectJob: job,
    job: { opId: job?.op?.id ?? sortOrder },
    renderPass: Math.max(0, Math.min(0xff,
      (effect.pass | 0) + (job?.passAdjust | 0))),
    pass: { usage: effect.usage || 'other' },
    sortMaterial: 0,
    sortOrder,
  };
}

function appendMeshRenderItems(items, job, geometry, lights, matrices,
    modelMatrices = null, stats = null, shadowFrusta = null, shadowZFailVolumes = null,
    preparedViewPlanes = undefined) {
  // BuildPaintJobs gives animated and shader-instanced mesh jobs an enormous
  // bbox. They consequently bypass both the ordinary camera-frustum test and
  // every per-light spatial rejection in the released engine.
  const nativeUnboundedBounds = meshHasAnimation(job?.mesh) ||
    Boolean(job?.instances?.length);
  let worldBounds = null, worldBoundsReady = false, lightVisible = null;
  const ensureWorldBounds = () => {
    if (!worldBoundsReady) {
      worldBoundsReady = true;
      worldBounds = nativeUnboundedBounds
        ? null : meshJobWorldBounds(geometry, job, modelMatrices);
    }
    return worldBounds;
  };
  const viewPlanes = preparedViewPlanes === undefined
    ? viewFrustumPlanes(matrices?.viewProjection) : preparedViewPlanes;
  const ensureLightVisibility = () => {
    if (lightVisible) return;
    const bounds = ensureWorldBounds();
    lightVisible = lights.map(light => lightIntersectsWorldBounds(light, bounds));
  };
  const shadowGroups = [];
  let firstShadow = null;
  const count = key => { if (stats) stats[key] = (stats[key] || 0) + 1; };

  for (const group of geometry.groups) {
    // Native ordinary jobs use their material/cluster bbox, unlike the shared
    // shadow job below. Older embedders without prepared group bounds fall
    // back to the whole mesh, which can only retain excess work.
    const groupWorldBounds = nativeUnboundedBounds ? null : meshJobWorldBounds({
      bounds: group.bounds || geometry.bounds,
    }, job, modelMatrices);
    // Unknown/projective bounds remain visible. A false positive only costs a
    // draw, while rejecting them could remove authored geometry. The tolerance
    // in boundsOutsidePlanes additionally retains float-borderline boxes.
    const ordinaryVisible = nativeUnboundedBounds || !viewPlanes?.length ||
      !groupWorldBounds || !boundsOutsidePlanes(groupWorldBounds, viewPlanes);
    let groupLightVisible = null;
    const ensureGroupLightVisibility = () => {
      if (!groupLightVisible) {
        groupLightVisible = lights.map(light =>
          lightIntersectsWorldBounds(light, groupWorldBounds));
      }
    };
    const materialSlot = materialSlotFromGroup(geometry, group);
    const material = materialFromGroup(geometry, group);
    const passes = material?.passes?.length ? material.passes : [DEFAULT_RENDER_PASS];
    for (const pass of passes) {
      const sortMaterial = materialPassSortIdentity(material, pass);
      const materialPass = ((pass.renderPass | 0) + groupRenderPass(geometry, group)) & 0xff;
      const renderPass = Math.max(0, Math.min(0xff, materialPass + (job.passAdjust | 0)));
      if (pass.usage === 'shadow') {
        shadowGroups.push(group);
        firstShadow ||= { materialSlot, material, pass, renderPass, sortMaterial };
      } else if (pass.usage === 'light') {
        count('candidateViewItems');
        if (!ordinaryVisible) {
          count('culledViewItems');
          continue;
        }
        ensureGroupLightVisibility();
        for (let lightIndex = 0; lightIndex < lights.length; lightIndex++) {
          count('candidateLightItems');
          if (!groupLightVisible[lightIndex]) {
            count('culledLightItems');
            continue;
          }
          items.push({
            job, geometry, group, material, pass, renderPass,
            light: lights[lightIndex], lightIndex, sortMaterial,
          });
        }
      } else {
        count('candidateViewItems');
        if (!ordinaryVisible) {
          count('culledViewItems');
          continue;
        }
        items.push({ job, geometry, group, material, pass, renderPass, sortMaterial });
      }
    }
  }
  if (firstShadow && shadowGroups.length) {
    ensureLightVisibility();
    for (let lightIndex = 0; lightIndex < lights.length; lightIndex++) {
      if (!(lights[lightIndex].flags & 2)) continue;
      count('candidateShadowItems');
      const shadowVisible = lightVisible[lightIndex] && shadowCasterMayAffectView(
        lights[lightIndex], worldBounds, matrices,
        shadowFrusta ? shadowFrusta[lightIndex] : undefined,
      );
      if (!shadowVisible) {
        count(lightVisible[lightIndex] ? 'culledShadowFrustumItems' : 'culledShadowSphereItems');
        continue;
      }
      items.push({
        job, geometry, groups: shadowGroups,
        material: firstShadow.material, pass: firstShadow.pass,
        renderPass: firstShadow.renderPass,
        light: lights[lightIndex], lightIndex, shadowVolume: true,
        shadowZFail: shadowCasterUsesZFail(worldBounds,
          shadowZFailVolumes?.[lightIndex]),
        sortMaterial: firstShadow.sortMaterial,
      });
    }
  }
  return items;
}

function normalize3(x, y, z, fallback = [1, 0, 0]) {
  const length = Math.hypot(x, y, z);
  if (!(length > 1e-10)) return fallback.slice();
  return [x / length, y / length, z / length];
}

// Exec_Effect_ChainLine emits its marker-transformed Verlet points directly in
// world space. Its MODELVIEW query therefore supplies the world-space camera
// axis; the deferred effect-job matrix must not be applied to either the axis
// or the generated vertices a second time.
function buildChainRibbon(effect, matrices) {
  const points = effect?.points || [];
  const count = points.length;
  const positions = new Float32Array(count * 2 * 3);
  const uvs = new Float32Array(count * 2 * 2);
  const view = matrices?.view || mat4Identity();
  const viewAxis = normalize3(view[2], view[6], view[10], [0, 0, 1]);
  const halfThickness = Number(effect.thickness || 0) * 0.5;

  for (let index = 0; index < count; index++) {
    const before = points[Math.max(0, index - 1)];
    const after = points[Math.min(count - 1, index + 1)];
    const tangent = normalize3(
      before[0] - after[0], before[1] - after[1], before[2] - after[2],
    );
    // Native normalizes d and s, but deliberately leaves cross(d,s) at its
    // natural length before multiplying by half the authored width. Cables
    // viewed along their tangent consequently foreshorten instead of keeping
    // a constant screen-facing width.
    const side = [
      tangent[1] * viewAxis[2] - tangent[2] * viewAxis[1],
      tangent[2] * viewAxis[0] - tangent[0] * viewAxis[2],
      tangent[0] * viewAxis[1] - tangent[1] * viewAxis[0],
    ];
    const positionOffset = index * 6;
    positions[positionOffset] = points[index][0] + side[0] * halfThickness;
    positions[positionOffset + 1] = points[index][1] + side[1] * halfThickness;
    positions[positionOffset + 2] = points[index][2] + side[2] * halfThickness;
    positions[positionOffset + 3] = points[index][0] - side[0] * halfThickness;
    positions[positionOffset + 4] = points[index][1] - side[1] * halfThickness;
    positions[positionOffset + 5] = points[index][2] - side[2] * halfThickness;
    const uvOffset = index * 4;
    uvs[uvOffset] = uvs[uvOffset + 2] = index;
    uvs[uvOffset + 1] = 0; uvs[uvOffset + 3] = 1;
  }

  const indexValues = [];
  for (let index = 0; index < count - 1; index++) {
    if (index === (effect.ripped | 0)) continue;
    const vertex = index * 2;
    // sQuad(ip,i*2+3,i*2+2,i*2+0,i*2+1)
    indexValues.push(vertex + 3, vertex + 2, vertex, vertex + 3, vertex, vertex + 1);
  }
  return { positions, uvs, indices: new Uint16Array(indexValues) };
}

function weldKey(positions, index) {
  const offset = index * 3;
  const x = Object.is(positions[offset], -0) ? 0 : positions[offset];
  const y = Object.is(positions[offset + 1], -0) ? 0 : positions[offset + 1];
  const z = Object.is(positions[offset + 2], -0) ? 0 : positions[offset + 2];
  return `${x},${y},${z}`;
}

// The native shadow job is prepared once for the whole mesh, not once per
// material. `groups` therefore contains every range whose material has an
// MPP_SHADOW pass. Position welding recreates GenMesh's shared topology when
// UV/normal wedges use separate render vertices at the same point.
function prepareShadowTopology(geometry, groups = geometry.groups) {
  const positions = geometry.positions;
  const indices = geometry.indices;
  const canonical = [];
  const canonicalSources = [];
  const sourceToCanonical = new Map();
  const welded = new Map();
  let usesShadowVertexMap = Boolean(geometry.shadowVertexMap);
  const canonicalIndex = sourceIndex => {
    let result = sourceToCanonical.get(sourceIndex);
    if (result !== undefined) return result;
    const topologyIndex = geometry.shadowVertexMap?.[sourceIndex];
    if (topologyIndex === undefined) usesShadowVertexMap = false;
    const key = topologyIndex === undefined ? weldKey(positions, sourceIndex) : topologyIndex;
    result = welded.get(key);
    if (result === undefined) {
      result = canonical.length / 3;
      const offset = sourceIndex * 3;
      canonical.push(positions[offset], positions[offset + 1], positions[offset + 2]);
      canonicalSources.push(sourceIndex);
      welded.set(key, result);
    }
    sourceToCanonical.set(sourceIndex, result);
    return result;
  };
  const faces = [];
  for (const group of groups || []) {
    const end = Math.min(indices.length, group.start + group.count);
    for (let cursor = group.start; cursor + 2 < end; cursor += 3) {
      if (geometry.shadowTriangleMask && !geometry.shadowTriangleMask[cursor / 3]) continue;
      const a = canonicalIndex(indices[cursor]);
      const b = canonicalIndex(indices[cursor + 1]);
      const c = canonicalIndex(indices[cursor + 2]);
      if (a !== b && b !== c && c !== a) faces.push(a, b, c);
    }
  }
  const edges = new Map();
  const vertexCount = canonical.length / 3;
  const addEdge = (a, b, face) => {
    const min = Math.min(a, b), max = Math.max(a, b);
    const numericKey = min * vertexCount + max;
    const key = Number.isSafeInteger(numericKey) ? numericKey : `${min}:${max}`;
    let records = edges.get(key);
    if (!records) edges.set(key, records = []);
    records.push({ a, b, face });
  };
  for (let face = 0; face < faces.length / 3; face++) {
    const offset = face * 3, a = faces[offset], b = faces[offset + 1], c = faces[offset + 2];
    addEdge(a, b, face); addEdge(b, c, face); addEdge(c, a, face);
  }
  const edgeRecords = Array.from(edges.values());
  let boundaryEdges = 0, nonManifoldEdges = 0, windingConflictEdges = 0;
  let maxEdgeIncidence = 0;
  for (const records of edgeRecords) {
    maxEdgeIncidence = Math.max(maxEdgeIncidence, records.length);
    if (records.length === 1) boundaryEdges++;
    else if (records.length > 2) nonManifoldEdges++;
    else if (records[0].a === records[1].a && records[0].b === records[1].b) {
      // A consistently wound manifold traverses a shared edge in opposite
      // directions. Equal directions expose a local winding inversion that
      // can make the shadow volume open even though incidence is exactly two.
      windingConflictEdges++;
    }
  }
  const volumePositions = new Float32Array(canonical.length * 2);
  const extrusions = new Uint8Array(canonical.length / 3 * 2);
  for (let index = 0; index < canonical.length / 3; index++) {
    const source = index * 3, target = index * 6;
    const x = canonical[source], y = canonical[source + 1], z = canonical[source + 2];
    volumePositions[target] = volumePositions[target + 3] = x;
    volumePositions[target + 1] = volumePositions[target + 4] = y;
    volumePositions[target + 2] = volumePositions[target + 5] = z;
    extrusions[index * 2 + 1] = 1;
  }
  return {
    positions: new Float32Array(canonical),
    sourceIndices: new Uint32Array(canonicalSources),
    usesShadowVertexMap,
    faces: new Uint32Array(faces),
    edges: edgeRecords,
    boundaryEdges,
    nonManifoldEdges,
    windingConflictEdges,
    maxEdgeIncidence,
    volumePositions,
    extrusions,
  };
}

function refreshShadowTopologyPositions(topology, positions) {
  const sourceIndices = topology?.sourceIndices;
  const canonical = topology?.positions;
  const volume = topology?.volumePositions;
  if (!sourceIndices || canonical?.length !== sourceIndices.length * 3 ||
      volume?.length !== sourceIndices.length * 6) return false;
  for (let index = 0; index < sourceIndices.length; index++) {
    const sourceIndex = sourceIndices[index];
    const source = sourceIndex * 3, target = index * 3, volumeTarget = index * 6;
    if (sourceIndex >= positions.length / 3) return false;
    const x = positions[source], y = positions[source + 1], z = positions[source + 2];
    canonical[target] = x; canonical[target + 1] = y; canonical[target + 2] = z;
    volume[volumeTarget] = volume[volumeTarget + 3] = x;
    volume[volumeTarget + 1] = volume[volumeTarget + 4] = y;
    volume[volumeTarget + 2] = volume[volumeTarget + 5] = z;
  }
  return true;
}

const SHADOW_GROUP_STRUCTURE_FIELDS = Object.freeze([
  'material', 'materialIndex', 'cluster', 'pass', 'renderPass', 'start', 'count',
]);

function shadowGroupStructureMatches(previous, next) {
  if (!Array.isArray(previous) || !Array.isArray(next) || previous.length !== next.length) {
    return false;
  }
  for (let index = 0; index < previous.length; index++) {
    for (const field of SHADOW_GROUP_STRUCTURE_FIELDS) {
      if (previous[index]?.[field] !== next[index]?.[field]) return false;
    }
  }
  return true;
}

function refreshedShadowTopologies(entry, normalized, dirty) {
  const previous = entry.shadowTopologies;
  if (!(previous instanceof Map) || !dirty || dirty.has('indices') ||
      entry.indices !== normalized.indices ||
      entry.shadowVertexMap !== normalized.shadowVertexMap ||
      entry.shadowTriangleMask !== normalized.shadowTriangleMask ||
      !shadowGroupStructureMatches(entry.groups, normalized.groups)) return new Map();
  const positionsChanged = dirty.has('positions') || entry.positions !== normalized.positions;
  if (!positionsChanged) return previous;
  // Position-derived welding can split or merge as vertices move. Only an
  // authored stable shadow map proves that the cached face/edge structure is
  // still valid for a moving geometry source.
  for (const topology of previous.values()) {
    if (!topology.usesShadowVertexMap ||
        !refreshShadowTopologyPositions(topology, normalized.positions)) return new Map();
  }
  return previous;
}

function shadowScratchArray(scratch, key, Type, length) {
  if (!scratch) return new Type(length);
  const requiredBytes = length * Type.BYTES_PER_ELEMENT;
  if (requiredBytes > MAX_RETAINED_SHADOW_SCRATCH_BYTES) return new Type(length);
  let storage = scratch[key];
  if (!(storage instanceof Type) || storage.length < length || storage.length > Math.max(1, length) * 4) {
    const maximum = Math.floor(MAX_RETAINED_SHADOW_SCRATCH_BYTES / Type.BYTES_PER_ELEMENT);
    const capacity = length ? Math.min(maximum, 2 ** Math.ceil(Math.log2(length))) : 0;
    storage = new Type(capacity);
    scratch[key] = storage;
  }
  return storage.subarray(0, length);
}

function buildShadowVolume(geometry, groups, light, model = mat4Identity(), includeCaps = true,
    scratch = null) {
  const topology = groups?.faces && groups?.edges ? groups : prepareShadowTopology(geometry, groups);
  const vertexCount = topology.positions.length / 3;
  const positions = topology.positions;
  const worldLightPosition = light?.position || light || [0, 0, 0];
  const lightPosition = shadowScratchArray(scratch, 'lightPosition', Float32Array, 3);
  // Engine_::RenderPaintJobs and sMaterial11::Set both call TransR before the
  // shadow job. TransR is not an affine inverse: it keeps authored object
  // scale in c4, and both plane selection and extrusion consume that value.
  legacyTransRVector(model, worldLightPosition, true, lightPosition);
  const faceFront = shadowScratchArray(scratch, 'faceFront', Uint8Array,
    topology.faces.length / 3);
  for (let face = 0; face < faceFront.length; face++) {
    const offset = face * 3;
    const ia = topology.faces[offset] * 3;
    const ib = topology.faces[offset + 1] * 3;
    const ic = topology.faces[offset + 2] * 3;
    const abx = positions[ib] - positions[ia];
    const aby = positions[ib + 1] - positions[ia + 1];
    const abz = positions[ib + 2] - positions[ia + 2];
    const acx = positions[ic] - positions[ia];
    const acy = positions[ic + 1] - positions[ia + 1];
    const acz = positions[ic + 2] - positions[ia + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    faceFront[face] = nx * (lightPosition[0] - positions[ia]) +
      ny * (lightPosition[1] - positions[ia + 1]) +
      nz * (lightPosition[2] - positions[ia + 2]) >= 0 ? 1 : 0;
  }

  const IndexType = vertexCount * 2 > 0xffff ? Uint32Array : Uint16Array;
  const maximumIndexCount = topology.edges.length * 6 +
    (includeCaps ? faceFront.length * 3 : 0);
  const indexValues = shadowScratchArray(scratch, 'indices', IndexType, maximumIndexCount);
  let indexCount = 0;
  for (const records of topology.edges) {
    const first = records[0];
    const firstFront = Boolean(faceFront[first.face]);
    const second = records[1] || null;
    const secondFront = second ? Boolean(faceFront[second.face]) : true;
    if (firstFront === secondFront) continue;
    let a, b;
    if (firstFront) { a = first.a; b = first.b; }
    else if (second) { a = second.a; b = second.b; }
    else { a = first.b; b = first.a; }
    a *= 2; b *= 2;
    indexValues[indexCount++] = a; indexValues[indexCount++] = b;
    indexValues[indexCount++] = b + 1; indexValues[indexCount++] = a;
    indexValues[indexCount++] = b + 1; indexValues[indexCount++] = a + 1;
  }
  const silhouetteIndexCount = indexCount;
  if (includeCaps) {
    for (let face = 0; face < faceFront.length; face++) {
      const offset = face * 3, side = faceFront[face];
      indexValues[indexCount++] = topology.faces[offset] * 2 + side;
      indexValues[indexCount++] = topology.faces[offset + 1] * 2 + side;
      indexValues[indexCount++] = topology.faces[offset + 2] * 2 + side;
    }
  }
  return {
    positions: topology.volumePositions,
    extrusions: topology.extrusions,
    indices: indexValues.subarray(0, indexCount),
    silhouetteIndexCount,
    capIndexCount: indexCount - silhouetteIndexCount,
    faceFront,
    lightPosition,
    topology,
  };
}

class GeometryCache {
  constructor(gl, instanceBuffer, options = {}) {
    this.gl = gl;
    this.instanceBuffer = instanceBuffer;
    this.entries = new WeakMap();
    this.animatedPools = new WeakMap();
    this.allEntries = new Set();
    this.frame = 0;
    this.diagnostics = Boolean(options.diagnostics);
  }

  beginFrame() {
    this.frame++;
  }

  updateEntry(entry, normalized, dynamic = false) {
    const gl = this.gl;
    const names = ['positions', 'normals', 'uvs', 'colors', 'tangents', 'uv1', 'indices'];
    const values = [normalized.positions, normalized.normals, normalized.uvs,
      normalized.colors, normalized.tangents, normalized.uv1, normalized.indices];
    const declared = normalized.dynamicAttributes;
    const partial = Array.isArray(declared);
    const dirty = partial ? new Set(declared) : null;
    // UV1 aliases UV0 on legacy meshes and Water. They live in separate GPU
    // buffers, so a changed shared source has to refresh both bindings.
    if (dirty?.has('uvs') && normalized.uv1 === normalized.uvs) dirty.add('uv1');
    // A producer cannot safely leave a buffer stale if its representation or
    // size changed, even when it accidentally omitted that field from its
    // dirty list.
    if (dirty) for (let index = 0; index < values.length; index++) {
      const previous = entry[names[index]], next = values[index];
      if (previous?.byteLength !== next?.byteLength || previous?.constructor !== next?.constructor) {
        dirty.add(names[index]);
      }
    }
    const shadowTopologies = refreshedShadowTopologies(entry, normalized, dirty);
    // ELEMENT_ARRAY_BUFFER is VAO state. Bind the entry while refreshing it so
    // an animated/effect mesh cannot replace the index buffer of the mesh that
    // happened to be drawn immediately before it.
    gl.bindVertexArray(entry.vao);
    for (let index = 0; index < values.length; index++) {
      if (dirty && !dirty.has(names[index])) continue;
      gl.bindBuffer(index === 6 ? gl.ELEMENT_ARRAY_BUFFER : gl.ARRAY_BUFFER, entry.buffers[index]);
      gl.bufferData(index === 6 ? gl.ELEMENT_ARRAY_BUFFER : gl.ARRAY_BUFFER, values[index],
        dynamic || partial ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
      // Animated producers normally retain their color representation, but a
      // Uint8/Float32 transition changes both the attribute type and normalize
      // bit and therefore must update the VAO pointer as well as its storage.
      if (index === 3) {
        const packed = values[index] instanceof Uint8Array;
        gl.vertexAttribPointer(3, 4, packed ? gl.UNSIGNED_BYTE : gl.FLOAT, packed, 0, 0);
      }
    }
    gl.bindVertexArray(null);
    Object.assign(entry, normalized, {
      dynamic: entry.dynamic || dynamic || partial,
      uploadBytes: values.reduce((sum, value) => sum + (value?.byteLength || 0), 0),
      version: normalized.source.version,
      shadowTopologies,
      indexType: normalized.indices instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT,
      indexBytes: normalized.indices instanceof Uint16Array ? 2 : 4,
    });
    if (this.diagnostics) entry.diagnostics = geometryTopologyStats(normalized);
    return entry;
  }

  createEntry(normalized, dynamic = false) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    const buffers = [];
    gl.bindVertexArray(vao);
    const dynamicAttributes = Array.isArray(normalized.dynamicAttributes)
      ? new Set(normalized.dynamicAttributes) : null;
    if (dynamicAttributes?.has('uvs') && normalized.uv1 === normalized.uvs) {
      dynamicAttributes.add('uv1');
    }
    const attribute = (name, location, size, values, normalizedAttribute = false) => {
      const buffer = gl.createBuffer(); buffers.push(buffer);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, values,
        dynamic || dynamicAttributes?.has(name) ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, values instanceof Uint8Array ? gl.UNSIGNED_BYTE : gl.FLOAT,
        normalizedAttribute, 0, 0);
    };
    attribute('positions', 0, 3, normalized.positions);
    attribute('normals', 1, 3, normalized.normals);
    attribute('uvs', 2, 2, normalized.uvs);
    attribute('colors', 3, 4, normalized.colors, normalized.colors instanceof Uint8Array);
    attribute('tangents', 4, 4, normalized.tangents);
    attribute('uv1', 9, 2, normalized.uv1);
    const indexBuffer = gl.createBuffer(); buffers.push(indexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, normalized.indices,
      dynamic || dynamicAttributes?.has('indices') ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    for (let column = 0; column < 4; column++) {
      const location = 5 + column;
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 4, gl.FLOAT, false, 64, column * 16);
      gl.vertexAttribDivisor(location, 1);
    }
    gl.bindVertexArray(null);
    const entry = {
      ...normalized, vao, buffers, dynamic: dynamic || Boolean(dynamicAttributes),
      uploadBytes: [normalized.positions, normalized.normals, normalized.uvs,
        normalized.colors, normalized.tangents, normalized.uv1, normalized.indices]
        .reduce((sum, value) => sum + (value?.byteLength || 0), 0),
      version: normalized.source.version,
      shadowTopologies: new Map(),
      indexType: normalized.indices instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT,
      indexBytes: normalized.indices instanceof Uint16Array ? 2 : 4,
    };
    if (this.diagnostics) entry.diagnostics = geometryTopologyStats(normalized);
    this.allEntries.add(entry);
    return entry;
  }

  resourceStats() {
    let gpuBytes = 0, cpuReferencedBytes = 0, shadowBytes = 0;
    let vertices = 0, triangles = 0, animatedEntries = 0, shadowTopologies = 0;
    const diagnostics = {
      triangleCount: 0, auditedTriangles: 0, unauditedTriangles: 0,
      degenerateTriangles: 0, duplicateTriangles: 0,
      oppositeDuplicateTriangles: 0, sameOrientationDuplicateTriangles: 0,
      exactDuplicateTriangles: 0, exactOppositeDuplicateTriangles: 0,
      exactSameOrientationDuplicateTriangles: 0,
      nearOnlyDuplicateTriangles: 0, nearOnlySameOrientationDuplicateTriangles: 0,
      exactSameGroupIdenticalAttributeTriangles: 0,
      exactSameGroupAttributeVariantTriangles: 0,
      exactCrossGroupSameMaterialTriangles: 0, exactCrossMaterialTriangles: 0,
      exactDegenerateSameOrientationTriangles: 0,
      normalAlignedWinding: 0, normalOpposedWinding: 0, unexpectedWinding: 0,
      alignedWinding: 0, reversedWinding: 0,
      indeterminateWinding: 0, truncatedEntries: 0,
      shadowBoundaryEdges: 0, shadowNonManifoldEdges: 0,
      shadowWindingConflictEdges: 0, shadowMaxEdgeIncidence: 0,
      offenders: [],
    };
    const countBytes = value => value?.byteLength || 0;
    for (const entry of this.allEntries) {
      gpuBytes += entry.uploadBytes || 0;
      cpuReferencedBytes += entry.uploadBytes || 0;
      vertices += entry.vertexCount || 0;
      triangles += (entry.indices?.length || 0) / 3;
      if (entry.dynamic) animatedEntries++;
      if (entry.diagnostics) {
        diagnostics.triangleCount += entry.diagnostics.triangleCount || 0;
        for (const key of ['auditedTriangles', 'degenerateTriangles', 'duplicateTriangles',
          'oppositeDuplicateTriangles', 'sameOrientationDuplicateTriangles',
          'exactDuplicateTriangles', 'exactOppositeDuplicateTriangles',
          'exactSameOrientationDuplicateTriangles', 'nearOnlyDuplicateTriangles',
          'nearOnlySameOrientationDuplicateTriangles',
          'exactSameGroupIdenticalAttributeTriangles',
          'exactSameGroupAttributeVariantTriangles',
          'exactCrossGroupSameMaterialTriangles', 'exactCrossMaterialTriangles',
          'exactDegenerateSameOrientationTriangles',
          'normalAlignedWinding', 'normalOpposedWinding',
          'unexpectedWinding', 'alignedWinding', 'reversedWinding',
          'indeterminateWinding']) diagnostics[key] += entry.diagnostics[key] || 0;
        if (entry.diagnostics.truncated) diagnostics.truncatedEntries++;
      }
      for (const topology of entry.shadowTopologies?.values?.() || []) {
        shadowTopologies++;
        const bytes = countBytes(topology.positions) + countBytes(topology.faces) +
          countBytes(topology.sourceIndices) + countBytes(topology.volumePositions) +
          countBytes(topology.extrusions);
        shadowBytes += bytes;
        diagnostics.shadowBoundaryEdges += topology.boundaryEdges || 0;
        diagnostics.shadowNonManifoldEdges += topology.nonManifoldEdges || 0;
        diagnostics.shadowWindingConflictEdges += topology.windingConflictEdges || 0;
        diagnostics.shadowMaxEdgeIncidence = Math.max(
          diagnostics.shadowMaxEdgeIncidence, topology.maxEdgeIncidence || 0);
      }
      if (entry.diagnostics) {
        let shadowBoundaryEdges = 0, shadowNonManifoldEdges = 0;
        let shadowWindingConflictEdges = 0, shadowMaxEdgeIncidence = 0;
        for (const topology of entry.shadowTopologies?.values?.() || []) {
          shadowBoundaryEdges += topology.boundaryEdges || 0;
          shadowNonManifoldEdges += topology.nonManifoldEdges || 0;
          shadowWindingConflictEdges += topology.windingConflictEdges || 0;
          shadowMaxEdgeIncidence = Math.max(
            shadowMaxEdgeIncidence, topology.maxEdgeIncidence || 0);
        }
        const issueCount = (entry.diagnostics.degenerateTriangles || 0) +
          (entry.diagnostics.duplicateTriangles || 0) +
          (entry.diagnostics.unexpectedWinding || 0) +
          shadowBoundaryEdges + shadowNonManifoldEdges + shadowWindingConflictEdges;
        if (issueCount) diagnostics.offenders.push({
          sourceIds: Array.from(entry.sourceIds || []).sort((a, b) => a - b),
          sourceKind: entry.source?.sourceKind || entry.source?.kind || 'geometry',
          vertices: entry.vertexCount || 0,
          triangles: (entry.indices?.length || 0) / 3,
          degenerateTriangles: entry.diagnostics.degenerateTriangles || 0,
          duplicateTriangles: entry.diagnostics.duplicateTriangles || 0,
          oppositeDuplicateTriangles: entry.diagnostics.oppositeDuplicateTriangles || 0,
          sameOrientationDuplicateTriangles:
            entry.diagnostics.sameOrientationDuplicateTriangles || 0,
          exactDuplicateTriangles: entry.diagnostics.exactDuplicateTriangles || 0,
          exactOppositeDuplicateTriangles: entry.diagnostics.exactOppositeDuplicateTriangles || 0,
          exactSameOrientationDuplicateTriangles:
            entry.diagnostics.exactSameOrientationDuplicateTriangles || 0,
          nearOnlyDuplicateTriangles: entry.diagnostics.nearOnlyDuplicateTriangles || 0,
          nearOnlySameOrientationDuplicateTriangles:
            entry.diagnostics.nearOnlySameOrientationDuplicateTriangles || 0,
          exactSameGroupIdenticalAttributeTriangles:
            entry.diagnostics.exactSameGroupIdenticalAttributeTriangles || 0,
          exactSameGroupAttributeVariantTriangles:
            entry.diagnostics.exactSameGroupAttributeVariantTriangles || 0,
          exactCrossGroupSameMaterialTriangles:
            entry.diagnostics.exactCrossGroupSameMaterialTriangles || 0,
          exactCrossMaterialTriangles: entry.diagnostics.exactCrossMaterialTriangles || 0,
          exactDegenerateSameOrientationTriangles:
            entry.diagnostics.exactDegenerateSameOrientationTriangles || 0,
          unexpectedWinding: entry.diagnostics.unexpectedWinding || 0,
          normalAlignedWinding: entry.diagnostics.normalAlignedWinding || 0,
          normalOpposedWinding: entry.diagnostics.normalOpposedWinding || 0,
          indeterminateWinding: entry.diagnostics.indeterminateWinding || 0,
          shadowBoundaryEdges,
          shadowNonManifoldEdges,
          shadowWindingConflictEdges,
          shadowMaxEdgeIncidence,
        });
      }
    }
    diagnostics.offenders.sort((a, b) =>
      (b.shadowNonManifoldEdges - a.shadowNonManifoldEdges) ||
      (b.shadowWindingConflictEdges - a.shadowWindingConflictEdges) ||
      (b.exactSameGroupIdenticalAttributeTriangles -
        a.exactSameGroupIdenticalAttributeTriangles) ||
      (b.sameOrientationDuplicateTriangles - a.sameOrientationDuplicateTriangles) ||
      (b.unexpectedWinding - a.unexpectedWinding) ||
      (b.oppositeDuplicateTriangles - a.oppositeDuplicateTriangles) ||
      (b.duplicateTriangles - a.duplicateTriangles) ||
      ((a.sourceIds[0] ?? Infinity) - (b.sourceIds[0] ?? Infinity)));
    diagnostics.offenderCount = diagnostics.offenders.length;
    diagnostics.offenders.length = Math.min(diagnostics.offenders.length, 24);
    diagnostics.unauditedTriangles = Math.max(0,
      diagnostics.triangleCount - diagnostics.auditedTriangles);
    return {
      entries: this.allEntries.size, animatedEntries, vertices, triangles,
      shadowTopologies, shadowBytes,
      gpuBytes, cpuReferencedBytes,
      diagnostics,
    };
  }

  tagSource(entry, sourceId) {
    if (Number.isSafeInteger(sourceId)) {
      entry.sourceIds ||= new Set();
      if (entry.sourceIds.size < 16) entry.sourceIds.add(sourceId);
    }
    return entry;
  }

  get(mesh, time, sourceId = null) {
    if (time !== undefined && meshHasAnimation(mesh)) {
      let pool = this.animatedPools.get(mesh);
      if (!pool) {
        pool = { frame: -1, used: 0, byTime: new Map(), entries: [] };
        this.animatedPools.set(mesh, pool);
      }
      if (pool.frame !== this.frame) {
        pool.frame = this.frame;
        pool.used = 0;
        pool.byTime.clear();
      }
      const key = Number(time) || 0;
      const cached = pool.byTime.get(key);
      if (cached) return this.tagSource(cached, sourceId);
      // Match MinMesh's bounded output scratch to the renderer's bounded VAO
      // pool. Distinct animation times in one viewport keep separate CPU
      // arrays until their later sorted shadow/material passes have consumed
      // them, while the same slots are reused on the next frame.
      const normalized = normalizePreparedGeometry(mesh, {
        time: key, animationSlot: pool.used,
      });
      let entry = pool.entries[pool.used];
      if (entry) this.updateEntry(entry, normalized, true);
      else {
        entry = this.createEntry(normalized, true);
        pool.entries.push(entry);
      }
      pool.used++;
      pool.byTime.set(key, entry);
      return this.tagSource(entry, sourceId);
    }

    let entry = this.entries.get(mesh);
    // Static Mesh/MinMesh producers expose their immutable prepared buffer on
    // _prepared, while direct geometry producers (for example Water) are
    // their own source.  When both that source identity and its generation
    // still match, normalization cannot change the cached entry.  Check this
    // before normalizePreparedGeometry: besides prepare(), normalization maps
    // groups and rebuilds several small wrapper/bounds arrays on every draw.
    const preparedSource = typeof mesh?.prepare === 'function' ? mesh._prepared : mesh;
    if (entry && preparedSource && entry.source === preparedSource &&
        entry.version === preparedSource.version) {
      return this.tagSource(entry, sourceId);
    }
    const normalized = normalizePreparedGeometry(mesh);
    const version = normalized.source.version;
    if (entry && entry.source === normalized.source && entry.version === version) {
      return this.tagSource(entry, sourceId);
    }
    if (entry && entry.source === normalized.source) {
      return this.tagSource(this.updateEntry(entry, normalized), sourceId);
    }
    entry = this.createEntry(normalized);
    this.entries.set(mesh, entry);
    return this.tagSource(entry, sourceId);
  }

  dispose() {
    const gl = this.gl;
    for (const entry of this.allEntries) {
      for (const buffer of entry.buffers) gl.deleteBuffer(buffer);
      gl.deleteVertexArray(entry.vao);
    }
    this.allEntries.clear();
    this.entries = new WeakMap();
    this.animatedPools = new WeakMap();
  }
}

// Material11's PS1.1 light path used generated lookup textures instead of
// arithmetic normalize/attenuation instructions. Keep these resources outside
// TextureCache: they are fixed renderer infrastructure, not KX bitmap nodes,
// and have cube/3D targets with source-specific sampler state.
class Material11LookupTextures {
  constructor(gl) {
    this.gl = gl;
    this.normalizerCube = gl.createTexture();
    this.attenuationVolume = gl.createTexture();

    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (gl.UNPACK_FLIP_Y_WEBGL !== undefined) {
      // Typed rows from MakeCubeNormalizer must remain unflipped. WebGL's cube
      // face t convention then maps the native top row to the same direction.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    }

    const cube = makeLegacyCubeNormalizer();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.normalizerCube);
    for (let face = 0; face < 6; face++) {
      gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, gl.RGBA8,
        LEGACY_NORMALIZER_CUBE_SIZE, LEGACY_NORMALIZER_CUBE_SIZE, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, legacyCubeNormalizerFace(cube, face));
    }
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // material11.cpp sets WRAP on the cube normalizers' U/V sampler axes.
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.REPEAT);
    if (gl.TEXTURE_BASE_LEVEL !== undefined) {
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_BASE_LEVEL, 0);
    }
    if (gl.TEXTURE_MAX_LEVEL !== undefined) {
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAX_LEVEL, 0);
    }

    const volume = makeLegacyAttenuationVolume();
    gl.bindTexture(gl.TEXTURE_3D, this.attenuationVolume);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8,
      LEGACY_ATTENUATION_VOLUME_SIZE, LEGACY_ATTENUATION_VOLUME_SIZE,
      LEGACY_ATTENUATION_VOLUME_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, volume);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    if (gl.TEXTURE_BASE_LEVEL !== undefined) {
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_BASE_LEVEL, 0);
    }
    if (gl.TEXTURE_MAX_LEVEL !== undefined) {
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAX_LEVEL, 0);
    }
  }

  bind(normalizerUnit, attenuationUnit) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + normalizerUnit);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.normalizerCube);
    gl.activeTexture(gl.TEXTURE0 + attenuationUnit);
    gl.bindTexture(gl.TEXTURE_3D, this.attenuationVolume);
  }

  resourceStats() {
    const normalizerCubeBytes = this.normalizerCube
      ? LEGACY_NORMALIZER_CUBE_BYTE_LENGTH : 0;
    const attenuationVolumeBytes = this.attenuationVolume
      ? LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH : 0;
    return {
      textures: Number(Boolean(this.normalizerCube)) +
        Number(Boolean(this.attenuationVolume)),
      normalizerCubeBytes,
      attenuationVolumeBytes,
      estimatedBytes: normalizerCubeBytes + attenuationVolumeBytes,
    };
  }

  dispose() {
    if (this.normalizerCube) this.gl.deleteTexture(this.normalizerCube);
    if (this.attenuationVolume) this.gl.deleteTexture(this.attenuationVolume);
    this.normalizerCube = null;
    this.attenuationVolume = null;
  }
}

class Renderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.gl = options.gl || createWebGL2(canvas, options.contextAttributes);
    this.effectExecutor = options.effectExecutor ?? executeEffectJob;
    const gl = this.gl;
    this.program = linkProgram(gl, VERTEX_SOURCE, FRAGMENT_SOURCE, 'Debris material');
    this.uniforms = uniformLocations(gl, this.program, [
      'uViewProjection', 'uUVTransform', 'uTexture0', 'uTexture1', 'uTexture2', 'uTexture3',
      'uPrelightTexture', 'uM11NormalizerCube', 'uM11AttenuationVolume',
      'uTextureMask',
      'uMode', 'uVertexColorMode', 'uUsePrelight',
      'uMaterial20', 'uM20Flags', 'uM20EnvironmentFlags',
      'uM20RuntimeEnvironmentFlags',
      'uM20SamplerFlags', 'uM20SamplerScales',
      'uM20UVTransform1[0]', 'uM20UVTransform2[0]',
      'uMaterial11', 'uM11MultipassLight', 'uM11View', 'uM11WorldToModel',
      'uM11ModelToWorld', 'uM11SpecialFlags',
      'uLegacyLightingMode', 'uConditionLegacyBasis',
      'uM11SamplerFlags', 'uM11SamplerScales',
      'uM11UVTransform1[0]', 'uM11UVTransform2[0]',
      'uDetailOps', 'uCombiners[0]', 'uAlphaCombiner', 'uMaterialColors[0]',
      'uBaseColor', 'uAmbient', 'uCameraPosition', 'uMaterialCameraPosition', 'uFogColor',
      'uFogRange', 'uFogEnabled', 'uAlphaCutoff', 'uSpecularPower', 'uSpecularColor',
      'uSpecularStrength', 'uLightCount',
      'uLightPosition[0]', 'uLightAttenuation[0]', 'uLightColor[0]', 'uLightSpecular[0]',
      'uM11LightConstant',
    ]);
    this.shadowProgram = linkProgram(gl, SHADOW_VERTEX_SOURCE, SHADOW_FRAGMENT_SOURCE, 'Debris shadow volume');
    this.shadowUniforms = uniformLocations(gl, this.shadowProgram, [
      'uViewProjection', 'uModel', 'uLightPosition',
    ]);
    this.shadowVAO = gl.createVertexArray();
    this.shadowPositionBuffer = gl.createBuffer();
    this.shadowExtrusionBuffer = gl.createBuffer();
    this.shadowIndexBuffer = gl.createBuffer();
    this.fullscreenProgram = linkProgram(gl, FULLSCREEN_VERTEX_SOURCE, FULLSCREEN_FRAGMENT_SOURCE, 'Debris IPP');
    this.fullscreenUniforms = uniformLocations(gl, this.fullscreenProgram, [
      'uImage', 'uMaterialTexture', 'uMode', 'uColor0', 'uColor1', 'uParameters',
      'uColorCorrect[0]', 'uTexel', 'uUVRect',
    ]);
    this.fullscreenVAO = gl.createVertexArray();
    this.instanceBuffer = gl.createBuffer();
    this.layerVAO = null;
    this.layerPositionBuffer = null;
    this.layerUVBuffer = null;
    this.layerIndexBuffer = null;
    this.layerPositions = new Float32Array(12);
    this.layerUVs = new Float32Array(8);
    const layerIdentity = mat4Identity();
    this.layerMatrices = { view: layerIdentity, projection: layerIdentity, viewProjection: layerIdentity };
    this.layerViewport = {
      ambientLight: 0,
      camera: {
        cameraSpace: layerIdentity, fogColor: 0xff808080,
        fogStart: 0, fogEnd: 4096,
      },
      lightJobs: [],
    };
    this.geometry = new GeometryCache(gl, this.instanceBuffer, { diagnostics: options.diagnostics });
    this.textures = new TextureCache(gl, { dxt5Mode: options.dxt5Mode });
    this.material11Lookups = new Material11LookupTextures(gl);
    // Samplers of different dimensionality cannot alias the same unit in a
    // linked WebGL program, even when a dynamic material branch does not read
    // them. Reserve two units above the four authored maps and prelight copy.
    gl.useProgram(this.program);
    this.material11Lookups.bind(M11_NORMALIZER_TEXTURE_UNIT,
      M11_ATTENUATION_TEXTURE_UNIT);
    gl.uniform1i(this.uniforms.uM11NormalizerCube, M11_NORMALIZER_TEXTURE_UNIT);
    gl.uniform1i(this.uniforms.uM11AttenuationVolume, M11_ATTENUATION_TEXTURE_UNIT);
    // Native IPP copies every root into a persistent master viewport. Keep it
    // outside the depth-indexed scratch pool so nested Layer2D graphs and post
    // ping-pong targets can never alias the sequence destination.
    this.masterTarget = new RenderTarget(gl, 1, 1);
    // Material20's REALMAT path grabs the completed texture/prelight phase
    // and samples it from every subsequent per-light pass.
    this.prelightTarget = new RenderTarget(gl, 1, 1, { depth: false });
    this.currentPrelightTexture = null;
    this.targets = [];
    // Glare's two fixed native levels plus one same-size ping-pong surface are
    // allocated lazily, so frames without the effect pay no memory cost.
    this.glareTargets = [];
    this.drawCalls = 0;
    this.triangles = 0;
    this.cullingStats = {};
    this.instanceMatrixCache = new WeakMap();
    this.instanceUploadCache = new WeakMap();
    this.instanceUploadRecords = [];
    this.instanceUploadUsedBytes = 0;
    this.instanceBufferCapacity = 0;
    this.instanceUploadActive = false;
    this.shadowVolumeScratch = {};
    this.materialViews = new WeakMap();
    this.defaultMaterialViews = new Map();
    this.materialScratch = {
      baseColor: new Float32Array(4), color: new Float32Array(4),
      materialColors: new Float32Array(16),
      specularColor: new Float32Array(3), ambient: new Float32Array(3),
      materialCameraPosition: new Float32Array(3),
      fogColor: new Float32Array(3), lightColor: new Float32Array(3),
      lightPositions: new Float32Array(MAX_LIGHTS * 4),
      lightAttenuation: new Float32Array(MAX_LIGHTS * 4),
      lightColors: new Float32Array(MAX_LIGHTS * 4),
      lightSpecular: new Float32Array(MAX_LIGHTS),
      m11LightConstant: new Float32Array(4),
      selectedTextures: new Array(4), singleLight: new Array(1),
      m20SamplerFlags: new Int32Array(4), m20SamplerScales: new Float32Array(4),
      m11SamplerFlags: new Int32Array(4), m11SamplerScales: new Float32Array(4),
      identityUVTransform: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]),
      identityMatrix: mat4Identity(), m11WorldToModel: new Float32Array(16),
      emptyCombiners: new Int32Array(13), emptyDetailOps: new Int32Array(2),
    };
    this.pixelRatio = options.pixelRatio || 0;
    this.maxPixelRatio = options.maxPixelRatio || 2;
    this.shadows = options.shadows !== false;
    this.width = this.height = 1;
    this.canvasWidth = this.canvasHeight = 1;
    this.presentationRegion = { x: 0, y: 0, width: 1, height: 1 };
    this.contextLost = false;
    canvas.addEventListener?.('webglcontextlost', event => {
      event.preventDefault(); this.contextLost = true;
    });
  }

  resize(width = this.canvas.clientWidth, height = this.canvas.clientHeight, pixelRatio = 0,
      presentationAspect = 0) {
    const ratio = pixelRatio || this.pixelRatio || Math.min(globalThis.devicePixelRatio || 1, this.maxPixelRatio);
    width = Math.max(1, Math.round(width * ratio));
    height = Math.max(1, Math.round(height * ratio));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    this.canvasWidth = width; this.canvasHeight = height;
    this.presentationRegion = fitAspectRegion(width, height, presentationAspect);
    this.width = this.presentationRegion.width;
    this.height = this.presentationRegion.height;
    this.masterTarget.resize(this.width, this.height);
    this.prelightTarget.resize(this.width, this.height);
    for (const target of this.targets) target.resize(this.width, this.height);
    return {
      width, height, pixelRatio: ratio,
      presentationRegion: { ...this.presentationRegion },
    };
  }

  target(index) {
    while (this.targets.length <= index) {
      this.targets.push(new RenderTarget(this.gl, this.width, this.height, { depth: false }));
    }
    return this.targets[index];
  }

  glareTarget(index, width, height) {
    while (this.glareTargets.length <= index) {
      this.glareTargets.push(new RenderTarget(this.gl, 1, 1, { depth: false }));
    }
    const target = this.glareTargets[index];
    target.resize(width, height);
    return target;
  }

  bindDestination(destination) {
    const gl = this.gl;
    if (destination) destination.bind();
    else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.width, this.height);
    }
    gl.disable(gl.SCISSOR_TEST);
  }

  setViewportRegion(viewport) {
    const region = viewportRegion(viewport, this.width, this.height);
    const gl = this.gl;
    gl.viewport(region.x, region.y, region.width, region.height);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(region.x, region.y, region.width, region.height);
    return region;
  }

  setLightScissor(rectangle, viewport) {
    const region = lightScissorRegion(rectangle, viewport);
    const gl = this.gl;
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(region.x, region.y, region.width, region.height);
    return region;
  }

  postTarget(depth, destination) {
    let index = Math.max(0, depth | 0);
    while (true) {
      const candidate = this.target(index++);
      if (candidate !== destination) return candidate;
    }
  }

  captureFramebuffer(target, source = null) {
    const gl = this.gl;
    // Sequence rendering uses same-size RGBA8 single-sample targets, making
    // this the direct WebGL2 equivalent of RenderTargetManager::GrabToTarget.
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, source?.framebuffer || null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, target.framebuffer);
    gl.blitFramebuffer(0, 0, this.width, this.height, 0, 0, this.width, this.height,
      gl.COLOR_BUFFER_BIT, gl.NEAREST);
  }

  present(image) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, this.canvasWidth, this.canvasHeight);
    gl.colorMask(true, true, true, true);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.configureFullscreenState(null);
    const region = this.presentationRegion;
    gl.viewport(region.x, region.y, region.width, region.height);
    this.drawFullscreen(image);
  }

  clearMasterTarget() {
    const gl = this.gl;
    this.bindDestination(this.masterTarget);
    // mainplayer.cpp clears its X8R8G8B8 master viewport before executing
    // every Demo root. RGBA8 needs alpha one to emulate that opaque X channel.
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.stencilMask(0xff);
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(1);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
  }

  render(output, environment, options = {}) {
    const outputs = resolveIPPOutputs(output, environment);
    if (this.contextLost) return { drawCalls: 0, triangles: 0, outputs: 0 };
    this.resize(options.width, options.height, options.pixelRatio, options.presentationAspect);
    this.drawCalls = 0; this.triangles = 0;
    this.cullingStats = {};
    this.instanceMatrixCache = new WeakMap();
    this.geometry.beginFrame();
    this.clearMasterTarget();
    for (const node of outputs) this.renderNode(node, environment, this.masterTarget, 0);
    this.present(this.masterTarget.color);
    return {
      drawCalls: this.drawCalls, triangles: this.triangles, outputs: outputs.length,
      width: this.canvasWidth, height: this.canvasHeight,
      renderWidth: this.width, renderHeight: this.height,
      presentationRegion: { ...this.presentationRegion },
      culling: { ...this.cullingStats },
    };
  }

  resourceStats() {
    const geometry = this.geometry.resourceStats();
    const bitmapTextures = this.textures.resourceStats();
    const material11Lookups = this.material11Lookups.resourceStats();
    const textures = {
      ...bitmapTextures,
      textures: bitmapTextures.textures + material11Lookups.textures,
      lookupTextures: material11Lookups.textures,
      lookupEstimatedBytes: material11Lookups.estimatedBytes,
      estimatedBytes: bitmapTextures.estimatedBytes + material11Lookups.estimatedBytes,
    };
    const targets = [this.masterTarget, this.prelightTarget, ...this.targets, ...this.glareTargets];
    const uniqueTargets = Array.from(new Set(targets));
    const renderTargetBytes = uniqueTargets.reduce((sum, target) =>
      sum + (target?.estimatedBytes?.() || 0), 0);
    const canvasBytes = this.canvasWidth * this.canvasHeight * 8;
    const totalEstimatedBytes = geometry.gpuBytes + geometry.cpuReferencedBytes + geometry.shadowBytes +
      textures.estimatedBytes + renderTargetBytes + canvasBytes;
    return {
      width: this.canvasWidth, height: this.canvasHeight,
      renderWidth: this.width, renderHeight: this.height,
      presentationRegion: { ...this.presentationRegion },
      geometry, textures, material11Lookups,
      renderTargets: uniqueTargets.length,
      renderTargetBytes, canvasBytes,
      totalEstimatedBytes,
    };
  }

  warmupResidentBytes() {
    let geometryBytes = 0;
    for (const entry of this.geometry?.allEntries || []) {
      geometryBytes += entry.uploadBytes || 0;
      for (const topology of entry.shadowTopologies?.values?.() || []) {
        geometryBytes += shadowTopologyBytes(topology);
      }
    }
    const targets = [this.masterTarget, this.prelightTarget,
      ...(this.targets || []), ...(this.glareTargets || [])];
    const targetBytes = Array.from(new Set(targets))
      .reduce((sum, target) => sum + (target?.estimatedBytes?.() || 0), 0);
    return geometryBytes + (this.textures?.estimatedBytes || 0) + targetBytes;
  }

  async prewarmResources(plan, options = {}) {
    const gl = this.gl;
    const budgetMilliseconds = Math.max(1, Math.min(12,
      Number(options.budgetMilliseconds) || 8));
    const maxResidentBytes = Math.max(0,
      Number(options.maxResidentBytes ?? 256 * 1024 * 1024));
    const maxResourceBytes = Math.max(0,
      Number(options.maxResourceBytes ?? 64 * 1024 * 1024));
    const now = typeof options.now === 'function' ? options.now
      : () => globalThis.performance?.now?.() ?? Date.now();
    const yieldThread = typeof options.yield === 'function' ? options.yield : () =>
      new Promise(resolve => {
        if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
        else setTimeout(resolve, 0);
      });
    const shouldAbort = () => Boolean(this.contextLost || options.signal?.aborted ||
      options.shouldAbort?.());

    // Expand material tasks into exactly the texture identities selected by
    // their compiled pass maps. This excludes linked-but-unused bitmap slots
    // without evaluating a timeline frame or invoking any effect handler.
    const tasks = [];
    const textureIdentities = new Set();
    for (const task of plan?.tasks || []) {
      if (task?.kind !== 'material') {
        tasks.push(task);
        continue;
      }
      tasks.push(task);
      const material = task.value;
      const passes = material?.passes?.length ? material.passes : [null];
      for (const pass of passes) {
        const view = this.viewMaterial(material, pass);
        for (const sourceIndex of view.textureMap || []) {
          if (sourceIndex === null || sourceIndex === undefined) continue;
          const bitmap = view.textures?.[sourceIndex];
          if (!bitmap || textureIdentities.has(bitmap)) continue;
          textureIdentities.add(bitmap);
          tasks.push({ kind: 'texture', value: bitmap });
        }
      }
    }
    tasks.push({ kind: 'infrastructure', value: null });

    const baselineResidentBytes = this.warmupResidentBytes();
    const stats = {
      enabled: true,
      plannedTasks: tasks.length,
      plannedMeshes: tasks.filter(task => task?.kind === 'mesh').length,
      plannedMaterials: tasks.filter(task => task?.kind === 'material').length,
      plannedTextures: textureIdentities.size,
      completedTasks: 0,
      warmedMeshes: 0,
      cachedMeshes: 0,
      warmedTextures: 0,
      cachedTextures: 0,
      warmedShadowTopologies: 0,
      skippedAnimatedMeshes: 0,
      skippedLargeResources: 0,
      skippedBudgetResources: 0,
      yields: 0,
      aborted: false,
      baselineResidentBytes,
      residentBytes: baselineResidentBytes,
      newResidentBytes: 0,
      maxResidentBytes,
      maxResourceBytes,
    };
    let deadline = now() + budgetMilliseconds;
    const publish = task => options.onProgress?.({ ...stats, phase: task?.kind || 'unknown' });
    const refreshResidentBytes = () => {
      stats.residentBytes = this.warmupResidentBytes();
      stats.newResidentBytes = Math.max(0, stats.residentBytes - baselineResidentBytes);
      return stats.newResidentBytes;
    };
    const admitted = estimate => {
      estimate = Math.max(0, Number(estimate) || 0);
      if (estimate > maxResourceBytes) {
        stats.skippedLargeResources++;
        return false;
      }
      if (stats.newResidentBytes + estimate > maxResidentBytes) {
        stats.skippedBudgetResources++;
        return false;
      }
      return true;
    };

    for (const task of tasks) {
      if (shouldAbort()) {
        stats.aborted = true;
        break;
      }
      if (task?.kind === 'mesh') {
        const mesh = task.value;
        if (meshHasAnimation(mesh)) stats.skippedAnimatedMeshes++;
        else {
          const existing = this.geometry.entries.get(mesh);
          const summary = mesh?.storageSummary?.() || {};
          const shadowEstimate = this.shadows
            ? Math.max(0, Number(summary.vertices) || 0) * 28 +
              Math.max(0, Number(summary.faces) || 0) * 24
            : 0;
          const estimate = existing ? 0 : geometryWarmupEstimateBytes(mesh) + shadowEstimate;
          if (existing || admitted(estimate)) {
            const entry = this.geometry.get(mesh, undefined, task.sourceId);
            if (existing) stats.cachedMeshes++;
            else stats.warmedMeshes++;
            if (this.shadows) {
              const shadowGroups = shadowGroupsForGeometry(entry);
              if (shadowGroups.length) {
                const key = shadowTopologyKey(shadowGroups);
                if (!entry.shadowTopologies.has(key)) {
                  entry.shadowTopologies.set(key, prepareShadowTopology(entry, shadowGroups));
                  stats.warmedShadowTopologies++;
                }
              }
            }
            refreshResidentBytes();
          }
        }
      } else if (task?.kind === 'texture') {
        const bitmap = task.value;
        const estimate = this.textures.estimatedUploadBytes(bitmap);
        if (!estimate) stats.cachedTextures++;
        else if (admitted(estimate)) {
          this.textures.get(bitmap);
          stats.warmedTextures++;
          refreshResidentBytes();
        }
      } else if (task?.kind === 'infrastructure') {
        const estimate = Math.max(0,
          this.width * this.height * 4 + 512 * 256 * 8 + 256 * 128 * 4);
        if (admitted(estimate)) {
          this.ensureLayerGeometry();
          this.textures.fallbackTexture();
          this.target(0);
          this.glareTarget(0, 512, 256);
          this.glareTarget(1, 256, 128);
          this.glareTarget(2, 512, 256);
          refreshResidentBytes();
        }
      }
      stats.completedTasks++;
      publish(task);
      if (now() >= deadline) {
        gl.flush?.();
        stats.yields++;
        await yieldThread();
        deadline = now() + budgetMilliseconds;
      }
    }
    gl.flush?.();
    if (!stats.aborted && options.finish !== false) gl.finish?.();
    refreshResidentBytes();
    publish({ kind: stats.aborted ? 'aborted' : 'complete' });
    return stats;
  }

  renderNode(node, environment, destination, depth) {
    if (!node) {
      this.bindDestination(destination);
      this.gl.colorMask(true, true, true, true);
      this.gl.depthMask(true);
      this.gl.stencilMask(0xff);
      this.gl.clearColor(0, 0, 0, 1);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT | this.gl.STENCIL_BUFFER_BIT);
      return;
    }
    if (node.type === 'layer2d') return this.renderLayer(node, environment, destination, depth);
    if (node.type === 'viewport') return this.renderViewport(node, environment, destination, depth);
    throw new Error(`unsupported IPP node ${node.type || node.kind || typeof node}`);
  }

  renderLayer(layer, environment, destination, depth) {
    // The Debris player is compiled with GENOVER_RTSIZES=0. Layer allocation
    // therefore aliases the master viewport and Copy() is a no-op for its
    // Bitmap==0 input: the child has already produced color, depth and stencil
    // directly in the destination used by the overlay quad.
    if (layer.input) this.renderNode(layer.input, environment, destination, depth + 1);
    this.bindDestination(destination);
    const flags = layer.clearFlags | 0;
    if (flags) {
      this.gl.clearColor(0, 0, 0, 1);
      let mask = 0;
      if (flags & 1) {
        this.gl.colorMask(true, true, true, true);
        mask |= this.gl.COLOR_BUFFER_BIT;
      }
      if (flags & 2) {
        this.gl.depthMask(true);
        this.gl.stencilMask(0xff);
        mask |= this.gl.DEPTH_BUFFER_BIT | this.gl.STENCIL_BUFFER_BIT;
      }
      this.gl.clear(mask);
    }
    this.currentPrelightTexture = null;
    for (const pass of layer.material?.passes || []) this.drawLayerQuad(layer, layer.material, pass);
  }

  ensureLayerGeometry() {
    if (this.layerVAO) return;
    const gl = this.gl;
    this.layerVAO = gl.createVertexArray();
    this.layerPositionBuffer = gl.createBuffer();
    this.layerUVBuffer = gl.createBuffer();
    this.layerIndexBuffer = gl.createBuffer();
    gl.bindVertexArray(this.layerVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.layerPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.layerPositions.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.layerUVBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.layerUVs.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(9);
    gl.vertexAttribPointer(9, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.layerIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    for (const location of [1, 3, 4, 5, 6, 7, 8]) {
      gl.disableVertexAttribArray(location);
      gl.vertexAttribDivisor(location, 0);
    }
    gl.bindVertexArray(null);
  }

  drawLayerQuad(layer, material, pass) {
    const gl = this.gl, uniforms = this.uniforms;
    this.ensureLayerGeometry();
    layerQuadGeometry(layer, this.layerPositions, this.layerUVs);
    const view = this.viewMaterial(material, pass);
    this.configureMaterialState(view);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.layerVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.layerPositionBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.layerPositions);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.layerUVBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.layerUVs);
    // sVertexTSpace3's authored quad uses packed defaults and color zero.
    // These channels are immaterial for Debris' unlit fade pass, but keeping
    // deterministic constants also makes other linked materials well-defined.
    gl.vertexAttrib3f(1, 0, 0, -1);
    gl.vertexAttrib4f(3, 0, 0, 0, 0);
    gl.vertexAttrib4f(4, 0, 1, 0, 1);
    gl.vertexAttrib4f(5, 1, 0, 0, 0);
    gl.vertexAttrib4f(6, 0, 1, 0, 0);
    gl.vertexAttrib4f(7, 0, 0, 1, 0);
    gl.vertexAttrib4f(8, 0, 0, 0, 1);
    gl.uniformMatrix4fv(uniforms.uViewProjection, false, this.layerMatrices.viewProjection);
    gl.uniform4f(uniforms.uUVTransform, 1, 1, 0, 0);
    this.bindMaterial(view, this.layerViewport, this.layerMatrices, [], null);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    this.drawCalls++;
    this.triangles += 2;
  }

  renderViewport(viewport, environment, destination, depth) {
    const frame = environment?.frame || { effectGeometry: [], postJobs: [] };
    frame.effectGeometry ||= []; frame.postJobs ||= [];
    frame.effectGeometry.length = 0; frame.postJobs.length = 0;
    this.bindDestination(destination);
    this.drawViewportScene(viewport, environment, destination, depth);
  }

  drawViewportScene(viewport, environment, destination = null, depth = 0) {
    const gl = this.gl;
    const frame = environment?.frame || { effectGeometry: [], postJobs: [] };
    frame.effectGeometry ||= []; frame.postJobs ||= [];
    this.currentPrelightTexture = null;
    const clear = colorARGB(viewport.clearColor >>> 0);
    const viewportPixels = this.setViewportRegion(viewport);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.colorMask(true, true, true, true);
    gl.disable(gl.BLEND);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(clear[0], clear[1], clear[2], clear[3]);
    gl.clearDepth(1);
    gl.clearStencil(0);
    gl.stencilMask(0xff);
    const clearFlags = viewportClearFlags(viewport);
    let clearMask = 0;
    if (clearFlags & 1) clearMask |= gl.COLOR_BUFFER_BIT;
    if (clearFlags & 2) clearMask |= gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT;
    if (clearMask) gl.clear(clearMask);

    const matrices = cameraMatrices(viewport.camera);
    const viewPlanes = viewFrustumPlanes(matrices.viewProjection);
    const items = [];
    const lights = selectActiveLights(viewport, MAX_LIGHTS, viewPlanes);
    // InsertLightJob computes this rectangle once, then shares it between
    // SetScissor and ZFailVolume. Reusing it here avoids subtly different
    // floating-point edges between raster clipping and stencil selection.
    const lightRectangles = lights.map(light =>
      lightSphereBounds(light, matrices.view, viewport.camera?.zoomX, viewport.camera?.zoomY));
    // The native SVFrustum belongs to the selected light, not the mesh.
    // Preparing these five planes once avoids meshCount*lightCount temporary
    // plane sets during paint-job construction.
    const shadowFrusta = lights.map(light => light.kind === 'directional' ? null
      : shadowViewFrustumPlanes(matrices.viewProjection, light.position));
    const shadowZFailVolumes = lights.map((light, index) =>
      shadowZFailVolumePlanes(viewport.camera, light, matrices, lightRectangles[index]));
    forEachNativeHeadInsertedJob(viewport.effectJobs, (job, index) => {
      items.push(effectRenderItem(job, index));
    });
    forEachNativeHeadInsertedJob(viewport.meshJobs, job => {
      if (!job.mesh) return;
      const geometry = this.geometry.get(job.mesh, job.time, job.opId);
      appendMeshRenderItems(items, job, geometry, lights, matrices,
        this.instanceMatrices(job), this.cullingStats, shadowFrusta, shadowZFailVolumes,
        viewPlanes);
    });
    // The native radix sort is stable. Items are already in BuildPaintJobs
    // order, so this final field only makes equal-key stability explicit for
    // engines whose Array#sort implementation predates the ES2019 guarantee.
    for (let index = 0; index < items.length; index++) items[index].sortOrder = index;
    sortRenderItems(items);
    this.prepareInstanceUploads(items);
    const materialInsertTracker = createMaterialInsertTracker(items);
    const needCurrentRenderPasses = new Set(items
      .filter(item => item.effectJob?.effect?.needCurrentRender)
      .map(item => item.renderPass | 0));
    let stencilGroup = null;
    let previousItem = null;
    let currentRenderPass = -1;
    let currentRenderTarget = null;
    let viewportScissorActive = true;
    const restoreViewportScissor = () => {
      if (!viewportScissorActive) this.setViewportRegion(viewport);
      viewportScissorActive = true;
    };
    let prelightPending = false;
    let prelightRenderPass = -1;
    const capturePrelight = () => {
      if (!prelightPending) return;
      // GrabToTarget runs after SetScissor(0) in native RenderPaintJobs.
      restoreViewportScissor();
      this.captureFramebuffer(this.prelightTarget, destination);
      this.currentPrelightTexture = this.prelightTarget.color;
      this.bindDestination(destination);
      this.setViewportRegion(viewport);
      viewportScissorActive = true;
      prelightPending = false;
    };
    for (const item of items) {
      const usage = item.pass?.usage;
      if (usage === 'shadow' && !this.shadows) continue;
      // Every native pass/usage/light transition first removes the previous
      // light rectangle. AfterUsage inserts and clears must cover the complete
      // viewport, not just the light that happened to render before them.
      const usageChanged = renderUsageChanged(previousItem, item);
      if (usageChanged) restoreViewportScissor();
      const insertAction = materialInsertTracker.transition(item);
      if (insertAction) {
        restoreViewportScissor();
        this.drawMaterial11Insert(insertAction);
      }
      if (prelightPending && (usage !== 'prelight' || item.renderPass !== prelightRenderPass)) {
        capturePrelight();
      }
      if (item.renderPass !== currentRenderPass) {
        currentRenderPass = item.renderPass;
        currentRenderTarget = null;
        if (needCurrentRenderPasses.has(currentRenderPass)) {
          // RenderPaintJobs performs GrabToTarget after the previous pass's
          // AfterUsage insert and before the new pass's BeforeUsage/job. Keep
          // this pass snapshot stable for every NeedCurrentRender effect in
          // the pass; ColorCorrection samples it when its sorted job runs.
          restoreViewportScissor();
          currentRenderTarget = this.postTarget(depth, destination);
          this.captureFramebuffer(currentRenderTarget, destination);
          this.bindDestination(destination);
          this.setViewportRegion(viewport);
          viewportScissorActive = true;
        }
      }
      if (usage === 'shadow' || usage === 'light') {
        const key = `${item.renderPass}:${item.lightIndex}`;
        if (key !== stencilGroup) {
          restoreViewportScissor();
          gl.stencilMask(0xff);
          gl.clear(gl.STENCIL_BUFFER_BIT);
          stencilGroup = key;
        }
      } else if (stencilGroup !== null) {
        restoreViewportScissor();
        gl.stencilMask(0xff);
        gl.clear(gl.STENCIL_BUFFER_BIT);
        stencilGroup = null;
      }
      // Native applies LightRect only after BeforeUsage/AfterUsage work and
      // stencil clearing have completed for the new group.
      if (usageChanged && item.light && (usage === 'shadow' || usage === 'light')) {
        this.setLightScissor(lightRectangles[item.lightIndex], viewportPixels);
        viewportScissorActive = false;
      }
      if (item.effectJob) this.drawEffectItem(
        item, frame, environment, viewport, matrices, destination, depth, lights,
        currentRenderTarget,
      );
      else this.drawMeshItem(item, viewport, matrices, lights);
      if (usage === 'prelight' && item.material?.system === '2.0') {
        prelightPending = true;
        prelightRenderPass = item.renderPass;
      }
      previousItem = item;
    }
    restoreViewportScissor();
    const finalInsertAction = materialInsertTracker.finish();
    if (finalInsertAction) {
      restoreViewportScissor();
      this.drawMaterial11Insert(finalInsertAction);
    }
    capturePrelight();
    if (stencilGroup !== null) {
      restoreViewportScissor();
      gl.stencilMask(0xff);
      gl.clear(gl.STENCIL_BUFFER_BIT);
    }
    // RenderPaintJobs leaves scissoring at the complete active viewport.
    restoreViewportScissor();
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.disable(gl.STENCIL_TEST);
    gl.stencilMask(0xff);
    this.instanceUploadActive = false;
  }

  drawEffectItem(item, frame, environment, viewport, matrices, destination, depth,
      activeLights = null, currentRenderTarget = null) {
    frame.effectGeometry.length = 0;
    frame.postJobs.length = 0;
    (this.effectExecutor ?? executeEffectJob)?.(environment, item.effectJob);
    for (const geometry of frame.effectGeometry) {
      this.drawEffectGeometry(geometry, viewport, matrices, activeLights);
    }
    for (const postJob of frame.postJobs) {
      const usePassSnapshot = Boolean(item.effectJob?.effect?.needCurrentRender &&
        currentRenderTarget);
      const scratch = usePassSnapshot
        ? currentRenderTarget : this.postTarget(depth, destination);
      if (!usePassSnapshot) {
        this.setViewportRegion(viewport);
        this.captureFramebuffer(scratch, destination);
      }
      this.bindDestination(destination);
      const region = this.setViewportRegion(viewport);
      this.applyPost(scratch.color, postJob, {
        uvRect: region.uvRect, destination, viewport,
      });
      // Post effects change targets/viewports internally; restore the active
      // viewport for any later jobs in the same native pass stream.
      this.bindDestination(destination);
      this.setViewportRegion(viewport);
    }
    frame.effectGeometry.length = 0;
    frame.postJobs.length = 0;
  }

  instanceMatrices(job) {
    const cached = this.instanceMatrixCache.get(job);
    if (cached) return cached;
    const result = composeInstanceMatrices(job);
    this.instanceMatrixCache.set(job, result);
    return result;
  }

  instanceBufferSize(requiredBytes) {
    requiredBytes = Math.max(INSTANCE_MATRIX_BYTES, Number(requiredBytes) || 0);
    const aligned = Math.ceil(requiredBytes / INSTANCE_MATRIX_BYTES) * INSTANCE_MATRIX_BYTES;
    if (aligned > MAX_RETAINED_INSTANCE_BUFFER_BYTES) return aligned;
    const rounded = Math.min(MAX_RETAINED_INSTANCE_BUFFER_BYTES,
      2 ** Math.ceil(Math.log2(aligned)));
    const retained = this.instanceBufferCapacity || 0;
    // Avoid carrying a viewport's peak indefinitely, while also avoiding
    // reallocating for small fluctuations around the current demand.
    if (retained >= aligned && retained <= rounded * 4 &&
        retained <= MAX_RETAINED_INSTANCE_BUFFER_BYTES) return retained;
    return rounded;
  }

  prepareInstanceUploads(items) {
    const jobs = [];
    const seen = new Set();
    let requiredBytes = 0, effectSlots = 0;
    for (const item of items || []) {
      if (item?.effectJob) {
        // Each currently supported geometry effect emits at most one raster
        // matrix. Reserve its slot without executing the effect ahead of its
        // native sorted position.
        effectSlots++;
        continue;
      }
      if (item?.pass?.usage === 'shadow' || !item?.group?.count) continue;
      const job = item.job;
      if (!job || seen.has(job)) continue;
      seen.add(job);
      const instances = this.instanceMatrices(job);
      if (!instances.length) continue;
      jobs.push({ job, instances });
      requiredBytes += instances.byteLength;
    }
    requiredBytes += effectSlots * INSTANCE_MATRIX_BYTES;
    this.instanceUploadCache = new WeakMap();
    this.instanceUploadRecords = [];
    this.instanceUploadUsedBytes = 0;
    this.instanceUploadActive = requiredBytes > 0;
    if (!this.instanceUploadActive) return;

    const gl = this.gl;
    this.instanceBufferCapacity = this.instanceBufferSize(requiredBytes);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    // One orphan per viewport prevents a busy previous frame from stalling
    // the CPU while retaining the same bounded buffer object and VAO wiring.
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceBufferCapacity, gl.DYNAMIC_DRAW);
    for (const { job, instances } of jobs) this.uploadInstanceMatrices(job, instances);
  }

  ensureInstanceUploadCapacity(requiredBytes) {
    if (requiredBytes <= this.instanceBufferCapacity) return;
    const gl = this.gl;
    this.instanceBufferCapacity = this.instanceBufferSize(requiredBytes);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceBufferCapacity, gl.DYNAMIC_DRAW);
    // This path is only needed for an effect that emits more matrices than its
    // conservative reservation. Refill the new store so later sorted passes
    // can still address every already-cached job at its original offset.
    for (const record of this.instanceUploadRecords) {
      gl.bufferSubData(gl.ARRAY_BUFFER, record.offset, record.instances);
    }
  }

  uploadInstanceMatrices(job, instances) {
    const cacheable = job && (typeof job === 'object' || typeof job === 'function');
    const cached = cacheable ? this.instanceUploadCache.get(job) : null;
    if (cached?.instances === instances) return cached;
    const offset = this.instanceUploadUsedBytes;
    this.ensureInstanceUploadCapacity(offset + instances.byteLength);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, offset, instances);
    const record = { instances, offset };
    this.instanceUploadRecords.push(record);
    this.instanceUploadUsedBytes += instances.byteLength;
    if (cacheable) this.instanceUploadCache.set(job, record);
    return record;
  }

  bindInstanceMatrices(job, instances) {
    const gl = this.gl;
    let offset = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (this.instanceUploadActive) offset = this.uploadInstanceMatrices(job, instances).offset;
    else gl.bufferData(gl.ARRAY_BUFFER, instances, gl.DYNAMIC_DRAW);
    // Attribute pointers are VAO state. GeometryCache enables these locations
    // once; only their packed-block byte offsets vary between draw jobs.
    for (let column = 0; column < 4; column++) {
      const location = 5 + column;
      gl.vertexAttribPointer(location, 4, gl.FLOAT, false,
        INSTANCE_MATRIX_BYTES, offset + column * 16);
    }
  }

  drawMeshItem(item, viewport, matrices, activeLights = null) {
    const { geometry, group, material, pass, job } = item;
    if (pass.usage === 'shadow') return this.drawShadowItem(item, matrices);
    if (!group?.count) return;
    const instances = this.instanceMatrices(job);
    if (!instances.length) return;
    const gl = this.gl, uniforms = this.uniforms;
    const view = this.viewMaterial(material, pass);
    this.configureMaterialState(view);
    gl.useProgram(this.program);
    gl.bindVertexArray(geometry.vao);
    this.bindInstanceMatrices(job, instances);
    gl.uniformMatrix4fv(uniforms.uViewProjection, false, matrices.viewProjection);
    const uvScale = Number.isFinite(view.uvScale) && view.uvScale !== 0 ? view.uvScale : 1;
    gl.uniform4f(uniforms.uUVTransform, uvScale, uvScale, 0, 0);
    let selectedLights = activeLights;
    if (item.light) {
      this.materialScratch.singleLight[0] = item.light;
      selectedLights = this.materialScratch.singleLight;
    }
    this.bindMaterial(view, viewport, matrices, selectedLights, job.matrix,
      meshHasAnimation(job.mesh), pass.program === 'instances');
    const instanceCount = instances.length / 16;
    gl.drawElementsInstanced(gl.TRIANGLES, group.count, geometry.indexType,
      group.start * geometry.indexBytes, instanceCount);
    this.drawCalls++;
    this.triangles += group.count / 3 * instanceCount;
  }

  drawShadowItem(item, matrices) {
    if (!item.light || !item.groups?.length) return;
    const gl = this.gl, uniforms = this.shadowUniforms;
    const instanceMatrices = this.instanceMatrices(item.job);
    const topologyKey = shadowTopologyKey(item.groups);
    let topology = item.geometry.shadowTopologies?.get(topologyKey);
    if (!topology) {
      topology = prepareShadowTopology(item.geometry, item.groups);
      item.geometry.shadowTopologies?.set(topologyKey, topology);
    }
    gl.useProgram(this.shadowProgram);
    gl.bindVertexArray(this.shadowVAO);
    gl.uniformMatrix4fv(uniforms.uViewProjection, false, matrices.viewProjection);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(false);
    gl.colorMask(false, false, false, false);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    // WZ3's authored faces are clockwise. Its default D3DCULL_CCW state
    // discards the opposite winding; our projection preserves X/Y signs, so
    // WebGL must likewise classify clockwise triangles as front-facing.
    gl.frontFace(gl.CW);
    gl.enable(gl.STENCIL_TEST);
    gl.stencilMask(0xff);
    gl.stencilFunc(gl.ALWAYS, 0, 0xff);
    const useZFail = item.shadowZFail !== false;
    if (useZFail) {
      // Carmack's reverse: sMBF_STENCILINDE|sMBF_STENCILZFAIL.
      gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.DECR_WRAP, gl.KEEP);
      gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.INCR_WRAP, gl.KEEP);
    } else {
      // A caster outside the native near-plane volume takes the cheaper
      // depth-pass path and needs no front/back caps.
      gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP);
      gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.KEEP, gl.DECR_WRAP);
    }

    let topologyUploaded = false;
    for (let offset = 0; offset < instanceMatrices.length; offset += 16) {
      const model = instanceMatrices.subarray(offset, offset + 16);
      const volume = buildShadowVolume(
        item.geometry, topology, item.light, model, useZFail, this.shadowVolumeScratch);
      if (!volume.indices.length) continue;
      gl.uniform3fv(uniforms.uLightPosition, volume.lightPosition);
      if (!topologyUploaded) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.shadowPositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, topology.volumePositions, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.shadowExtrusionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, topology.extrusions, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 1, gl.UNSIGNED_BYTE, false, 0, 0);
        topologyUploaded = true;
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.shadowIndexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, volume.indices, gl.DYNAMIC_DRAW);
      gl.uniformMatrix4fv(uniforms.uModel, false, model);
      gl.drawElements(gl.TRIANGLES, volume.indices.length,
        volume.indices instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT, 0);
      this.drawCalls++;
      this.triangles += volume.indices.length / 3;
    }
  }

  configureMaterialState(view) {
    const gl = this.gl;
    const state = materialState(view);
    if (state.depthTest) gl.enable(gl.DEPTH_TEST); else gl.disable(gl.DEPTH_TEST);
    gl.depthFunc(glDepthFunction(gl, state.depthFunc));
    gl.depthMask(state.depthWrite);
    const depthBias = view.baseFlags & MBF_ZBIASFORE ? -LEGACY_DEPTH_BIAS
      : view.baseFlags & MBF_ZBIASBACK ? LEGACY_DEPTH_BIAS : 0;
    if (depthBias) {
      gl.enable(gl.POLYGON_OFFSET_FILL);
      // D3D9 adds both authored values directly in normalized depth:
      //   maxSlope * f + f, where f = +/-1/65536.
      // WebGL instead evaluates maxSlope * factor + r * units. Scene depth is
      // DEPTH24_STENCIL8, so 256 depth-buffer units reproduce the constant
      // 1/65536 term while factor retains the released slope-scale value.
      gl.polygonOffset(depthBias, Math.sign(depthBias) * DEPTH24_OFFSET_UNITS);
    } else gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.frontFace(gl.CW);
    if (state.cull === 'none') gl.disable(gl.CULL_FACE);
    else {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(state.cull === 'front' ? gl.FRONT : gl.BACK);
    }
    if (!state.blend) gl.disable(gl.BLEND);
    else {
      gl.enable(gl.BLEND);
      gl.blendEquation(state.blend[2] === 'reverse-subtract' ? gl.FUNC_REVERSE_SUBTRACT : gl.FUNC_ADD);
      gl.blendFunc(glBlendFactor(gl, state.blend[0]), glBlendFactor(gl, state.blend[1]));
    }
    gl.colorMask(state.colorWrite, state.colorWrite, state.colorWrite, state.colorWrite);
    if (state.stencilTest) {
      gl.enable(gl.STENCIL_TEST);
      gl.stencilMask(0);
      gl.stencilFunc(gl.EQUAL, 0, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    } else gl.disable(gl.STENCIL_TEST);
  }

  viewMaterial(material, pass = null) {
    const key = `${pass?.usage || 'other'}\u0000${pass?.state || ''}`;
    if (!material || typeof material !== 'object') {
      let view = this.defaultMaterialViews.get(key);
      if (!view) this.defaultMaterialViews.set(key, view = compiledMaterialView(material, pass));
      return view;
    }
    let views = this.materialViews.get(material);
    if (!views) this.materialViews.set(material, views = new Map());
    const version = Number.isSafeInteger(material.version) ? material.version : 0;
    let cached = views.get(key);
    if (!cached || cached.version !== version) {
      cached = { version, view: compiledMaterialView(material, pass) };
      views.set(key, cached);
    }
    return cached.view;
  }

  bindMaterial(view, viewport, matrices, selectedLights = null, modelMatrix = null,
      animatedMesh = false, shaderInstanced = false) {
    const gl = this.gl, uniforms = this.uniforms;
    const scratch = this.materialScratch;
    const baseColor = colorARGB(
      view.system === '2.0' && view.usage === 'ambient'
        ? 0xffffffff : view.color >>> 0,
      scratch.baseColor,
    );
    if (view.generatedBaseAlpha === 'zero') baseColor[3] = 0;
    else if (view.generatedBaseAlpha === 'texture0') baseColor[3] = 1;
    const lightScale = view.lightScale || 1;
    if (lightScale !== 1) {
      baseColor[0] *= lightScale; baseColor[1] *= lightScale; baseColor[2] *= lightScale;
    }
    gl.uniform4fv(uniforms.uBaseColor, baseColor);
    const mode = renderMode(view);
    gl.uniform1i(uniforms.uMode, mode);
    gl.uniform1i(uniforms.uVertexColorMode, view.vertexColorMode || 0);
    const material20 = view.system === '2.0';
    const material11 = view.system === '1.1';
    const legacyMaterialMatrix = (material20 || material11) && modelMatrix?.length >= 16
      ? modelMatrix : null;
    const legacyLighting = mode === 2 && (material20 || material11);
    const legacyEnvironment = mode === 4 && material20;
    // Engine.cpp selects BoneData variant 1 first, then lets the shader-
    // instances program override it with raw transformed variant 2.
    const conditionAnimatedBasis = animatedMesh && !shaderInstanced;
    gl.uniform1i(uniforms.uLegacyLightingMode,
      legacyLighting ? (conditionAnimatedBasis ? 2 : 1) : legacyEnvironment ? 3 : 0);
    gl.uniform1i(uniforms.uConditionLegacyBasis,
      material11 && conditionAnimatedBasis ? 1 : 0);
    gl.uniform1i(uniforms.uMaterial20, material20 ? 1 : 0);
    gl.uniform1i(uniforms.uM20Flags, material20 ? view.flags >>> 0 : 0);
    gl.uniform1i(uniforms.uM20EnvironmentFlags,
      material20 ? view.environmentFlags >>> 0 : 0);
    gl.uniform1i(uniforms.uM20RuntimeEnvironmentFlags,
      material20 ? (view.runtimeEnvironmentFlags ?? view.environmentFlags) >>> 0 : 0);
    gl.uniform1i(uniforms.uMaterial11, material11 ? 1 : 0);
    gl.uniform1i(uniforms.uM11MultipassLight,
      material11 && mode === 2 ? 1 : 0);
    gl.uniformMatrix4fv(uniforms.uM11View, false, matrices.view);
    let worldToModel = scratch.identityMatrix;
    if (modelMatrix?.length >= 16) {
      worldToModel = mat4Inverse(modelMatrix, scratch.m11WorldToModel) || scratch.identityMatrix;
    }
    gl.uniformMatrix4fv(uniforms.uM11WorldToModel, false, worldToModel);
    gl.uniformMatrix4fv(uniforms.uM11ModelToWorld, false,
      legacyMaterialMatrix || scratch.identityMatrix);
    gl.uniform1i(uniforms.uM11SpecialFlags,
      material11 ? view.specialFlags >>> 0 : 0);
    gl.uniform2iv(uniforms.uDetailOps, view.detailOps || scratch.emptyDetailOps);
    gl.uniform1iv(uniforms['uCombiners[0]'], view.combiners || scratch.emptyCombiners);
    gl.uniform1i(uniforms.uAlphaCombiner, view.alphaCombiner || 0);
    const materialColors = scratch.materialColors;
    for (let index = 0; index < 4; index++) {
      colorARGB((view.colors?.[index] ?? 0xffffffff) >>> 0, scratch.color);
      materialColors.set(scratch.color, index * 4);
    }
    gl.uniform4fv(uniforms['uMaterialColors[0]'], materialColors);
    gl.uniform1f(uniforms.uAlphaCutoff, view.alphaCutoff || -1);
    gl.uniform1f(uniforms.uSpecularPower, view.specularPower || 8);
    const specularColor = colorRGB(view.specularColor >>> 0, scratch.specularColor);
    if (lightScale !== 1) {
      specularColor[0] *= lightScale; specularColor[1] *= lightScale; specularColor[2] *= lightScale;
    }
    gl.uniform3fv(uniforms.uSpecularColor, specularColor);
    gl.uniform1f(uniforms.uSpecularStrength, view.specularStrength || 0);
    const ambient = colorRGB(viewport.ambientLight >>> 0, scratch.ambient);
    gl.uniform3fv(uniforms.uAmbient, ambient);
    const cameraSpace = viewport.camera.cameraSpace;
    gl.uniform3f(uniforms.uCameraPosition, cameraSpace[12], cameraSpace[13], cameraSpace[14]);
    const materialCameraPosition = scratch.materialCameraPosition;
    if (legacyMaterialMatrix) {
      legacyTransRVector(legacyMaterialMatrix,
        [cameraSpace[12], cameraSpace[13], cameraSpace[14]], true,
        materialCameraPosition);
    } else {
      materialCameraPosition[0] = cameraSpace[12];
      materialCameraPosition[1] = cameraSpace[13];
      materialCameraPosition[2] = cameraSpace[14];
    }
    gl.uniform3fv(uniforms.uMaterialCameraPosition, materialCameraPosition);
    gl.uniform3fv(uniforms.uFogColor,
      colorRGB(viewport.camera.fogColor >>> 0, scratch.fogColor));
    gl.uniform2f(uniforms.uFogRange, viewport.camera.fogStart, viewport.camera.fogEnd);
    gl.uniform1i(uniforms.uFogEnabled, Boolean(view.baseFlags & 0x40));

    const textures = view.textures || [];
    const textureMap = view.textureMap || [0, 1, 2, 3];
    const selected = scratch.selectedTextures;
    const samplerFlags = scratch.m20SamplerFlags;
    const samplerScales = scratch.m20SamplerScales;
    for (let index = 0; index < 4; index++) {
      const sourceIndex = textureMap[index];
      selected[index] = sourceIndex === null || sourceIndex === undefined
        ? null : textures[sourceIndex];
      if (!material20 || sourceIndex === null || sourceIndex === undefined) {
        samplerFlags[index] = 0;
        samplerScales[index] = 1;
      } else {
        const lighting = sourceIndex >= 4;
        const slot = lighting ? sourceIndex - 4 : sourceIndex;
        samplerFlags[index] = (lighting ? view.lightFlags?.[slot] :
          view.textureFlags?.[slot]) >>> 0;
        const scale = Number(lighting ? view.lightScales?.[slot] :
          view.textureScales?.[slot]);
        samplerScales[index] = Number.isFinite(scale) ? scale : 1;
      }
    }
    gl.uniform4iv(uniforms.uM20SamplerFlags, samplerFlags);
    gl.uniform4fv(uniforms.uM20SamplerScales, samplerScales);
    gl.uniform4fv(uniforms['uM20UVTransform1[0]'],
      material20 ? view.uvTransform1 : scratch.identityUVTransform);
    gl.uniform4fv(uniforms['uM20UVTransform2[0]'],
      material20 ? view.uvTransform2 : scratch.identityUVTransform);
    const m11SamplerFlags = scratch.m11SamplerFlags;
    const m11SamplerScales = scratch.m11SamplerScales;
    for (let index = 0; index < 4; index++) {
      m11SamplerFlags[index] = material11 ? view.slotTextureFlags?.[index] >>> 0 : 0;
      const scale = Number(view.slotTextureScales?.[index]);
      m11SamplerScales[index] = material11 && Number.isFinite(scale) ? scale : 1;
    }
    gl.uniform4iv(uniforms.uM11SamplerFlags, m11SamplerFlags);
    gl.uniform4fv(uniforms.uM11SamplerScales, m11SamplerScales);
    gl.uniform4fv(uniforms['uM11UVTransform1[0]'],
      material11 ? view.uvTransform1 : scratch.identityUVTransform);
    gl.uniform4fv(uniforms['uM11UVTransform2[0]'],
      material11 ? view.uvTransform2 : scratch.identityUVTransform);
    let mask = 0;
    for (let index = 0; index < 4; index++) {
      if (selected[index]) mask |= 1 << index;
      const sourceIndex = textureMap[index];
      const flags = sourceIndex === null || sourceIndex === undefined ? 1 :
        (sourceIndex < 4 ? view.textureFlags[sourceIndex] >>> 0 :
          view.extraTextureFlags?.[sourceIndex - 4] >>> 0);
      this.textures.bind(selected[index], index, {
        clamp: Boolean(flags & 0x100), filterMode: flags & 7,
        filterProfile: material11 ? 'material11' : 'material20',
      });
      gl.uniform1i(uniforms[`uTexture${index}`], index);
    }
    gl.uniform1i(uniforms.uTextureMask, mask);
    const usePrelight = Boolean(view.usePrelight && this.currentPrelightTexture);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, usePrelight ? this.currentPrelightTexture : this.textures.fallbackTexture());
    gl.uniform1i(uniforms.uPrelightTexture, 4);
    gl.uniform1i(uniforms.uUsePrelight, usePrelight ? 1 : 0);

    const positions = scratch.lightPositions;
    const attenuationCenters = scratch.lightAttenuation;
    const colors = scratch.lightColors;
    const lightSpecular = scratch.lightSpecular;
    const lights = selectedLights ?? selectActiveLights(viewport);
    if (mode === 2) for (let index = 0; index < lights.length; index++) {
      const light = lights[index], offset = index * 4;
      if (light.kind === 'directional') {
        // material11.vsh consumes EngLight::Direction directly as the
        // surface-to-light vector. The shadow path's +direction*1e6 point is
        // the same convention.
        if (legacyMaterialMatrix) {
          legacyTransRVector(legacyMaterialMatrix, light.direction, false, positions, offset);
        } else {
          positions[offset] = light.direction[0];
          positions[offset + 1] = light.direction[1];
          positions[offset + 2] = light.direction[2];
        }
        positions[offset + 3] = 0;
      } else {
        if (legacyMaterialMatrix) {
          legacyTransRVector(legacyMaterialMatrix, light.position, true, positions, offset);
        } else {
          positions[offset] = light.position[0]; positions[offset + 1] = light.position[1];
          positions[offset + 2] = light.position[2];
        }
        positions[offset + 3] = 1;
      }
      if (material20 && light.kind === 'directional') {
        // Material20 uses direction for L but the independently transformed
        // EngLight::Position for its attenuation coordinates.
        if (legacyMaterialMatrix) {
          legacyTransRVector(legacyMaterialMatrix, light.position, true,
            attenuationCenters, offset);
        } else {
          attenuationCenters[offset] = light.position[0];
          attenuationCenters[offset + 1] = light.position[1];
          attenuationCenters[offset + 2] = light.position[2];
        }
      } else {
        // Point lights, plus Material11's unusual directional-volume path,
        // use the same point/direction constant already supplied for L.
        attenuationCenters[offset] = positions[offset];
        attenuationCenters[offset + 1] = positions[offset + 1];
        attenuationCenters[offset + 2] = positions[offset + 2];
      }
      const color = colorRGB(light.color >>> 0, scratch.lightColor);
      colors[offset] = color[0] * light.amplify;
      colors[offset + 1] = color[1] * light.amplify;
      colors[offset + 2] = color[2] * light.amplify;
      const inverseRange = light.range > 1e12 ? 0 : 1 / Math.max(1e-5, light.range);
      colors[offset + 3] = inverseRange;
      attenuationCenters[offset + 3] = inverseRange;
      lightSpecular[index] = ((light.color >>> 24) & 255) / 255 * light.amplify;
    }
    const m11LightConstant = scratch.m11LightConstant;
    m11LightConstant.fill(0);
    if (material11 && legacyLighting && lights.length) {
      material11LightConstant(view.color, lights[0].color, lights[0].amplify,
        m11LightConstant);
    }
    gl.uniform1i(uniforms.uLightCount, lights.length);
    if (mode === 2 && lights.length) {
      const vectorLength = lights.length * 4;
      gl.uniform4fv(uniforms['uLightPosition[0]'], positions, 0, vectorLength);
      gl.uniform4fv(uniforms['uLightAttenuation[0]'], attenuationCenters, 0, vectorLength);
      gl.uniform4fv(uniforms['uLightColor[0]'], colors, 0, vectorLength);
      gl.uniform1fv(uniforms['uLightSpecular[0]'], lightSpecular, 0, lights.length);
    }
    gl.uniform4fv(uniforms.uM11LightConstant, m11LightConstant);
  }

  drawEffectGeometry(effect, viewport, matrices, activeLights = null) {
    if (effect.kind === 'water' && effect.geometry) {
      const geometry = this.geometry.get(effect.geometry, undefined, effect.opId);
      const pass = effect.material?.passes?.[0] || { usage: 'other', renderPass: 0 };
      this.drawMeshItem({
        geometry, group: geometry.groups[0], material: effect.material, pass,
        job: { opId: effect.opId, matrix: effect.matrix, instances: null, passAdjust: 0 },
        renderPass: pass.renderPass | 0,
      }, viewport, matrices, activeLights);
      return;
    }
    if (effect.kind !== 'chain-line' || effect.points?.length < 2) return;
    const gl = this.gl;
    this.dynamicChainVAO ||= gl.createVertexArray();
    this.dynamicChainPositionBuffer ||= gl.createBuffer();
    this.dynamicChainUVBuffer ||= gl.createBuffer();
    this.dynamicChainIndexBuffer ||= gl.createBuffer();
    const ribbon = buildChainRibbon(effect, matrices);
    gl.bindVertexArray(this.dynamicChainVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamicChainPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ribbon.positions, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.disableVertexAttribArray(1); gl.vertexAttrib3f(1, 0, 1, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamicChainUVBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ribbon.uvs, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(9); gl.vertexAttribPointer(9, 2, gl.FLOAT, false, 0, 0);
    gl.disableVertexAttribArray(3); gl.vertexAttrib4f(3, 64 / 255, 64 / 255, 64 / 255, 1);
    gl.disableVertexAttribArray(4); gl.vertexAttrib4f(4, 1, 0, 0, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.dynamicChainIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ribbon.indices, gl.DYNAMIC_DRAW);
    // ChainLine's marker endpoints and generated ribbon are already in world
    // space. Water remains on drawMeshItem above and keeps its authored model
    // matrix; only this effect binds an identity instance transform.
    const matrix = this.materialScratch?.identityMatrix || mat4Identity();
    for (let column = 0; column < 4; column++) {
      const location = 5 + column;
      gl.disableVertexAttribArray(location);
      gl.vertexAttrib4fv(location, matrix.subarray(column * 4, column * 4 + 4));
      gl.vertexAttribDivisor(location, 0);
    }
    const material = this.viewMaterial(effect.material, effect.material?.passes?.[0]);
    this.configureMaterialState(material);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uniforms.uViewProjection, false, matrices.viewProjection);
    gl.uniform4f(this.uniforms.uUVTransform, 1, 1, 0, 0);
    this.bindMaterial(material, viewport, matrices, activeLights, matrix);
    gl.drawElements(gl.TRIANGLES, ribbon.indices.length, gl.UNSIGNED_SHORT, 0);
    this.drawCalls++;
    this.triangles += ribbon.indices.length / 3;
  }

  drawMaterial11Insert(action) {
    if (action !== 'clear-destination-alpha' && action !== 'add-destination-alpha') return;
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    if (action === 'clear-destination-alpha') gl.blendFunc(gl.ZERO, gl.SRC_COLOR);
    else gl.blendFunc(gl.DST_ALPHA, gl.ONE);
    gl.colorMask(true, true, true, true);
    this.drawFullscreen(this.textures.fallbackTexture(), 7);
  }

  configureFullscreenState(material) {
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.CULL_FACE);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.STENCIL_TEST);
    const flags = material?.parameters?.[0] >>> 0;
    const blend = flags & 0xf000;
    if (!blend) gl.disable(gl.BLEND);
    else {
      gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD);
      if (blend === 0x1000) gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      else if (blend === 0x2000) gl.blendFunc(gl.ONE, gl.ONE);
      else if (blend === 0x3000) gl.blendFunc(gl.DST_COLOR, gl.ZERO);
      else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_COLOR);
    }
    gl.colorMask(true, true, true, true);
  }

  materialColor(material) {
    const color = material?.system === '2.0' ? material.parameters?.[1] : material?.parameters?.[28];
    return colorARGB(color === undefined ? 0xffffffff : color >>> 0);
  }

  drawFullscreen(image, mode = 0, options = {}) {
    const gl = this.gl, uniforms = this.fullscreenUniforms;
    gl.useProgram(this.fullscreenProgram);
    gl.bindVertexArray(this.fullscreenVAO);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, image);
    gl.uniform1i(uniforms.uImage, 0);
    if (options.secondImage) {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, options.secondImage);
    } else if (options.materialTexture?.data) this.textures.bind(options.materialTexture, 1, { clamp: true });
    else { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures.fallbackTexture()); }
    gl.uniform1i(uniforms.uMaterialTexture, 1);
    gl.uniform1i(uniforms.uMode, mode);
    gl.uniform4fv(uniforms.uColor0, options.color0 || [1, 1, 1, 1]);
    gl.uniform4fv(uniforms.uColor1, options.color1 || [1, 1, 1, 1]);
    gl.uniform4fv(uniforms.uParameters, options.parameters || [0, 0, 0, 0]);
    gl.uniform4fv(uniforms['uColorCorrect[0]'], options.colorCorrect || new Float32Array(28));
    const texel = options.texel || [1 / this.width, 1 / this.height];
    gl.uniform2f(uniforms.uTexel, texel[0], texel[1]);
    gl.uniform4fv(uniforms.uUVRect, options.uvRect || [0, 0, 1, 1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.drawCalls++;
  }

  applyGlare(image, parameters, options = {}) {
    const plan = glarePlan(parameters);
    const sizes = [[512, 256], [256, 128]];
    let current = {
      color: image, width: this.width, height: this.height,
      uvRect: options.uvRect || [0, 0, 1, 1],
    };

    for (let level = 0; level < 2; level++) {
      if (!plan.downsample[level]) continue;
      const [width, height] = sizes[level];
      const target = this.glareTarget(level, width, height);
      this.bindDestination(target);
      this.drawFullscreen(current.color, plan.copyDownsample ? 0 : 1, {
        uvRect: current.uvRect,
        texel: [1 / current.width, 1 / current.height],
      });
      current = { color: target.color, width, height, target, uvRect: [0, 0, 1, 1] };
    }

    const selected = current;
    const ping = this.glareTarget(2, selected.width, selected.height);
    let read = selected;
    let write = { color: ping.color, width: ping.width, height: ping.height, target: ping };
    const swapPass = (mode, drawOptions) => {
      this.bindDestination(write.target);
      this.drawFullscreen(read.color, mode, drawOptions);
      const previous = read;
      read = { color: write.color, width: write.width, height: write.height,
        target: write.target, uvRect: [0, 0, 1, 1] };
      write = { color: previous.color, width: previous.width, height: previous.height,
        target: previous.target };
    };

    swapPass(4, {
      color0: plan.threshold,
      parameters: [plan.grayScale, plan.grayMix, 0, 0],
      uvRect: [0, 0, 1, 1],
    });
    const aspect = selected.width / selected.height;
    for (const stage of plan.stages) {
      if (!(stage.blur > 0)) continue;
      swapPass(5, {
        parameters: [stage.blur, 0, 1, 0], uvRect: [0, 0, 1, 1],
      });
      swapPass(5, {
        parameters: [0, stage.blur * aspect, stage.amplify, 0], uvRect: [0, 0, 1, 1],
      });
    }

    this.bindDestination(options.destination ?? null);
    if (options.viewport) this.setViewportRegion(options.viewport);
    this.drawFullscreen(read.color, 6, {
      secondImage: image,
      color0: plan.glareColor,
      color1: plan.originalColor,
      parameters: [plan.addSmooth, 0, 0, 0],
      uvRect: options.uvRect || [0, 0, 1, 1],
    });
  }

  applyPost(image, job, options = {}) {
    this.configureFullscreenState(null);
    const p = job.parameters || [];
    if (job.kind === 'glare') {
      this.applyGlare(image, p, options);
    } else if (job.kind === 'color-correction') {
      this.drawFullscreen(image, 2, {
        colorCorrect: colorCorrectionConstants(p),
        uvRect: options.uvRect,
      });
    } else this.drawFullscreen(image, 0, { uvRect: options.uvRect });
  }

  dispose() {
    const gl = this.gl;
    this.masterTarget.dispose();
    this.prelightTarget.dispose();
    for (const target of this.targets) target.dispose();
    this.targets.length = 0;
    for (const target of this.glareTargets) target.dispose();
    this.glareTargets.length = 0;
    this.geometry.dispose();
    this.textures.dispose();
    this.material11Lookups.dispose();
    gl.deleteBuffer(this.instanceBuffer);
    gl.deleteBuffer(this.shadowPositionBuffer);
    gl.deleteBuffer(this.shadowExtrusionBuffer);
    gl.deleteBuffer(this.shadowIndexBuffer);
    gl.deleteVertexArray(this.shadowVAO);
    if (this.dynamicChainPositionBuffer) gl.deleteBuffer(this.dynamicChainPositionBuffer);
    if (this.dynamicChainUVBuffer) gl.deleteBuffer(this.dynamicChainUVBuffer);
    if (this.dynamicChainIndexBuffer) gl.deleteBuffer(this.dynamicChainIndexBuffer);
    if (this.dynamicChainVAO) gl.deleteVertexArray(this.dynamicChainVAO);
    if (this.layerPositionBuffer) gl.deleteBuffer(this.layerPositionBuffer);
    if (this.layerUVBuffer) gl.deleteBuffer(this.layerUVBuffer);
    if (this.layerIndexBuffer) gl.deleteBuffer(this.layerIndexBuffer);
    if (this.layerVAO) gl.deleteVertexArray(this.layerVAO);
    gl.deleteVertexArray(this.fullscreenVAO);
    gl.deleteProgram(this.fullscreenProgram);
    gl.deleteProgram(this.shadowProgram);
    gl.deleteProgram(this.program);
  }
}

export {
  appendMeshRenderItems,
  boundsIntersectsSphere,
  boundsOutsidePlanes,
  composeInstanceMatrices,
  createMaterialInsertTracker,
  GeometryCache,
  Material11LookupTextures,
  Renderer,
  buildShadowVolume,
  buildChainRibbon,
  colorCorrectionConstants,
  evaluateGlareComposite,
  evaluateGlareTone,
  evaluateMaterial11Combiner,
  evaluateMaterial11Specular,
  evaluateMaterial20Prelight,
  evaluateColorCorrection,
  groupRenderPass,
  geometryTopologyStats,
  glarePlan,
  effectRenderItem,
  forEachNativeHeadInsertedJob,
  layerQuadGeometry,
  legacyTransRVector,
  legacyVertexLighting,
  lightIntersectsWorldBounds,
  lightIntersectsViewFrustum,
  lightScissorRegion,
  lightSphereBounds,
  materialPassSortIdentity,
  materialInsertKind,
  materialInsertPlan,
  material11InsertAction,
  material11LightConstant,
  materialState,
  materialLightAttenuation,
  material11EnvironmentNormals,
  material20EnvironmentBumpDirection,
  material20EnvironmentDirection,
  material20EnvironmentEye,
  material20EnvironmentUV,
  material20UV,
  material20UVTransforms,
  material11UV,
  material11UVTransforms,
  VERTEX_SOURCE as materialVertexSource,
  FRAGMENT_SOURCE as materialFragmentSource,
  SHADOW_VERTEX_SOURCE as shadowVertexSource,
  FULLSCREEN_FRAGMENT_SOURCE as fullscreenFragmentSource,
  materialView,
  materialSlotFromGroup,
  meshHasAnimation,
  meshJobWorldBounds,
  normalizeGeometryBounds,
  normalizePreparedGeometry,
  prepareShadowTopology,
  renderMode,
  selectActiveLights,
  shadowCasterMayAffectView,
  shadowCasterUsesZFail,
  shadowGroupsForGeometry,
  shadowTopologyKey,
  shadowViewFrustumPlanes,
  shadowZFailVolumePlanes,
  sortRenderItems,
  tangentBasisNormal,
  transformGeometryBounds,
  resolveIPPOutputs,
  viewportClearFlags,
  viewFrustumPlanes,
  fitAspectRegion,
  viewportRegion,
};
