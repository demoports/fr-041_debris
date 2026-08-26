// Semantic representation of Debris' material 1.1 and 2.0 operators. This
// preserves the old state words and pass split for the WebGL2 shader compiler.

const USAGE = Object.freeze({
  BASE: 'base',
  PRELIGHT: 'prelight',
  SHADOW: 'shadow',
  LIGHT: 'light',
  POSTLIGHT: 'postlight',
  POSTLIGHT2: 'postlight2',
  AMBIENT: 'ambient',
  OTHER: 'other',
});

const PROGRAM = Object.freeze({
  STATIC: 'static',
  INSTANCES: 'instances',
  SPRITES: 'sprites',
  THICK_LINES: 'thick-lines',
  SHADOW: 'shadow',
});

function cloneParameters(parameters) {
  return Array.from(parameters, value => value == null ? null : value);
}

function textureSummary(texture) {
  if (!texture) return null;
  return {
    kind: texture.kind || null,
    width: texture.width ?? texture.sizeX ?? null,
    height: texture.height ?? texture.sizeY ?? null,
    opId: texture.opId ?? null,
  };
}

class Material {
  constructor(system, parameters, textures) {
    this.kind = 'material';
    this.system = system;
    this.parameters = cloneParameters(parameters);
    // Material11 compiles words 0..23 and pass topology once during Init.
    // Exec only mutates selected float/color fields in each compiled pass.
    this.initialParameters = cloneParameters(parameters);
    this.textures = textures.slice();
    this.passes = [];
    // Renderer-side material views are intentionally cached because compiling
    // the legacy state words allocates several small arrays.  Exec can replace
    // animated parameters after precalc, so give that cache an explicit
    // generation to validate instead of keying only by object identity.
    this.version = 0;
  }

  update(parameters) {
    if (this.parameters.length === parameters.length) {
      let unchanged = true;
      for (let index = 0; index < parameters.length; index++) {
        const value = parameters[index] == null ? null : parameters[index];
        if (!Object.is(this.parameters[index], value)) { unchanged = false; break; }
      }
      if (unchanged) return false;
    }
    this.parameters = cloneParameters(parameters);
    this.version++;
    return true;
  }

  summary() {
    return {
      kind: this.kind,
      system: this.system,
      textures: this.textures.map(textureSummary),
      passes: this.passes.map(pass => ({
        usage: pass.usage,
        program: pass.program,
        renderPass: pass.renderPass,
        size: pass.size,
        aspect: pass.aspect,
      })),
      parameters: this.parameters,
    };
  }
}

function resolveLinkedTextures(call, count) {
  const textures = new Array(count).fill(null);
  for (let i = 0; i < count; i++) {
    const value = call.links[i];
    if (value?.kind === 'bitmap' || value?.pixels || value?.data) textures[i] = value;
  }
  return textures;
}

function pass(usage, program, renderPass = 0, size = 1, aspect = 1, state = null) {
  return { usage, program, renderPass: renderPass | 0, size, aspect, state };
}

function setPassMaterial(materialPass, material) {
  Object.defineProperty(materialPass, 'material', {
    configurable: true,
    writable: true,
    enumerable: false,
    value: material,
  });
  return materialPass;
}

function configureMaterial11(material) {
  const p = material.parameters;
  const flags = (p[48] || 0) >>> 0;
  const renderPass = (p[49] || 0) | 0;
  material.multiFlags = flags;
  material.shaderMask = (p[50] || 0) >>> 0;
  material.passes.length = 0;

  if ((flags & 0x1e) === 0) {
    let program = flags & 0x800 ? PROGRAM.INSTANCES : PROGRAM.STATIC;
    let size = 1;
    let aspect = 1;
    if (flags & 0x30000) {
      size = p[60] ?? 1;
      aspect = p[62] ?? 1;
      const finalizer = (flags >>> 16) & 3;
      if (finalizer === 1) program = PROGRAM.SPRITES;
      else if (finalizer === 2) program = PROGRAM.THICK_LINES;
    }
    const phase = (flags >>> 20) & 15;
    const usages = [
      USAGE.BASE, USAGE.PRELIGHT, USAGE.AMBIENT, USAGE.SHADOW,
      USAGE.LIGHT, USAGE.POSTLIGHT, USAGE.POSTLIGHT2, USAGE.OTHER,
    ];
    material.passes.push(pass(
      phase === 0 ? USAGE.OTHER : usages[phase - 1] || USAGE.OTHER,
      program,
      renderPass,
      size,
      aspect,
      'material11-single',
    ));
    return;
  }

  const program = flags & 0x800 ? PROGRAM.INSTANCES : PROGRAM.STATIC;
  // The original always creates its depth/base phase once any multipass bit
  // is enabled, then conditionally appends the remaining phases.
  material.passes.push(pass(USAGE.BASE, program, renderPass, 1, 1, 'material11-base'));
  if (flags & 0x02) material.passes.push(pass(USAGE.LIGHT, program, renderPass, 1, 1, 'material11-light'));
  if (flags & 0x04) material.passes.push(pass(USAGE.SHADOW, PROGRAM.SHADOW, renderPass, 1, 1, 'shadow-volume'));
  if (flags & 0x08) material.passes.push(pass(USAGE.POSTLIGHT, program, renderPass, 1, 1, 'material11-postlight'));
  if (flags & 0x10) material.passes.push(pass(USAGE.POSTLIGHT2, program, renderPass, 1, 1, 'material11-postlight2'));
}

function initMaterial11(call) {
  const textures = resolveLinkedTextures(call, 7);
  const material = new Material('1.1', call.parameters, textures);
  material.base = {
    flags: (call.parameters[0] || 0) >>> 0,
    specialFlags: (call.parameters[1] || 0) >>> 0,
    lightFlags: (call.parameters[2] || 0) >>> 0,
    textureFlags: call.parameters.slice(4, 8).map(value => (value || 0) >>> 0),
    combiners: call.parameters.slice(8, 21).map(value => (value || 0) >>> 0),
    alphaCombiner: (call.parameters[23] || 0) >>> 0,
  };
  configureMaterial11(material);

  // GenMaterialPass owns a reference to the exact sMaterial instance that
  // created it. Preserve that ownership when Material_Add chains materials;
  // animated downstream parameters must not overwrite an upstream pass.
  for (const ownPass of material.passes) setPassMaterial(ownPass, material);

  const upstream = call.inputs[0];
  if (upstream?.kind === 'material') {
    material.passes.unshift(...upstream.passes.map(value =>
      setPassMaterial({ ...value }, value.material || upstream)));
  }
  return material;
}

function execMaterial11(call) {
  call.op.execInputs(call.environment);
  call.op.cache.update(call.parameters);
  const flags = (call.parameters[48] || 0) >>> 0;
  if (flags & 0x30000) {
    for (const materialPass of call.op.cache.passes) {
      if (materialPass.usage !== USAGE.OTHER) continue;
      materialPass.size = call.parameters[60] ?? 1;
      materialPass.aspect = call.parameters[62] ?? 1;
    }
  }
  // Pass topology stays fixed after Init, as in the original. Dynamic words
  // are consumed directly by the shader state compiler at draw time.
}

function material20View(parameters, compiledParameters = parameters) {
  return {
    // sMaterial20Base compiles flags, sampler state and source usage in its
    // constructor. UpdatePara replaces the values read by Set(), but does not
    // rebuild those setups or shaders.
    flags: (compiledParameters[0] || 0) >>> 0,
    runtimeFlags: (parameters[0] || 0) >>> 0,
    diffuse: (parameters[1] || 0) >>> 0,
    specular: (parameters[2] || 0) >>> 0,
    specularPower: parameters[3] || 0,
    parallaxStrength: parameters[4] || 0,
    textureFlags: compiledParameters.slice(8, 12).map(value => (value || 0) >>> 0),
    textureScale: parameters.slice(12, 16),
    lightFlags: compiledParameters.slice(16, 20).map(value => (value || 0) >>> 0),
    lightScale: parameters.slice(20, 24),
    srt1: parameters.slice(24, 33),
    srt2: parameters.slice(33, 38),
    colorCorrection: parameters.slice(38, 42),
    environmentFlags: (compiledParameters[18] || 0) >>> 0,
    runtimeEnvironmentFlags: (parameters[18] || 0) >>> 0,
  };
}

function initMaterial20(call) {
  const textures = resolveLinkedTextures(call, 7);
  const inputMap = [8, 9, 10, 11, 16, 17, 18];
  for (let i = 0; i < textures.length; i++) {
    const inputIndex = ((((call.parameters[inputMap[i]] || 0) >>> 0) & 0x03000000) >>> 24) - 1;
    if (inputIndex >= 0 && inputIndex < 3) {
      const input = call.inputs[inputIndex];
      if (input?.kind === 'bitmap' || input?.pixels || input?.data) textures[i] = input;
    }
  }

  const material = new Material('2.0', call.parameters, textures);
  const view = material20View(call.parameters, material.initialParameters);
  material.view = view;
  material.passes.push(pass(USAGE.BASE, PROGRAM.STATIC, 0, 1, 1, 'material20-zfill'));
  material.passes.push(pass(USAGE.PRELIGHT, PROGRAM.STATIC, 0, 1, 1, 'material20-texture'));
  material.passes.push(pass(USAGE.AMBIENT, PROGRAM.STATIC, 0, 1, 1, 'material20-ambient'));
  if (view.flags & 0x10) material.passes.push(pass(USAGE.SHADOW, PROGRAM.SHADOW, 0, 1, 1, 'shadow-volume'));
  material.passes.push(pass(USAGE.LIGHT, PROGRAM.STATIC, 0, 1, 1, 'material20-light'));
  if (textures[6]) material.passes.push(pass(USAGE.POSTLIGHT, PROGRAM.STATIC, 0, 1, 1, 'material20-environment'));
  return material;
}

function execMaterial20(call) {
  call.op.execInputs(call.environment);
  const material = call.op.cache;
  if (material.update(call.parameters)) {
    material.view = material20View(call.parameters, material.initialParameters);
  }
}

const materialHandlers = {
  0x00d0: { init: initMaterial11, exec: execMaterial11 },
  0x00d3: { init: initMaterial20, exec: execMaterial20 },
};

export {
  Material,
  PROGRAM as MATERIAL_PROGRAM,
  USAGE as MATERIAL_USAGE,
  materialHandlers,
};
