import assert from 'node:assert/strict';
import * as MaterialAPI from '../src/material.js';

const handlers = new Map(Object.entries(MaterialAPI.materialHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...MaterialAPI, handlers };
const bitmap = id => ({ kind: 'bitmap', opId: id, width: 8, height: 8 });
const op = { cache: null, execInputs() {} };

const p11 = new Array(64).fill(0);
p11[34] = p11[35] = p11[36] = 1;
p11[43] = p11[44] = 1;
p11[60] = 2;
p11[62] = 0.5;
const call11 = {
  op,
  environment: {},
  inputs: [],
  links: [bitmap(0), null, null, null, null, null, null],
  parameters: p11,
};
const h11 = D.handlers.get(0xd0);
op.cache = h11.init(call11);
assert.equal(op.cache.system, '1.1');
assert.equal(op.cache.version, 0);
assert.equal(op.cache.passes.length, 1);
assert.equal(op.cache.passes[0].usage, 'other');
assert.equal(op.cache.passes[0].state, 'material11-single');
assert.equal(op.cache.textures[0].opId, 0);
const updated11 = p11.slice();
updated11[28] = 0xff123456;
h11.exec({ ...call11, parameters: updated11 });
assert.equal(op.cache.version, 1);
assert.equal(op.cache.parameters[28], 0xff123456);

p11[48] = 0x1e;
const multipass = h11.init({ ...call11, op: { execInputs() {} } });
assert.deepEqual(
  multipass.passes.map(value => value.usage),
  ['base', 'light', 'shadow', 'postlight', 'postlight2'],
);
assert.deepEqual(
  multipass.passes.map(value => value.state),
  ['material11-base', 'material11-light', 'shadow-volume', 'material11-postlight', 'material11-postlight2'],
);

p11[48] = 0x10000;
const sprites = h11.init({ ...call11, op: { execInputs() {} } });
assert.equal(sprites.passes[0].program, 'sprites');
assert.equal(sprites.passes[0].size, 2);
assert.equal(sprites.passes[0].aspect, 0.5);
const animatedSpriteOp = { execInputs() {} };
const animatedSprites = h11.init({ ...call11, op: animatedSpriteOp });
animatedSpriteOp.cache = animatedSprites;
const animatedSpriteParameters = p11.slice();
animatedSpriteParameters[60] = 3;
animatedSpriteParameters[62] = 0.75;
h11.exec({ ...call11, op: animatedSpriteOp, parameters: animatedSpriteParameters });
assert.equal(animatedSprites.passes[0].size, 3);
assert.equal(animatedSprites.passes[0].aspect, 0.75);

// MultiFlags phase insertion stores native ENGU_* + 1. AMBIENT occupies slot
// two; omitting it shifts every later phase to the wrong renderer bucket.
const nativePhaseUsages = [
  'base', 'prelight', 'ambient', 'shadow',
  'light', 'postlight', 'postlight2', 'other',
];
for (let phase = 1; phase <= nativePhaseUsages.length; phase++) {
  const parameters = p11.slice();
  parameters[48] = phase << 20;
  const phased = h11.init({ ...call11, parameters, op: { execInputs() {} } });
  assert.equal(phased.passes[0].usage, nativePhaseUsages[phase - 1]);
}

// Material_Add retains each GenMaterialPass' own material state. A downstream
// animated Material11 must not recolor/retexture passes inherited from input 0.
const upstreamParameters = p11.slice();
upstreamParameters[48] = 0;
upstreamParameters[28] = 0xff112233;
const upstream = h11.init({ ...call11, parameters: upstreamParameters,
  op: { execInputs() {} } });
const downstreamParameters = p11.slice();
downstreamParameters[48] = 0;
downstreamParameters[28] = 0xffaabbcc;
const downstream = h11.init({ ...call11, inputs: [upstream],
  parameters: downstreamParameters, op: { execInputs() {} } });
assert.equal(downstream.passes.length, 2);
assert.equal(downstream.passes[0].material, upstream);
assert.equal(downstream.passes[1].material, downstream);
assert.equal(downstream.passes[0].material.parameters[28], 0xff112233);
assert.equal(downstream.passes[1].material.parameters[28], 0xffaabbcc);
assert.doesNotThrow(() => JSON.stringify(downstream),
  'pass ownership stays non-enumerable and does not make material state cyclic');

const p20 = new Array(38).fill(0);
p20[0] = 0x10;
p20[8] = 0x01000000;
const inputBitmap = bitmap(10);
const environment = {};
const op20 = { execInputs() {} };
const call20 = {
  op: op20,
  environment,
  inputs: [inputBitmap],
  links: [bitmap(20), null, null, null, null, null, bitmap(26)],
  parameters: p20,
};
const h20 = D.handlers.get(0xd3);
op20.cache = h20.init(call20);
assert.equal(op20.cache.version, 0);
assert.equal(op20.cache.textures[0], inputBitmap);
assert.deepEqual(
  op20.cache.passes.map(value => value.usage),
  ['base', 'prelight', 'ambient', 'shadow', 'light', 'postlight'],
);
assert.deepEqual(
  op20.cache.passes.map(value => value.state),
  ['material20-zfill', 'material20-texture', 'material20-ambient', 'shadow-volume', 'material20-light', 'material20-environment'],
);
p20[3] = 64;
h20.exec(call20);
assert.equal(op20.cache.version, 1);
assert.equal(op20.cache.view.specularPower, 64);
const stableVersion = op20.cache.version;
const stableView = op20.cache.view;
h20.exec(call20);
assert.equal(op20.cache.version, stableVersion);
assert.equal(op20.cache.view, stableView);
p20[3] = 32;
h20.exec(call20);
assert.equal(op20.cache.version, stableVersion + 1);
assert.equal(op20.cache.view.specularPower, 32);

// UpdatePara changes Set-time constants, but native shader/setup compilation
// and sampler selection stay owned by the constructor's initial parameter set.
p20[0] = 0x40;
p20[8] = 0x3000;
p20[12] = 7;
p20[16] = 0x2301;
p20[18] = 0x10000;
p20[20] = 9;
p20[24] = 2;
h20.exec(call20);
assert.equal(op20.cache.view.flags, 0x10);
assert.equal(op20.cache.view.runtimeFlags, 0x40);
assert.equal(op20.cache.view.textureFlags[0], 0x01000000);
assert.equal(op20.cache.view.textureScale[0], 7);
assert.equal(op20.cache.view.lightFlags[0], 0);
assert.equal(op20.cache.view.lightScale[0], 9);
assert.equal(op20.cache.view.srt1[0], 2);
assert.equal(op20.cache.view.environmentFlags, 0);
assert.equal(op20.cache.view.runtimeEnvironmentFlags, 0x10000);
assert.deepEqual(op20.cache.passes.map(value => value.usage),
  ['base', 'prelight', 'ambient', 'shadow', 'light', 'postlight'],
  'compiled pass topology is unchanged when runtime flags move');

console.log('material pass and texture-resolution tests passed');
