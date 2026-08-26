import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mat4Identity } from '../src/core.js';
import { parseKX } from '../src/kx.js';
import {
  appendMeshRenderItems,
  composeInstanceMatrices,
  materialView,
  renderMode,
} from '../src/renderer.js';

const document = parseKX(readFileSync(new URL(
  '../assets/debris_party.kx', import.meta.url,
)));

function dependencyPath(root, target) {
  const queue = [root];
  const parent = new Map([[root, null]]);
  while (queue.length) {
    const id = queue.shift();
    if (id === target) {
      const path = [];
      for (let cursor = id; cursor !== null; cursor = parent.get(cursor)) path.push(cursor);
      return path.reverse();
    }
    const operation = document.operations[id];
    for (const reference of [...operation.inputs, ...operation.links]) {
      if (!Number.isInteger(reference) || parent.has(reference)) continue;
      parent.set(reference, id);
      queue.push(reference);
    }
  }
  return null;
}

function eventIndicesUsing(operation) {
  const result = [];
  for (let index = 0; index < document.events.length; index++) {
    if (dependencyPath(document.events[index].operation, operation)) result.push(index);
  }
  return result;
}

const fontGreetings = document.operations.filter(operation =>
  operation.classId === 0x0133 && operation.strings.some(text => /andromeda|hund/i.test(text)));
assert.deepEqual(fontGreetings.map(operation => operation.id), [
  12325, 12362, 12377, 12453, 13825, 13909, 14141,
]);

// First chronological "andromeda software": event 47, beat 832..848
// (04:14.694..04:19.592). Its direct Center/Transform path deliberately has
// no MatLink, so native renders Font3D cluster 1 with DefaultMat.
assert.equal(document.operations[14141].strings[0], 'andromedasoftwaredevelopment.');
assert.deepEqual(eventIndicesUsing(14141), [3, 47]);
assert.deepEqual(dependencyPath(document.events[47].operation, 14141), [
  14189, 14188, 14185, 14181, 14178, 14143, 14142, 14141,
]);
assert.deepEqual([
  document.events[47].start / 65536,
  document.events[47].end / 65536,
], [832, 848]);

// The next occurrence reuses the same Font3D source through Explode and an
// explicit Material11 MatInput. A fix must not special-case the text/op id.
assert.deepEqual(dependencyPath(document.events[3].operation, 14141), [
  14175, 14174, 14172, 14169, 14168, 14167, 14166, 14165, 14163, 14143, 14142, 14141,
]);
assert.equal(document.operations[14165].classId, 0x0117); // MatInput
assert.deepEqual(document.operations[14165].inputs, [14163, 14164]);
assert.equal(document.operations[14164].classId, 0x00d0); // Material11
assert.deepEqual([
  document.events[3].start / 65536,
  document.events[3].end / 65536,
], [848, 880]);

// First "hund": event 5, beat 1020..1036 (05:12.245..05:17.143). The large
// Georgia word (12377) is another unlinked default cluster. The same scene
// also contains a separately authored Arial copy (12362 -> M20 12364), which
// must retain its explicit material.
assert.deepEqual(eventIndicesUsing(12377), [5]);
assert.deepEqual(dependencyPath(document.events[5].operation, 12377), [
  12424, 12423, 12420, 12381, 12380, 12379, 12378, 12377,
]);
assert.deepEqual(dependencyPath(document.events[5].operation, 12362), [
  12424, 12423, 12420, 12381, 12368, 12367, 12366, 12365, 12362,
]);
assert.equal(document.operations[12365].classId, 0x0110); // MatLink
assert.deepEqual(document.operations[12365].links, [12364]);
assert.equal(document.operations[12364].classId, 0x00d3); // Material20
assert.deepEqual([
  document.events[5].start / 65536,
  document.events[5].end / 65536,
], [1020, 1036]);

// Later authored variants: 12453 remains DefaultMat; 13825/13909 are explicit
// M20 variants (the bright 13825 destruction scene intentionally blooms).
assert.deepEqual(eventIndicesUsing(12453), [2]);
assert.deepEqual(eventIndicesUsing(13825), [8, 50]);
assert.deepEqual(eventIndicesUsing(13909), [21]);
assert.deepEqual(document.operations[13828].links, [13827]);
assert.deepEqual(document.operations[13912].links, [13911]);
assert.equal(document.operations[13827].parameters[1] >>> 0, 0xffb8c8d0);
assert.equal(document.operations[13911].parameters[1] >>> 0, 0xff809098);
assert.deepEqual([
  document.events[2].start / 65536, document.events[2].end / 65536,
  document.events[8].start / 65536, document.events[8].end / 65536,
  document.events[21].start / 65536, document.events[21].end / 65536,
  document.events[50].start / 65536, document.events[50].end / 65536,
], [1036, 1056, 1056, 1088, 1088, 1136, 1232, 1248]);

// The long Font3D greetings strip is dormant in the shipped event graph.
assert.equal(document.operations[12325].strings[0],
  'andromedasoftwaredevelopment.fairlight.bauknecht.traction.equinox.');
assert.deepEqual(eventIndicesUsing(12325), []);

// Glare really is active beside both first scene events; removing/skipping it
// would break windows and the later explicit glowing text. The material path,
// not the IPP schedule, is the source of the false contribution.
assert.deepEqual(dependencyPath(document.events[77].operation, 16416), [
  16419, 16418, 16417, 16416,
]);
assert.deepEqual(dependencyPath(document.events[86].operation, 16444), [
  16447, 16446, 16445, 16444,
]);
assert.deepEqual([
  document.events[77].start / 65536, document.events[77].end / 65536,
  document.events[86].start / 65536, document.events[86].end / 65536,
], [832, 848, 1020, 1036]);
assert.equal(document.operations[16416].parameters[2] >>> 0, 0xff808080);
assert.equal(document.operations[16444].parameters[2] >>> 0, 0xff808080);

// Lock the released constructor chain that gives every non-deleted MinMesh
// cluster its light-dependent fallback material.
const nativeMinMesh = readFileSync(new URL('../vendor/wz3/genminmesh.cpp', import.meta.url), 'utf8');
const nativeMesh = readFileSync(new URL('../vendor/wz3/genmesh.cpp', import.meta.url), 'utf8');
const nativeOverlay = readFileSync(new URL('../vendor/wz3/genoverlay.cpp', import.meta.url), 'utf8');
const nativeMaterial11 = readFileSync(new URL(
  '../vendor/wz3/materials/material11.cpp', import.meta.url,
), 'utf8');
const nativeMaterial11Header = readFileSync(new URL(
  '../vendor/wz3/materials/material11.hpp', import.meta.url,
), 'utf8');
assert.match(nativeMesh,
  /Mtrl\.Count = 2;[\s\S]*?Mtrl\[0\]\.Material = 0;[\s\S]*?Mtrl\[1\]\.Material = GenOverlayManager->DefaultMat;/);
assert.match(nativeMinMesh,
  /Clusters\.Add\(\)->Init\(0\);[\s\S]*?AddCluster\(0,0\);/);
assert.match(nativeMinMesh,
  /sInt GenMinMesh::AddCluster[\s\S]*?if\(mtrl==0\)\s+mtrl = GenOverlayManager->DefaultMat;/);
assert.match(nativeMinMesh,
  /font3DBeginCB[\s\S]*?face->Cluster = 1;/);
assert.match(nativeOverlay,
  /Mtrl\[GENOVER_DEFAULT\]->LightFlags = sMLF_BUMPX;[\s\S]*?Color\[0\] = 0x00c0c0c0;[\s\S]*?Color\[1\] = 0x00404040;[\s\S]*?SpecPower = 32\.0f;[\s\S]*?DefaultMat->AddPass\(Mtrl\[GENOVER_DEFAULT\],ENGU_LIGHT,MPP_STATIC\);/);
assert.match(nativeMaterial11, /TFlags\[i\] = sMTF_MIPMAPS;/);
assert.match(nativeMaterial11Header, /#define sMTF_MIPMAPS\s+0x0002/);

const fallbackView = materialView(null, { usage: 'other', renderPass: 0 });
assert.equal(fallbackView.system, '1.1');
assert.equal(fallbackView.usage, 'light');
assert.equal(fallbackView.color, 0x00c0c0c0);
assert.equal(fallbackView.lightFlags, 0x0005);
assert.deepEqual(fallbackView.textureFlags, [2, 2, 2, 2]);
assert.equal(renderMode(fallbackView), 2);

// The renderer-wide unresolved-material fallback is intentional: native
// GenMesh puts DefaultMat in active slot 1, and native GenMinMesh resolves
// every non-deleted null cluster in AddCluster (and while loading streams).
// Thus an active prepared group without a material always denotes DefaultMat,
// not a separate unlit material class. The explicit mesh sentinel follows the
// same path and must not retain an embedder-provided placeholder pass.
assert.equal(materialView({
  kind: 'material', system: 'default', passes: [],
}, { usage: 'other' }).usage, 'light');

// The paint planner must duplicate DefaultMat per intersecting light. Before
// the fix this yielded one unlit `other` item with constant RGB 0xb8b8b8,
// which necessarily survived the active 0x808080 glare threshold.
const identity = mat4Identity();
const bounds = { minimum: [-0.5, -0.5, 0], maximum: [0.5, 0.5, 1] };
const geometry = {
  bounds,
  groups: [{ material: null, start: 0, count: 3, bounds }],
  materials: [],
};
const job = { opId: 14141, mesh: {}, matrix: identity, passAdjust: 0 };
const light = {
  opId: 14182, kind: 'point', position: [0, 0, 2], range: 8,
  amplify: 1, color: 0xffffffff, flags: 0,
};
const items = [];
appendMeshRenderItems(items, job, geometry, [light], {
  viewProjection: identity,
}, composeInstanceMatrices(job));
assert.equal(items.length, 1);
assert.equal(items[0].pass.usage, 'light');
assert.equal(items[0].light, light);

console.log('greeting occurrence and DefaultMat glare regressions passed');
