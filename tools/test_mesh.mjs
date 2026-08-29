import assert from 'node:assert/strict';
import * as MeshAPI from '../src/mesh.js';

const handlers = new Map(Object.entries(MeshAPI.meshHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...MeshAPI, handlers };
const identitySRT = [1, 1, 1, 0, 0, 0, 0, 0, 0];
const arrayBytes = value => new Uint8Array(value.buffer, value.byteOffset, value.byteLength);

function invoke(id, parameters, inputs = [], links = []) {
  const handler = D.handlers.get(id);
  assert.equal(typeof handler, 'function', `class 0x${id.toString(16)} handler`);
  return handler({ runtime: null, environment: {}, op: {}, inputs, links, parameters, strings: [], splines: [] });
}

function verifyTopology(mesh) {
  for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex++) {
    const face = mesh.faces[faceIndex];
    if (face.edge < 0) continue;
    const loop = mesh.faceEdges(faceIndex);
    assert.ok(loop.length >= (face.material ? 3 : 1), `face ${faceIndex} degree`);
    for (const halfedge of loop) {
      assert.equal(mesh.getFaceId(halfedge), faceIndex, `face ${faceIndex} owns halfedge ${halfedge}`);
      assert.ok(mesh.getVertId(halfedge) >= 0 && mesh.getVertId(halfedge) < mesh.vertices.length);
      assert.equal(mesh.prevFaceEdge(mesh.nextFaceEdge(halfedge)), halfedge);
    }
  }
}

const cubeParameters = [1, 1, 1, 1, ...identitySRT];
const cube = invoke(0x81, cubeParameters);
assert.equal(cube.kind, 'mesh');
assert.equal(cube.Vert, cube.vertices);
assert.equal(cube.Edge, cube.edges);
assert.equal(cube.Face, cube.faces);
assert.deepEqual(
  { vertices: cube.vertices.length, edges: cube.edges.length, faces: cube.faces.length },
  { vertices: 24, edges: 12, faces: 6 },
);
assert.deepEqual(cube.faces.map(face => face.mask), [1, 2, 4, 8, 16, 32]);
assert.ok(cube.vertices.every(vertex => vertex.select));
assert.ok(cube.faces.every(face => face.select));
assert.ok(cube.edges.every(edge => edge.select));
assert.ok(cube.edges.every(edge => {
  const faces = edge.face.map(index => cube.faces[index]);
  return edge.mask === (faces[0].mask | faces[1].mask);
}), 'cube edges retain both native incident face-selection bits');
assert.deepEqual(cube.summary().bounds, { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] });
verifyTopology(cube);

const cubeOracle = cube.summary();
assert.deepEqual(
  { topologyHash: cubeOracle.topologyHash, vertexHash: cubeOracle.vertexHash },
  { topologyHash: 0xb304c922, vertexHash: 0xcb340fc5 },
);
assert.deepEqual(invoke(0x81, cubeParameters).summary(), cubeOracle);

// Dormant caches round-trip through flat storage without changing the public
// record shape or a single topology/attribute bit.
const compactRoundTrip = cube.clone();
compactRoundTrip.compact();
assert.deepEqual(compactRoundTrip.storageSummary(), {
  compact: true, released: false, vertices: 24, edges: 12, faces: 6,
});
assert.equal(compactRoundTrip._compact.vertexDirections, null, 'all-zero normal/tangent channel is implicit');
assert.equal(compactRoundTrip._compact.vertexColors, null, 'all-zero color channel is implicit');
assert.equal(compactRoundTrip._compact.vertexWeights, null, 'default weights are implicit');
assert.equal(compactRoundTrip._compact.vertexMatrices, null, 'default matrices are implicit');
assert.equal(compactRoundTrip.Vert, compactRoundTrip.vertices, 'classic alias expands the same record array');
assert.equal(compactRoundTrip.storageSummary().compact, false);
assert.deepEqual(compactRoundTrip.summary(), cubeOracle);

const compactChannels = cube.clone();
compactChannels.vertices[0].normal.set([-0, 2, Number.NaN, 0]);
compactChannels.vertices[0].tangent.set([3, 4, 5, 6]);
compactChannels.vertices[0].color.set([0.25, 0.5, 0.75, 1]);
compactChannels.vertices[0].weights.set([1, 2, 3, 4]);
compactChannels.vertices[0].matrices.set([5, 6, 7, 8]);
const channelOracle = compactChannels.summary();
compactChannels.compact();
assert.ok(compactChannels._compact.vertexDirections instanceof Float32Array);
assert.ok(compactChannels._compact.vertexColors instanceof Float32Array);
assert.ok(compactChannels._compact.vertexWeights instanceof Uint8Array);
assert.ok(compactChannels._compact.vertexMatrices instanceof Uint8Array);
compactChannels.ensureExpanded();
assert.ok(Object.is(compactChannels.vertices[0].normal[0], -0), 'negative zero survives flat storage');
assert.ok(Number.isNaN(compactChannels.vertices[0].normal[2]), 'NaN survives flat storage');
assert.deepEqual(Array.from(compactChannels.vertices[0].weights), [1, 2, 3, 4]);
assert.deepEqual(Array.from(compactChannels.vertices[0].matrices), [5, 6, 7, 8]);
assert.deepEqual(compactChannels.summary(), channelOracle);

const compactUV1 = cube.clone();
compactUV1.vertexMask |= D.MESH_FEATURE.UV1;
compactUV1.vertices[0].uv1.set([0.25, 0.75, 0, 1]);
compactUV1.compact();
assert.ok(compactUV1._compact.vertexUV1s instanceof Float32Array,
  'old GenMesh compact storage retains authored UV1');
assert.deepEqual(Array.from(compactUV1.prepare().uv1s.slice(0, 2)), [0.25, 0.75]);
const zeroUV1 = cube.clone();
zeroUV1.vertexMask |= D.MESH_FEATURE.UV1;
zeroUV1.compact();
assert.ok(zeroUV1.prepare().uv1s instanceof Float32Array,
  'an all-zero but declared UV1 channel remains available to rendering');
const noUV1Transform = D.Mesh_TransformEx(cube, 0, identitySRT,
  D.MESH_ATTR.UV1, D.MESH_ATTR.POS);
assert.deepEqual(noUV1Transform.vertices.map(vertex => Array.from(vertex.position)),
  cube.vertices.map(vertex => Array.from(vertex.position)),
  'VertMap rejects an attribute absent from the native vertex format');

const prepared = cube.prepare();
assert.ok(prepared.positions instanceof Float32Array);
assert.ok(prepared.normals instanceof Float32Array);
assert.ok(prepared.uvs instanceof Float32Array);
assert.ok(prepared.colors instanceof Float32Array);
assert.ok(prepared.indices instanceof Uint32Array);
assert.ok(prepared.shadowVertexMap instanceof Uint32Array);
assert.ok(prepared.shadowTriangleMask instanceof Uint8Array);
assert.equal(prepared.indices.length, 36);
assert.equal(prepared.shadowVertexMap.length, prepared.positions.length / 3);
assert.equal(prepared.shadowTriangleMask.length, prepared.indices.length / 3);
assert.deepEqual(prepared.groups.map(group => [group.materialIndex, group.start, group.count]), [[1, 0, 36]]);
assert.equal(cube.prepare(), prepared, 'prepare cache is stable');

// Old EngMesh serializes both explicit face slots of every SilEdge. A deleted
// neighbour before the next caster borrows that caster's Temp+1 plane instead
// of becoming an open boundary. Keep the native job before topology release.
const legacyShadowMaterial = { kind: 'material', passes: [{ usage: 'shadow' }] };
const makeLegacyShadowBoundary = () => {
  const mesh = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
  mesh.materials[1].material = legacyShadowMaterial;
  const incident = mesh.edges[0].face;
  const caster = Math.max(...incident);
  for (const face of mesh.faces) face.material = 0;
  mesh.faces[caster].material = 1;
  return mesh;
};
const expandedLegacyShadow = makeLegacyShadowBoundary().prepare().nativeShadow;
const compactLegacyMesh = makeLegacyShadowBoundary().compact();
const compactLegacyPrepared = compactLegacyMesh.prepare({ releaseTopology: true });
const compactLegacyShadow = compactLegacyPrepared.nativeShadow;
assert.equal(compactLegacyShadow.kind, 'genmesh-shadow-job');
assert.equal(compactLegacyShadow.planeCount, 2, 'reserved plane plus one caster plane');
assert.ok(Array.from(compactLegacyShadow.edgePlanes)
  .some((plane, index, values) => !(index & 1) && plane === 1 && values[index + 1] === 1),
'deleted neighbour aliases the following caster plane');
for (const key of ['sourceIndices', 'faces', 'trianglePlanes', 'planes',
  'edgeVertices', 'edgePlanes']) {
  assert.deepEqual(arrayBytes(compactLegacyShadow[key]), arrayBytes(expandedLegacyShadow[key]),
    `compact GenMesh preserves native shadow ${key}`);
}
assert.equal(compactLegacyMesh.topologyReleasedForPlayback, true);
assert.equal(compactLegacyPrepared.nativeShadow, compactLegacyShadow,
  'released playback geometry retains its self-contained native shadow job');

const playbackOnlyCube = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
playbackOnlyCube.compact();
const playbackPrepared = playbackOnlyCube.prepare({ releaseTopology: true });
assert.equal(playbackOnlyCube.storageSummary().preparedOnly, true);
assert.equal(playbackOnlyCube.prepare(), playbackPrepared,
  'immutable playback mesh keeps its self-contained render buffers');
assert.equal(playbackPrepared.shadowVertexMap.length, playbackPrepared.positions.length / 3);
assert.equal(playbackPrepared.shadowTriangleMask.length, playbackPrepared.indices.length / 3);
assert.throws(() => playbackOnlyCube.vertices, /released for immutable playback/,
  'released procedural topology cannot be silently reused by a modifier');
const playbackOnlyStats = D.meshStorageStats({ operations: [{ cache: playbackOnlyCube }] });
assert.equal(playbackOnlyStats.preparedOnly, 1);
assert.equal(playbackOnlyStats.expanded, 0);
assert.ok(playbackOnlyStats.preparedBytes > 0);

// A dormant old Mesh must be renderable without reconstructing its large
// vertex/edge/face object graph. Compare every render channel bit-for-bit with
// the legacy expanded route; the spy makes an accidental expansion explicit.
const preparedArrayKeys = [
  'positions', 'normals', 'tangents', 'colors', 'uvs', 'indices',
  'triangleMaterials', 'shadowVertexMap', 'shadowTriangleMask',
];
function assertPreparedEquivalent(actual, expected, label) {
  assert.equal(actual.kind, expected.kind, `${label} kind`);
  for (const key of preparedArrayKeys) {
    assert.equal(actual[key].constructor, expected[key].constructor, `${label} ${key} type`);
    assert.deepEqual(arrayBytes(actual[key]), arrayBytes(expected[key]), `${label} ${key} bits/order`);
  }
  assert.equal(actual.groups.length, expected.groups.length, `${label} group count`);
  for (let index = 0; index < actual.groups.length; index++) {
    const actualGroup = actual.groups[index];
    const expectedGroup = expected.groups[index];
    assert.equal(actualGroup.material, expectedGroup.material, `${label} group ${index} material`);
    assert.deepEqual(
      [actualGroup.materialIndex, actualGroup.pass, actualGroup.start, actualGroup.count],
      [expectedGroup.materialIndex, expectedGroup.pass, expectedGroup.start, expectedGroup.count],
      `${label} group ${index} range`,
    );
  }
  assert.deepEqual(arrayBytes(actual.bounds.min), arrayBytes(expected.bounds.min), `${label} minimum bits`);
  assert.deepEqual(arrayBytes(actual.bounds.max), arrayBytes(expected.bounds.max), `${label} maximum bits`);
  assert.deepEqual(
    actual.materials.map(slot => [slot.material, slot.pass | 0]),
    expected.materials.map(slot => [slot.material, slot.pass | 0]),
    `${label} material slots`,
  );
  assert.equal(actual.nativeShadow?.kind || null, expected.nativeShadow?.kind || null,
    `${label} native shadow kind`);
  if (actual.nativeShadow && expected.nativeShadow) {
    for (const key of ['sourceIndices', 'faces', 'trianglePlanes', 'planes',
      'edgeVertices', 'edgePlanes']) {
      assert.deepEqual(arrayBytes(actual.nativeShadow[key]), arrayBytes(expected.nativeShadow[key]),
        `${label} native shadow ${key}`);
    }
  }
}

function compareCompactPreparation(source, label, options = {}) {
  const compactMesh = source.clone().compact();
  const retainedStorage = compactMesh._compact;
  const expandedMesh = source.clone().compact();
  expandedMesh.ensureExpanded();
  const expected = expandedMesh.prepare();
  let expanded = false;
  const ensureExpanded = compactMesh.ensureExpanded;
  compactMesh.ensureExpanded = function () {
    expanded = true;
    return ensureExpanded.call(this);
  };
  const actual = compactMesh.prepare(options);
  assert.equal(expanded, false, `${label} stayed in compact storage`);
  assertPreparedEquivalent(actual, expected, label);
  if (options.releaseTopology) {
    assert.equal(compactMesh.topologyReleasedForPlayback, true, `${label} release marker`);
    assert.equal(compactMesh._compact, null, `${label} compact topology released`);
  } else {
    assert.equal(compactMesh._compact, retainedStorage, `${label} retained the original flat storage`);
  }
  return { actual, expected, compactMesh };
}

const compactGeneratedNormals = D.Mesh_Cube(2, 1, 1, 0, identitySRT);
compactGeneratedNormals.edges[0].crease |= D.MESH_FEATURE.NORMAL;
assert.equal(compactGeneratedNormals.gotNormals, false);
const generatedPair = compareCompactPreparation(
  compactGeneratedNormals, 'compact generated normals and first-vertex welding',
);
assert.equal(generatedPair.compactMesh.gotNormals, true);
assert.ok(generatedPair.compactMesh._compact.vertexDirections instanceof Float32Array);
assert.ok(Array.from(generatedPair.actual.shadowVertexMap).some((first, index) => first !== index),
  'compact preparation preserves welded physical first vertices');
generatedPair.compactMesh.ensureExpanded();
assert.equal(generatedPair.compactMesh.storageSummary().compact, false,
  'prepared compact topology still expands through the public record API');
const expandedGeneratedNormals = new Float32Array(generatedPair.compactMesh.vertices.length * 3);
for (let index = 0; index < generatedPair.compactMesh.vertices.length; index++) {
  expandedGeneratedNormals.set(generatedPair.compactMesh.vertices[index].normal.subarray(0, 3), index * 3);
}
assert.deepEqual(arrayBytes(expandedGeneratedNormals), arrayBytes(generatedPair.actual.normals),
  'directly generated compact normals survive public expansion exactly');
generatedPair.compactMesh.compact();
assert.equal(generatedPair.compactMesh.storageSummary().compact, true,
  'prepared topology can return to dormant compact storage');

const deferredReleaseMesh = D.Mesh_Cube(1, 1, 1, 0, identitySRT).compact();
let deferredReleasedGeometry = null;
Object.defineProperty(deferredReleaseMesh, '_playbackTopologyRelease', {
  configurable: true,
  writable: true,
  value: { release: geometry => { deferredReleasedGeometry = geometry; } },
});
const deferredPrepared = deferredReleaseMesh.prepare();
assert.equal(deferredReleasedGeometry, deferredPrepared,
  'compact preparation preserves the deferred playback release callback');
assert.equal(deferredReleaseMesh.topologyReleasedForPlayback, true);
assert.equal(deferredReleaseMesh._playbackTopologyRelease, null);

const materialA = { kind: 'material', name: 'compact group A' };
const materialB = { kind: 'material', name: 'compact group B' };
const groupedCompactSource = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
const testSlot = (material, pass) => ({
  material, pass, jobIds: new Int32Array(16), remap: 0,
});
groupedCompactSource.materials.push(testSlot(materialA, 7), testSlot(materialB, 11));
groupedCompactSource.faces[0].material = 2;
groupedCompactSource.faces[1].material = 3;
groupedCompactSource.faces[1].used = false;
groupedCompactSource.faces[2].material = 0;
groupedCompactSource.faces[3].material = 2;
groupedCompactSource.faces[3].used = false;
groupedCompactSource.faces[4].material = 0;
groupedCompactSource.faces[5].material = 3;
const groupedPair = compareCompactPreparation(
  groupedCompactSource, 'compact material buckets and shadow-use flags', { releaseTopology: true },
);
assert.deepEqual(groupedPair.actual.groups.map(group => [group.materialIndex, group.pass]), [[2, 7], [3, 11]]);
assert.deepEqual(Array.from(groupedPair.actual.shadowTriangleMask), [1, 1, 0, 0, 0, 0, 1, 1]);
assert.deepEqual(Array.from(groupedPair.actual.triangleMaterials), [2, 2, 2, 2, 3, 3, 3, 3]);

// When normals are already authored, the fast path copies every compact
// float bit instead of deriving them again (including negative zero).
const authoredDirections = D.Mesh_Cylinder(8, 2, 0, 1, 0);
authoredDirections.needNormals();
for (let index = 0; index < authoredDirections.vertices.length; index++) {
  const vertex = authoredDirections.vertices[index];
  vertex.color.set([Math.fround(index / 11), index === 0 ? -0 : 0.25, Math.fround(index * 0.125), 1]);
  vertex.uv[0] = Math.fround(index / 13);
  vertex.uv[1] = index === 0 ? -0 : Math.fround(index / 17);
}
authoredDirections.vertices[0].normal[0] = -0;
authoredDirections.vertices[0].tangent[3] = -0;
const authoredPair = compareCompactPreparation(authoredDirections, 'compact authored attribute channels');
assert.equal(new Uint32Array(authoredPair.actual.normals.buffer)[0], 0x80000000,
  'authored negative-zero normal bit survives compact preparation');
assert.equal(new Uint32Array(authoredPair.actual.uvs.buffer)[1], 0x80000000,
  'authored negative-zero UV bit survives compact preparation');

// A material-only branch shares dormant immutable topology. Deriving missing
// normals must replace, not modify, a shared pre-normal direction channel.
const sharedDirectionSource = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
sharedDirectionSource.vertices[0].normal[0] = 0.25;
sharedDirectionSource.compact();
const sharedDirections = sharedDirectionSource._compact.vertexDirections;
const sharedDirectionBranch = D.Mesh_MatLink(sharedDirectionSource, materialA, 0, 3);
assert.equal(sharedDirectionBranch._compact.vertexDirections, sharedDirections);
sharedDirectionBranch.prepare();
assert.notEqual(sharedDirectionBranch._compact.vertexDirections, sharedDirections,
  'derived normals retain compact material-branch copy-on-write');
assert.equal(sharedDirections[0], 0.25, 'source compact directions were not modified');
assert.equal(sharedDirectionSource.gotNormals, false, 'source normal state was not modified');

// Every mutator has an explicit JS copy-on-write boundary.
const originalPosition = Array.from(cube.vertices[0].position);
const moved = invoke(0x88, [0, 1, 1, 1, 0, 0, 0, 2, -3, 4], [cube]);
assert.notEqual(moved, cube);
assert.deepEqual(Array.from(cube.vertices[0].position), originalPosition);
assert.deepEqual(moved.summary().bounds, { min: [1.5, -3.5, 3.5], max: [2.5, -2.5, 4.5] });

const material = { kind: 'material', name: 'linked cache' };
const linked = invoke(0x96, [0, 8], [cube], [material]);
assert.equal(linked.materials.length, 3);
assert.equal(linked.materials[2].material, material);
assert.ok(linked.faces.every(face => face.material === 2));
assert.ok(cube.faces.every(face => face.material === 1));

// Material-only branches of a dormant mesh share their large immutable flat
// topology while retaining private selection and material channels.
const compactMaterialSource = cube.clone().compact();
const compactMaterialBranch = D.Mesh_MatLink(compactMaterialSource, material, 0, 8);
assert.ok(compactMaterialBranch._compact);
assert.equal(compactMaterialBranch._compact.vertexPositions.buffer,
  compactMaterialSource._compact.vertexPositions.buffer);
assert.equal(compactMaterialBranch._compact.edgeInts.buffer,
  compactMaterialSource._compact.edgeInts.buffer);
assert.notEqual(compactMaterialBranch._compact.vertexBytes.buffer,
  compactMaterialSource._compact.vertexBytes.buffer);
assert.notEqual(compactMaterialBranch._compact.faceInts.buffer,
  compactMaterialSource._compact.faceInts.buffer);
assert.ok(Array.from(compactMaterialSource._compact.faceInts)
  .filter((_, index) => index % 5 === 0).every(materialIndex => materialIndex === 1));
assert.ok(Array.from(compactMaterialBranch._compact.faceInts)
  .filter((_, index) => index % 5 === 0).every(materialIndex => materialIndex === 2));
const sharedCompactStats = D.meshStorageStats({ operations: [
  { cache: compactMaterialSource }, { cache: compactMaterialBranch },
] });
const nominalCompactBytes = mesh => Object.values(mesh._compact)
  .reduce((bytes, value) => bytes + (ArrayBuffer.isView(value) ? value.byteLength : 0), 0);
assert.ok(sharedCompactStats.compactBytes <
  nominalCompactBytes(compactMaterialSource) + nominalCompactBytes(compactMaterialBranch),
  'storage telemetry counts shared backing buffers only once');
assert.equal(compactMaterialBranch.prepare().groups[0].material, material);
assert.notEqual(compactMaterialSource.prepare().groups[0].material, material);

const selected = invoke(0x86, [0x010100, D.MESH_SELECT.SET, 0, 0, 0, 2, 2, 2], [cube]);
assert.ok(selected.faces.every(face => face.mask & 1));
assert.ok(selected.vertices.every(vertex => vertex.mask & 1));
const deleted = invoke(0x93, [1], [selected]);
assert.equal(deleted.summary().activeFaces, 0);
assert.equal(selected.summary().activeFaces, 6);

const inverted = invoke(0xa3, [], [cube]);
const uninverted = invoke(0xa3, [], [inverted]);
assert.deepEqual(uninverted.summary().bounds, cube.summary().bounds);
verifyTopology(uninverted);

const projected = invoke(0xa5, [0, ...identitySRT, 1], [cube]);
assert.notEqual(projected.summary().vertexHash, cube.summary().vertexHash);
assert.deepEqual(invoke(0xa5, [0, ...identitySRT, 1], [cube]).summary(), projected.summary());

// Multiply exposes words 0..16 only. Altering packed word 17 must be inert.
const multiplyWords = [1, 1, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0x12345678];
const multipliedA = invoke(0x95, multiplyWords, [cube]);
const multipliedB = invoke(0x95, [...multiplyWords.slice(0, 17), 0x87654321], [cube]);
assert.deepEqual(multipliedA.summary(), multipliedB.summary());
assert.equal(multipliedA.vertices.length, 48);
assert.deepEqual(multipliedA.summary().bounds, { min: [-0.5, -0.5, -0.5], max: [1.5, 0.5, 0.5] });
assert.ok(multipliedA.vertices.slice(0, 24).every(vertex => !vertex.select));
assert.ok(multipliedA.vertices.slice(24).every(vertex => vertex.select),
  'Multiply leaves only its final appended instance selected');

const rangedMultiplySource = cube.clone();
const rangedMultiplyNormals = cube.clone().needNormals();
const rangedMultiply = D.Mesh_Multiply(
  rangedMultiplySource, identitySRT, 3, 1, 0.25, 0.5, [0, 0, 0], 0.125,
);
for (let instance = 0; instance < 3; instance++) {
  const vertex = rangedMultiply.vertices[instance * cube.vertices.length];
  const sourceVertex = rangedMultiplyNormals.vertices[0];
  assert.deepEqual(Array.from(vertex.position.slice(0, 3)), [0, 1, 2].map(axis => Math.fround(
    sourceVertex.position[axis] + sourceVertex.normal[axis] * instance * 0.125,
  )), 'ranged Multiply retains the extrude transform on every appended instance');
  assert.deepEqual(Array.from(vertex.uv.slice(0, 2)), [
    Math.fround(sourceVertex.uv[0] + instance * 0.25),
    Math.fround(sourceVertex.uv[1] + instance * 0.5),
  ], 'ranged Multiply retains per-instance UV translation');
}
assert.ok(rangedMultiply.vertices.slice(0, 48).every(vertex => !vertex.select));
assert.ok(rangedMultiply.vertices.slice(48).every(vertex => vertex.select));

const multiply2Words = [17, 2, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
const tiledA = invoke(0xb4, multiply2Words, [cube, moved]);
const tiledB = invoke(0xb4, multiply2Words, [cube, moved]);
assert.deepEqual(tiledA.summary(), tiledB.summary());
assert.equal(tiledA.vertices.length, 48);
assert.ok(tiledA.vertices.slice(0, 24).every(vertex => !vertex.select));
assert.ok(tiledA.vertices.slice(24).every(vertex => vertex.select),
  'Multiply2 leaves only its final appended instance selected');

const heterogeneousGrid = D.Mesh_Grid(0, 1, 1);
const heterogeneousMultiply2 = D.Mesh_Multiply2(
  [cube, heterogeneousGrid], 17,
  [4, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], 0,
  [1, 1, 1], [0, 0, 0], [],
);
assert.equal(heterogeneousMultiply2.vertices.length, 64,
  'heterogeneous Multiply2 preserves the deterministic source sequence');
assert.ok(heterogeneousMultiply2.vertices.slice(0, 40).every(vertex => !vertex.select));
assert.ok(heterogeneousMultiply2.vertices.slice(40).every(vertex => vertex.select),
  'a differently sized final instance owns the complete selected suffix');

const cylinder = invoke(0x82, [8, 1, 0, 1, 0]);
assert.deepEqual(
  [cylinder.vertices.length, cylinder.edges.length, cylinder.faces.length],
  [38, 40, 24],
);
const torus = invoke(0x83, [8, 6, 0.5, 0.125, 0, 1, 0]);
assert.deepEqual([torus.vertices.length, torus.edges.length, torus.faces.length], [63, 96, 48]);
assert.equal(torus.edges.filter(edge => edge.crease & D.MESH_FEATURE.UV0).length, 14,
  'closed torus keeps both native UV seam families');
// Debris uses this exact open half-torus. Native extrudes the terminal ring to
// arclen*2pi and retains outward-facing start/end caps; wrapping the terminal
// sample like a closed torus folds the final strip back onto the first ring.
const openTorus = invoke(0x83, [3, 3, 0.5, 0.125, 0.5, 0.5, 0]);
const openStart = openTorus.vertices[0].position;
const openEnd = openTorus.vertices[3 * (3 + 1)].position;
assert.ok(openStart[0] < 0 && openEnd[0] > 0, 'open torus reaches its authored half-arc endpoint');
assert.ok(Math.abs(openStart[0] + openEnd[0]) < 1e-6);
const capNormal = faceIndex => {
  const ids = openTorus.faceVertices(openTorus.faces[faceIndex]);
  const a = openTorus.vertices[ids[0]].position;
  const b = openTorus.vertices[ids[1]].position;
  const c = openTorus.vertices[ids[2]].position;
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
};
assert.ok(capNormal(openTorus.faces.length - 2)[2] < 0, 'open torus start cap faces outward');
assert.ok(capNormal(openTorus.faces.length - 1)[2] < 0, 'open torus end cap faces outward');
assert.deepEqual([openTorus.vertices.length, openTorus.edges.length, openTorus.faces.length], [22, 21, 11]);
assert.equal(openTorus.edges.filter(edge => edge.crease & D.MESH_FEATURE.UV0).length, 9,
  'open torus creases both cap rims plus the swept minor seam');
const sphere = invoke(0x84, [8, 4]);
assert.deepEqual([sphere.vertices.length, sphere.edges.length, sphere.faces.length], [49, 88, 48]);
assert.equal(sphere.edges.filter(edge => edge.crease & D.MESH_FEATURE.UV0).length, 6,
  'sphere seam reaches both separately creased pole wedges');
const grid = invoke(0x9d, [3, 2, 2]);
assert.deepEqual([grid.vertices.length, grid.faces.length, grid.summary().activeFaces], [18, 8, 8]);
assert.deepEqual(
  Array.from(grid.vertices[0].position.subarray(0, 3)),
  [-0.5, 0, 0.5],
  'native Grid maps its top V=0 row to spatial +Z',
);
assert.ok(grid.vertices.every(vertex =>
  vertex.position[2] === Math.fround(0.5 - vertex.uv[1])),
  'front and back Grid vertices preserve the native spatial Z = 0.5 - V relation',
);
assert.deepEqual(Array.from(grid.vertices[0].uv.subarray(0, 2)), [1, 0],
  'native Grid mirrors U on its -Y face');
assert.deepEqual(Array.from(grid.vertices[9].uv.subarray(0, 2)), [0, 0],
  'native Grid keeps U on its +Y face');
grid.needNormals();
assert.equal(grid.vertices[0].first, grid.vertices[9].first,
  'double-sided grid perimeter retains the native physical seam link');
assert.notEqual(grid.vertices[4].first, grid.vertices[13].first,
  'double-sided grid interiors are distinct physical smoothing groups');
assert.deepEqual(Array.from(grid.vertices[4].normal.subarray(0, 3)), [0, -1, 0]);
assert.deepEqual(Array.from(grid.vertices[13].normal.subarray(0, 3)), [0, 1, 0]);
assert.deepEqual(Array.from(grid.vertices[4].tangent.subarray(0, 3)), [-1, 0, 0],
  'mirrored UV0.u produces the native -X tangent on the -Y face');
assert.deepEqual(Array.from(grid.vertices[13].tangent.subarray(0, 3)), [1, 0, 0],
  'increasing UV0.u produces the native +X tangent on the +Y face');
const compactGridDirections = invoke(0x9d, [3, 2, 2]).compact().prepare();
assert.deepEqual(
  Array.from(compactGridDirections.tangents.slice(4 * 4, 4 * 4 + 3)),
  [-1, 0, 0],
  'dormant compact GenMesh derives the same UV tangent without expanding topology',
);
const gridOne = invoke(0x9d, [1, 1, 1]);
gridOne.needNormals();
const gridOnePrepared = gridOne.prepare();
const triangleKey = triangle => {
  const keys = [];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = gridOnePrepared.indices[triangle * 3 + corner] * 3;
    keys.push(`${gridOnePrepared.positions[vertex]},${gridOnePrepared.positions[vertex + 1]},${gridOnePrepared.positions[vertex + 2]}`);
  }
  return keys.sort().join('|');
};
assert.equal(new Set(Array.from({ length: 4 }, (_, triangle) => triangleKey(triangle))).size, 4,
  'front and back grid fans use opposite diagonals without coincident triangles');
const shadowIncidence = new Map();
for (let triangle = 0; triangle < 4; triangle++) {
  const canonical = [];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = gridOnePrepared.indices[triangle * 3 + corner];
    canonical.push(gridOne.vertices[vertex].first);
  }
  for (const [a, b] of [[canonical[0], canonical[1]], [canonical[1], canonical[2]], [canonical[2], canonical[0]]]) {
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    shadowIncidence.set(key, (shadowIncidence.get(key) || 0) + 1);
  }
}
assert.ok(Array.from(shadowIncidence.values()).every(incidence => incidence === 2),
  'double-sided 1x1 grid shadow topology is closed and manifold');

// Ops 11733..11772 build the five wall labels visible in the 32..64 beat
// scene from this shared double-sided grid. Preserve their exact authored
// atlas strips: changing transform order or planar-projection signs moves the
// quads onto blank rows of the 512x512 text bitmap.
const wallLabelMaterial = { kind: 'material', name: 'op 2722 wall labels' };
function wallLabelMesh(atlasX, atlasY) {
  let mesh = invoke(0x9d, [3, 1, 1]);
  mesh = invoke(0x96, [0, 9], [mesh], [wallLabelMaterial]);
  mesh = invoke(0x88, [0, 1, 1, 1, 0.25, 0, 0, 0, 0, 0], [mesh]);
  mesh = invoke(0x88, [0, 1.75, 0.5, 1, 0, 0, 0, 0, 0, 0], [mesh]);
  return invoke(0xa5,
    [0, 0.125, 0.125, 0.125, 0, 0, 0, atlasX, atlasY, 0, 0], [mesh]);
}
const wallLabelStrips = [
  // The first Transform is InitSRT -> InitEulerPI2. These are the native
  // SSE_SinCos4 float results, not host Math.sin/cos rounded afterward.
  ['code', 0.5, -0.5999908447265625, 0.5687409043312073, 0.6312407851219177],
  ['ryg', 0.5, -0.12299919128417969, 0.09174926578998566, 0.15424911677837372],
  ['chaos', 0.5, -0.042999267578125, 0.01174933835864067, 0.07424919307231903],
  ['fried', 0.18999862670898438, -0.2519989013671875, 0.22074897587299347, 0.2832488417625427],
  ['kb', 0.5, -0.18199920654296875, 0.15074928104877472, 0.21324913203716278],
];
let codeLabelMesh;
for (const [name, atlasX, atlasY, expectedMinV, expectedMaxV] of wallLabelStrips) {
  const mesh = wallLabelMesh(atlasX, atlasY);
  const v = mesh.vertices.map(vertex => vertex.uv[1]);
  assert.deepEqual([Math.min(...v), Math.max(...v)], [expectedMinV, expectedMaxV],
    `${name} occupies its authored atlas row`);
  if (name === 'code') codeLabelMesh = mesh;
}
const codeU = codeLabelMesh.vertices.map(vertex => vertex.uv[0]);
assert.deepEqual([Math.min(...codeU), Math.max(...codeU)],
  [-0.6093747615814209, 0.6093747615814209]);
const codeLabelGroups = codeLabelMesh.prepare().groups;
assert.equal(codeLabelGroups.length, 1);
assert.deepEqual([codeLabelGroups[0].pass, codeLabelGroups[0].count], [9, 12]);
assert.equal(codeLabelGroups[0].material, wallLabelMaterial);

// CubicProjection repeatedly creases selected projection islands. A smooth
// tetra starts with one record per physical vertex, then gains independent UV
// wedges wherever neighboring faces classify to different cube directions.
const cubicSeamProbe = new D.Mesh();
for (const position of [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]]) {
  cubicSeamProbe.addVertex(position, [0, 0]);
}
cubicSeamProbe.setPolygons([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]);
cubicSeamProbe.linkVertexCopies();
const cubicProjected = D.Mesh_UVProjection(cubicSeamProbe, 0, identitySRT, 0);
assert.equal(cubicProjected.vertices.length, 12,
  'native cube-direction boundaries duplicate the tetra UV wedges');
assert.ok(cubicProjected.edges.every(edge => edge.crease & D.MESH_FEATURE.UV0));
for (const face of cubicProjected.faces) {
  if (!face.material) continue;
  const ids = cubicProjected.faceVertices(face);
  const p0 = cubicProjected.vertices[ids[ids.length - 1]].position;
  const p1 = cubicProjected.vertices[ids[0]].position;
  const p2 = cubicProjected.vertices[ids[1]].position;
  const a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
  const normal = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
  let maximum = normal[0], axis = 0;
  if (Math.abs(normal[1]) > Math.abs(maximum)) { maximum = normal[1]; axis = 2; }
  if (Math.abs(normal[2]) > Math.abs(maximum)) { maximum = normal[2]; axis = 4; }
  if (maximum > 0) axis++;
  const sign = axis & 1 ? 1 : -1;
  for (const id of ids) {
    const vertex = cubicProjected.vertices[id], point = vertex.position;
    const expected = axis >> 1 === 0 ? [point[2] * sign, -point[1]]
      : axis >> 1 === 1 ? [point[0], -point[2] * sign]
        : [-point[0] * sign, -point[1]];
    assert.deepEqual(Array.from(vertex.uv.slice(0, 2)), expected);
  }
}

for (const mesh of [cylinder, torus, openTorus, sphere, grid]) verifyTopology(mesh);

const subdivided = invoke(0x87, [0, 1, 1], [cube]);
assert.equal(subdivided.faces.length, 24);
assert.equal(subdivided.edges.length, 48);
assert.ok(subdivided.vertices.length > cube.vertices.length);
verifyTopology(subdivided);
const sourceEdge = cube.edges[0];
const sourceHalfedge = 0;
const sourceA = cube.vertices[cube.getVertId(sourceHalfedge)];
const sourceB = cube.vertices[cube.getVertId(cube.nextFaceEdge(sourceHalfedge))];
const sourceCenters = sourceEdge.face.map(faceIndex => {
  const ids = cube.faceVertices(faceIndex), center = new Float32Array(4);
  for (const id of ids) for (let component = 0; component < 4; component++) {
    center[component] = Math.fround(center[component] + cube.vertices[id].position[component] / ids.length);
  }
  return center;
});
const edgePoint = subdivided.vertices[cube.vertices.length + cube.faces.length];
const expectedEdgePosition = Array.from({ length: 4 }, (_, component) => Math.fround(
  (sourceA.position[component] + sourceB.position[component]) * 0.25 +
  (sourceCenters[0][component] + sourceCenters[1][component]) * 0.25,
));
assert.deepEqual(Array.from(edgePoint.position), expectedEdgePosition,
  'a UV/normal crease does not harden the Catmull-Clark position channel');
assert.deepEqual(Array.from(edgePoint.uv), Array.from(sourceA.uv, (value, component) =>
  Math.fround((value + sourceB.uv[component]) * 0.5)),
  'the creased UV channel uses its face-side edge midpoint');
assert.deepEqual(Array.from(subdivided.vertices[0].uv), Array.from(cube.vertices[0].uv),
  'closed even vertices retain a creased UV channel');
assert.ok(subdivided.edges.some(edge => edge.vert.includes(cube.vertices.length + cube.faces.length) &&
  edge.crease === sourceEdge.crease), 'both split edge halves retain native crease metadata');

// GenMesh's boundary even-vertex rule takes the two half-edges that bound one
// UNSELECTED sector of the one-ring (genmesh.cpp:1721-1727), not the two edges
// running along the selection border. Where a vertex has two or more adjacent
// unselected faces those differ: one of the reference's two edges points out of
// the selected region and has no selected face at all.
{
  const grid = D.Mesh_Grid(0, 2, 2);
  for (const face of grid.faces) face.select = false;
  grid.faces[0].select = true;
  const subdivided = D.Mesh_Subdivide(grid, 1, 1, 1);
  const at = target => subdivided.vertices.findIndex((_, index) =>
    grid.vertices[index].position.subarray(0, 3).every((value, axis) => value === target[axis]));
  const border = at([-0.5, 0, 0]);
  assert.ok(border >= 0);
  // alpha 1 gives w1 = 0.125 and w2 = 0.75. The reference pair here is the
  // corner (-0.5,0,-0.5) plus the grid centre (0,0,0), so the vertex moves to
  // 0.75*(-0.5,0,0) + 0.125*((-0.5,0,-0.5) + (0,0,0)) = (-0.4375,0,-0.0625).
  assert.deepEqual(
    Array.from(subdivided.vertices[border].position.subarray(0, 3)),
    [-0.4375, 0, -0.0625],
    'boundary even vertex uses the sector-bounding pair, not the two collinear border edges');
}

// Debris ops 15654..15660 selectively subdivide the destructible road twice.
// Split points turn neighboring quads into pentagons; their cyclic Face.Edge
// origin must still make a consistently wound fan because Mesh_ToMin and the
// following Explode operator preserve and triangulate that exact order.
let debrisRoad = D.Mesh_Grid(2, 8, 32);
debrisRoad = D.Mesh_Transform(debrisRoad, 0,
  [24, 24, 24, 0.5, 0, 0, 0, 0, 0]);
debrisRoad = D.Mesh_SelectCube(debrisRoad, 65792, 2,
  [0, 0, 0], [24, 12, 11]);
debrisRoad = D.Mesh_Subdivide(debrisRoad, 1, 1, 1);
debrisRoad = D.Mesh_SelectCube(debrisRoad, 65792, 2,
  [-2, 0, 0], [6, 1.25, 6]);
debrisRoad = D.Mesh_Subdivide(debrisRoad, 1, 1, 1);
const debrisRoadFanAreas = { positive: 0, negative: 0, zero: 0 };
for (const face of debrisRoad.faces) {
  if (!face.material || face.edge < 0) continue;
  const ids = debrisRoad.faceVertices(face);
  const position = id => {
    const vertex = debrisRoad.vertices[id];
    return debrisRoad.vertices[vertex.first]?.position || vertex.position;
  };
  const first = position(ids[0]);
  for (let index = 2; index < ids.length; index++) {
    const second = position(ids[index - 1]), third = position(ids[index]);
    const areaY = (second[2] - first[2]) * (third[0] - first[0]) -
      (second[0] - first[0]) * (third[2] - first[2]);
    if (areaY > 1e-8) debrisRoadFanAreas.positive++;
    else if (areaY < -1e-8) debrisRoadFanAreas.negative++;
    else debrisRoadFanAreas.zero++;
  }
}
assert.deepEqual(debrisRoadFanAreas, { positive: 1526, negative: 0, zero: 0 },
  'selective subdivision retains a valid native fan origin for every road face');

const shadowless = invoke(0xb0, [0], [cube]);
assert.ok(shadowless.faces.every(face => !face.used));
assert.ok(cube.faces.every(face => face.used));

const centered = invoke(0xa6, [0, 7], [moved]);
assert.deepEqual(centered.summary().bounds, cube.summary().bounds);
assert.ok(centered.faces.filter(face => face.material).every(face => face.select),
  'CalcBBox leaves every live face selected');
assert.ok(centered.vertices.every(vertex => vertex.select),
  'CalcBBox followed by Center leaves every vertex selected');

const pivotSource = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
pivotSource.pivot = 0;
const pivotPosition = Array.from(pivotSource.vertices[0].position);
const pivotScaled = D.Mesh_Transform(pivotSource, 0, [2, 3, 4, 0, 0, 0, 0, 0, 0]);
assert.deepEqual(Array.from(pivotScaled.vertices[0].position), pivotPosition,
  'Transform applies SRT around the stored native pivot');
assert.deepEqual(Array.from(pivotScaled.vertices[1].position), [-0.5, -0.5, 3.5, 1]);

// A dormant mask-zero Transform stays in flat storage. Its compact COW path
// must retain the expanded operator's exact pivot, selection and metadata
// semantics while sharing only channels that remain immutable.
const compactTransformFixture = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
compactTransformFixture.pivot = 3;
compactTransformFixture.vertices[3].first = 0;
compactTransformFixture.faces[0].material = 0;
compactTransformFixture.vertices[0].select = false;
compactTransformFixture.edges[0].select = false;
compactTransformFixture.faces[1].select = false;
compactTransformFixture.vertices[0].position[3] = -0;
compactTransformFixture.vertices[0].normal[0] = 0.25;
compactTransformFixture.vertices[0].color.set([0.25, 0.5, 0.75, 1]);
compactTransformFixture.vertices[0].uv.set([0.125, 0.875, 0, 1]);
compactTransformFixture.collisions.push({
  vert: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), mode: 4,
});
compactTransformFixture.lights.push(5);
const compactTransformSRT = [1.25, 0.75, 1.5, 0.125, -0.25, 0.375, 2, -3, 4];
const expandedTransformOracle = D.Mesh_Transform(compactTransformFixture.clone(), 0, compactTransformSRT);
const compactTransformSource = compactTransformFixture.clone().compact();
const sourcePositionBytes = Uint8Array.from(arrayBytes(compactTransformSource._compact.vertexPositions));
const sourceSelectionBytes = Uint8Array.from(arrayBytes(compactTransformSource._compact.vertexBytes));
const sharedVertexInts = compactTransformSource._compact.vertexInts;
const sharedEdgeInts = compactTransformSource._compact.edgeInts;
const sharedFaceInts = compactTransformSource._compact.faceInts;
const sharedTransformDirections = compactTransformSource._compact.vertexDirections;
const sharedTransformUVs = compactTransformSource._compact.vertexUVs;
compactTransformSource.ensureExpanded = () => {
  throw new Error('compact Transform expanded its source');
};
const compactTransformed = D.Mesh_Transform(compactTransformSource, 0, compactTransformSRT);
assert.ok(compactTransformed._compact, 'compact Transform returns dormant storage');
assert.notEqual(compactTransformed, compactTransformSource, 'compact Transform retains direct-call COW');
assert.notEqual(compactTransformed._compact.vertexPositions.buffer,
  compactTransformSource._compact.vertexPositions.buffer);
assert.notEqual(compactTransformed._compact.vertexBytes.buffer,
  compactTransformSource._compact.vertexBytes.buffer);
assert.notEqual(compactTransformed._compact.edgeBytes.buffer,
  compactTransformSource._compact.edgeBytes.buffer);
assert.notEqual(compactTransformed._compact.faceBytes.buffer,
  compactTransformSource._compact.faceBytes.buffer);
assert.equal(compactTransformed._compact.vertexInts, sharedVertexInts);
assert.equal(compactTransformed._compact.edgeInts, sharedEdgeInts);
assert.equal(compactTransformed._compact.faceInts, sharedFaceInts);
assert.equal(compactTransformed._compact.vertexDirections, sharedTransformDirections);
assert.equal(compactTransformed._compact.vertexUVs, sharedTransformUVs);
assert.deepEqual(arrayBytes(compactTransformSource._compact.vertexPositions), sourcePositionBytes,
  'compact Transform leaves fork-source position bits unchanged');
assert.deepEqual(arrayBytes(compactTransformSource._compact.vertexBytes), sourceSelectionBytes,
  'compact Transform leaves fork-source selection bits unchanged');
assertPreparedEquivalent(compactTransformed.prepare(), expandedTransformOracle.prepare(),
  'compact mask-zero Transform');
compactTransformed.ensureExpanded();
for (let index = 0; index < compactTransformed.vertices.length; index++) {
  assert.deepEqual(arrayBytes(compactTransformed.vertices[index].position),
    arrayBytes(expandedTransformOracle.vertices[index].position), `compact Transform vertex ${index} bits`);
}
assert.ok(compactTransformed.vertices.every(vertex => vertex.select));
assert.ok(compactTransformed.edges.every(edge => edge.select));
assert.ok(compactTransformed.faces.every(face => face.select === (face.material !== 0)));
assert.equal(compactTransformed.pivot, compactTransformFixture.pivot);
assert.deepEqual(compactTransformed.lights, compactTransformFixture.lights);
compactTransformed.collisions[0].vert[0] = 9;
assert.equal(compactTransformSource.collisions[0].vert[0], 0,
  'compact Transform metadata remains copy-on-write');

const sharedPositionSource = cube.clone().compact();
const sharedPositionBytes = Uint8Array.from(arrayBytes(sharedPositionSource._compact.vertexPositions));
const sharedPositionBranch = D.Mesh_MatLink(sharedPositionSource, material, 0, 8);
assert.equal(D.Mesh_Transform(sharedPositionBranch, 0,
  [1, 1, 1, 0, 0, 0, 3, 0, 0], true), sharedPositionBranch);
assert.deepEqual(arrayBytes(sharedPositionSource._compact.vertexPositions), sharedPositionBytes,
  'owned compact Transform detaches a position buffer shared by an older branch');

const sharedMaterialSource = cube.clone().compact();
const sharedMaterialBytes = Uint8Array.from(arrayBytes(sharedMaterialSource._compact.faceInts));
const sharedMaterialBranch = D.Mesh_Transform(sharedMaterialSource, 0,
  [1, 1, 1, 0, 0, 0, 3, 0, 0]);
assert.equal(D.Mesh_MatLink(sharedMaterialBranch, material, 0, 8, true), sharedMaterialBranch);
assert.deepEqual(arrayBytes(sharedMaterialSource._compact.faceInts), sharedMaterialBytes,
  'owned compact MatLink detaches face materials shared by an older Transform branch');

const maskedCompactTransform = compactTransformFixture.clone().compact();
let maskedTransformExpansions = 0;
const expandMaskedTransform = maskedCompactTransform.ensureExpanded.bind(maskedCompactTransform);
maskedCompactTransform.ensureExpanded = () => {
  maskedTransformExpansions++;
  return expandMaskedTransform();
};
D.Mesh_Transform(maskedCompactTransform, 1, compactTransformSRT);
assert.ok(maskedTransformExpansions > 0, 'nonzero Transform masks retain the expanded selection path');

const metadataA = cube.clone();
metadataA.collisions.push({ vert: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), mode: 1 });
metadataA.lights.push(2);
const metadataB = cube.clone();
metadataB.collisions.push({ Vert: [0, 1, 2, 3, 4, 5, 6, 7], mode: 2 });
metadataB.lights.push(3);
const metadataAdded = D.Mesh_Add([metadataA, metadataB]);
assert.deepEqual(Array.from(metadataAdded.collisions[1].Vert),
  [24, 25, 26, 27, 28, 29, 30, 31]);
assert.deepEqual(metadataAdded.lights, [2, 27]);
metadataAdded.collisions[0].vert[0] = 9;
assert.equal(metadataA.collisions[0].vert[0], 0,
  'copy-on-write collision extrema do not alias the source mesh');

// Runtime dispatch clones a shared input for the first branch, then transfers
// the same cache identity to its final consumer. Dormant snapshots are compact
// between consumers; exported direct calls above remain ordinary COW.
const sourceOp = { id: 1000, classId: 0x81, inputs: [], links: [] };
const firstConsumer = { id: 1001, classId: 0x88, inputs: [sourceOp], links: [] };
const finalConsumer = { id: 1002, classId: 0x88, inputs: [sourceOp], links: [] };
const mockRuntime = {
  operations: [sourceOp, firstConsumer, finalConsumer], roots: [], events: [],
};
const runtimeSource = invoke(0x81, cubeParameters);
const transformHandler = D.handlers.get(0x88);
const runtimeCall = (op, x) => ({
  runtime: mockRuntime, environment: {}, op, inputs: [runtimeSource], links: [],
  parameters: [0, 1, 1, 1, 0, 0, 0, x, 0, 0], strings: [], splines: [],
});
const firstBranch = transformHandler(runtimeCall(firstConsumer, 1));
assert.notEqual(firstBranch, runtimeSource, 'non-final branch keeps COW identity');
assert.equal(runtimeSource.storageSummary().compact, true, 'retained branch snapshot is flat');
assert.deepEqual(firstBranch.summary().bounds, {
  min: [0.5, -0.5, -0.5], max: [1.5, 0.5, 0.5],
});
assert.deepEqual(runtimeSource.summary().bounds, cubeOracle.bounds, 'first branch did not mutate source');
const finalBranch = transformHandler(runtimeCall(finalConsumer, 2));
assert.equal(finalBranch, runtimeSource, 'last graph consumer transfers cache identity');
assert.equal(finalBranch.storageSummary().compact, true, 'terminal result returns to flat storage');
assert.deepEqual(finalBranch.summary().bounds, {
  min: [1.5, -0.5, -0.5], max: [2.5, 0.5, 0.5],
});

// Keep a separate runtime fixture dormant throughout both consumers so an
// accidental dispatch expansion cannot be hidden by summary()/public getters.
const compactSourceOp = { id: 1010, classId: 0x81, inputs: [], links: [] };
const compactFirstConsumer = { id: 1011, classId: 0x88, inputs: [compactSourceOp], links: [] };
const compactFinalConsumer = { id: 1012, classId: 0x88, inputs: [compactSourceOp], links: [] };
const compactRuntime = {
  operations: [compactSourceOp, compactFirstConsumer, compactFinalConsumer], roots: [], events: [],
};
const compactRuntimeSource = invoke(0x81, cubeParameters).compact();
const compactRuntimeSourceBits = Uint8Array.from(arrayBytes(compactRuntimeSource._compact.vertexPositions));
compactRuntimeSource.ensureExpanded = () => {
  throw new Error('runtime compact Transform expanded its input');
};
const compactRuntimeCall = (op, x) => ({
  runtime: compactRuntime, environment: {}, op, inputs: [compactRuntimeSource], links: [],
  parameters: [0, 1, 1, 1, 0, 0, 0, x, 0, 0], strings: [], splines: [],
});
const compactFirstBranch = transformHandler(compactRuntimeCall(compactFirstConsumer, 1));
assert.notEqual(compactFirstBranch, compactRuntimeSource);
assert.ok(compactFirstBranch._compact);
assert.deepEqual(arrayBytes(compactRuntimeSource._compact.vertexPositions), compactRuntimeSourceBits,
  'first compact runtime branch leaves its source unchanged');
const compactFinalBranch = transformHandler(compactRuntimeCall(compactFinalConsumer, 2));
assert.equal(compactFinalBranch, compactRuntimeSource,
  'final compact runtime consumer mutates the transferred cache identity');
assert.ok(compactFinalBranch._compact);
delete compactRuntimeSource.ensureExpanded;
assert.deepEqual(compactFinalBranch.summary().bounds, {
  min: [1.5, -0.5, -0.5], max: [2.5, 0.5, 0.5],
});

console.log('old GenMesh topology, COW, compact storage, packing, and deterministic constructor tests passed');
