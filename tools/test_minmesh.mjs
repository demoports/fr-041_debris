import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { CLASS_REGISTRY } from '../src/classes.js';
import * as BitmapAPI from '../src/bitmap.js';
import * as CoreAPI from '../src/core.js';
import { FONT3D_FAMILIES } from '../src/font3d_glyphs.js';
import { parseKX } from '../src/kx.js';
import * as MeshAPI from '../src/mesh.js';
import { Mesh_ToMin, meshToMinHandlers } from '../src/mesh_to_min.js';
import * as MinMeshAPI from '../src/minmesh.js';
import {
  geometryTopologyStats,
  normalizePreparedGeometry,
  prepareShadowTopology,
} from '../src/renderer.js';
import * as RuntimeAPI from '../src/runtime.js';

const handlers = new Map([
  ...Object.entries(RuntimeAPI.runtimeHandlers),
  ...Object.entries(MeshAPI.meshHandlers),
  ...Object.entries(meshToMinHandlers),
  ...Object.entries(MinMeshAPI.minMeshHandlers),
].map(([id, handler]) => [Number(id), handler]));
const D = {
  ...CoreAPI,
  ...BitmapAPI,
  ...RuntimeAPI,
  ...MeshAPI,
  ...MinMeshAPI,
  CLASS_REGISTRY,
  geometryTopologyStats,
  normalizePreparedGeometry,
  prepareShadowTopology,
  handlers,
};
const identitySRT = [1, 1, 1, 0, 0, 0, 0, 0, 0];

function referenceAnimatedChannels(mesh, time) {
  mesh.calcNormals();
  const matrices = mesh.evaluateAnimation(time);
  const positions = new Float32Array(mesh.vertices.length * 3);
  const normals = new Float32Array(mesh.vertices.length * 3);
  const tangents = new Float32Array(mesh.vertices.length * 4);
  const transform = (matrix, source, point) => new Float32Array([
    Math.fround(matrix[0] * source[0] + matrix[4] * source[1] + matrix[8] * source[2] + (point ? matrix[12] : 0)),
    Math.fround(matrix[1] * source[0] + matrix[5] * source[1] + matrix[9] * source[2] + (point ? matrix[13] : 0)),
    Math.fround(matrix[2] * source[0] + matrix[6] * source[1] + matrix[10] * source[2] + (point ? matrix[14] : 0)),
  ]);
  for (let index = 0; index < mesh.vertices.length; index++) {
    const vertex = mesh.vertices[index];
    const position = new Float32Array(3);
    const normal = new Float32Array(3);
    const tangent = new Float32Array(3);
    if (matrices && vertex.boneCount) {
      for (let bone = 0; bone < vertex.boneCount; bone++) {
        const matrix = matrices[vertex.matrices[bone]];
        if (!matrix) continue;
        const weight = vertex.weights[bone];
        const transformedPosition = transform(matrix, vertex.position, true);
        const transformedNormal = transform(matrix, vertex.normal, false);
        const transformedTangent = transform(matrix, vertex.tangent, false);
        for (let component = 0; component < 3; component++) {
          position[component] = Math.fround(position[component] + transformedPosition[component] * weight);
          normal[component] = Math.fround(normal[component] + transformedNormal[component] * weight);
          tangent[component] = Math.fround(tangent[component] + transformedTangent[component] * weight);
        }
      }
    } else {
      position.set(vertex.position);
      normal.set(vertex.normal);
      tangent.set(vertex.tangent);
    }
    positions.set(position, index * 3);
    normals.set(normal, index * 3);
    tangents.set(tangent, index * 4);
    tangents[index * 4 + 3] = 1;
  }
  return { positions, normals, tangents };
}

// Source anchors for the two intentionally non-obvious playback rules below.
// Cluster 0 is a render/material sentinel, not a CalcNormals filter, and the
// native CPU bone loop defers all T-space conditioning to the vertex shader.
const nativeMinMeshSource = readFileSync(
  new URL('../vendor/wz3/genminmesh.cpp', import.meta.url), 'utf8');
const nativeCalcNormalsStart = nativeMinMeshSource.indexOf('void GenMinMesh::CalcNormals()');
const nativeCalcNormalsEnd = nativeMinMeshSource.indexOf('\n/*', nativeCalcNormalsStart);
const nativeCalcNormals = nativeMinMeshSource.slice(nativeCalcNormalsStart, nativeCalcNormalsEnd);
assert.match(nativeCalcNormals, /else if\(mf\[i\]\.Count>=3\)/);
assert.doesNotMatch(nativeCalcNormals, /Cluster/,
  'native CalcNormals does not exclude the draw-suppressed cluster 0');

const nativeEngineSource = readFileSync(
  new URL('../vendor/wz3/engine.cpp', import.meta.url), 'utf8');
const nativeFullBoneStart = nativeEngineSource.indexOf('static void FullBoneInner(');
const nativeFullBoneEnd = nativeEngineSource.indexOf('static void RigidInnerSSE_asm', nativeFullBoneStart);
const nativeFullBone = nativeEngineSource.slice(nativeFullBoneStart, nativeFullBoneEnd);
assert.match(nativeFullBone,
  /the vertex shader does renormalization and orthogonalization now/);
assert.doesNotMatch(nativeFullBone, /\.UnitSafe\s*\(|\bnormalize\s*\(|\borthogonalize\s*\(/i,
  'FullBoneInner writes raw weighted normal/tangent channels');

const rendererSource = readFileSync(new URL('../src/renderer.js', import.meta.url), 'utf8');
assert.match(rendererSource,
  /if \(uLegacyLightingMode == 2\) \{[\s\S]*?legacyNormal = normalize\(legacyNormal\);[\s\S]*?legacyTangent = normalize\(legacyTangent - legacyNormal \*[\s\S]*?dot\(legacyNormal, legacyTangent\)\);/,
  'the animated legacy vertex shader is the sole N/T conditioning stage');

const portMinMeshSource = readFileSync(new URL('../src/minmesh.js', import.meta.url), 'utf8');
const compactSkinStart = portMinMeshSource.indexOf('function skinCompactGeometry(');
const compactSkinEnd = portMinMeshSource.indexOf('\nclass MinMesh', compactSkinStart);
assert.doesNotMatch(portMinMeshSource.slice(compactSkinStart, compactSkinEnd),
  /normalize3\((?:normal|tangent)\)|lengthSq\s*=\s*(?:nx|tx)\s*\*/,
  'compact CPU skinning must upload raw weighted N/T');
const expandedSkinStart = portMinMeshSource.indexOf(
  'const position = vector3(), normal = vector3(), tangent = vector3();', compactSkinEnd);
const expandedSkinEnd = portMinMeshSource.indexOf('\n    const clusters = this.clusters;', expandedSkinStart);
assert.doesNotMatch(portMinMeshSource.slice(expandedSkinStart, expandedSkinEnd),
  /normalize3\((?:normal|tangent)\)/,
  'expanded CPU skinning must also leave weighted N/T raw');

const grid = D.MinMesh_Grid(2, 2, 2);
assert.equal(grid.kind, 'minmesh');
assert.equal(grid.vertices.length, 9);
assert.equal(grid.faces.length, 4);
assert.deepEqual(Array.from(grid.bounds().minimum), [-0.5, 0, -0.5]);
assert.deepEqual(Array.from(grid.bounds().maximum), [0.5, 0, 0.5]);
assert.equal(grid.prepare().indices.length, 24);
assert.equal(grid.prepare(), grid.prepare(), 'static prepare result is cached');

const doubleGrid = D.MinMesh_Grid(1, 2, 2);
assert.equal(doubleGrid.vertices.length, 18);
assert.equal(doubleGrid.faces.length, 8);

const cube = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
assert.equal(cube.vertices.length, 24);
assert.equal(cube.faces.length, 6);
assert.equal(cube.prepare().triangleCount, 12);
assert.equal(cube.prepare().tangents.length, cube.vertices.length * 4);
assert.ok(cube.prepare().colors instanceof Uint8Array);
assert.equal(cube.prepare().colors.length, cube.vertices.length * 4);
assert.equal(new Set(cube.vertices.map(vertex => Array.from(vertex.position).join(','))).size, 8,
  'all six cube faces occupy the surface rather than four degenerate midplanes');
assert.equal(new Set(cube.prepare().shadowVertexMap).size, 8,
  'cube face wedges merge to the eight native shadow-adjacency vertices');
const cubeShadow = D.prepareShadowTopology(D.normalizePreparedGeometry(cube));
assert.equal(cubeShadow.positions.length / 3, 8);
assert.equal(cubeShadow.faces.length / 3, 12);
assert.equal(cubeShadow.edges.length, 18, 'twelve cube edges plus six face diagonals');
assert.equal(cubeShadow.boundaryEdges, 0);
assert.equal(cubeShadow.nonManifoldEdges, 0);
assert.equal(cubeShadow.windingConflictEdges, 0);
assert.ok(cubeShadow.edges.every(records => records.length === 2),
  'closed cube has no false shadow boundary at a UV/normal seam');
assert.deepEqual(cube.summary().bounds, { minimum: [-0.5, -0.5, -0.5], maximum: [0.5, 0.5, 0.5] });
for (const normal of cube.vertices.map(vertex => vertex.normal)) {
  assert.ok(Math.abs(Math.hypot(...normal) - 1) < 1e-6);
}

const weightedSeam = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
const firstPosition = weightedSeam.vertices[0].position;
const duplicateIndex = weightedSeam.vertices.findIndex((vertex, index) => index > 0 &&
  vertex.position.every((value, axis) => Object.is(value, firstPosition[axis]) || value === firstPosition[axis]));
assert.ok(duplicateIndex > 0);
weightedSeam.vertices[duplicateIndex].weights[0] = 1;
weightedSeam.vertices[duplicateIndex].matrices[0] = 7;
const weightedMap = weightedSeam.prepare().shadowVertexMap;
assert.notEqual(weightedMap[0], weightedMap[duplicateIndex],
  'coincident vertices with different skinning remain separate like CalcMergeVerts');

const sphere = D.MinMesh_Sphere(8, 4);
assert.equal(sphere.vertices.length, 47);
assert.equal(sphere.faces.length, 53); // 32 quads, 16 cap tris, five stitch records.
assert.equal(sphere.prepare().triangleCount, 80);

const cylinder = D.MinMesh_Cylinder(8, 1, 0, 1, 0);
assert.equal(cylinder.vertices.length, 38);
assert.equal(cylinder.faces.length, 26);
assert.equal(cylinder.prepare().triangleCount, 32);

// Arc cylinders insert the sector center at x==tx-1 in every generated grid,
// not just in the side wall. Cylinder caps deliberately use MakeGrid(tx,0)
// for tz==1, whose otherwise-unused UV1.v is the native 0/0 NaN.
const arcCylinder = D.MinMesh_Cylinder(8, 1, 0, 1, 3);
assert.equal(arcCylinder.vertices.length, 34);
for (const index of [6, 14, 22, 31]) {
  assert.equal(arcCylinder.vertices[index].position[0], 0);
  assert.equal(arcCylinder.vertices[index].position[2], 0);
}
assert.ok(Number.isNaN(arcCylinder.vertices[16].uv[1][1]));
assert.ok(Number.isNaN(arcCylinder.vertices[25].uv[1][1]));

// Flat normal derivation is bit-identical to the editable object path and
// keeps cold static preparation compact throughout.
const expandedNormalProbe = D.MinMesh_Cube(3, 2, 4, 0, identitySRT);
const compactNormalProbe = expandedNormalProbe.clone().compact();
expandedNormalProbe.calcNormals();
compactNormalProbe.calcNormals();
assert.equal(compactNormalProbe.storageSummary().compact, true);
compactNormalProbe.ensureExpanded();
assert.deepEqual(
  compactNormalProbe.vertices.map(vertex => [Array.from(vertex.normal), Array.from(vertex.tangent)]),
  expandedNormalProbe.vertices.map(vertex => [Array.from(vertex.normal), Array.from(vertex.tangent)]),
);
assert.deepEqual(
  compactNormalProbe.faces.map(face => Array.from(face.normal)),
  expandedNormalProbe.faces.map(face => Array.from(face.normal)),
);

// Cluster 0 faces do not emit draw indices, but native CalcNormals still lets
// them influence shared render vertices. Keep this true in both expanded and
// compact derivation, including UnitSafe's +X result for a degenerate face.
function makeClusterZeroNormalProbe() {
  const mesh = new D.MinMesh();
  for (const position of [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1],
    [2, 0, 0], [3, 0, 0], [4, 0, 0],
  ]) {
    const vertex = D.makeMinMeshVertex();
    vertex.position.set(position);
    mesh.vertices.push(vertex);
  }
  mesh.faces.push(
    D.makeMinMeshFace([0, 1, 2], 1), // +Z, visible
    D.makeMinMeshFace([0, 1, 3], 0), // +Y, draw-suppressed
    D.makeMinMeshFace([4, 5, 6], 0), // degenerate -> UnitSafe +X
  );
  return mesh;
}

const expandedClusterZeroProbe = makeClusterZeroNormalProbe();
const compactClusterZeroProbe = expandedClusterZeroProbe.clone().compact();
expandedClusterZeroProbe.calcNormals();
compactClusterZeroProbe.calcNormals();
assert.equal(compactClusterZeroProbe.storageSummary().compact, true);
compactClusterZeroProbe.ensureExpanded();
assert.deepEqual(
  compactClusterZeroProbe.vertices.map(vertex => [Array.from(vertex.normal), Array.from(vertex.tangent)]),
  expandedClusterZeroProbe.vertices.map(vertex => [Array.from(vertex.normal), Array.from(vertex.tangent)]),
  'compact and expanded CalcNormals include cluster 0 identically',
);
assert.deepEqual(
  compactClusterZeroProbe.faces.map(face => Array.from(face.normal)),
  expandedClusterZeroProbe.faces.map(face => Array.from(face.normal)),
);
const sharedDeletedNormal = expandedClusterZeroProbe.vertices[0].normal;
assert.ok(Math.abs(sharedDeletedNormal[0]) < 1e-7 &&
  Math.abs(sharedDeletedNormal[1] - Math.SQRT1_2) < 1e-6 &&
  Math.abs(sharedDeletedNormal[2] - Math.SQRT1_2) < 1e-6,
  'a deleted +Y face remains in the shared vertex average with the visible +Z face');
assert.deepEqual(Array.from(expandedClusterZeroProbe.faces[2].normal), [1, 0, 0]);
assert.deepEqual(Array.from(expandedClusterZeroProbe.vertices[4].normal), [1, 0, 0]);
assert.equal(expandedClusterZeroProbe.prepare().indices.length, 3,
  'cluster 0 affects derivation without becoming drawable geometry');

// Lazy flat storage is lossless for every topology channel and material slot.
const compactProbe = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
const probeMaterial = { kind: 'material', name: 'compact-probe' };
compactProbe.clusters[1].material = probeMaterial;
compactProbe.clusters[1].renderPass = -7;
compactProbe.clusters[1].id = 0x12345678;
compactProbe.clusters[1].animType = 2;
compactProbe.clusters[1].animMatrix = 19;
const probeVertex = compactProbe.vertices[0], probeFace = compactProbe.faces[0];
probeVertex.select = -3; probeVertex.boneCount = 4; probeVertex.tempByte = 0x1234; probeVertex.mergeTag = -99;
probeVertex.color = 0x89abcdef;
probeVertex.position.set([-0, NaN, 3.25]);
probeVertex.normal.set([0.25, -0, -2]);
probeVertex.tangent.set([Infinity, -Infinity, NaN]);
probeVertex.uv[0].set([-0, 0.75]); probeVertex.uv[1].set([2.5, -3.5]);
probeVertex.weights.set([-0, 0.125, NaN, 2]); probeVertex.matrices.set([0, 255, 256, 65535]);
probeFace.select = -1; probeFace.temp = -123456; probeFace.flags = 0xfedcba98;
probeFace.normal.set([-0, NaN, 8]);
probeFace.adjacent = probeFace.adjacent.map((_, index) => index * 17 - 9);
const probeSnapshot = {
  position: probeVertex.position.slice(), normal: probeVertex.normal.slice(), tangent: probeVertex.tangent.slice(),
  uv0: probeVertex.uv[0].slice(), uv1: probeVertex.uv[1].slice(), weights: probeVertex.weights.slice(),
  matrices: probeVertex.matrices.slice(), color: probeVertex.color, select: probeVertex.select,
  boneCount: probeVertex.boneCount, tempByte: probeVertex.tempByte, mergeTag: probeVertex.mergeTag,
  faceVertices: probeFace.vertices.slice(), faceAdjacent: probeFace.adjacent.slice(), faceNormal: probeFace.normal.slice(),
  faceSelect: probeFace.select, faceCount: probeFace.count, faceCluster: probeFace.cluster,
  faceTemp: probeFace.temp, faceFlags: probeFace.flags,
};
compactProbe.compact();
assert.equal(compactProbe.storageSummary().compact, true);
assert.ok(compactProbe.storageSummary().compactBytes > 0);
assert.equal(compactProbe._vertices.length, 0);
compactProbe.ensureExpanded();
const expandedVertex = compactProbe.vertices[0], expandedFace = compactProbe.faces[0];
assert.deepEqual(expandedVertex.position, probeSnapshot.position);
assert.deepEqual(expandedVertex.normal, probeSnapshot.normal);
assert.deepEqual(expandedVertex.tangent, probeSnapshot.tangent);
assert.deepEqual(expandedVertex.uv[0], probeSnapshot.uv0);
assert.deepEqual(expandedVertex.uv[1], probeSnapshot.uv1);
assert.deepEqual(expandedVertex.weights, probeSnapshot.weights);
assert.deepEqual(expandedVertex.matrices, probeSnapshot.matrices);
assert.deepEqual([
  expandedVertex.color, expandedVertex.select, expandedVertex.boneCount,
  expandedVertex.tempByte, expandedVertex.mergeTag,
], [probeSnapshot.color, probeSnapshot.select, probeSnapshot.boneCount, probeSnapshot.tempByte, probeSnapshot.mergeTag]);
assert.deepEqual(expandedFace.vertices, probeSnapshot.faceVertices);
assert.deepEqual(expandedFace.adjacent, probeSnapshot.faceAdjacent);
assert.deepEqual(expandedFace.normal, probeSnapshot.faceNormal);
assert.deepEqual([
  expandedFace.select, expandedFace.count, expandedFace.cluster, expandedFace.temp, expandedFace.flags,
], [probeSnapshot.faceSelect, probeSnapshot.faceCount, probeSnapshot.faceCluster, probeSnapshot.faceTemp, probeSnapshot.faceFlags]);
assert.equal(compactProbe.clusters[1].material, probeMaterial);
assert.deepEqual([
  compactProbe.clusters[1].renderPass, compactProbe.clusters[1].id,
  compactProbe.clusters[1].animType, compactProbe.clusters[1].animMatrix,
], [-7, 0x12345678, 2, 19]);

// Static prepare emits self-contained renderer buffers and immediately drops
// the expanded object graph again; cached prepare does not re-expand it.
compactProbe.compact();
const compactPrepared = compactProbe.prepare();
assert.equal(compactProbe.storageSummary().compact, true);

const playbackOnlyMinMesh = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
playbackOnlyMinMesh.compact();
const playbackOnlyPrepared = playbackOnlyMinMesh.prepare({ releaseTopology: true });
assert.equal(playbackOnlyMinMesh.storageSummary().preparedOnly, true);
assert.equal(playbackOnlyMinMesh.prepare(), playbackOnlyPrepared,
  'immutable playback MinMesh keeps its self-contained render buffers');
assert.throws(() => playbackOnlyMinMesh.vertices, /released for immutable playback/);
const playbackOnlyStats = D.minMeshStorageStats({ operations: [{ cache: playbackOnlyMinMesh }] });
assert.equal(playbackOnlyStats.preparedOnly, 1);
assert.equal(playbackOnlyStats.expanded, 0);
assert.ok(playbackOnlyStats.preparedBytes > 0);
assert.equal(compactPrepared.shadowTriangleMask.length, compactPrepared.triangleCount);
assert.equal(compactProbe.prepare(), compactPrepared);
assert.equal(compactProbe.storageSummary().compact, true);

// Modifiers use copy-on-write: graph branches may safely share their input.
const moved = D.MinMesh_TransformEx(cube, 0, 1, 1, 1, 0, 0, 0, 2, 3, 4);
assert.notEqual(moved, cube);
assert.deepEqual(cube.summary().bounds, { minimum: [-0.5, -0.5, -0.5], maximum: [0.5, 0.5, 0.5] });
assert.deepEqual(moved.summary().bounds, { minimum: [1.5, 2.5, 3.5], maximum: [2.5, 3.5, 4.5] });
const compactCowSource = D.MinMesh_Grid(2, 2, 2);
compactCowSource.compact();
const compactCowMoved = D.MinMesh_TransformEx(compactCowSource, 0, 1, 1, 1, 0, 0, 0, 4, 0, 0);
assert.notEqual(compactCowMoved, compactCowSource);
assert.deepEqual(Array.from(compactCowSource.bounds().minimum), [-0.5, 0, -0.5]);
assert.deepEqual(Array.from(compactCowMoved.bounds().minimum), [3.5, 0, -0.5]);

let selected = D.MinMesh_SelectCube(cube, 2, 0, 0, 0, 0.25, 2, 2);
assert.equal(selected.vertices.filter(vertex => vertex.select).length, 0);
selected = D.MinMesh_SelectCube(cube, 6, 0, 0, 0, 2, 2, 2);
assert.equal(selected.faces.filter(face => face.select).length, 6);
const deleted = D.MinMesh_DeleteFaces(selected);
assert.equal(deleted.prepare().triangleCount, 0);
assert.equal(cube.prepare().triangleCount, 12);

const material = { kind: 'material', name: 'test' };
const linked = D.MinMesh_MatLink(cube, material, 0, 7);
assert.equal(linked.clusters.length, 3);
assert.equal(linked.clusters[2].material, material);
assert.equal(linked.clusters[2].renderPass, 7);
assert.ok(linked.faces.every(face => face.cluster === 2));
assert.equal(linked.vertices, cube.vertices, 'material-only branches share immutable vertex topology');
const detachedLinked = D.MinMesh_TransformEx(linked, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0);
assert.notEqual(detachedLinked.vertices, cube.vertices);
assert.deepEqual(cube.summary().bounds, { minimum: [-0.5, -0.5, -0.5], maximum: [0.5, 0.5, 0.5] });

// Sharing is bidirectional: mutating the original after making a material
// branch must detach it as well.
const sharedSource = D.MinMesh_Grid(2, 2, 2);
const sharedBranch = D.MinMesh_MatLink(sharedSource, material);
assert.equal(sharedSource.vertices, sharedBranch.vertices);
sharedSource.transform(0, D.mat4SRT(new Float32Array([1, 1, 1, 0, 0, 0, 3, 0, 0])));
assert.notEqual(sharedSource.vertices, sharedBranch.vertices);
assert.deepEqual(Array.from(sharedBranch.bounds().minimum), [-0.5, 0, -0.5]);

// Dormant material-only branches retain the same native-style sharing after
// compaction: only their face cluster words and cluster tables are private.
const compactMaterialSource = D.MinMesh_Grid(2, 8, 8).compact();
const compactMaterialBranch = D.MinMesh_MatLink(compactMaterialSource, material, 0, 7);
assert.ok(compactMaterialBranch._compact);
assert.equal(compactMaterialBranch._compact.vertexFloats.buffer,
  compactMaterialSource._compact.vertexFloats.buffer);
assert.equal(compactMaterialBranch._compact.faceVertices.buffer,
  compactMaterialSource._compact.faceVertices.buffer);
assert.notEqual(compactMaterialBranch._compact.faceInts.buffer,
  compactMaterialSource._compact.faceInts.buffer);
assert.ok(Array.from(compactMaterialSource._compact.faceInts)
  .filter((_, index) => index % 4 === 2).every(cluster => cluster === 1));
assert.ok(Array.from(compactMaterialBranch._compact.faceInts)
  .filter((_, index) => index % 4 === 2).every(cluster => cluster === 2));
const sharedCompactStats = D.minMeshStorageStats({ operations: [
  { cache: compactMaterialSource }, { cache: compactMaterialBranch },
] });
assert.ok(sharedCompactStats.compactBytes <
  compactMaterialSource.storageSummary().compactBytes +
  compactMaterialBranch.storageSummary().compactBytes,
  'storage telemetry counts shared backing buffers only once');
assert.equal(compactMaterialBranch.prepare().groups[0].material, material);
assert.equal(compactMaterialSource.prepare().groups[0].material, null);

const added = D.MinMesh_Add([grid, grid]);
assert.equal(added.vertices.length, 18);
assert.equal(added.faces.length, 8);
assert.equal(added.prepare().triangleCount, 16);

const triangulated = D.MinMesh_Triangulate(cube);
assert.equal(triangulated.faces.length, 12);
assert.ok(triangulated.faces.every(face => face.count === 3));

const multiplied = D.MinMesh_Multiply(grid, identitySRT, 3, 1, 0.25, 0.5, [0, 0, 0], 0);
assert.equal(multiplied.vertices.length, 27);
assert.equal(multiplied.prepare().triangleCount, 24);
assert.equal(multiplied.vertices[18].uv[0][0], grid.vertices[0].uv[0][0] + 0.5);

// sMatrix::MulA advances the authored transform on the left while the local
// Euler recurrence advances independently.  Rotation/non-uniform scale makes
// the order observable after the second copy (the production graph happens
// to use only translation in its seven Multiply nodes).
const multiplySRT = [1.25, 0.75, 1.5, 0.125, -0.0625, 0.2, 0.5, -0.25, 0.75];
const multiplyLocalRotation = [0.05, 0.1, -0.075];
const multiplyOrderProbe = D.MinMesh_Multiply(
  grid, multiplySRT, 4, 0, 0, 0, multiplyLocalRotation, 0,
);
const multiplyStep = D.mat4SRT(multiplySRT);
const multiplyLocalStep = D.mat4EulerTurns(multiplyLocalRotation);
let multiplyTransform = D.mat4Identity();
let multiplyLocal = D.mat4Identity();
for (let copy = 0; copy < 4; copy++) {
  const expected = D.mat4TransformPoint(
    D.mat4Mul(multiplyTransform, multiplyLocal), grid.vertices[0].position,
  );
  assert.deepEqual(
    Array.from(multiplyOrderProbe.vertices[copy * grid.vertices.length].position),
    Array.from(expected.subarray(0, 3)),
    `Multiply copy ${copy} uses the native xform/local recurrence order`,
  );
  multiplyTransform = D.mat4Mul(multiplyStep, multiplyTransform);
  multiplyLocal = D.mat4Mul(multiplyLocalStep, multiplyLocal);
}

// InitRandomSRT samples scale between identity and the authored value; it is
// not a symmetric range around identity.
const randomScaleSRT = [3, 1, 1, 0, 0, 0, 0, 0, 0];
const randomScaleProbe = D.MinMesh_Multiply(grid, randomScaleSRT, 2, 2, 0, 0, [0, 0, 0], 0);
const randomScaleReference = new D.Random(); randomScaleReference.setSeed(2);
const expectedRandomScale = Math.fround(1 + randomScaleReference.float() * 2);
assert.equal(
  randomScaleProbe.vertices[grid.vertices.length].position[0],
  Math.fround(grid.vertices[0].position[0] * expectedRandomScale),
  'Multiply random scale spans identity to the authored scale',
);

const animatedMultiplySource = D.MinMesh_BoneChain(
  D.MinMesh_Cube(1, 1, 1, 0, identitySRT), 0, 0, -1, 0, 0, 1, 3, 0,
);
const multipliedAnimationDeclaration = animatedMultiplySource.clusters[1].animType;
const multipliedAnimated = D.MinMesh_Multiply(
  animatedMultiplySource, identitySRT, 1, 0, 0, 0, [0, 0, 0], 0, true,
);
assert.equal(multipliedAnimated.animation, null,
  'Multiply Add() copies cluster declarations but not the source skeleton');
assert.equal(multipliedAnimated.completelyRigid, false);
assert.equal(multipliedAnimated.clusters[1].animType, multipliedAnimationDeclaration);

// RenderAutoMap packs enabled projection directions into horizontal atlas
// fields. Disabled directions inherit fields through the native fallback
// chain, while retaining their own projection orientation.
const autoMapped = D.MinMesh_AutoMap(D.MinMesh_Cube(1, 1, 1, 0, identitySRT), 0x21);
for (const face of autoMapped.faces) {
  const u = face.vertices.map(index => autoMapped.vertices[index].uv[0][0]);
  assert.equal(Math.max(...u) - Math.min(...u), 0.5);
}
assert.deepEqual(autoMapped.faces[0].vertices.map(index => autoMapped.vertices[index].uv[0][0]),
  [0.5, 0.5, 1, 1]);
assert.deepEqual(autoMapped.faces[3].vertices.map(index => autoMapped.vertices[index].uv[0][0]),
  [0, 0, 0.5, 0.5]);

const invalidSelectDomain = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
invalidSelectDomain.faces.forEach((face, index) => { face.select = index & 1; });
const invalidSelectSnapshot = invalidSelectDomain.faces.map(face => face.select);
assert.deepEqual(
  D.MinMesh_SelectCube(invalidSelectDomain, 14, 0, 0, 0, 2, 2, 2).faces.map(face => face.select),
  invalidSelectSnapshot,
  'SelectCube domain 12 is the native no-op switch case',
);

const noisyA = D.MinMesh_Perlin(grid, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3);
const noisyB = D.MinMesh_Perlin(grid, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3);
assert.equal(noisyA.summary().hash, noisyB.summary().hash);
assert.notEqual(noisyA.summary().hash, grid.summary().hash);

// FullBoneInner uploads raw weighted N/T. A non-uniform bone transform makes
// any accidental CPU normalization directly observable: +Z must have length
// four and +X length two until the renderer's animated shader conditions it.
const rawSkinProbe = new D.MinMesh();
for (const [position, uv] of [
  [[0, 0, 0], [0, 0]], [[1, 0, 0], [1, 0]], [[0, 1, 0], [0, 1]],
]) {
  const vertex = D.makeMinMeshVertex();
  vertex.position.set(position);
  vertex.uv[0].set(uv);
  vertex.boneCount = 1;
  vertex.matrices[0] = 0;
  vertex.weights[0] = 1;
  rawSkinProbe.vertices.push(vertex);
}
rawSkinProbe.faces.push(D.makeMinMeshFace([0, 1, 2], 1));
const rawSkinMatrix = D.mat4Identity();
rawSkinMatrix[0] = 2; rawSkinMatrix[5] = 3; rawSkinMatrix[10] = 4;
rawSkinMatrix[12] = 5; rawSkinMatrix[13] = 7; rawSkinMatrix[14] = 11;
rawSkinProbe.animation = { matrices: [{
  basePose: D.mat4Identity(), parent: -1, factor: 1, spread: 0, offset: 0,
  noAnimation: rawSkinMatrix, spline: null,
}] };
const rawSkinPrepared = rawSkinProbe.prepare({ time: 0, animationSlot: 0 });
assert.deepEqual(Array.from(rawSkinPrepared.positions), [
  5, 7, 11, 7, 7, 11, 5, 10, 11,
]);
assert.deepEqual(Array.from(rawSkinPrepared.normals), [
  0, 0, 4, 0, 0, 4, 0, 0, 4,
]);
assert.deepEqual(Array.from(rawSkinPrepared.tangents), [
  2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1,
]);

const chained = D.MinMesh_BoneChain(cube, 0, 0, -1, 0, 0, 1, 4, 0);
assert.equal(chained.animation.matrices.length, 4);
assert.ok(chained.vertices.every(vertex => vertex.boneCount > 0));
const trainSpline = {
  eval(time) {
    const matrix = D.mat4Identity(); matrix[12] = time;
    return { matrix, zoom: 1 };
  },
};
const trained = D.MinMesh_BoneTrain(chained, trainSpline, 0.5, 0, 0);
assert.equal(trained.hasAnimation(), true);
const trainedWeights = trained.vertices[0].weights.slice(), trainedMatrices = trained.vertices[0].matrices.slice();
const trainedReferenceAtOne = referenceAnimatedChannels(trained, 1);
const trainedReferenceAtZero = referenceAnimatedChannels(trained, 0);
trained.compact();
assert.equal(trained.hasAnimation(), true);
assert.equal(trained.storageSummary().bones, 4);
const trainedAtOne = trained.prepare({ time: 1 });
assert.equal(trained.storageSummary().compact, true);
const trainedAtZero = trained.prepare({ time: 0 });
assert.equal(trained.storageSummary().compact, true);
assert.deepEqual(trainedAtOne.positions, trainedReferenceAtOne.positions);
assert.deepEqual(trainedAtOne.normals, trainedReferenceAtOne.normals);
assert.deepEqual(trainedAtOne.tangents, trainedReferenceAtOne.tangents);
assert.deepEqual(trainedAtZero.positions, trainedReferenceAtZero.positions);
assert.deepEqual(trainedAtZero.normals, trainedReferenceAtZero.normals);
assert.deepEqual(trainedAtZero.tangents, trainedReferenceAtZero.tangents);
assert.notDeepEqual(Array.from(trainedAtOne.positions), Array.from(trainedAtZero.positions));
const trainedSlotZero = trained.prepare({ time: 1, animationSlot: 0 });
const trainedSlotOne = trained.prepare({ time: 0, animationSlot: 1 });
const slotOneSnapshot = trainedSlotOne.positions.slice();
const reusedSlotZero = trained.prepare({ time: 0, animationSlot: 0 });
assert.equal(reusedSlotZero, trainedSlotZero, 'renderer animation slots reuse bounded output storage');
assert.notEqual(trainedSlotZero.positions, trainedSlotOne.positions,
  'distinct same-frame animation slots retain distinct skinned channels');
assert.deepEqual(trainedSlotOne.positions, slotOneSnapshot,
  'reusing one animation slot leaves other same-frame geometry untouched');
assert.deepEqual(reusedSlotZero.positions, trainedReferenceAtZero.positions);
assert.deepEqual(reusedSlotZero.dynamicAttributes, ['positions', 'normals', 'tangents']);

// Renderer slots are deliberately bounded. Overflow callers still receive a
// complete result, but each call owns ephemeral channels and cannot overwrite
// any geometry retained by the eight reusable slots.
const boundedSlotZeroPositions = trainedSlotZero.positions;
const boundedSlotOnePositions = trainedSlotOne.positions;
const boundedSlotZeroSnapshot = trainedSlotZero.positions.slice();
const boundedSlotOneSnapshot = trainedSlotOne.positions.slice();
const overflowAtOne = trained.prepare({ time: 1, animationSlot: 8 });
const overflowAtOneSnapshot = overflowAtOne.positions.slice();
const overflowAtZero = trained.prepare({ time: 0, animationSlot: 23 });
assert.notEqual(overflowAtOne, overflowAtZero,
  'animation slots at or above the limit return independent ephemeral geometry');
assert.notEqual(overflowAtOne.positions, overflowAtZero.positions);
assert.notEqual(overflowAtOne.positions, boundedSlotZeroPositions);
assert.notEqual(overflowAtZero.positions, boundedSlotOnePositions);
assert.deepEqual(overflowAtOne.positions, overflowAtOneSnapshot,
  'a later overflow prepare does not mutate an earlier ephemeral result');
assert.deepEqual(overflowAtOne.positions, trainedReferenceAtOne.positions);
assert.deepEqual(overflowAtZero.positions, trainedReferenceAtZero.positions);
assert.equal(trainedSlotZero.positions, boundedSlotZeroPositions);
assert.equal(trainedSlotOne.positions, boundedSlotOnePositions);
assert.deepEqual(trainedSlotZero.positions, boundedSlotZeroSnapshot);
assert.deepEqual(trainedSlotOne.positions, boundedSlotOneSnapshot,
  'overflow prepares leave bounded animation slots untouched');
assert.equal(trained._animatedPrepareScratch.slots.length, 2,
  'overflow slot numbers are not retained in the bounded slot table');
assert.equal(Object.keys(trained).includes('_animatedPrepareScratch'), false,
  'derivable animated scratch is excluded from runtime snapshots');
trained.ensureExpanded();
assert.equal(trained.animation.matrices[0].spline, trainSpline);
assert.deepEqual(trained.vertices[0].weights, trainedWeights);
assert.deepEqual(trained.vertices[0].matrices, trainedMatrices);

const shardSource = D.MinMesh_Cube(1, 1, 1, 0, identitySRT);
const exploded = D.MinMesh_Explode(shardSource, null,
  0, 0, 0, 0.1, 1, 0, -1, 0,
  0.1, 0.1, 0.1, 0, 1, 0, 0, 0,
  0, 0, 1, 0, 0);
assert.equal(exploded.vertices.length, 144);
assert.equal(exploded.faces.length, 48);
assert.equal(exploded.animation.matrices.length, 6);
assert.equal(exploded.completelyRigid, true);
const kineticAngle = exploded.animation.matrices[0].spline.angle;
const kineticPosition = exploded.animation.matrices[0].spline.position.slice();
const explodedReferenceAtHalf = referenceAnimatedChannels(exploded, 0.5);
exploded.compact();
const explodedAtHalf = exploded.prepare({ time: 0.5 });
assert.equal(exploded.storageSummary().compact, true);
assert.deepEqual(explodedAtHalf.positions, explodedReferenceAtHalf.positions);
assert.deepEqual(explodedAtHalf.normals, explodedReferenceAtHalf.normals);
assert.deepEqual(explodedAtHalf.tangents, explodedReferenceAtHalf.tangents);
const explodedAtHalfAgain = exploded.prepare({ time: 0.5 });
assert.deepEqual(explodedAtHalfAgain.positions, explodedAtHalf.positions);
exploded.ensureExpanded();
assert.equal(exploded.animation.matrices[0].spline.angle, kineticAngle);
assert.deepEqual(exploded.animation.matrices[0].spline.position, kineticPosition);

const axisExploded = D.MinMesh_Explode(D.MinMesh_Cube(1, 1, 1, 0, identitySRT), null,
  0, 0, 0, 0.1, 1, 0, -1, 0,
  0.1, 0.1, 0.1, 0, 1, 0, 0, 4,
  0, 0, 1, 0, 0.5);
const axisExplodedReference = referenceAnimatedChannels(axisExploded, 0.37);
axisExploded.compact();
const axisExplodedPrepared = axisExploded.prepare({ time: 0.37, animationSlot: 0 });
assert.deepEqual(axisExplodedPrepared.positions, axisExplodedReference.positions,
  'axis-angle kinetic animation stays bit-identical in compact playback');
assert.deepEqual(axisExplodedPrepared.normals, axisExplodedReference.normals);
assert.deepEqual(axisExplodedPrepared.tangents, axisExplodedReference.tangents);

// Explode mode bit 2 rotates a quad before fan triangulation when the other
// diagonal has the better native area-vector dot product.
const explodeDiagonalProbe = new D.MinMesh();
for (const position of [[1, 0, 0], [1, 2, 0], [-5, 1, 0], [-1, 0, 0]]) {
  const vertex = D.makeMinMeshVertex();
  vertex.position.set(position);
  explodeDiagonalProbe.vertices.push(vertex);
}
explodeDiagonalProbe.faces.push(D.makeMinMeshFace([0, 1, 2, 3], 1));
const diagonalExploded = D.MinMesh_Explode(explodeDiagonalProbe, null,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 0, 0, 2,
  0, 0, 1, 0, 0);
assert.deepEqual(
  diagonalExploded.vertices.slice(0, 4).map(vertex => Array.from(vertex.position)),
  [[1, 2, 0], [-5, 1, 0], [-1, 0, 0], [1, 0, 0]],
  'Explode mode bit 2 selects the native quad diagonal',
);
assert.deepEqual(diagonalExploded.faces.slice(0, 2).map(face => face.vertices),
  [[0, 1, 2], [0, 2, 3]]);

// The released operator accidentally scales both texture coordinates by
// XSize.  Preserve that observable behavior for non-square delay maps.
const explodeDelayMap = new D.Bitmap(4, 2);
for (let x = 0; x < 4; x++) {
  explodeDelayMap.data[x * 4] = 8192;
  explodeDelayMap.data[(4 + x) * 4] = 24576;
}
const explodeDelayProbe = new D.MinMesh();
for (const [position, uv] of [
  [[0, 0, 0], [0, 0.5]], [[1, 0, 0], [0, 0.5]], [[0, 1, 0], [0, 0.5]],
]) {
  const vertex = D.makeMinMeshVertex(); vertex.position.set(position); vertex.uv[0].set(uv);
  explodeDelayProbe.vertices.push(vertex);
}
explodeDelayProbe.faces.push(D.makeMinMeshFace([0, 1, 2], 1));
const delayMappedExplosion = D.MinMesh_Explode(explodeDelayProbe, explodeDelayMap,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 1, 0, 0, 0,
  0, 0, 1, 0, 0);
assert.equal(delayMappedExplosion.animation.matrices[0].offset,
  Math.fround(-(1 - 8192 / 32767)),
  'Explode delay-map V uses the native XSize coordinate scale');

// Non-browser callers may install an explicit adapter; production never
// silently substitutes block geometry when real outline extraction fails.
D.setMinMeshFontAdapter(({ height, extrude }) => D.fontPolygonsToMinMesh([{
  outer: [[0, 0], [0, height], [height, height], [height, 0]], holes: [],
}], extrude));
const text = D.MinMesh_Font3D(1, 0.1, 0.05, 'a.', 'sans');
D.setMinMeshFontAdapter(null);
assert.equal(text.hasAnimation(), false);
assert.equal(text.vertices.length, 16);
assert.equal(text.prepare().triangleCount, 12);

// Debris' authored Arial/Georgia subset is vector data, not a browser Canvas
// result. Arimo/Gelasio retain the original horizontal metrics while compact
// reference bounds fit the tessellated glyphs to their Windows bearings and
// visible size without changing topology.
const arialGlyph = D.font3DVectorGlyph('arial', 1, 0.04998779296875, 'a');
const halfArialGlyph = D.font3DVectorGlyph('arial', 0.5, 0.04998779296875, 'a');
const georgiaGlyph = D.font3DVectorGlyph('georgia', 1, 0.04998779296875, 'a');
assert.equal(arialGlyph.advance, 1139 / 2288);
assert.equal(georgiaGlyph.advance, 1032 / 2327);
assert.deepEqual(arialGlyph.referenceBounds, [74, -24, 1052, 1086]);
assert.deepEqual(georgiaGlyph.referenceBounds, [80, -25, 1006, 1014]);
assert.equal(arialGlyph.ppem, 114);
assert.equal(halfArialGlyph.ppem, 55);
assert.equal(georgiaGlyph.ppem, 112);
assert.equal(D.font3DGlyphAdvance(arialGlyph, 1), 64 / 128);
assert.equal(D.font3DGlyphAdvance(georgiaGlyph, 1), 57 / 128);
assert.equal(D.font3DGlyphAdvance(halfArialGlyph, 0.5), 31 / 128);
assert.equal(D.font3DGlyphAdvance(halfArialGlyph, 0.501), 31 / 128,
  'CreateFontA truncates the requested logical height before metric scaling');
assert.ok(arialGlyph.deterministic && georgiaGlyph.deterministic);

for (const [family, data] of Object.entries(FONT3D_FAMILIES)) {
  const heights = family === 'arial' ? [0.5, 1] : [1];
  for (const height of heights) for (const [character, reference] of
    Object.entries(data.referenceBounds)) {
    const mesh = D.MinMesh_Font3D(height, 0, 0.04998779296875, character, family);
    const bounds = mesh.bounds();
    const glyph = D.font3DVectorGlyph(family, height, 0.04998779296875, character);
    const expected = reference.map(value =>
      D.font3DPointFXCoordinate(value, glyph.coordinateUnits, glyph.coordinatePpem));
    assert.deepEqual([
      bounds.minimum[0], bounds.minimum[1], bounds.maximum[0], bounds.maximum[1],
    ], expected, `${family} ${JSON.stringify(character)} matches its Windows outline bounds`);
  }
}

// GGO_UNHINTED points cross FreeType's signed 26.6 grid, whose six fractional
// bits Wine repeats into POINTFX 16.16 before native /128 and Float32 storage.
function expectedFont3DPointFX(value, unitsPerEm, ppem) {
  const fixed26_6 = Math.round(value * ppem * 64 / unitsPerEm);
  const integer = fixed26_6 >> 6;
  let fraction = (fixed26_6 & 0x3f) << 10;
  fraction |= (fraction >> 6) | (fraction >> 12);
  return Math.fround(Math.fround(integer + fraction / 65536) / 128);
}
assert.equal(D.font3DPointFXCoordinate(37, 1000, 128),
  expectedFont3DPointFX(37, 1000, 128));
assert.equal(D.font3DPointFXCoordinate(-37, 1000, 128),
  expectedFont3DPointFX(-37, 1000, 128));

// A unit quadratic has exactly the native recursive midpoint subdivision
// counts at these two tolerances: 8 and 16 segments, plus the starting point.
const quadraticContour = [0, 0, 1, 50, 100, 0, 100, 0, 1];
const coarseQuadratic = D.flattenFont3DContour(quadraticContour, 100, 1, 0.5);
const fineQuadratic = D.flattenFont3DContour(quadraticContour, 100, 1, 0.1);
assert.equal(coarseQuadratic.length, 9);
assert.equal(fineQuadratic.length, 17);
assert.ok(fineQuadratic.flat().every(value => Math.fround(value) === value),
  'native curve subdivision stores every generated point as Float32');
const irregularF32Curve = D.flattenFont3DContour([
  37, 11, 1, 413, 997, 0, 991, -23, 1, 750, -311, 1,
], 1000, 1, 0.5);
assert.deepEqual(irregularF32Curve.slice(0, 3), [
  [0.037078261375427246, 0.01103663444519043],
  [0.13423267006874084, 0.22619041800498962],
  [0.23769985139369965, 0.3786581754684448],
], 'irregular POINTFX curve retains native single-precision subdivision points');

// TrueType contours may begin off-curve. Cover both native start rules: reuse
// the final on-curve point, or synthesize the first/last off-curve midpoint.
const firstOffLastOn = D.flattenFont3DContour([
  50, 100, 0, 100, 0, 1, 100, -100, 1, 0, 0, 1,
], 100, 1, 20);
assert.deepEqual(firstOffLastOn, [[0, 0], [1, 0], [1, -1]]);
const firstAndLastOff = D.flattenFont3DContour([
  0, 100, 0, 100, 0, 1, 200, 0, 1, 200, 100, 0,
], 100, 1, 20);
assert.deepEqual(firstAndLastOff, [[1, 1], [1, 0], [2, 0]]);

const expectedGenericTolerance = Math.fround(Math.fround(
  Math.fround(0.501) * Math.fround(0.1),
) * Math.fround(0.5));
assert.equal(D.font3DToleranceSq(0.501, 0.5),
  Math.fround(expectedGenericTolerance * expectedGenericTolerance));
assert.notEqual(D.font3DToleranceSq(0.501, 0.5), D.font3DToleranceSq(0.5, 0.5),
  'curve tolerance uses native untruncated height, unlike CreateFontA sizing');

assert.throws(() => D.font3DVectorGlyph('georgia', 1, 0, 's'), /exceeds curve limit/,
  'pathological curve tolerances stop at a deterministic memory bound');

const previousDeterministicCanvas = globalThis.OffscreenCanvas;
let deterministicArial, repeatedDeterministicArial;
try {
  globalThis.OffscreenCanvas = class HostFontMustNotBeObserved {
    constructor() { throw new Error('host Canvas font path was observed'); }
  };
  deterministicArial = D.MinMesh_Font3D(1, 0.1, 0.04998779296875, 'hund.', 'arial.');
  repeatedDeterministicArial = D.MinMesh_Font3D(1, 0.1, 0.04998779296875, 'hund.', 'ARIAL');
  assert.throws(() => D.MinMesh_Font3D(1, 0.1, 0.04998779296875, 'A', 'arial'),
    /deterministic arial subset has no glyph "A"/,
    'known production families never silently fall through to a host font');
} finally {
  if (previousDeterministicCanvas === undefined) delete globalThis.OffscreenCanvas;
  else globalThis.OffscreenCanvas = previousDeterministicCanvas;
}
const deterministicArialGeometry = deterministicArial.prepare();
const repeatedDeterministicArialGeometry = repeatedDeterministicArial.prepare();
assert.ok(Math.abs(deterministicArial.bounds().maximum[0] -
  (4 * 64 / 128 + D.font3DPointFXCoordinate(391, 2048, 114))) < 1e-6,
  'Font3D accumulates native integer gmCellIncX positions before /128');
assert.deepEqual([
  deterministicArial.bounds().minimum[1], deterministicArial.bounds().maximum[1],
], [
  D.font3DPointFXCoordinate(-24, 2048, 114),
  D.font3DPointFXCoordinate(1466, 2048, 114),
], 'Font3D fits the final Windows glyph bounds after native tessellation');

let andromedaAdvance = 0;
for (const character of 'andromedasoftwaredevelopment.') {
  const glyph = D.font3DVectorGlyph('arial', 0.5, 0.04998779296875, character);
  andromedaAdvance += D.font3DGlyphAdvance(glyph, 0.5);
}
assert.equal(andromedaAdvance * 128, 830,
  'the intact Andromeda greeting uses Arial 2.82 VDMX and integer GDI advances');
assert.deepEqual([deterministicArial.vertices.length, deterministicArial.faces.length], [804, 587],
  'bounds fitting preserves the established Font3D topology and face order');
for (const vertex of deterministicArial.vertices) {
  assert.equal(vertex.uv[0][0], Math.fround(vertex.position[0] + vertex.position[2]));
  assert.equal(vertex.uv[0][1], vertex.position[1]);
}
assert.deepEqual(repeatedDeterministicArialGeometry.positions,
  deterministicArialGeometry.positions);
assert.deepEqual(repeatedDeterministicArialGeometry.indices,
  deterministicArialGeometry.indices,
  'authored Font3D geometry is bit-deterministic across aliases and repeats');

const glyphMask = new Uint8Array([
  1, 1, 1,
  1, 0, 0,
  1, 1, 1,
]);
const glyphRectangles = D.fontMaskRectangles(glyphMask, 3, 3);
assert.ok(glyphRectangles.length <= 3);
const glyphCoverage = new Uint8Array(glyphMask.length);
for (const [x0, y0, x1, y1] of glyphRectangles) {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    assert.equal(glyphCoverage[y * 3 + x], 0, 'font rectangles do not overlap');
    glyphCoverage[y * 3 + x] = 1;
  }
}
assert.deepEqual(glyphCoverage, glyphMask);
assert.equal(D.fontMaskRectangles(glyphMask, 3, 3, 1), null,
  'font outline complexity has a hard rectangle limit');

// The Canvas Font3D path traces one outer contour plus explicit holes, then
// triangulates the filled polygon as one shared surface.  A square ring keeps
// every expected edge and winding exact enough to audit without a browser.
const ringMask = new Uint8Array([
  1, 1, 1, 1, 1,
  1, 0, 0, 0, 1,
  1, 0, 0, 0, 1,
  1, 0, 0, 0, 1,
  1, 1, 1, 1, 1,
]);
const contourArea = contour => contour.reduce((area, point, index) => {
  const next = contour[(index + 1) % contour.length];
  return area + point[0] * next[1] - next[0] * point[1];
}, 0) * 0.5;

// The smooth Font3D path triangulates vector-like contours through the
// released GLU-compatible tessellator.
// Keep this separate from the mask-band fixtures below: both paths must retain
// holes, native -Z/CW front winding, and closed extrusion topology.
const smoothOuter = [
  [0, 3], [0.75, 1.25], [2, 0.25], [4, 0], [6, 0.25], [7.25, 1.25],
  [8, 3], [7.25, 4.75], [6, 5.75], [4, 6], [2, 5.75], [0.75, 4.75],
];
const smoothHole = [
  [3, 2], [5, 2], [5.5, 3], [5, 4], [3, 4], [2.5, 3],
];
const smoothRegions = [{ outer: smoothOuter, holes: [smoothHole] }];
const smoothPlanar = D.fontPolygonsToMinMesh(smoothRegions, 0);
assert.ok(smoothPlanar, 'smooth Font3D polygons triangulate with a hole');
const smoothPrepared = smoothPlanar.prepare();
const insideContour = (point, contour) => {
  let inside = false;
  for (let index = 0, previous = contour.length - 1;
    index < contour.length; previous = index++) {
    const a = contour[index], b = contour[previous];
    if ((a[1] > point[1]) !== (b[1] > point[1]) &&
      point[0] < (b[0] - a[0]) * (point[1] - a[1]) /
        (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
};
let smoothCoveredArea = 0;
for (let offset = 0; offset < smoothPrepared.indices.length; offset += 3) {
  const ia = smoothPrepared.indices[offset] * 3;
  const ib = smoothPrepared.indices[offset + 1] * 3;
  const ic = smoothPrepared.indices[offset + 2] * 3;
  const ax = smoothPrepared.positions[ia], ay = smoothPrepared.positions[ia + 1];
  const bx = smoothPrepared.positions[ib], by = smoothPrepared.positions[ib + 1];
  const cx = smoothPrepared.positions[ic], cy = smoothPrepared.positions[ic + 1];
  const signedArea = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  assert.ok(signedArea < 0, 'smooth Font3D planar front uses native CW winding');
  smoothCoveredArea -= signedArea * 0.5;
  assert.equal(insideContour([(ax + bx + cx) / 3, (ay + by + cy) / 3], smoothHole), false,
    'smooth Font3D triangulation leaves the glyph hole empty');
}
const expectedSmoothArea = Math.abs(contourArea(smoothOuter)) - Math.abs(contourArea(smoothHole));
assert.ok(Math.abs(smoothCoveredArea - expectedSmoothArea) < 1e-6,
  'smooth Font3D triangles cover exactly the outer contour minus its hole');
const repeatedSmoothPrepared = D.fontPolygonsToMinMesh(smoothRegions, 0).prepare();
assert.deepEqual(repeatedSmoothPrepared.positions, smoothPrepared.positions);
assert.deepEqual(repeatedSmoothPrepared.indices, smoothPrepared.indices,
  'smooth Font3D triangulation order is deterministic');
assert.equal(D.fontPolygonsToMinMesh(smoothRegions, 0, { triangleLimit: 1 }), null,
  'smooth Font3D triangulation observes its bounded triangle limit');

const smoothExtruded = D.fontPolygonsToMinMesh(smoothRegions, 0.25);
assert.ok(smoothExtruded, 'smooth Font3D polygons extrude');
const smoothShadow = D.prepareShadowTopology(D.normalizePreparedGeometry(smoothExtruded));
assert.equal(smoothShadow.boundaryEdges, 0, 'smooth extruded glyph is position-manifold');
assert.equal(smoothShadow.nonManifoldEdges, 0);
assert.equal(smoothShadow.windingConflictEdges, 0,
  'smooth glyph front, back, and side winding agree');

const rasterStaircaseContour = [
  [0, 3], [0, 2], [1, 2], [1, 1], [2, 1], [2, 0], [4, 0],
  [4, 1], [5, 1], [5, 2], [6, 2], [6, 4], [5, 4], [5, 5],
  [4, 5], [4, 6], [2, 6], [2, 5], [1, 5], [1, 4], [0, 4],
];
const smoothStaircaseContour = D.simplifyClosedFontContour(rasterStaircaseContour, 0.5);
assert.ok(smoothStaircaseContour.length < rasterStaircaseContour.length,
  'closed contour simplification collapses raster staircase points');
assert.equal(Math.sign(contourArea(smoothStaircaseContour)),
  Math.sign(contourArea(rasterStaircaseContour)),
  'closed contour simplification preserves winding orientation');
assert.ok(smoothStaircaseContour.some((point, index) => {
  const next = smoothStaircaseContour[(index + 1) % smoothStaircaseContour.length];
  return point[0] !== next[0] && point[1] !== next[1];
}), 'closed contour simplification replaces staircase runs with non-axis-aligned segments');

// Exercise the actual Canvas Font3D route without a browser or native Canvas
// dependency. The fake context returns a deterministic slanted oval ring at
// whatever bounded resolution Font3D requests, so this stays independent of a
// particular production resolution while retaining raster staircase pressure.
class FakeFont3DContext {
  constructor(canvas) {
    this.canvas = canvas;
    this.font = '10px sans-serif';
    this.fontKerning = 'auto';
    this.textRendering = 'auto';
  }

  fontSize() {
    const match = /([0-9.]+)px/.exec(this.font);
    return Math.max(1, Number(match?.[1]) || 10);
  }

  measureText() {
    const size = this.fontSize();
    return {
      width: size * 0.78,
      actualBoundingBoxLeft: 0,
      actualBoundingBoxRight: size * 0.78,
      actualBoundingBoxAscent: size * 0.8,
      actualBoundingBoxDescent: size * 0.2,
    };
  }

  clearRect() {}
  fillText() {}

  getImageData(_x, _y, width, height) {
    const data = new Uint8ClampedArray(width * height * 4);
    const centerX = width * 0.5, centerY = height * 0.5;
    const radiusX = Math.max(2, width * 0.43), radiusY = Math.max(2, height * 0.43);
    const cosine = 0.9393727128473789, sine = 0.34289780745545134;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const dx = x + 0.5 - centerX, dy = y + 0.5 - centerY;
      const rotatedX = (dx * cosine + dy * sine) / radiusX;
      const rotatedY = (-dx * sine + dy * cosine) / radiusY;
      const outer = rotatedX * rotatedX + rotatedY * rotatedY <= 1;
      const innerX = rotatedX / 0.53, innerY = rotatedY / 0.57;
      const inner = innerX * innerX + innerY * innerY < 1;
      if (outer && !inner) data[(y * width + x) * 4 + 3] = 255;
    }
    return { data };
  }
}

class FakeFont3DCanvas {
  constructor(width, height) { this.width = width; this.height = height; }
  getContext(kind) { return kind === '2d' ? new FakeFont3DContext(this) : null; }
}

const previousOffscreenCanvas = globalThis.OffscreenCanvas;
const fakeCanvasDepth = 0.1;
let fakeCanvasFontMesh;
try {
  globalThis.OffscreenCanvas = FakeFont3DCanvas;
  fakeCanvasFontMesh = D.MinMesh_Font3D(
    1, fakeCanvasDepth, 0.04998779296875, 'z', 'font3d-regression',
  );
} finally {
  if (previousOffscreenCanvas === undefined) delete globalThis.OffscreenCanvas;
  else globalThis.OffscreenCanvas = previousOffscreenCanvas;
}
assert.ok(fakeCanvasFontMesh.vertices.length > 32,
  'Canvas Font3D uses the traced smooth contour rather than the box fallback');
const fakeCanvasSideFaces = fakeCanvasFontMesh.faces.filter(face => {
  const depths = face.vertices.map(index => fakeCanvasFontMesh.vertices[index].position[2]);
  return Math.min(...depths) !== Math.max(...depths);
});
assert.ok(fakeCanvasSideFaces.length > 12 && fakeCanvasSideFaces.length < 512,
  'Canvas Font3D smooths the raster outline to bounded contour complexity');
for (const face of fakeCanvasSideFaces) {
  const positions = face.vertices.map(index => fakeCanvasFontMesh.vertices[index].position);
  const front = positions.filter(position => Math.abs(position[2]) < 1e-7);
  const back = positions.filter(position => Math.abs(position[2] - fakeCanvasDepth) < 1e-7);
  assert.equal(front.length, 2);
  assert.equal(back.length, 2);
  for (const position of front) {
    assert.ok(back.some(candidate =>
      Math.abs(candidate[0] - position[0]) < 1e-6 &&
      Math.abs(candidate[1] - position[1]) < 1e-6),
    'native Font3D extrusion displaces each back point only along Z');
  }
}
for (const vertex of fakeCanvasFontMesh.vertices) {
  assert.equal(vertex.uv[0][0], Math.fround(vertex.position[0] + vertex.position[2]),
    'native Font3D projects position through k.x=1 into UV0.u');
  assert.equal(vertex.uv[0][1], vertex.position[1],
    'native Font3D projects position.y into UV0.v');
}
assert.ok(fakeCanvasSideFaces.some(face => {
  const front = face.vertices
    .map(index => fakeCanvasFontMesh.vertices[index].position)
    .filter(position => position[2] === 0);
  return front.length >= 2 && Math.abs(front[0][0] - front[1][0]) > 1e-6 &&
    Math.abs(front[0][1] - front[1][1]) > 1e-6;
}), 'Canvas Font3D preserves smooth non-axis-aligned contour segments');
const fakeCanvasGeometry = D.normalizePreparedGeometry(fakeCanvasFontMesh);
const fakeCanvasDiagnostics = D.geometryTopologyStats(fakeCanvasGeometry);
assert.equal(fakeCanvasDiagnostics.degenerateTriangles, 0,
  'Canvas Font3D emits no degenerate triangles after Float32 conversion');
assert.equal(fakeCanvasDiagnostics.unexpectedWinding, 0,
  'Canvas Font3D retains native-facing winding after Float32 conversion');
const fakeCanvasShadow = D.prepareShadowTopology(fakeCanvasGeometry);
assert.equal(fakeCanvasShadow.boundaryEdges, 0);
assert.equal(fakeCanvasShadow.nonManifoldEdges, 0);
assert.equal(fakeCanvasShadow.windingConflictEdges, 0,
  'Canvas Font3D extrusion remains a closed consistently wound shadow caster');
assert.equal(fakeCanvasShadow.maxEdgeIncidence, 2);

// Raster contours can place a distinct point exactly on a long edge. In
// normalized doubles the corrective tessellation triangle is near-zero, and
// Float32 rounding can reverse it. The triangulator must remove or repair that
// condition without opening extrusion seams.
const nearCollinearScale = 997;
const nearCollinearOuter = [
  [76 / nearCollinearScale, 121 / nearCollinearScale],
  [63 / nearCollinearScale, 13 / nearCollinearScale],
  [30 / nearCollinearScale, 19 / nearCollinearScale],
  [30 / nearCollinearScale, 112 / nearCollinearScale],
  [29 / nearCollinearScale, 115 / nearCollinearScale],
];
const nearCollinearMesh = D.fontPolygonsToMinMesh([
  { outer: nearCollinearOuter, holes: [] },
], 0.1);
assert.ok(nearCollinearMesh, 'near-collinear Font3D contour triangulates');
const nearCollinearGeometry = D.normalizePreparedGeometry(nearCollinearMesh);
const nearCollinearDiagnostics = D.geometryTopologyStats(nearCollinearGeometry);
assert.equal(nearCollinearDiagnostics.degenerateTriangles, 0);
assert.equal(nearCollinearDiagnostics.unexpectedWinding, 0,
  'near-collinear repair keeps front/back orientation stable after Float32 conversion');
const nearCollinearShadow = D.prepareShadowTopology(nearCollinearGeometry);
assert.equal(nearCollinearShadow.boundaryEdges, 0);
assert.equal(nearCollinearShadow.nonManifoldEdges, 0);
assert.equal(nearCollinearShadow.windingConflictEdges, 0,
  'near-collinear repair leaves extrusion topology closed and consistently wound');
assert.equal(nearCollinearShadow.maxEdgeIncidence, 2);

const ringContours = D.fontMaskContours(ringMask, 5, 5);
assert.equal(ringContours.length, 2);
assert.deepEqual(ringContours.map(contourArea).sort((a, b) => a - b), [-9, 25],
  'mask contours retain the 3x3 hole and 5x5 outer boundary');
const ringPolygons = D.fontMaskPolygons(ringContours);
assert.equal(ringPolygons.length, 1);
assert.equal(ringPolygons[0].holes.length, 1);
assert.equal(D.fontMaskContours(ringMask, 5, 5, 31), null,
  'font contour edge storage has a deterministic hard limit');

const planarRing = D.fontMaskToMinMesh(ringMask, 5, 5, 0);
assert.ok(planarRing);
assert.equal(planarRing.faces.length, 16);
assert.ok(planarRing.vertices.every(vertex => vertex.position[2] === 0),
  'extrude=0 emits only the native planar front');
const planarPrepared = planarRing.prepare();
for (let offset = 0; offset < planarPrepared.indices.length; offset += 3) {
  const ia = planarPrepared.indices[offset] * 3;
  const ib = planarPrepared.indices[offset + 1] * 3;
  const ic = planarPrepared.indices[offset + 2] * 3;
  const ax = planarPrepared.positions[ia], ay = planarPrepared.positions[ia + 1];
  const bx = planarPrepared.positions[ib], by = planarPrepared.positions[ib + 1];
  const cx = planarPrepared.positions[ic], cy = planarPrepared.positions[ic + 1];
  const signedArea = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  assert.ok(signedArea < 0, 'Font3D planar front uses renderer-facing CW winding');
  const centroidX = (ax + bx + cx) / 3, centroidY = (ay + by + cy) / 3;
  assert.ok(!(centroidX > 1 && centroidX < 4 && centroidY > 1 && centroidY < 4),
    'front triangulation does not cover the glyph hole');
}

const extrudedRing = D.fontMaskToMinMesh(ringMask, 5, 5, 0.25);
assert.ok(extrudedRing);
const sideFaces = extrudedRing.faces.filter(face => {
  const z = face.vertices.map(index => extrudedRing.vertices[index].position[2]);
  return Math.min(...z) !== Math.max(...z);
});
assert.ok(sideFaces.length > 0);
for (const face of sideFaces) {
  const points = face.vertices.map(index => extrudedRing.vertices[index].position);
  const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
  const sameX = xs.every(value => value === xs[0]), sameY = ys.every(value => value === ys[0]);
  const middleX = (Math.min(...xs) + Math.max(...xs)) * 0.5;
  const middleY = (Math.min(...ys) + Math.max(...ys)) * 0.5;
  const onOuter = sameX && (xs[0] === 0 || xs[0] === 5) && middleY >= 0 && middleY <= 5 ||
    sameY && (ys[0] === 0 || ys[0] === 5) && middleX >= 0 && middleX <= 5;
  const onHole = sameX && (xs[0] === 1 || xs[0] === 4) && middleY >= 1 && middleY <= 4 ||
    sameY && (ys[0] === 1 || ys[0] === 4) && middleX >= 1 && middleX <= 4;
  assert.ok(onOuter || onHole, 'extrusion walls exist only on outer or hole contours');
}
const ringShadow = D.prepareShadowTopology(D.normalizePreparedGeometry(extrudedRing));
assert.equal(ringShadow.boundaryEdges, 0, 'extruded glyph is position-manifold');
assert.equal(ringShadow.nonManifoldEdges, 0);
assert.equal(ringShadow.windingConflictEdges, 0, 'front, back, and side winding agree');

const diagonalMask = new Uint8Array([
  0, 1,
  1, 0,
]);
assert.equal(D.fontMaskContours(diagonalMask, 2, 2).length, 2,
  'corner-touching pixels remain separate outline components');
const diagonalMesh = D.fontMaskToMinMesh(diagonalMask, 2, 2, 0.25);
const diagonalShadow = D.prepareShadowTopology(D.normalizePreparedGeometry(diagonalMesh));
assert.equal(diagonalShadow.boundaryEdges, 0);
assert.equal(diagonalShadow.nonManifoldEdges, 0,
  'ambiguous diagonal mask corners do not share a four-face extrusion edge');
assert.equal(diagonalShadow.windingConflictEdges, 0);

const staircaseMask = new Uint8Array([
  1, 1, 1,
  1, 1, 0,
  1, 0, 0,
]);
const staircaseMesh = D.fontMaskToMinMesh(staircaseMask, 3, 3, 0.25);
const staircaseShadow = D.prepareShadowTopology(D.normalizePreparedGeometry(staircaseMesh));
assert.equal(staircaseShadow.boundaryEdges, 0);
assert.equal(staircaseShadow.nonManifoldEdges, 0,
  'run-end splits propagate through adjacent bands without T-junction walls');
assert.equal(staircaseShadow.windingConflictEdges, 0);
assert.equal(D.fontMaskToMinMesh(ringMask, 5, 5, 0.25, { triangleLimit: 1 }), null,
  'font surface triangulation has a deterministic hard limit');

assert.throws(() => D.MinMesh_Font3D(1, 0, 0.05, 'a.', 'sans'),
  /Canvas outline extraction or an explicit font adapter/,
  'Font3D never silently replaces unavailable outlines with block glyphs');

function makePipeProbeMesh(vCoordinates = [0.25, 0.5, 0.75]) {
  const mesh = new D.MinMesh();
  for (let index = 0; index < 3; index++) {
    const vertex = D.makeMinMeshVertex();
    vertex.position.set([0, 0, index / 2]);
    vertex.uv[0].set([0, vCoordinates[index]]);
    mesh.vertices.push(vertex);
  }
  mesh.faces.push(D.makeMinMeshFace([0, 1, 2], 1));
  return mesh;
}

const quarterTurn = Math.fround(Math.SQRT1_2);
const pipeKey = (position, quaternion) => ({
  select: 0, time: 0,
  px: position[0], py: position[1], pz: position[2],
  zoom: quaternion[0], rx: quaternion[1], ry: quaternion[2], rz: quaternion[3],
});
const rightAnglePipe = {
  mode: 4,
  pipe: {
    count: 2,
    keys: [
      { x: 0, y: 2, z: 0, radius: 0.5, flags: 0 },
      { x: 2, y: 2, z: 0, radius: 0.5, flags: 0 },
    ],
  },
  keys: [
    pipeKey([0, 0, 0], [quarterTurn, quarterTurn, 0, 0]),
    pipeKey([0, 1.5, 0], [quarterTurn, quarterTurn, 0, 0]),
    pipeKey([0.5, 2, 0], [quarterTurn, 0, -quarterTurn, 0]),
    pipeKey([1.5, 2, 0], [quarterTurn, 0, -quarterTurn, 0]),
  ],
};
assert.equal(D.MinMesh_Pipe({ mode: 4, keys: rightAnglePipe.keys },
  makePipeProbeMesh(), null, null, 0, 1, 1, 3), null,
'Pipe requires the retained BlobPipe parameters, not only a generic spline');

const rightAngleResult = D.MinMesh_Pipe(rightAnglePipe,
  makePipeProbeMesh(), makePipeProbeMesh(), null, 5, 1, 1, 3);
assert.equal(rightAngleResult.vertices.length, 9);
const rightAngleCurve = rightAngleResult.vertices.slice(3, 6);
assert.ok(Math.abs(rightAngleCurve[0].position[0] - 0) < 1e-6 &&
  Math.abs(rightAngleCurve[0].position[1] - 1.5) < 1e-6);
assert.ok(Math.abs(rightAngleCurve[1].position[0] - (0.5 - Math.SQRT1_2 / 2)) < 1e-6 &&
  Math.abs(rightAngleCurve[1].position[1] - (1.5 + Math.SQRT1_2 / 2)) < 1e-6,
'Pipe bends mesh1 around the native analytic circular center');
assert.ok(Math.abs(rightAngleCurve[2].position[0] - 0.5) < 1e-6 &&
  Math.abs(rightAngleCurve[2].position[1] - 2) < 1e-6);
assert.deepEqual(rightAngleCurve.map(vertex => vertex.uv[0][1]), [0.25, 0.5, 0.75],
  'flags=5 preserves authored bend UVs because (flags & 17) is not 17');

const rightAngleArcUV = D.MinMesh_Pipe(rightAnglePipe,
  makePipeProbeMesh(), makePipeProbeMesh(), null, 17, 1, 1, 3).vertices.slice(3, 6);
const quarterArcLength = Math.fround(0.5 * Math.fround(Math.PI / 2));
assert.ok(Math.abs(rightAngleArcUV[0].uv[0][1]) < 1e-7);
assert.ok(Math.abs(rightAngleArcUV[1].uv[0][1] - quarterArcLength / 2) < 1e-6);
assert.ok(Math.abs(rightAngleArcUV[2].uv[0][1] - quarterArcLength) < 1e-6,
  'bend UV distance uses radius*gamma arc length');

// Production op15780 is the only authored Pipe with flags=5. Its long
// straights are split into repeated copies, while each copy must retain its
// accumulated fraction along that straight and each bend must retain mesh1 V.
const productionDocument = parseKX(readFileSync(
  new URL('../assets/debris_party.kx', import.meta.url),
));
const productionPipeOp = productionDocument.operations[15780];
assert.deepEqual(productionPipeOp.parameters, [5, 1, 1, 3]);
const productionRuntime = new D.Runtime(productionDocument, {
  handlers: D.handlers, strictHandlers: false,
});
const productionPipeSpline = productionRuntime.operations[15759].precalc();
const productionRepeatCounts = [];
for (let segment = 0; segment < productionPipeSpline.pipe.count; segment++) {
  const first = productionPipeSpline.keys[segment * 2];
  const second = productionPipeSpline.keys[segment * 2 + 1];
  const distance = Math.fround(Math.hypot(
    second.px - first.px, second.py - first.py, second.pz - first.pz,
  ));
  productionRepeatCounts.push(Math.max(1, Math.min(64,
    Math.trunc(distance / productionPipeOp.parameters[3] + 0.5))));
}
assert.deepEqual(productionRepeatCounts, [9, 5, 1, 3, 6, 2, 1, 2, 1, 11],
  'op15780 retains the native repeated-straight counts');
const productionPipeResult = D.MinMesh_Pipe(
  productionPipeSpline, makePipeProbeMesh(), makePipeProbeMesh(), null,
  ...productionPipeOp.parameters,
);
const expectedProductionCopies = productionRepeatCounts.reduce((sum, count) => sum + count, 0) +
  productionPipeSpline.pipe.count - 1;
assert.equal(productionPipeResult.vertices.length, expectedProductionCopies * 3);
const firstStraightCopies = productionRepeatCounts[0];
for (let copy = 0; copy + 1 < firstStraightCopies; copy++) {
  const previousEnd = productionPipeResult.vertices[copy * 3 + 2].uv[0][1];
  const nextStart = productionPipeResult.vertices[(copy + 1) * 3].uv[0][1];
  assert.equal(previousEnd, nextStart,
    'repeated straight copies continue V instead of restarting it');
}
assert.equal(productionPipeResult.vertices[firstStraightCopies * 3 - 1].uv[0][1], 27.5,
  'the final repeated copy reaches the complete first-straight distance');
assert.deepEqual(productionPipeResult.vertices
  .slice(firstStraightCopies * 3, firstStraightCopies * 3 + 3)
  .map(vertex => vertex.uv[0][1]), [0.25, 0.5, 0.75],
  'op15780 flags=5 preserves the authored mesh1 bend V values');

const converted = D.meshToMin({
  kind: 'mesh',
  prepare() {
    return {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      indices: new Uint16Array([0, 1, 2]),
      materials: [material],
    };
  },
});
assert.equal(converted.vertices.length, 3);
assert.equal(converted.faces.length, 1);
assert.equal(converted.prepare().triangleCount, 1);

// Native Mesh_ToMin packs a present COLOR0 sVector even when it is all zero;
// white is reserved for old vertex formats where COLOR0 is genuinely absent.
const oldColorCube = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
oldColorCube.vertices[0].color.set([1.2, 0.5, -0.2, 0.25]);
const colorConverted = D.meshToMin(oldColorCube);
assert.equal(colorConverted.vertices[0].color, 0x3fff7f00);
assert.equal(colorConverted.vertices[1].color, 0x00000000);
const compactZeroColorCube = D.Mesh_Cube(1, 1, 1, 0, identitySRT).compact();
assert.equal(compactZeroColorCube._compact.vertexColors, null,
  'all-zero compact COLOR0 is represented by an implicit channel');
compactZeroColorCube.ensureExpanded = () => {
  throw new Error('implicit compact COLOR0 conversion expanded its source');
};
assert.ok(D.meshToMin(compactZeroColorCube).vertices.every(vertex => vertex.color === 0),
  'declared compact COLOR0 packs implicit zero instead of missing-channel white');
const noColorCube = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
noColorCube.vertexMask &= ~D.MESH_FEATURE.COLOR0;
assert.equal(D.meshToMin(noColorCube).vertices[0].color, 0xffffffff);

// Mesh_ToMin first triangulates selected faces whose degree reaches the
// eight-index MinMesh limit.  Its centroid averages every old vertex vector,
// including the second UV channel.
const oldOctagon = new D.Mesh(D.MESH_DEFAULT_VERTEX_MASK | D.MESH_FEATURE.UV1);
for (let index = 0; index < 8; index++) {
  const angle = index * Math.PI * 2 / 8;
  const vertexIndex = oldOctagon.addVertex([Math.cos(angle), Math.sin(angle), 0], [index, index + 0.25]);
  oldOctagon.vertices[vertexIndex].uv1.set([index + 10, index + 20, 0, 1]);
}
oldOctagon.setPolygons([{ verts: [0, 1, 2, 3, 4, 5, 6, 7], material: 1, select: true, used: false }]);
const unselectedExpandedNinegon = new D.Mesh(D.MESH_DEFAULT_VERTEX_MASK);
for (let index = 0; index < 9; index++) {
  const angle = index * Math.PI * 2 / 9;
  unselectedExpandedNinegon.addVertex([Math.cos(angle), Math.sin(angle), 0], [index, 0]);
}
unselectedExpandedNinegon.setPolygons([{
  verts: [0, 1, 2, 3, 4, 5, 6, 7, 8], material: 1, select: false, used: false,
}]);
const unselectedCompactNinegon = unselectedExpandedNinegon.clone().compact();
oldOctagon.compact();
oldOctagon.ensureExpanded = () => {
  throw new Error('compact Mesh_ToMin expanded its source');
};
const convertedOctagon = D.meshToMin(oldOctagon);
assert.ok(oldOctagon._compact, 'compact Mesh_ToMin leaves a non-owned source dormant');
assert.equal(convertedOctagon.vertices.length, 9);
assert.equal(convertedOctagon.faces.length, 8);
assert.ok(convertedOctagon.faces.every(face => face.count === 3 && face.flags === 1));
assert.deepEqual(Array.from(convertedOctagon.vertices[3].uv[1]), [13, 23]);
assert.deepEqual(Array.from(convertedOctagon.vertices[8].uv[1]), [13.5, 23.5]);
assert.deepEqual(convertedOctagon.faces.map(face => face.vertices), [
  [8, 0, 1], [8, 1, 2], [8, 2, 3], [8, 3, 4],
  [8, 4, 5], [8, 5, 6], [8, 6, 7], [8, 7, 0],
]);

// The preliminary native Triangulate call deliberately leaves an unselected
// high-degree face alone. All2Sel happens afterwards, so Mesh_ToMin retains
// all selected vertices but its fixed KMM_MAXVERT face loop copies only the
// original first eight corners instead of adding a centroid fan.
for (const [label, source] of [
  ['expanded', unselectedExpandedNinegon],
  ['compact', unselectedCompactNinegon],
]) {
  const convertedUnselectedNinegon = D.meshToMin(source);
  assert.equal(convertedUnselectedNinegon.vertices.length, 9,
    `${label} unselected nine-gon retains all old selected vertices without a centroid`);
  assert.equal(convertedUnselectedNinegon.faces.length, 1,
    `${label} unselected nine-gon remains one face`);
  assert.deepEqual(convertedUnselectedNinegon.faces[0].vertices,
    [0, 1, 2, 3, 4, 5, 6, 7],
    `${label} unselected nine-gon truncates to the native first-eight corner order`);
}

const conversionFloatBits = value => Array.from(new Uint32Array(
  value.buffer, value.byteOffset, value.byteLength / Uint32Array.BYTES_PER_ELEMENT,
));
const conversionSnapshot = mesh => ({
  vertices: mesh.vertices.map(vertex => ({
    select: vertex.select | 0,
    boneCount: vertex.boneCount | 0,
    tempByte: vertex.tempByte | 0,
    mergeTag: vertex.mergeTag | 0,
    color: vertex.color >>> 0,
    position: conversionFloatBits(vertex.position),
    normal: conversionFloatBits(vertex.normal),
    tangent: conversionFloatBits(vertex.tangent),
    uv0: conversionFloatBits(vertex.uv[0]),
    uv1: conversionFloatBits(vertex.uv[1]),
    weights: conversionFloatBits(vertex.weights),
    matrices: Array.from(vertex.matrices),
  })),
  faces: mesh.faces.map(face => ({
    select: face.select | 0,
    count: face.count | 0,
    cluster: face.cluster | 0,
    flags: face.flags >>> 0,
    vertices: Array.from(face.vertices),
  })),
  clusters: mesh.clusters.map(cluster => ({
    material: cluster.material,
    renderPass: cluster.renderPass | 0,
    id: cluster.id | 0,
    animType: cluster.animType | 0,
    animMatrix: cluster.animMatrix | 0,
  })),
});

// Exercise every old-record choice made by the direct flat converter: physical
// First position versus per-corner attributes, multiple materials, deleted
// faces, shadow flags, an eight-corner fan, UV1 and exact packed COLOR0 values.
const conversionFixture = new D.Mesh(D.MESH_DEFAULT_VERTEX_MASK | D.MESH_FEATURE.UV1);
const conversionMaterial = { kind: 'material', name: 'compact conversion material' };
conversionFixture.materials.push({ material: conversionMaterial, pass: 9 });
for (let index = 0; index < 8; index++) {
  const angle = index * Math.PI * 2 / 8;
  const vertexIndex = conversionFixture.addVertex(
    [Math.cos(angle) * (index + 1), Math.sin(angle) * (index + 1), index * 0.25, 1],
    [index === 0 ? -0 : index / 7, (7 - index) / 7],
  );
  const vertex = conversionFixture.vertices[vertexIndex];
  vertex.color.set(index === 0
    ? [1.2, 0.5, -0.2, 0.25]
    : [index / 7, (7 - index) / 7, index / 14, index % 2]);
  vertex.uv1.set([index + 10.25, index === 1 ? -0 : index + 20.5, 0, 1]);
}
conversionFixture.setPolygons([
  { verts: [0, 1, 2, 3, 4, 5, 6, 7], material: 1, used: false },
  { verts: [0, 2, 4], material: 2, used: true },
  { verts: [1, 3, 5], material: 0, used: false },
]);
conversionFixture.vertices[3].first = 0;
const expandedConversion = D.meshToMin(conversionFixture.clone());
const compactConversionSource = conversionFixture.clone().compact();
compactConversionSource.ensureExpanded = () => {
  throw new Error('direct compact conversion reconstructed old records');
};
const compactConversion = D.meshToMin(compactConversionSource);
const expandedConversionSnapshot = conversionSnapshot(expandedConversion);
assert.deepEqual(conversionSnapshot(compactConversion), expandedConversionSnapshot,
  'direct compact conversion is bitwise-equivalent to expanded old records');
assert.deepEqual(compactConversion.prepare(), expandedConversion.prepare(),
  'direct compact conversion prepares identical topology and attributes');
assert.ok(compactConversionSource._compact, 'direct compact conversion is read-only');

const bridgeConversionSource = conversionFixture.clone().compact();
bridgeConversionSource.ensureExpanded = () => {
  throw new Error('Mesh_ToMin bridge expanded its compact source');
};
assert.deepEqual(conversionSnapshot(Mesh_ToMin(bridgeConversionSource)),
  expandedConversionSnapshot, 'Mesh_ToMin bridge preserves the direct compact path');

const disposableCompactConversion = conversionFixture.clone().compact();
disposableCompactConversion.ensureExpanded = () => {
  throw new Error('unique compact conversion expanded its source');
};
const forcedCompactMeshToMin = D.createMeshToMinHandler(() => true);
assert.deepEqual(conversionSnapshot(forcedCompactMeshToMin({ inputs: [disposableCompactConversion] })),
  expandedConversionSnapshot);
assert.equal(disposableCompactConversion.released, true,
  'unique compact old-Mesh storage is released after direct conversion');

const noUV1Octagon = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
noUV1Octagon.vertices[0].uv1.set([0.25, 0.75]);
noUV1Octagon.compact();
assert.ok(noUV1Octagon._compact.vertexUV1s,
  'dormant UV1 data survives even when the old vertex format omits UV1');
noUV1Octagon.ensureExpanded = () => {
  throw new Error('absent compact UV1 conversion expanded its source');
};
assert.deepEqual(Array.from(D.meshToMin(noUV1Octagon).vertices[0].uv[1]), [0, 0],
  'UV1 data is ignored when the old vertex format omits UV1');

const noUV0Cube = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
noUV0Cube.vertexMask &= ~D.MESH_FEATURE.UV0;
noUV0Cube.vertices[0].uv.set([0.25, 0.75]);
assert.deepEqual(Array.from(D.meshToMin(noUV0Cube.clone()).vertices[0].uv[0]), [0, 0],
  'expanded conversion ignores UV0 data absent from the old vertex format');
noUV0Cube.compact();
assert.ok(noUV0Cube._compact.vertexUVs,
  'dormant UV0 data survives even when the old vertex format omits UV0');
noUV0Cube.ensureExpanded = () => {
  throw new Error('absent compact UV0 conversion expanded its source');
};
assert.deepEqual(Array.from(D.meshToMin(noUV0Cube).vertices[0].uv[0]), [0, 0],
  'compact conversion ignores UV0 data absent from the old vertex format');

const weightedOldMesh = D.Mesh_Cube(1, 1, 1, 0, identitySRT);
weightedOldMesh.vertices[0].weights.set([255, 0, 0, 0]);
weightedOldMesh.vertices[0].matrices.set([7, 255, 255, 255]);
weightedOldMesh.vertices[0].normal.set([0.25, 0.5, 0.75]);
const unskinnedConversion = D.meshToMin(weightedOldMesh);
assert.equal(unskinnedConversion.vertices[0].boneCount, 0,
  'Mesh_ToMin discards old GenMesh skinning metadata');
assert.deepEqual(Array.from(unskinnedConversion.vertices[0].weights), [0, 0, 0, 0]);
assert.deepEqual(Array.from(unskinnedConversion.vertices[0].normal), [0, 0, 0],
  'Mesh_ToMin leaves derived channels for GenMinMesh::CalcNormals');

let releasedAtConversion = false;
const disposableOldMesh = {
  kind: 'mesh',
  prepare() {
    return {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint16Array([0, 1, 2]), materials: [material],
    };
  },
  releaseStorage() { releasedAtConversion = true; },
};
const forcedUniqueMeshToMin = D.createMeshToMinHandler(() => true);
assert.equal(forcedUniqueMeshToMin({ inputs: [disposableOldMesh] }).faces.length, 1);
assert.equal(releasedAtConversion, true, 'unique old-Mesh storage is released at the conversion seam');

function makeRuntimeDocument(specifications, root = specifications.length - 1, extraClasses = {}) {
  const ids = Array.from(new Set(specifications.map(value => value.classId)));
  const registry = { ...D.CLASS_REGISTRY, ...extraClasses };
  const classes = ids.map(id => {
    const info = registry[id];
    return { id, convention: info.convention, packing: '-'.repeat(info.dataWords || 0) };
  });
  const classIndices = new Map(classes.map((value, index) => [value.id, index]));
  const operations = specifications.map((value, id) => ({
    id, classId: value.classId, classIndex: classIndices.get(value.classId),
    inputs: value.inputs || [], links: value.links || [], parameters: value.parameters || [],
    strings: [], splines: [], animation: new Uint8Array([1]),
  }));
  return {
    document: { classes, operations, roots: [root], events: [], splines: [], songBPMFixed: 196 * 65536, buzzTiming: true },
    registry,
  };
}

// Runtime final-consumer transfer preserves both sides of a fan-out. MatInput
// first shares the Grid vertices; the later Transform must detach the source,
// and Add may then adopt/release both finished branches.
const testMaterialClass = {
  id: 0x700, name: 'TestMaterial', convention: 0, dataWords: 0,
  outputClass: 'KC_MATERIAL', init: 'TestMaterial', exec: 'Exec_Misc_Nop',
};
const branchFixture = makeRuntimeDocument([
  { classId: 0x101, parameters: [2, 2, 2] },
  { classId: 0x700 },
  { classId: 0x117, inputs: [0, 1], parameters: [0, 0] },
  { classId: 0x120, inputs: [0], parameters: [0, 1, 1, 1, 0, 0, 0, 2, 0, 0] },
  { classId: 0x111, inputs: [2, 3] },
], 4, { 0x700: testMaterialClass });
const branchHandlers = new Map(D.handlers);
branchHandlers.set(0x700, () => material);
const branchRuntime = new D.Runtime(branchFixture.document, { registry: branchFixture.registry, handlers: branchHandlers });
const branchResult = branchRuntime.precalc();
assert.equal(branchResult.storageSummary().compact, true, 'terminal runtime MinMesh is compact');
assert.equal(branchResult.vertices.length, 18);
assert.deepEqual(Array.from(branchResult.vertices[0].position), [0.5, 0, -0.5]);
assert.deepEqual(Array.from(branchResult.vertices[9].position), [2.5, 0, -0.5]);
assert.equal(branchRuntime.operations[2].cache, branchResult, 'Add adopts its first final-consumer input');
assert.equal(branchRuntime.operations[0].cache, branchRuntime.operations[3].cache);
assert.equal(branchRuntime.operations[3].cache.released, true, 'copied fan-in storage is released');
assert.equal(branchRuntime.operations[3].cache.vertices.length, 0);

// Multiply can adopt a unique runtime input while the exported direct helper
// above remains copy-on-write.
const multiplyFixture = makeRuntimeDocument([
  { classId: 0x101, parameters: [2, 2, 2] },
  { classId: 0x12f, inputs: [0], parameters: [...identitySRT, 3, 0, 0, 0, 0, 0, 0, 0] },
]);
const multiplyRuntime = new D.Runtime(multiplyFixture.document, {
  registry: multiplyFixture.registry,
  handlers: D.handlers,
});
const runtimeMultiplied = multiplyRuntime.precalc();
assert.equal(runtimeMultiplied, multiplyRuntime.operations[0].cache);
assert.equal(runtimeMultiplied.storageSummary().compact, true);
assert.equal(runtimeMultiplied.vertices.length, 27);
assert.equal(runtimeMultiplied.faces.length, 12);

// Registration follows the exact Debris player dispatch (not editor labels).
for (const id of [0xb5, 0x101, 0x104, 0x110, 0x111, 0x113, 0x114, 0x117,
  0x120, 0x121, 0x124, 0x125, 0x128, 0x129, 0x12a, 0x12c, 0x12f,
  0x131, 0x133, 0x134]) assert.ok(D.handlers.has(id), `missing handler ${id.toString(16)}`);

const handlerGrid = D.handlers.get(0x101)({ parameters: [2, 2, 2], inputs: [], links: [], strings: [] });
assert.equal(handlerGrid.prepare().triangleCount, 8);

console.log('minmesh topology, modifiers, animation, conversion and dispatch tests passed');
