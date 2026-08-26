#!/usr/bin/env node
// Audits every authored Debris Font3D call. The production Arial/Georgia
// subset is deterministic and needs no Canvas; --canvas-module remains useful
// for explicitly auditing the unsupported-font fallback.

import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

import { parseKX } from '../src/kx.js';
import { MinMesh_Font3D, font3DResolution } from '../src/minmesh.js';
import {
  geometryTopologyStats,
  normalizePreparedGeometry,
  prepareShadowTopology,
} from '../src/renderer.js';

const argument = process.argv.slice(2).find(value => value.startsWith('--canvas-module='));
const canvasModulePath = argument?.slice('--canvas-module='.length);
if (canvasModulePath) {
  const canvasModule = await import(pathToFileURL(canvasModulePath).href);
  if (typeof canvasModule.createCanvas !== 'function') {
    throw new TypeError('the supplied Canvas module does not export createCanvas');
  }
  globalThis.OffscreenCanvas = class NodeCanvasOffscreenCanvas {
    constructor(width, height) { return canvasModule.createCanvas(width, height); }
  };
}

const bytes = new Uint8Array(await readFile(new URL('../assets/debris_party.kx', import.meta.url)));
const document = parseKX(bytes);
const calls = document.operations.filter(operation => operation.classId === 0x0133);
const megabytes = value => Number((value / 1048576).toFixed(3));
const preparedGeometryBytes = geometry => {
  const seen = new Set();
  let bytes = 0;
  for (const value of [
    geometry.positions, geometry.normals, geometry.tangents,
    geometry.uvs, geometry.uv1, geometry.colors, geometry.indices,
    geometry.boneWeights, geometry.boneIndices,
    geometry.shadowVertexMap, geometry.shadowTriangleMask,
  ]) if (ArrayBuffer.isView(value) && !seen.has(value.buffer)) {
    seen.add(value.buffer); bytes += value.byteLength;
  }
  return bytes;
};

const started = performance.now();
let sampledPeakRSS = process.memoryUsage().rss;
let totalMilliseconds = 0, totalVertices = 0, totalTriangles = 0;
let totalPreparedBytes = 0, worst = null;
const results = [];
for (const operation of calls) {
  const callStarted = performance.now();
  const [height, extrude, maxError] = operation.parameters;
  const [text, font] = operation.strings;
  const mesh = MinMesh_Font3D(height, extrude, maxError, text, font);
  const geometry = normalizePreparedGeometry(mesh);
  const diagnostics = geometryTopologyStats(geometry);
  const shadow = prepareShadowTopology(geometry);
  const milliseconds = performance.now() - callStarted;
  const preparedBytes = preparedGeometryBytes(geometry);
  const result = {
    id: operation.id, text, font, height,
    milliseconds: Number(milliseconds.toFixed(3)),
    vertices: geometry.vertexCount,
    triangles: geometry.indices.length / 3,
    preparedBytes,
    degenerateTriangles: diagnostics.degenerateTriangles,
    unexpectedWinding: diagnostics.unexpectedWinding,
    boundaryEdges: shadow.boundaryEdges,
    nonManifoldEdges: shadow.nonManifoldEdges,
    windingConflictEdges: shadow.windingConflictEdges,
    maxEdgeIncidence: shadow.maxEdgeIncidence,
  };
  results.push(result);
  totalMilliseconds += milliseconds;
  totalVertices += result.vertices;
  totalTriangles += result.triangles;
  totalPreparedBytes += preparedBytes;
  if (!worst || milliseconds > worst.milliseconds) worst = result;
  sampledPeakRSS = Math.max(sampledPeakRSS, process.memoryUsage().rss);
  mesh.releaseStorage();
}

const bad = results.filter(result => result.degenerateTriangles || result.unexpectedWinding ||
  result.boundaryEdges || result.nonManifoldEdges || result.windingConflictEdges ||
  result.maxEdgeIncidence !== 2);
const maxRSS = process.resourceUsage?.().maxRSS || 0;
const report = {
  ok: calls.length === 29 && bad.length === 0,
  calls: calls.length,
  resolution: font3DResolution(0.04998779296875),
  elapsedMilliseconds: Number((performance.now() - started).toFixed(3)),
  operatorMilliseconds: Number(totalMilliseconds.toFixed(3)),
  totalVertices,
  totalTriangles,
  totalPreparedBytes,
  totalPreparedMB: megabytes(totalPreparedBytes),
  sampledPeakRSSMB: megabytes(sampledPeakRSS),
  resourceMaxRSSMB: Number((maxRSS / 1024).toFixed(3)),
  finalMemory: Object.fromEntries(Object.entries(process.memoryUsage())
    .map(([name, value]) => [name + 'MB', megabytes(value)])),
  slowest: worst,
  bad,
  results,
};
console.log(JSON.stringify(report, null, 2));
if (!report.ok) process.exitCode = 1;
