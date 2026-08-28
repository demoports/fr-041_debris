// Direct JavaScript port of the compact GenMinMesh representation used by
// fr-041.  Faces remain polygons while operators are evaluated; prepare()
// performs the final fan triangulation into renderer-friendly typed arrays.
import { BilinearContext } from './bitmap.js';
import { KC_MINMESH } from './abi.js';
import {
  Random,
  f32,
  mat4Euler,
  mat4EulerTurns,
  mat4Identity,
  mat4Mul,
  mat4SRT,
} from './core.js';
import { FONT3D_FAMILIES } from './font3d_glyphs.js';
import { libtess } from './libtess.js';

const TAU = Math.PI * 2;
const EPSILON = 1e-20;
const MMU_ALL = 0;
const MMU_SELECTED = 1;
const MMU_UNSELECTED = 2;
const MMS_ADD = 0;
const MMS_SUB = 1;
const MMS_SET = 2;
const MMS_SETNOT = 3;
const MMS_VERTEX = 0;
const MMS_FULLFACE = 4;
const MMS_PARTFACE = 8;
const MINMESH_SELECT = Object.freeze({
  MMU_ALL,
  MMU_SELECTED,
  MMU_UNSELECTED,
  MMS_ADD,
  MMS_SUB,
  MMS_SET,
  MMS_SETNOT,
  MMS_VERTEX,
  MMS_FULLFACE,
  MMS_PARTFACE,
});
const SHADOW_POSITION_COMPONENTS = [0, 1, 2];
const SHADOW_WEIGHT_COMPONENTS = [13, 14, 15, 16];
const SHADOW_FLOAT_COMPONENTS = [0, 1, 2, 13, 14, 15, 16];

function vector3(x = 0, y = 0, z = 0) {
  return new Float32Array([f32(x), f32(y), f32(z)]);
}

function cloneVector(value) {
  return new Float32Array(value || [0, 0, 0]);
}

function copy3(out, value) {
  out[0] = f32(value[0]); out[1] = f32(value[1]); out[2] = f32(value[2]);
  return out;
}

function add3(out, value, scale = 1) {
  out[0] = f32(out[0] + value[0] * scale);
  out[1] = f32(out[1] + value[1] * scale);
  out[2] = f32(out[2] + value[2] * scale);
  return out;
}

function sub3(a, b, out = vector3()) {
  out[0] = f32(a[0] - b[0]); out[1] = f32(a[1] - b[1]); out[2] = f32(a[2] - b[2]);
  return out;
}

function scale3(out, scale) {
  out[0] = f32(out[0] * scale); out[1] = f32(out[1] * scale); out[2] = f32(out[2] * scale);
  return out;
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b, out = vector3()) {
  out[0] = f32(a[1] * b[2] - a[2] * b[1]);
  out[1] = f32(a[2] * b[0] - a[0] * b[2]);
  out[2] = f32(a[0] * b[1] - a[1] * b[0]);
  return out;
}

function normalize3(value, out = value) {
  const lengthSq = dot3(value, value);
  if (lengthSq < EPSILON) {
    out[0] = 1; out[1] = 0; out[2] = 0;
  } else {
    const inverse = 1 / Math.sqrt(lengthSq);
    out[0] = f32(value[0] * inverse); out[1] = f32(value[1] * inverse); out[2] = f32(value[2] * inverse);
  }
  return out;
}

function makeVertex(source = null) {
  if (source) {
    return {
      select: source.select | 0,
      boneCount: source.boneCount | 0,
      tempByte: source.tempByte | 0,
      mergeTag: source.mergeTag | 0,
      color: source.color >>> 0,
      position: cloneVector(source.position),
      normal: cloneVector(source.normal),
      tangent: cloneVector(source.tangent),
      uv: [new Float32Array(source.uv?.[0] || [0, 0]), new Float32Array(source.uv?.[1] || [0, 0])],
      weights: new Float32Array(source.weights || 4),
      matrices: new Uint16Array(source.matrices || 4),
    };
  }
  return {
    select: 0, boneCount: 0, tempByte: 0, mergeTag: 0, color: 0,
    position: vector3(), normal: vector3(), tangent: vector3(),
    uv: [new Float32Array(2), new Float32Array(2)],
    weights: new Float32Array(4), matrices: new Uint16Array(4),
  };
}

function makeFace(vertices = [], cluster = 1, select = 0, source = null) {
  return {
    select: source ? source.select | 0 : select | 0,
    count: vertices.length,
    cluster: source ? source.cluster | 0 : cluster | 0,
    temp: source ? source.temp | 0 : 0,
    normal: source ? cloneVector(source.normal) : vector3(),
    flags: source ? source.flags >>> 0 : 0,
    vertices: Array.from(vertices, value => value | 0),
    adjacent: source ? Array.from(source.adjacent || [], value => value | 0) : new Array(vertices.length).fill(-1),
  };
}

function cloneFace(face, vertexOffset = 0, clusterOffset = 0) {
  const result = makeFace(face.vertices.map(value => value + vertexOffset), face.cluster, face.select, face);
  if (result.cluster > 0) result.cluster += clusterOffset;
  result.count = result.vertices.length;
  result.adjacent.fill(-1);
  return result;
}

function makeCluster(material = null, renderPass = 0, id = 0, animType = 0, animMatrix = -1) {
  return { material, renderPass: renderPass | 0, id: id | 0, animType: animType | 0, animMatrix: animMatrix | 0 };
}

function cloneMatrixRecord(record) {
  if (!record) return null;
  return {
    basePose: new Float32Array(record.basePose), parent: record.parent | 0,
    factor: f32(record.factor ?? 1), spread: f32(record.spread || 0), offset: f32(record.offset || 0),
    noAnimation: new Float32Array(record.noAnimation), spline: record.spline || null,
  };
}

function compactAnimation(animation) {
  const records = animation?.matrices;
  if (!records?.length) return null;
  const count = records.length;
  const matrices = new Float32Array(count * 32);
  const parents = new Int32Array(count);
  // Every GenMinMatrix scalar is sF32 in the released source. Keeping the
  // compact representation single precision also avoids changing animation
  // semantics merely because topology was compacted for playback.
  const parameters = new Float32Array(count * 3);
  const splineKinds = new Uint8Array(count); // 0 none, 1 external, 2 kinetic
  const splines = new Array(count).fill(null);
  let hasKinetic = false;
  for (const record of records) if (record?.spline instanceof KineticSpline) { hasKinetic = true; break; }
  const kineticVectors = hasKinetic ? new Float32Array(count * 15) : null;
  const kineticAngles = hasKinetic ? new Float32Array(count) : null;
  const kineticAxes = hasKinetic ? new Uint8Array(count) : null;
  for (let index = 0; index < count; index++) {
    const record = records[index];
    matrices.set(record.basePose, index * 32);
    matrices.set(record.noAnimation, index * 32 + 16);
    parents[index] = record.parent | 0;
    parameters[index * 3] = record.factor;
    parameters[index * 3 + 1] = record.spread;
    parameters[index * 3 + 2] = record.offset;
    const spline = record.spline;
    if (spline instanceof KineticSpline) {
      splineKinds[index] = 2;
      const offset = index * 15;
      kineticVectors.set(spline.position, offset);
      kineticVectors.set(spline.speed, offset + 3);
      kineticVectors.set(spline.gravity, offset + 6);
      kineticVectors.set(spline.rotation, offset + 9);
      if (spline.axis) { kineticAxes[index] = 1; kineticVectors.set(spline.axis, offset + 12); }
      kineticAngles[index] = spline.angle;
    } else if (spline) {
      splineKinds[index] = 1;
      splines[index] = spline;
    }
  }
  return { count, matrices, parents, parameters, splineKinds, splines, kineticVectors, kineticAngles, kineticAxes };
}

function expandAnimation(storage) {
  if (!storage) return null;
  const records = new Array(storage.count);
  for (let index = 0; index < records.length; index++) {
    const matrixOffset = index * 32, parameterOffset = index * 3;
    let spline = null;
    if (storage.splineKinds[index] === 1) spline = storage.splines[index];
    else if (storage.splineKinds[index] === 2) {
      const offset = index * 15, vectors = storage.kineticVectors;
      spline = new KineticSpline(
        vectors.subarray(offset, offset + 3),
        vectors.subarray(offset + 3, offset + 6),
        vectors.subarray(offset + 6, offset + 9),
        vectors.subarray(offset + 9, offset + 12),
        storage.kineticAxes[index] ? vectors.subarray(offset + 12, offset + 15) : null,
        storage.kineticAngles[index],
      );
    }
    records[index] = {
      basePose: storage.matrices.slice(matrixOffset, matrixOffset + 16),
      parent: storage.parents[index] | 0,
      factor: storage.parameters[parameterOffset],
      spread: storage.parameters[parameterOffset + 1],
      offset: storage.parameters[parameterOffset + 2],
      noAnimation: storage.matrices.slice(matrixOffset + 16, matrixOffset + 32),
      spline,
    };
  }
  return { matrices: records };
}

function identityMatrix() {
  return mat4Identity();
}

function translationMatrix(x, y, z) {
  const matrix = identityMatrix();
  matrix[12] = f32(x); matrix[13] = f32(y); matrix[14] = f32(z);
  return matrix;
}

function transformPoint(matrix, value, out = vector3()) {
  const x = value[0], y = value[1], z = value[2];
  out[0] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]);
  out[1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]);
  out[2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]);
  return out;
}

function transformDirection(matrix, value, out = vector3()) {
  const x = value[0], y = value[1], z = value[2];
  out[0] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z);
  out[1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z);
  out[2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z);
  return out;
}

// sMatrix::TransR is the inverse used by Bend2: transpose the orthonormal
// 3x3 part, negate the translation, then rotate that translation.  Using a
// generic cofactor inverse is mathematically equivalent here but does not
// preserve the native single-precision operation order.
function rigidInverse(matrix, out = new Float32Array(16)) {
  out.fill(0);
  out[0] = matrix[0]; out[1] = matrix[4]; out[2] = matrix[8];
  out[4] = matrix[1]; out[5] = matrix[5]; out[6] = matrix[9];
  out[8] = matrix[2]; out[9] = matrix[6]; out[10] = matrix[10];
  const tx = matrix[12], ty = matrix[13], tz = matrix[14];
  out[12] = f32(-(out[0] * tx + out[4] * ty + out[8] * tz));
  out[13] = f32(-(out[1] * tx + out[5] * ty + out[9] * tz));
  out[14] = f32(-(out[2] * tx + out[6] * ty + out[10] * tz));
  out[15] = 1;
  return out;
}

function matrixMul(a, b, out) {
  return mat4Mul(a, b, out);
}

function matrixSRT(values) {
  return mat4SRT(new Float32Array(values));
}

function selected(selection, value) {
  return selection === MMU_ALL || selection === MMU_SELECTED && value || selection === MMU_UNSELECTED && !value;
}

function selectLogic(oldValue, newValue, mode) {
  switch (mode & 3) {
    case MMS_ADD: return newValue ? 1 : oldValue | 0;
    case MMS_SUB: return newValue ? 0 : oldValue | 0;
    case MMS_SET: return newValue ? 1 : 0;
    case MMS_SETNOT: return newValue ? 0 : 1;
    default: return oldValue | 0;
  }
}

function fnvFloat(hash, value) {
  const buffer = new ArrayBuffer(4), view = new DataView(buffer);
  view.setFloat32(0, value, true);
  return Math.imul(hash ^ view.getUint32(0, true), 0x01000193) >>> 0;
}

const ANIMATED_PREPARE_SLOT_LIMIT = 8;

function compactMinMeshBounds(storage) {
  const minimum = vector3(Infinity, Infinity, Infinity);
  const maximum = vector3(-Infinity, -Infinity, -Infinity);
  let found = false;
  for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
    const start = storage.faceVertexOffsets[faceIndex];
    const end = storage.faceVertexOffsets[faceIndex + 1];
    for (let corner = start; corner < end; corner++) {
      const vertexIndex = storage.faceVertices[corner];
      if (vertexIndex < 0 || vertexIndex >= storage.vertexCount) continue;
      const offset = vertexIndex * 17;
      const x = storage.vertexFloats[offset];
      const y = storage.vertexFloats[offset + 1];
      const z = storage.vertexFloats[offset + 2];
      found = true;
      minimum[0] = Math.min(minimum[0], x); maximum[0] = Math.max(maximum[0], x);
      minimum[1] = Math.min(minimum[1], y); maximum[1] = Math.max(maximum[1], y);
      minimum[2] = Math.min(minimum[2], z); maximum[2] = Math.max(maximum[2], z);
    }
  }
  if (!found) minimum.fill(0), maximum.fill(0);
  return { minimum, maximum };
}

// Everything except the three skinned T-space channels is invariant across
// animated frames. Build those renderer buffers directly from compact storage
// once, instead of expanding thousands of tiny vertex/face objects and
// rebuilding string-heavy shadow topology on every prepare(time).
function compactPreparedTemplate(mesh, storage) {
  const count = storage.vertexCount;
  const uv0 = new Float32Array(count * 2);
  const uv1 = new Float32Array(count * 2);
  const colors = new Uint8Array(count * 4);
  const boneWeights = new Float32Array(count * 4);
  const boneIndices = new Uint16Array(count * 4);
  const shadowVertexMap = new Uint32Array(count);
  const shadowVertices = new Map();
  const floatWords = new Uint32Array(storage.vertexFloats.buffer,
    storage.vertexFloats.byteOffset, storage.vertexFloats.length);
  const canonicalFloatWord = word => {
    const magnitude = word & 0x7fffffff;
    if (magnitude === 0) return 0; // String(-0) and String(+0) are both "0".
    if ((magnitude & 0x7f800000) === 0x7f800000 && (magnitude & 0x007fffff)) {
      return 0x7fc00000; // Every JavaScript NaN stringifies as the same key.
    }
    return word >>> 0;
  };
  const sameShadowVertex = (a, b) => {
    const af = a * 17, bf = b * 17, ai = a * 4, bi = b * 4;
    for (const component of SHADOW_FLOAT_COMPONENTS) {
      if (canonicalFloatWord(floatWords[af + component]) !==
          canonicalFloatWord(floatWords[bf + component])) return false;
    }
    for (let bone = 0; bone < 4; bone++) {
      if (storage.vertexMatrices[ai + bone] !== storage.vertexMatrices[bi + bone]) return false;
    }
    return true;
  };
  for (let index = 0; index < count; index++) {
    const floatOffset = index * 17;
    const boneOffset = index * 4;
    const uvOffset = index * 2;
    uv0[uvOffset] = storage.vertexFloats[floatOffset + 9];
    uv0[uvOffset + 1] = storage.vertexFloats[floatOffset + 10];
    uv1[uvOffset] = storage.vertexFloats[floatOffset + 11];
    uv1[uvOffset + 1] = storage.vertexFloats[floatOffset + 12];
    for (let bone = 0; bone < 4; bone++) {
      boneWeights[boneOffset + bone] = storage.vertexFloats[floatOffset + 13 + bone];
      boneIndices[boneOffset + bone] = storage.vertexMatrices[boneOffset + bone];
    }
    const color = storage.vertexColors[index] >>> 0;
    colors[boneOffset] = (color >>> 16) & 255;
    colors[boneOffset + 1] = (color >>> 8) & 255;
    colors[boneOffset + 2] = color & 255;
    colors[boneOffset + 3] = color >>> 24;
    let shadowHash = 0x811c9dc5;
    for (const component of SHADOW_POSITION_COMPONENTS) {
      shadowHash = Math.imul(shadowHash ^ canonicalFloatWord(floatWords[floatOffset + component]),
        0x01000193) >>> 0;
    }
    for (let bone = 0; bone < 4; bone++) {
      shadowHash = Math.imul(shadowHash ^ storage.vertexMatrices[boneOffset + bone],
        0x01000193) >>> 0;
    }
    for (const component of SHADOW_WEIGHT_COMPONENTS) {
      shadowHash = Math.imul(shadowHash ^ canonicalFloatWord(floatWords[floatOffset + component]),
        0x01000193) >>> 0;
    }
    let bucket = shadowVertices.get(shadowHash);
    let merged;
    if (bucket) {
      for (const candidate of bucket) {
        if (sameShadowVertex(candidate, index)) { merged = candidate; break; }
      }
    }
    if (merged === undefined) {
      shadowVertexMap[index] = index;
      if (bucket) bucket.push(index);
      else shadowVertices.set(shadowHash, [index]);
    } else shadowVertexMap[index] = merged;
  }

  const clusters = storage.clusters;
  const buckets = Array.from({ length: clusters.length }, () => []);
  const shadowBuckets = Array.from({ length: clusters.length }, () => []);
  for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
    const intOffset = faceIndex * 4;
    const faceCount = storage.faceInts[intOffset + 1];
    const cluster = storage.faceInts[intOffset + 2];
    if (cluster <= 0 || cluster >= buckets.length || faceCount < 3) continue;
    const start = storage.faceVertexOffsets[faceIndex];
    const available = storage.faceVertexOffsets[faceIndex + 1] - start;
    const first = available > 0 ? storage.faceVertices[start] : undefined;
    for (let corner = 2; corner < faceCount; corner++) {
      buckets[cluster].push(first,
        corner - 1 < available ? storage.faceVertices[start + corner - 1] : undefined,
        corner < available ? storage.faceVertices[start + corner] : undefined);
      shadowBuckets[cluster].push(storage.faceFlags[faceIndex] & 1 ? 0 : 1);
    }
  }
  const indexCount = buckets.reduce((sum, bucket) => sum + bucket.length, 0);
  const IndexType = count > 0xffff ? Uint32Array : Uint16Array;
  const indices = new IndexType(indexCount);
  const groups = [];
  let cursor = 0;
  for (let clusterIndex = 1; clusterIndex < buckets.length; clusterIndex++) {
    const bucket = buckets[clusterIndex];
    if (!bucket.length) continue;
    indices.set(bucket, cursor);
    const cluster = clusters[clusterIndex];
    groups.push({
      cluster: clusterIndex,
      material: cluster.material,
      renderPass: cluster.renderPass,
      start: cursor,
      count: bucket.length,
    });
    cursor += bucket.length;
  }
  const shadowTriangleMask = new Uint8Array(indexCount / 3);
  cursor = 0;
  for (let clusterIndex = 1; clusterIndex < shadowBuckets.length; clusterIndex++) {
    shadowTriangleMask.set(shadowBuckets[clusterIndex], cursor);
    cursor += shadowBuckets[clusterIndex].length;
  }
  const preparedClusters = clusters.map(cluster => makeCluster(
    cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
  ));
  return {
    kind: 'indexed-geometry', sourceKind: mesh.kind,
    uv0, uv1, uvs: uv0, colors, indices, boneWeights, boneIndices,
    groups, shadowVertexMap, shadowTriangleMask,
    materials: preparedClusters.map(cluster => cluster.material),
    clusters: preparedClusters,
    bounds: compactMinMeshBounds(storage),
    vertexCount: count, indexCount, triangleCount: indexCount / 3,
  };
}

function calculateCompactNormals(storage) {
  // Compact material branches may share these two buffers with their source.
  // Derived normals are private copy-on-write state, just like detachVertices
  // in the expanded implementation.
  const vertexFloats = storage.vertexFloats = storage.vertexFloats.slice();
  const faceNormals = storage.faceNormals = storage.faceNormals.slice();
  faceNormals.fill(0);
  for (let index = 0; index < storage.vertexCount; index++) {
    const offset = index * 17;
    vertexFloats[offset + 3] = 0; vertexFloats[offset + 4] = 0; vertexFloats[offset + 5] = 0;
    vertexFloats[offset + 6] = 0; vertexFloats[offset + 7] = 0; vertexFloats[offset + 8] = 0;
  }
  const d0 = vector3(), d1 = vector3(), normal = vector3(), tangent = vector3();
  for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
    const intOffset = faceIndex * 4;
    const count = storage.faceInts[intOffset + 1];
    const start = storage.faceVertexOffsets[faceIndex];
    const end = storage.faceVertexOffsets[faceIndex + 1];
    const available = end - start;
    if (count === 2) {
      const firstIndex = available > 0 ? storage.faceVertices[start] : -1;
      const secondIndex = available > 1 ? storage.faceVertices[start + 1] : -1;
      if (firstIndex < 0 || firstIndex >= storage.vertexCount ||
          secondIndex < 0 || secondIndex >= storage.vertexCount) continue;
      const first = firstIndex * 17, second = secondIndex * 17;
      for (let component = 3; component < 9; component++) {
        vertexFloats[first + component] = f32(
          vertexFloats[first + component] + vertexFloats[second + component]);
        vertexFloats[second + component] = vertexFloats[first + component];
      }
      continue;
    }
    // GenMinMesh::CalcNormals includes cluster 0. That cluster suppresses a
    // face at draw time, but deleted/hidden faces remain part of the authored
    // normal and tangent accumulation until topology itself is removed.
    if (count < 3 || available < 3) continue;
    const i0 = storage.faceVertices[start];
    const i1 = storage.faceVertices[start + 1];
    const i2 = storage.faceVertices[start + 2];
    if (i0 < 0 || i0 >= storage.vertexCount || i1 < 0 || i1 >= storage.vertexCount ||
        i2 < 0 || i2 >= storage.vertexCount) continue;
    const v0 = i0 * 17, v1 = i1 * 17, v2 = i2 * 17;
    d0[0] = f32(vertexFloats[v1] - vertexFloats[v0]);
    d0[1] = f32(vertexFloats[v1 + 1] - vertexFloats[v0 + 1]);
    d0[2] = f32(vertexFloats[v1 + 2] - vertexFloats[v0 + 2]);
    d1[0] = f32(vertexFloats[v2] - vertexFloats[v0]);
    d1[1] = f32(vertexFloats[v2 + 1] - vertexFloats[v0 + 1]);
    d1[2] = f32(vertexFloats[v2 + 2] - vertexFloats[v0 + 2]);
    normalize3(cross3(d0, d1, normal));
    faceNormals.set(normal, faceIndex * 3);
    const b1 = vertexFloats[v1 + 9] - vertexFloats[v0 + 9];
    const b2 = vertexFloats[v2 + 9] - vertexFloats[v0 + 9];
    const c1 = vertexFloats[v1 + 10] - vertexFloats[v0 + 10];
    const c2 = vertexFloats[v2 + 10] - vertexFloats[v0 + 10];
    const determinant = b1 * c2 - c1 * b2;
    tangent.fill(0);
    if (Math.abs(determinant) > EPSILON) {
      add3(tangent, d0, c2 / determinant);
      add3(tangent, d1, -c1 / determinant);
    }
    add3(tangent, normal, -dot3(tangent, normal));
    normalize3(tangent);
    for (let corner = start; corner < end; corner++) {
      const vertexIndex = storage.faceVertices[corner];
      if (vertexIndex < 0 || vertexIndex >= storage.vertexCount) continue;
      const offset = vertexIndex * 17;
      vertexFloats[offset + 3] = f32(vertexFloats[offset + 3] + normal[0]);
      vertexFloats[offset + 4] = f32(vertexFloats[offset + 4] + normal[1]);
      vertexFloats[offset + 5] = f32(vertexFloats[offset + 5] + normal[2]);
      vertexFloats[offset + 6] = f32(vertexFloats[offset + 6] + tangent[0]);
      vertexFloats[offset + 7] = f32(vertexFloats[offset + 7] + tangent[1]);
      vertexFloats[offset + 8] = f32(vertexFloats[offset + 8] + tangent[2]);
    }
  }
  for (let index = 0; index < storage.vertexCount; index++) {
    const offset = index * 17;
    normal[0] = vertexFloats[offset + 3]; normal[1] = vertexFloats[offset + 4]; normal[2] = vertexFloats[offset + 5];
    normalize3(normal);
    vertexFloats[offset + 3] = normal[0]; vertexFloats[offset + 4] = normal[1]; vertexFloats[offset + 5] = normal[2];
    tangent[0] = vertexFloats[offset + 6]; tangent[1] = vertexFloats[offset + 7]; tangent[2] = vertexFloats[offset + 8];
    normalize3(tangent);
    vertexFloats[offset + 6] = tangent[0]; vertexFloats[offset + 7] = tangent[1]; vertexFloats[offset + 8] = tangent[2];
  }
}

function compactKineticRecords(animation) {
  const result = new Array(animation.count).fill(null);
  for (let index = 0; index < animation.count; index++) {
    if (animation.splineKinds[index] !== 2) continue;
    const offset = index * 15;
    const axis = animation.kineticAxes[index]
      ? normalize3(animation.kineticVectors.slice(offset + 12, offset + 15)) : null;
    result[index] = {
      position: animation.kineticVectors.subarray(offset, offset + 3),
      speed: animation.kineticVectors.subarray(offset + 3, offset + 6),
      gravity: animation.kineticVectors.subarray(offset + 6, offset + 9),
      rotation: animation.kineticVectors.subarray(offset + 9, offset + 12),
      axis,
      angle: animation.kineticAngles[index],
    };
  }
  return result;
}

function evaluateCompactKinetic(record, time, out) {
  time = f32(Math.max(0, time));
  if (record.axis) {
    const radians = f32(f32(record.angle * time) * f32(TAU));
    const c = f32(Math.cos(radians)), s = f32(Math.sin(radians)), t = f32(1 - c);
    const x = record.axis[0], y = record.axis[1], z = record.axis[2];
    out[0] = f32(f32(f32(t*x)*x)+c); out[1] = f32(f32(f32(t*x)*y)+f32(s*z)); out[2] = f32(f32(f32(t*x)*z)-f32(s*y)); out[3] = 0;
    out[4] = f32(f32(f32(t*x)*y)-f32(s*z)); out[5] = f32(f32(f32(t*y)*y)+c); out[6] = f32(f32(f32(t*y)*z)+f32(s*x)); out[7] = 0;
    out[8] = f32(f32(f32(t*x)*z)+f32(s*y)); out[9] = f32(f32(f32(t*y)*z)-f32(s*x)); out[10] = f32(f32(f32(t*z)*z)+c); out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  } else {
    mat4Euler(
      f32(f32(record.rotation[0] * f32(TAU)) * time),
      f32(f32(record.rotation[1] * f32(TAU)) * time),
      f32(f32(record.rotation[2] * f32(TAU)) * time), out,
    );
  }
  const time2 = f32(time * time);
  out[12] = f32(f32(record.position[0] + f32(record.speed[0] * time)) + f32(record.gravity[0] * time2));
  out[13] = f32(f32(record.position[1] + f32(record.speed[1] * time)) + f32(record.gravity[1] * time2));
  out[14] = f32(f32(record.position[2] + f32(record.speed[2] * time)) + f32(record.gravity[2] * time2));
  return out;
}

function createAnimatedPrepareScratch(mesh, storage) {
  const animation = storage.animation;
  const matrixCount = animation.count;
  const temporaryBuffer = new Float32Array(matrixCount * 16);
  const outputBuffer = new Float32Array(matrixCount * 16);
  const localBuffer = new Float32Array(matrixCount * 16);
  const temporary = new Array(matrixCount);
  const output = new Array(matrixCount);
  const local = new Array(matrixCount);
  const basePose = new Array(matrixCount);
  const noAnimation = new Array(matrixCount);
  for (let index = 0; index < matrixCount; index++) {
    temporary[index] = temporaryBuffer.subarray(index * 16, index * 16 + 16);
    output[index] = outputBuffer.subarray(index * 16, index * 16 + 16);
    local[index] = localBuffer.subarray(index * 16, index * 16 + 16);
    basePose[index] = animation.matrices.subarray(index * 32, index * 32 + 16);
    noAnimation[index] = animation.matrices.subarray(index * 32 + 16, index * 32 + 32);
  }
  return {
    storage,
    template: compactPreparedTemplate(mesh, storage),
    temporary, output, local, basePose, noAnimation,
    kinetics: compactKineticRecords(animation),
    slots: [],
  };
}

function evaluateCompactAnimation(scratch, time = 0) {
  const animation = scratch.storage.animation;
  time = Math.max(0, time);
  const scale = animation.count > 1 ? 1 / (animation.count - 1) : 0;
  for (let index = 0; index < animation.count; index++) {
    const parameterOffset = index * 3;
    let local = scratch.noAnimation[index];
    const splineTime = time * animation.parameters[parameterOffset] +
      animation.parameters[parameterOffset + 2] +
      index * animation.parameters[parameterOffset + 1] * scale;
    const kind = animation.splineKinds[index];
    if (kind === 1) {
      const value = animation.splines[index]?.eval(splineTime, time);
      local = value?.matrix || value || local;
    } else if (kind === 2) {
      local = evaluateCompactKinetic(scratch.kinetics[index], splineTime, scratch.local[index]);
    }
    const parent = animation.parents[index];
    if (parent >= 0) matrixMul(scratch.temporary[parent], local, scratch.temporary[index]);
    else scratch.temporary[index].set(local);
    matrixMul(scratch.temporary[index], scratch.basePose[index], scratch.output[index]);
  }
  return scratch.output;
}

function createCompactGeometry(mesh, template, dynamic = false) {
  const count = template.vertexCount;
  const geometry = {
    ...template,
    positions: new Float32Array(count * 3),
    normals: new Float32Array(count * 3),
    tangents: new Float32Array(count * 4),
  };
  if (dynamic) geometry.dynamicAttributes = ['positions', 'normals', 'tangents'];
  Object.defineProperty(geometry, 'animation', { enumerable: true, get: () => mesh.animation });
  return geometry;
}

function skinCompactGeometry(storage, matrices, geometry) {
  const positions = geometry.positions;
  const normals = geometry.normals;
  const tangents = geometry.tangents;
  for (let index = 0; index < storage.vertexCount; index++) {
    const floatOffset = index * 17;
    const intOffset = index * 4;
    const positionOffset = index * 3;
    const tangentOffset = index * 4;
    const sx = storage.vertexFloats[floatOffset];
    const sy = storage.vertexFloats[floatOffset + 1];
    const sz = storage.vertexFloats[floatOffset + 2];
    const snx = storage.vertexFloats[floatOffset + 3];
    const sny = storage.vertexFloats[floatOffset + 4];
    const snz = storage.vertexFloats[floatOffset + 5];
    const stx = storage.vertexFloats[floatOffset + 6];
    const sty = storage.vertexFloats[floatOffset + 7];
    const stz = storage.vertexFloats[floatOffset + 8];
    const boneCount = storage.vertexInts[intOffset + 1];
    if (matrices && boneCount) {
      let px = 0, py = 0, pz = 0;
      let nx = 0, ny = 0, nz = 0;
      let tx = 0, ty = 0, tz = 0;
      for (let bone = 0; bone < boneCount; bone++) {
        // Expanded MinMesh has four fixed matrix/weight lanes. Values beyond
        // those lanes read as undefined and are skipped by the legacy loop.
        if (bone >= 4) continue;
        const matrix = matrices[storage.vertexMatrices[intOffset + bone]];
        if (!matrix) continue;
        const weight = storage.vertexFloats[floatOffset + 13 + bone];
        const p0 = f32(matrix[0] * sx + matrix[4] * sy + matrix[8] * sz + matrix[12]);
        const p1 = f32(matrix[1] * sx + matrix[5] * sy + matrix[9] * sz + matrix[13]);
        const p2 = f32(matrix[2] * sx + matrix[6] * sy + matrix[10] * sz + matrix[14]);
        const n0 = f32(matrix[0] * snx + matrix[4] * sny + matrix[8] * snz);
        const n1 = f32(matrix[1] * snx + matrix[5] * sny + matrix[9] * snz);
        const n2 = f32(matrix[2] * snx + matrix[6] * sny + matrix[10] * snz);
        const t0 = f32(matrix[0] * stx + matrix[4] * sty + matrix[8] * stz);
        const t1 = f32(matrix[1] * stx + matrix[5] * sty + matrix[9] * stz);
        const t2 = f32(matrix[2] * stx + matrix[6] * sty + matrix[10] * stz);
        px = f32(px + p0 * weight); py = f32(py + p1 * weight); pz = f32(pz + p2 * weight);
        nx = f32(nx + n0 * weight); ny = f32(ny + n1 * weight); nz = f32(nz + n2 * weight);
        tx = f32(tx + t0 * weight); ty = f32(ty + t1 * weight); tz = f32(tz + t2 * weight);
      }
      // FullBoneInner deliberately leaves the weighted normal and tangent
      // raw. The animated vertex-shader path performs its one normalization
      // and tangent orthogonalization after skinning.
      positions[positionOffset] = px; positions[positionOffset + 1] = py; positions[positionOffset + 2] = pz;
      normals[positionOffset] = nx; normals[positionOffset + 1] = ny; normals[positionOffset + 2] = nz;
      tangents[tangentOffset] = tx; tangents[tangentOffset + 1] = ty; tangents[tangentOffset + 2] = tz;
    } else {
      positions[positionOffset] = sx; positions[positionOffset + 1] = sy; positions[positionOffset + 2] = sz;
      normals[positionOffset] = snx; normals[positionOffset + 1] = sny; normals[positionOffset + 2] = snz;
      tangents[tangentOffset] = stx; tangents[tangentOffset + 1] = sty; tangents[tangentOffset + 2] = stz;
    }
    tangents[tangentOffset + 3] = 1;
  }
  return geometry;
}

class MinMesh {
  constructor() {
    this.kind = 'minmesh';
    this.classId = KC_MINMESH;
    this.outputClass = 'KC_MINMESH';
    this._compact = null;
    this._vertices = [];
    this._faces = [];
    this._clusters = [makeCluster(null), makeCluster(null)];
    this._animation = null;
    this.completelyRigid = false;
    this.stripped = false;
    this.normalsValid = false;
    this._prepared = null;
    this._sharedVertices = false;
    this.released = false;
    this.topologyReleasedForPlayback = false;
    Object.defineProperty(this, '_animatedPrepareScratch', {
      configurable: true, writable: true, value: null,
    });
  }

  get vertices() { this.ensureExpanded(); return this._vertices; }
  set vertices(value) { if (this._compact) this.ensureExpanded(); this._vertices = value; }
  get faces() { this.ensureExpanded(); return this._faces; }
  set faces(value) { if (this._compact) this.ensureExpanded(); this._faces = value; }
  get clusters() { this.ensureExpanded(); return this._clusters; }
  set clusters(value) { if (this._compact) this.ensureExpanded(); this._clusters = value; }
  get animation() { this.ensureExpanded(); return this._animation; }
  set animation(value) { if (this._compact) this.ensureExpanded(); this._animation = value; }
  get Vertices() { return this.vertices; }
  set Vertices(value) { this.vertices = value; }
  get Faces() { return this.faces; }
  set Faces(value) { this.faces = value; }
  get Clusters() { return this.clusters; }
  set Clusters(value) { this.clusters = value; }

  invalidate() {
    if (this.topologyReleasedForPlayback) {
      throw new Error('cannot mutate MinMesh after playback topology release');
    }
    this.normalsValid = false;
    this._prepared = null;
    this._animatedPrepareScratch = null;
    this.released = false;
    return this;
  }

  clone() {
    const result = new MinMesh();
    result.vertices = this.vertices.map(makeVertex);
    result.faces = this.faces.map(face => cloneFace(face));
    result.clusters = this.clusters.map(cluster => makeCluster(
      cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
    ));
    result.animation = this.animation ? { matrices: this.animation.matrices.map(cloneMatrixRecord) } : null;
    result.completelyRigid = this.completelyRigid;
    result.stripped = this.stripped;
    result.normalsValid = this.normalsValid;
    return result;
  }

  // Material linking changes only face cluster ids and the cluster table.
  // Sharing immutable vertex records here mirrors GenMinMesh's native
  // refcount/COW behavior and avoids copying large tessellated inputs merely
  // to attach a material. Geometry-changing instance methods detach first.
  cloneForMaterial() {
    const result = new MinMesh();
    const storage = this._compact;
    if (storage) {
      // A material link changes only the cluster table and the cluster/select
      // words in faceInts. Keep the immutable vertex, face-topology and
      // animation buffers shared just as the expanded path below shares the
      // vertex records. Without this path every material branch serialized a
      // complete duplicate of a dormant tessellated mesh.
      result._compact = {
        ...storage,
        faceInts: storage.faceInts.slice(),
        clusters: storage.clusters.map(cluster => makeCluster(
          cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
        )),
      };
      result.completelyRigid = this.completelyRigid;
      result.stripped = this.stripped;
      result.normalsValid = this.normalsValid;
      return result;
    }
    result.vertices = this.vertices;
    result.faces = this.faces.map(face => cloneFace(face));
    result.clusters = this.clusters.map(cluster => makeCluster(
      cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
    ));
    // Vertex topology is the expensive immutable part. Animation records are
    // small and later BoneTrain operators mutate them, so keep those private.
    result.animation = this.animation ? { matrices: this.animation.matrices.map(cloneMatrixRecord) } : null;
    result.completelyRigid = this.completelyRigid;
    result.stripped = this.stripped;
    result.normalsValid = this.normalsValid;
    this._sharedVertices = true;
    result._sharedVertices = true;
    return result;
  }

  detachVertices() {
    if (!this._sharedVertices) return this;
    this.vertices = this.vertices.map(makeVertex);
    this._sharedVertices = false;
    return this;
  }

  compact() {
    if (this._compact || this.released || this.topologyReleasedForPlayback) return this;
    const vertices = this._vertices, faces = this._faces;
    const vertexFloats = new Float32Array(vertices.length * 17);
    const vertexInts = new Int32Array(vertices.length * 4);
    const vertexColors = new Uint32Array(vertices.length);
    const vertexMatrices = new Uint16Array(vertices.length * 4);
    for (let index = 0; index < vertices.length; index++) {
      const vertex = vertices[index], floatOffset = index * 17, intOffset = index * 4;
      vertexFloats.set(vertex.position, floatOffset);
      vertexFloats.set(vertex.normal, floatOffset + 3);
      vertexFloats.set(vertex.tangent, floatOffset + 6);
      vertexFloats.set(vertex.uv[0], floatOffset + 9);
      vertexFloats.set(vertex.uv[1], floatOffset + 11);
      vertexFloats.set(vertex.weights, floatOffset + 13);
      vertexInts[intOffset] = vertex.select | 0;
      vertexInts[intOffset + 1] = vertex.boneCount | 0;
      vertexInts[intOffset + 2] = vertex.tempByte | 0;
      vertexInts[intOffset + 3] = vertex.mergeTag | 0;
      vertexColors[index] = vertex.color >>> 0;
      vertexMatrices.set(vertex.matrices, intOffset);
    }
    let vertexCornerCount = 0, adjacentCornerCount = 0;
    for (const face of faces) {
      vertexCornerCount += face.vertices.length;
      adjacentCornerCount += face.adjacent.length;
    }
    const faceInts = new Int32Array(faces.length * 4);
    const faceFlags = new Uint32Array(faces.length);
    const faceNormals = new Float32Array(faces.length * 3);
    const faceVertexOffsets = new Uint32Array(faces.length + 1);
    const faceAdjacentOffsets = new Uint32Array(faces.length + 1);
    const faceVertices = new Int32Array(vertexCornerCount);
    const faceAdjacent = new Int32Array(adjacentCornerCount);
    let vertexCursor = 0, adjacentCursor = 0;
    for (let index = 0; index < faces.length; index++) {
      const face = faces[index], intOffset = index * 4;
      faceInts[intOffset] = face.select | 0;
      faceInts[intOffset + 1] = face.count | 0;
      faceInts[intOffset + 2] = face.cluster | 0;
      faceInts[intOffset + 3] = face.temp | 0;
      faceFlags[index] = face.flags >>> 0;
      faceNormals.set(face.normal, index * 3);
      faceVertexOffsets[index] = vertexCursor;
      faceVertices.set(face.vertices, vertexCursor);
      vertexCursor += face.vertices.length;
      faceAdjacentOffsets[index] = adjacentCursor;
      faceAdjacent.set(face.adjacent, adjacentCursor);
      adjacentCursor += face.adjacent.length;
    }
    faceVertexOffsets[faces.length] = vertexCursor;
    faceAdjacentOffsets[faces.length] = adjacentCursor;
    const clusters = this._clusters.map(cluster => makeCluster(
      cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
    ));
    this._compact = {
      vertexCount: vertices.length, vertexFloats, vertexInts, vertexColors, vertexMatrices,
      faceCount: faces.length, faceInts, faceFlags, faceNormals,
      faceVertexOffsets, faceAdjacentOffsets, faceVertices, faceAdjacent,
      clusters, animation: compactAnimation(this._animation),
    };
    this._vertices = [];
    this._faces = [];
    this._clusters = [];
    this._animation = null;
    this._sharedVertices = false;
    this._animatedPrepareScratch = null;
    return this;
  }

  ensureExpanded() {
    if (this.topologyReleasedForPlayback) {
      throw new Error('MinMesh topology was released for immutable playback');
    }
    const storage = this._compact;
    if (!storage) return this;
    this._animatedPrepareScratch = null;
    this._compact = null;
    const vertices = new Array(storage.vertexCount);
    for (let index = 0; index < vertices.length; index++) {
      const vertex = makeVertex(), floatOffset = index * 17, intOffset = index * 4;
      vertex.position.set(storage.vertexFloats.subarray(floatOffset, floatOffset + 3));
      vertex.normal.set(storage.vertexFloats.subarray(floatOffset + 3, floatOffset + 6));
      vertex.tangent.set(storage.vertexFloats.subarray(floatOffset + 6, floatOffset + 9));
      vertex.uv[0].set(storage.vertexFloats.subarray(floatOffset + 9, floatOffset + 11));
      vertex.uv[1].set(storage.vertexFloats.subarray(floatOffset + 11, floatOffset + 13));
      vertex.weights.set(storage.vertexFloats.subarray(floatOffset + 13, floatOffset + 17));
      vertex.select = storage.vertexInts[intOffset];
      vertex.boneCount = storage.vertexInts[intOffset + 1];
      vertex.tempByte = storage.vertexInts[intOffset + 2];
      vertex.mergeTag = storage.vertexInts[intOffset + 3];
      vertex.color = storage.vertexColors[index] >>> 0;
      vertex.matrices.set(storage.vertexMatrices.subarray(intOffset, intOffset + 4));
      vertices[index] = vertex;
    }
    const faces = new Array(storage.faceCount);
    for (let index = 0; index < faces.length; index++) {
      const intOffset = index * 4;
      const vertexStart = storage.faceVertexOffsets[index], vertexEnd = storage.faceVertexOffsets[index + 1];
      const adjacentStart = storage.faceAdjacentOffsets[index], adjacentEnd = storage.faceAdjacentOffsets[index + 1];
      const face = makeFace(storage.faceVertices.subarray(vertexStart, vertexEnd),
        storage.faceInts[intOffset + 2], storage.faceInts[intOffset]);
      face.count = storage.faceInts[intOffset + 1];
      face.temp = storage.faceInts[intOffset + 3];
      face.flags = storage.faceFlags[index] >>> 0;
      face.normal.set(storage.faceNormals.subarray(index * 3, index * 3 + 3));
      face.adjacent = Array.from(storage.faceAdjacent.subarray(adjacentStart, adjacentEnd));
      faces[index] = face;
    }
    this._vertices = vertices;
    this._faces = faces;
    this._clusters = storage.clusters.map(cluster => makeCluster(
      cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
    ));
    this._animation = expandAnimation(storage.animation);
    return this;
  }

  storageSummary() {
    const storage = this._compact;
    if (this.topologyReleasedForPlayback) {
      return {
        compact: false, released: false, preparedOnly: true,
        vertices: this._prepared?.vertexCount || 0,
        faces: this._prepared?.triangleCount || 0,
        clusters: this._prepared?.clusters?.length || 0,
        bones: 0,
        compactBytes: 0,
      };
    }
    let compactBytes = 0;
    if (storage) {
      const visit = value => {
        if (ArrayBuffer.isView(value)) compactBytes += value.byteLength;
        else if (value && typeof value === 'object' && !Array.isArray(value)) {
          for (const nested of Object.values(value)) visit(nested);
        }
      };
      visit(storage);
    }
    return {
      compact: !!storage, released: !!this.released,
      vertices: storage ? storage.vertexCount : this._vertices.length,
      faces: storage ? storage.faceCount : this._faces.length,
      clusters: storage ? storage.clusters.length : this._clusters.length,
      bones: storage?.animation?.count || this._animation?.matrices?.length || 0,
      compactBytes,
    };
  }

  hasAnimation() {
    return Boolean(
      (this._compact?.animation?.count || 0) ||
      (this._animation?.matrices?.length || 0),
    );
  }

  // Runtime operator caches outlive native-style argument references. A
  // released shell remains inspectable through Runtime.operations, but its
  // topology storage is dropped once the final graph consumer has finished.
  // Reassigning (rather than clearing) arrays is important for material-only
  // branches that deliberately share the vertex array.
  releaseStorage() {
    this._compact = null;
    this._vertices = [];
    this._faces = [];
    this._clusters = [];
    this._animation = null;
    this._prepared = null;
    this._animatedPrepareScratch = null;
    this._sharedVertices = false;
    this.normalsValid = false;
    this.completelyRigid = false;
    this.released = true;
    this.topologyReleasedForPlayback = false;
  }

  releaseTopologyForPlayback() {
    if (!this._prepared || this.hasAnimation()) {
      throw new Error('only prepared static MinMesh topology can be released for playback');
    }
    const deferredRelease = this._playbackTopologyRelease || null;
    this._playbackTopologyRelease = null;
    this._compact = null;
    this._vertices = [];
    this._faces = [];
    this._clusters = [];
    this._animation = null;
    this._sharedVertices = false;
    this._animatedPrepareScratch = null;
    this.topologyReleasedForPlayback = true;
    this.released = false;
    deferredRelease?.release?.(this._prepared);
    return this;
  }

  copy() { return this.clone(); }

  addCluster(material, renderPass = 0, id = 0, animType = 0, animMatrix = -1) {
    for (let index = 1; index < this.clusters.length; index++) {
      const cluster = this.clusters[index];
      if (cluster.material === material && cluster.id === (id | 0) &&
        cluster.animType === (animType | 0) && cluster.animMatrix === (animMatrix | 0)) return index;
    }
    this.clusters.push(makeCluster(material, renderPass, id, animType, animMatrix));
    return this.clusters.length - 1;
  }

  makeGrid(tx, ty, flags = 0) {
    this.detachVertices();
    // Cylinder caps intentionally call MakeGrid(tx,0,flags). The C++ loop
    // still emits the one rim row plus its optional center vertex.
    tx = Math.max(1, tx | 0); ty = Math.max(0, ty | 0); flags |= 0;
    let bits = 0;
    if (flags & 1) bits++;
    if (flags & 2) bits++;
    const vertexStart = this.vertices.length;
    const regularCount = (tx + 1) * (ty + 1);
    const cluster = flags & 32 ? 0 : 1;
    for (let y = 0; y <= ty; y++) for (let x = 0; x <= tx; x++) {
      const vertex = makeVertex();
      // MakeGrid(tx,0) is intentional for one-ring cylinder caps. Native
      // floating division leaves UV1.v as NaN there; the cap operator replaces
      // position and UV0 but preserves that otherwise dormant channel.
      const yf = y / ty;
      vertex.position[0] = f32(x / tx - 0.5); vertex.position[1] = f32(yf - 0.5);
      vertex.uv[0][0] = f32(x / tx); vertex.uv[0][1] = f32(1 - yf);
      vertex.uv[1][0] = f32((x === tx ? 0 : x) / tx);
      vertex.uv[1][1] = f32(1 - (y === ty ? 0 : y) / ty);
      vertex.select = 1;
      this.vertices.push(vertex);
    }
    for (let i = 0; i < bits; i++) this.vertices.push(makeVertex());

    for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
      const indices = [
        vertexStart + y * (tx + 1) + x,
        vertexStart + (y + 1) * (tx + 1) + x,
        vertexStart + (y + 1) * (tx + 1) + x + 1,
        vertexStart + y * (tx + 1) + x + 1,
      ];
      if (flags & 16) { [indices[0], indices[3]] = [indices[3], indices[0]]; [indices[1], indices[2]] = [indices[2], indices[1]]; }
      this.faces.push(makeFace(indices, cluster, 1));
    }

    let center = vertexStart + regularCount;
    let border = vertexStart;
    for (let side = 0; side < 2; side++) {
      if (flags & (side + 1)) {
        const centerVertex = this.vertices[center];
        centerVertex.uv[0][0] = 1; centerVertex.uv[0][1] = 1 - side;
        for (let x = 0; x < tx; x++) {
          this.faces.push(makeFace([center, border + side, border + 1 - side], cluster, 1));
          border++;
        }
      }
      border = vertexStart + ty * (tx + 1);
      if (flags & 1) center++;
    }
    if (flags & 4) for (let x = 0; x <= tx; x++) {
      this.faces.push(makeFace([vertexStart + x, vertexStart + x + ty * (tx + 1)], 0, 0));
    }
    if (flags & 8) for (let y = 0; y <= ty; y++) {
      this.faces.push(makeFace([vertexStart + y * (tx + 1), vertexStart + y * (tx + 1) + tx], 0, 0));
    }
    this.invalidate();
    return vertexStart;
  }

  transform(selection, matrix, source = 0, destination = 0, operation = 0) {
    this.detachVertices();
    selection |= 0; source |= 0; destination |= 0; operation |= 0;
    const temporary = vector3();
    for (const vertex of this.vertices) {
      if (!selected(selection, vertex.select)) continue;
      if (source === 0) copy3(temporary, vertex.position);
      else {
        const uv = vertex.uv[source - 1];
        if (!uv) continue;
        temporary[0] = uv[0]; temporary[1] = uv[1]; temporary[2] = 0;
      }
      transformPoint(matrix, temporary, temporary);
      if (operation === 1) normalize3(temporary);
      else if (operation === 2) {
        const x = temporary[0], y = temporary[1], z = temporary[2];
        temporary[0] = f32(Math.atan2(x, z) / TAU + 0.5);
        temporary[1] = f32(0.5 - Math.atan2(y, Math.sqrt(x * x + z * z)) / Math.PI);
        temporary[2] = 0;
      }
      if (destination === 0) copy3(vertex.position, temporary);
      else if (vertex.uv[destination - 1]) {
        vertex.uv[destination - 1][0] = temporary[0]; vertex.uv[destination - 1][1] = temporary[1];
      }
    }
    return this.invalidate();
  }

  calcNormals() {
    if (this.normalsValid) return this;
    if (this._compact) {
      calculateCompactNormals(this._compact);
      this._animatedPrepareScratch = null;
      this._prepared = null;
      this.normalsValid = true;
      return this;
    }
    // Face deletion/triangulation branches can share vertices with a prior
    // material-only cache while requiring different derived normals.
    this.detachVertices();
    for (const vertex of this.vertices) { vertex.normal.fill(0); vertex.tangent.fill(0); }
    for (const face of this.faces) {
      face.normal.fill(0);
      if (face.count === 2) {
        const first = this.vertices[face.vertices[0]], second = this.vertices[face.vertices[1]];
        if (first && second) { add3(first.normal, second.normal); copy3(second.normal, first.normal); add3(first.tangent, second.tangent); copy3(second.tangent, first.tangent); }
      } else if (face.count >= 3) {
        const v0 = this.vertices[face.vertices[0]], v1 = this.vertices[face.vertices[1]], v2 = this.vertices[face.vertices[2]];
        if (!v0 || !v1 || !v2) continue;
        const d0 = sub3(v1.position, v0.position), d1 = sub3(v2.position, v0.position);
        const normal = normalize3(cross3(d0, d1));
        copy3(face.normal, normal);
        const b1 = v1.uv[0][0] - v0.uv[0][0], b2 = v2.uv[0][0] - v0.uv[0][0];
        const c1 = v1.uv[0][1] - v0.uv[0][1], c2 = v2.uv[0][1] - v0.uv[0][1];
        const determinant = b1 * c2 - c1 * b2;
        const tangent = vector3();
        if (Math.abs(determinant) > EPSILON) {
          add3(tangent, d0, c2 / determinant); add3(tangent, d1, -c1 / determinant);
        }
        add3(tangent, normal, -dot3(tangent, normal)); normalize3(tangent);
        for (const index of face.vertices) {
          const vertex = this.vertices[index]; if (!vertex) continue;
          add3(vertex.normal, normal); add3(vertex.tangent, tangent);
        }
      }
    }
    for (const vertex of this.vertices) { normalize3(vertex.normal); normalize3(vertex.tangent); }
    this.normalsValid = true;
    return this;
  }

  add(other) {
    if (!(other instanceof MinMesh)) return this;
    this.detachVertices();
    const vertexOffset = this.vertices.length;
    const clusterOffset = this.clusters.length - 1;
    this.faces.push(...other.faces.map(face => cloneFace(face, vertexOffset, clusterOffset)));
    this.vertices.push(...other.vertices.map(makeVertex));
    for (let index = 1; index < other.clusters.length; index++) {
      const cluster = other.clusters[index];
      this.clusters.push(makeCluster(cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix));
    }
    return this.invalidate();
  }

  mergeClusters() {
    const remap = new Int32Array(this.clusters.length).fill(-1);
    remap[0] = 0;
    for (const face of this.faces) if (face.cluster >= 0 && face.cluster < remap.length) remap[face.cluster] = 0;
    // Cluster zero is a deletion sentinel even when the browser's default
    // material is represented by null. It must never merge with a live slot.
    const zero = this.clusters[0] || makeCluster(null);
    const result = [makeCluster(zero.material, zero.renderPass, zero.id, zero.animType, zero.animMatrix)];
    for (let index = 1; index < this.clusters.length; index++) {
      if (remap[index] < 0) continue;
      const cluster = this.clusters[index];
      let destination = result.findIndex((value, resultIndex) => resultIndex > 0 && value.material === cluster.material &&
        value.renderPass === cluster.renderPass && value.id === cluster.id &&
        value.animType === cluster.animType && value.animMatrix === cluster.animMatrix);
      if (destination < 0) { destination = result.length; result.push(makeCluster(cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix)); }
      remap[index] = destination;
    }
    for (const face of this.faces) face.cluster = remap[face.cluster] >= 0 ? remap[face.cluster] : 0;
    this.clusters = result;
    this.Clusters = this.clusters;
    return this.invalidate();
  }

  invert() {
    for (const face of this.faces) face.vertices.reverse();
    return this.invalidate();
  }

  selectAll(mode) {
    mode >>>= 0;
    if (mode & 2) this.detachVertices();
    if (mode & 2) for (const vertex of this.vertices) vertex.select = mode & 1 ? 1 : 0;
    if (mode & 4) for (const face of this.faces) face.select = mode & 1 ? 1 : 0;
    return this.invalidate();
  }

  bounds() {
    const minimum = vector3(Infinity, Infinity, Infinity), maximum = vector3(-Infinity, -Infinity, -Infinity);
    let found = false;
    for (const face of this.faces) for (const index of face.vertices) {
      const position = this.vertices[index]?.position; if (!position) continue;
      found = true;
      for (let axis = 0; axis < 3; axis++) { minimum[axis] = Math.min(minimum[axis], position[axis]); maximum[axis] = Math.max(maximum[axis], position[axis]); }
    }
    if (!found) minimum.fill(0), maximum.fill(0);
    return { minimum, maximum };
  }

  triangulate() {
    const result = [];
    for (const face of this.faces) {
      if (face.count <= 3) result.push(cloneFace(face));
      else for (let index = 2; index < face.count; index++) {
        result.push(makeFace([face.vertices[0], face.vertices[index - 1], face.vertices[index]], face.cluster, face.select, face));
      }
    }
    this.faces = result; this.Faces = result;
    return this.invalidate();
  }

  evaluateAnimation(time = 0, metamorph = 0) {
    void metamorph;
    if (!this.animation?.matrices?.length) return [];
    time = Math.max(0, time);
    const records = this.animation.matrices, temporary = new Array(records.length), output = new Array(records.length);
    const scale = records.length > 1 ? 1 / (records.length - 1) : 0;
    for (let index = 0; index < records.length; index++) {
      const record = records[index];
      let local = record.noAnimation;
      if (record.spline) {
        const value = record.spline.eval(time * record.factor + record.offset + index * record.spread * scale, time);
        local = value?.matrix || value || record.noAnimation;
      }
      temporary[index] = record.parent >= 0 ? matrixMul(temporary[record.parent], local) : new Float32Array(local);
      output[index] = matrixMul(temporary[index], record.basePose);
    }
    return output;
  }

  prepareAnimated(options) {
    // Normal/tangent derivation is the only preparation step which still
    // needs the editable object representation. It runs at most once after a
    // mutation; all subsequent frames remain entirely in compact storage.
    if (!this._compact || !this.normalsValid) {
      this.ensureExpanded();
      this.calcNormals();
      this.compact();
    }
    const storage = this._compact;
    if (!storage?.animation?.count) return null;
    let scratch = this._animatedPrepareScratch;
    if (!scratch || scratch.storage !== storage) {
      scratch = createAnimatedPrepareScratch(this, storage);
      this._animatedPrepareScratch = scratch;
    }
    const slotIndex = Number.isInteger(options.animationSlot) && options.animationSlot >= 0 &&
      options.animationSlot < ANIMATED_PREPARE_SLOT_LIMIT ? options.animationSlot : -1;
    let geometry;
    if (slotIndex >= 0) {
      geometry = scratch.slots[slotIndex];
      if (!geometry) scratch.slots[slotIndex] = geometry = createCompactGeometry(this, scratch.template, true);
    } else geometry = createCompactGeometry(this, scratch.template, true);
    const matrices = evaluateCompactAnimation(scratch, options.time);
    return skinCompactGeometry(storage, matrices, geometry);
  }

  prepare(options = {}) {
    if (typeof options === 'number') options = { time: options };
    const releaseTopology = Boolean(options.releaseTopology || this._playbackTopologyRelease);
    if (options.time === undefined && this._prepared) {
      if (releaseTopology) this.releaseTopologyForPlayback();
      return this._prepared;
    }
    if (options.time !== undefined && this.hasAnimation()) {
      return this.prepareAnimated(options);
    }
    if (this._prepared) {
      if (releaseTopology && !this.topologyReleasedForPlayback) this.releaseTopologyForPlayback();
      return this._prepared;
    }
    // The final operator graph is already compact. Derive normals in that flat
    // representation and compile renderer buffers directly, avoiding a large
    // cold-transition object graph which would immediately be compacted again.
    this.calcNormals();
    if (!this._compact) this.compact();
    if (this._compact) {
      const template = compactPreparedTemplate(this, this._compact);
      const geometry = skinCompactGeometry(
        this._compact, null, createCompactGeometry(this, template, false));
      this._prepared = geometry;
      if (releaseTopology) this.releaseTopologyForPlayback();
      return geometry;
    }
    const animation = this.animation;
    const animated = options.time !== undefined && animation?.matrices?.length;
    if (!animated && this._prepared) return this._prepared;
    this.calcNormals();
    const matrices = animated ? this.evaluateAnimation(options.time, options.metamorph || 0) : null;
    const count = this.vertices.length;
    const positions = new Float32Array(count * 3), normals = new Float32Array(count * 3), tangents = new Float32Array(count * 4);
    const uv0 = new Float32Array(count * 2), uv1 = new Float32Array(count * 2), colors = new Uint8Array(count * 4);
    const boneWeights = new Float32Array(count * 4), boneIndices = new Uint16Array(count * 4);
    // GenMinMesh::CalcMergeVerts builds shadow adjacency from vertices with
    // byte-identical source position, matrix indices and weights. Render
    // vertices at cube/UV seams therefore share one shadow-topology vertex,
    // while coincident skinned vertices with different deformation do not.
    const shadowVertexMap = new Uint32Array(count);
    const shadowVertices = new Map();
    const cleanShadowValue = value => Object.is(value, -0) ? 0 : value;
    const position = vector3(), normal = vector3(), tangent = vector3();
    for (let index = 0; index < count; index++) {
      const vertex = this.vertices[index];
      const p = vertex.position, m = vertex.matrices, w = vertex.weights;
      const shadowKey = `${cleanShadowValue(p[0])},${cleanShadowValue(p[1])},${cleanShadowValue(p[2])}|` +
        `${m[0]},${m[1]},${m[2]},${m[3]}|${cleanShadowValue(w[0])},${cleanShadowValue(w[1])},` +
        `${cleanShadowValue(w[2])},${cleanShadowValue(w[3])}`;
      const merged = shadowVertices.get(shadowKey);
      if (merged === undefined) {
        shadowVertexMap[index] = index;
        shadowVertices.set(shadowKey, index);
      } else shadowVertexMap[index] = merged;
      if (matrices && vertex.boneCount) {
        position.fill(0); normal.fill(0); tangent.fill(0);
        for (let bone = 0; bone < vertex.boneCount; bone++) {
          const matrix = matrices[vertex.matrices[bone]]; if (!matrix) continue;
          add3(position, transformPoint(matrix, vertex.position, vector3()), vertex.weights[bone]);
          add3(normal, transformDirection(matrix, vertex.normal, vector3()), vertex.weights[bone]);
          add3(tangent, transformDirection(matrix, vertex.tangent, vector3()), vertex.weights[bone]);
        }
        // Match FullBoneInner: preserve the raw weighted T-space channels.
        // The renderer's animated vertex shader is their sole conditioning
        // stage, after the complete weighted transform has been evaluated.
      } else { copy3(position, vertex.position); copy3(normal, vertex.normal); copy3(tangent, vertex.tangent); }
      positions.set(position, index * 3); normals.set(normal, index * 3); tangents.set(tangent, index * 4); tangents[index * 4 + 3] = 1;
      uv0.set(vertex.uv[0], index * 2); uv1.set(vertex.uv[1], index * 2);
      const color = vertex.color >>> 0, colorOffset = index * 4;
      colors[colorOffset] = (color >>> 16) & 255; colors[colorOffset + 1] = (color >>> 8) & 255;
      colors[colorOffset + 2] = color & 255; colors[colorOffset + 3] = color >>> 24;
      boneWeights.set(vertex.weights, index * 4); boneIndices.set(vertex.matrices, index * 4);
    }
    const clusters = this.clusters;
    const buckets = Array.from({ length: clusters.length }, () => []);
    const shadowBuckets = Array.from({ length: clusters.length }, () => []);
    for (const face of this.faces) {
      if (face.cluster <= 0 || face.cluster >= buckets.length || face.count < 3) continue;
      for (let index = 2; index < face.count; index++) {
        buckets[face.cluster].push(face.vertices[0], face.vertices[index - 1], face.vertices[index]);
        shadowBuckets[face.cluster].push(face.flags & 1 ? 0 : 1);
      }
    }
    const indexCount = buckets.reduce((sum, bucket) => sum + bucket.length, 0);
    const IndexType = count > 0xffff ? Uint32Array : Uint16Array;
    const indices = new IndexType(indexCount), groups = [];
    let cursor = 0;
    for (let clusterIndex = 1; clusterIndex < buckets.length; clusterIndex++) {
      const bucket = buckets[clusterIndex]; if (!bucket.length) continue;
      indices.set(bucket, cursor);
      const cluster = clusters[clusterIndex];
      groups.push({ cluster: clusterIndex, material: cluster.material, renderPass: cluster.renderPass, start: cursor, count: bucket.length });
      cursor += bucket.length;
    }
    const shadowTriangleMask = new Uint8Array(indexCount / 3);
    cursor = 0;
    for (let clusterIndex = 1; clusterIndex < shadowBuckets.length; clusterIndex++) {
      shadowTriangleMask.set(shadowBuckets[clusterIndex], cursor);
      cursor += shadowBuckets[clusterIndex].length;
    }
    const bounds = this.bounds();
    const preparedClusters = clusters.map(cluster => makeCluster(
      cluster.material, cluster.renderPass, cluster.id, cluster.animType, cluster.animMatrix,
    ));
    const geometry = {
      kind: 'indexed-geometry', sourceKind: this.kind,
      positions, normals, tangents, uv0, uv1, uvs: uv0, colors, indices,
      boneWeights, boneIndices, groups, shadowVertexMap, shadowTriangleMask,
      materials: preparedClusters.map(cluster => cluster.material), clusters: preparedClusters,
      bounds, vertexCount: count, indexCount, triangleCount: indexCount / 3,
    };
    // Preserve the public field without pinning the expanded animation object
    // graph inside renderer geometry. Consumers that inspect it expand lazily.
    Object.defineProperty(geometry, 'animation', { enumerable: true, get: () => this.animation });
    if (!animated) this._prepared = geometry;
    if (!animated && releaseTopology) this.releaseTopologyForPlayback();
    else this.compact();
    return geometry;
  }

  summary() {
    let hash = 0x811c9dc5;
    for (const vertex of this.vertices) for (const value of vertex.position) hash = fnvFloat(hash, value);
    let triangles = 0, liveFaces = 0;
    for (const face of this.faces) if (face.cluster && face.count >= 3) { liveFaces++; triangles += face.count - 2; }
    const bounds = this.bounds();
    return {
      kind: this.kind, vertices: this.vertices.length, faces: this.faces.length, liveFaces,
      triangles, clusters: this.clusters.length, materialSlots: this.clusters.slice(1).map(cluster => cluster.material?.kind || null),
      animated: Boolean(this.animation), bones: this.animation?.matrices?.length || 0,
      bounds: { minimum: Array.from(bounds.minimum), maximum: Array.from(bounds.maximum) }, hash: hash >>> 0,
    };
  }
}

function requireMinMesh(value, name = 'mesh') {
  if (!(value instanceof MinMesh)) throw new TypeError(`${name} is not a MinMesh`);
  return value;
}

function writableMinMesh(value, owned = false, mutatesVertices = false) {
  const source = requireMinMesh(value);
  const mesh = owned ? source : source.clone();
  if (owned && mutatesVertices) mesh.detachVertices();
  mesh.released = false;
  return mesh;
}

function MinMesh_Grid(mode, tx, ty) {
  const mesh = new MinMesh();
  mesh.makeGrid(tx, ty, 0);
  if ((mode | 0) & 1) mesh.makeGrid(tx, ty, 16);
  const matrix = identityMatrix();
  matrix[0] = -1; matrix[4] = 0; matrix[5] = 0; matrix[6] = 1;
  mesh.transform(MMU_ALL, matrix);
  mesh.selectAll(2 | 4 | 1);
  return mesh;
}

function MinMesh_Cube(tx, ty, tz, flags = 0, ...srtValues) {
  tx = Math.max(1, tx | 0); ty = Math.max(1, ty | 0); tz = Math.max(1, tz | 0); flags |= 0;
  const srt = srtValues.length === 1 && srtValues[0]?.length ? Array.from(srtValues[0]) : srtValues;
  while (srt.length < 9) srt.push(srt.length < 3 ? 1 : 0);
  const tessellation = [tx, ty, tz];
  const definitions = [
    [0, 1, 1, 1, 0, 0, -1, 1, 0], [2, 1, 1, 1, -1, 0, 0, 0, 16],
    [0, 1, 1, 1, 0, 0, 1, 3, 16], [2, 1, 1, 1, 1, 0, 0, 2, 0],
    [0, 2, 1, 1, 0, 1, 0, 0, 0], [0, 2, 1, 1, 0, -1, 0, 0, 16],
  ];
  const mesh = new MinMesh();
  for (const definition of definitions) {
    mesh.makeGrid(tessellation[definition[0]], tessellation[definition[1]], definition[8]);
    const matrix = new Float32Array(16); matrix[15] = 1;
    // Native writes (&mat.i.x)[axis] and (&mat.j.x)[axis]. In our column-major
    // storage those are the selected rows of columns i (0) and j (4), not the
    // transposed column selected by `axis * 4`.
    matrix[definition[0]] = definition[2]; matrix[4 + definition[1]] = definition[3];
    matrix[12] = definition[4] * 0.5; matrix[13] = definition[5] * 0.5; matrix[14] = definition[6] * 0.5;
    mesh.transform(MMU_SELECTED, matrix);
    const uvMatrix = identityMatrix();
    if (definition[8]) { uvMatrix[12] = 1; uvMatrix[0] = -1; }
    if (flags & 2) uvMatrix[12] += definition[7];
    mesh.transform(MMU_SELECTED, uvMatrix, 1, 1);
    mesh.selectAll(2 | 4); // SETNOT with ALL clears both domains.
  }
  mesh.selectAll(2 | 4 | 1);
  mesh.transform(MMU_ALL, matrixSRT(srt));
  if (flags & 8) {
    const uvMatrix = identityMatrix(); uvMatrix[0] = srt[0]; uvMatrix[5] = srt[1]; uvMatrix[10] = srt[2];
    mesh.transform(MMU_ALL, uvMatrix, 1, 1);
  }
  if (flags & 4) mesh.transform(MMU_ALL, translationMatrix(0, srt[1] / 2, 0));
  return mesh;
}

function MinMesh_Sphere(tx, ty) {
  tx = Math.max(3, tx | 0); ty = Math.max(1, ty | 0);
  const mesh = new MinMesh(), start = mesh.makeGrid(tx, ty, 3 | 8);
  const regular = (tx + 1) * (ty + 1);
  for (let index = 0; index < regular; index++) {
    const vertex = mesh.vertices[start + index];
    const fx = (1 - vertex.uv[1][0]) * TAU;
    const fy = (0.5 / (ty + 1) + vertex.uv[0][1] * ty / (ty + 1)) * Math.PI;
    vertex.position[0] = f32(-Math.sin(fy) * Math.sin(fx) * 0.5);
    vertex.position[1] = f32(Math.cos(fy) * 0.5);
    vertex.position[2] = f32(-Math.sin(fy) * Math.cos(fx) * 0.5);
  }
  mesh.vertices[start + regular].position[1] = mesh.vertices[start].position[1];
  mesh.vertices[start + regular + 1].position[1] = mesh.vertices[start + regular - 1].position[1];
  return mesh.invalidate();
}

function MinMesh_Cylinder(tx, ty, flags = 0, tz = 1, arc = 0) {
  tx = Math.max(3, tx | 0); ty = Math.max(1, ty | 0); tz = Math.max(1, tz | 0); flags |= 0; arc |= 0;
  const count = tx; arc = Math.min(arc, tx - 1); if (arc > 0) tx = count - arc + 2;
  const mesh = new MinMesh();
  let start = mesh.makeGrid(tx, ty, 8);
  for (let y = 0; y <= ty; y++) for (let x = 0; x <= tx; x++) {
    const vertex = mesh.vertices[start++], angle = (x === tx ? 0 : x) * TAU / count, fy = y / ty;
    if (x !== tx - 1 || arc === 0) {
      vertex.position[0] = f32(Math.sin(angle) * 0.5); vertex.position[2] = f32(-Math.cos(angle) * 0.5);
    } else {
      vertex.position[0] = 0; vertex.position[2] = 0;
    }
    vertex.position[1] = f32(fy - 0.5); vertex.uv[0][0] = f32(x / tx); vertex.uv[0][1] = f32(1 - fy);
  }
  mesh.selectAll(4); // clear vertex selection, leave the middle's faces selected.
  start = mesh.makeGrid(tx, tz - 1, (flags & 1 ? 32 : 0) | 1);
  const bottomGridRows = tz;
  for (let y = 0; y < bottomGridRows; y++) for (let x = 0; x <= tx; x++) {
    const vertex = mesh.vertices[start++], angle = (x === tx ? 0 : x) * TAU / count;
    if (x !== tx - 1 || arc === 0) {
      vertex.position[0] = f32(Math.sin(angle) * 0.5); vertex.position[2] = f32(-Math.cos(angle) * 0.5);
    } else {
      vertex.position[0] = 0; vertex.position[2] = 0;
    }
    vertex.position[1] = -0.5; vertex.uv[0][0] = f32(vertex.position[0] + 0.5); vertex.uv[0][1] = f32(vertex.position[2] + 0.5);
  }
  const bottomCenter = mesh.vertices[start++]; bottomCenter.position[1] = -0.5; bottomCenter.uv[0].set([0.5, 0.5]);
  start = mesh.makeGrid(tx, tz - 1, (flags & 1 ? 32 : 0) | 2);
  for (let y = tz - 1; y >= 0; y--) for (let x = 0; x <= tx; x++) {
    const vertex = mesh.vertices[start++], angle = (x === tx ? 0 : x) * TAU / count;
    if (x !== tx - 1 || arc === 0) {
      vertex.position[0] = f32(Math.sin(angle) * 0.5); vertex.position[2] = f32(-Math.cos(angle) * 0.5);
    } else {
      vertex.position[0] = 0; vertex.position[2] = 0;
    }
    vertex.position[1] = 0.5; vertex.uv[0][0] = f32(vertex.position[0] + 0.5); vertex.uv[0][1] = f32(-vertex.position[2] + 0.5);
  }
  const topCenter = mesh.vertices[start]; topCenter.position[1] = 0.5; topCenter.uv[0].set([0.5, 0.5]);
  if (flags & 2) mesh.transform(MMU_ALL, translationMatrix(0, 0.5, 0));
  return mesh.invalidate();
}

function MinMesh_MatLink(input, material, selection = 0, renderPass = 0, owned = false) {
  const source = requireMinMesh(input);
  if (!material || material.kind !== 'material') return null;
  const mesh = owned ? source : source.cloneForMaterial();
  const storage = mesh._compact;
  if (storage) {
    let cluster = -1;
    for (let index = 1; index < storage.clusters.length; index++) {
      const value = storage.clusters[index];
      if (value.material === material && value.id === 0 &&
        value.animType === 0 && value.animMatrix === -1) {
        cluster = index;
        break;
      }
    }
    if (cluster < 0) {
      cluster = storage.clusters.length;
      storage.clusters.push(makeCluster(material, renderPass));
    }
    for (let index = 0; index < storage.faceCount; index++) {
      const offset = index * 4;
      const selectedFace = storage.faceInts[offset] !== 0;
      if ((selection | 0) === 0 || selection === 1 && selectedFace || selection === 2 && !selectedFace) {
        if (storage.faceInts[offset + 2] !== 0) storage.faceInts[offset + 2] = cluster;
      }
    }
    // GenMinMesh::ChangeTopo invalidates derived normals/prepared geometry,
    // even though the serialized position/topology buffers remain shareable.
    mesh.normalsValid = false;
    mesh._prepared = null;
    mesh.released = false;
    return mesh;
  }
  const cluster = mesh.addCluster(material, renderPass);
  for (const face of mesh.faces) {
    if ((selection | 0) === 0 || selection === 1 && face.select || selection === 2 && !face.select) {
      if (face.cluster !== 0) face.cluster = cluster;
    }
  }
  return mesh.invalidate();
}

function MinMesh_Add(inputs, ownership = []) {
  const valid = inputs.map((mesh, index) => ({ mesh, index })).filter(value => value.mesh instanceof MinMesh);
  if (!valid.length) return null;
  const first = valid[0];
  const mesh = ownership[first.index] ? first.mesh : first.mesh.clone();
  // Duplicate variable inputs refer to the pre-operator value. Preserve one
  // snapshot before adopting that value as the output accumulator.
  const adoptedSnapshot = mesh === first.mesh && valid.some((value, index) => index > 0 && value.mesh === mesh)
    ? mesh.clone() : null;
  for (let index = 1; index < valid.length; index++) {
    const source = valid[index].mesh === mesh ? adoptedSnapshot : valid[index].mesh;
    mesh.add(source);
  }
  return mesh.mergeClusters();
}

function MinMesh_SelectAll(input, mode, owned = false) {
  return writableMinMesh(input, owned, (mode & 2) !== 0).selectAll(mode);
}

function MinMesh_SelectCube(input, mode, cx, cy, cz, sx, sy, sz, owned = false) {
  const mesh = writableMinMesh(input, owned, true); mode |= 0;
  for (const vertex of mesh.vertices) vertex.tempByte =
    Math.abs(vertex.position[0] - cx) <= sx / 2 && Math.abs(vertex.position[1] - cy) <= sy / 2 && Math.abs(vertex.position[2] - cz) <= sz / 2 ? 1 : 0;
  switch (mode & 12) {
    case MMS_VERTEX:
      for (const vertex of mesh.vertices) vertex.select = selectLogic(vertex.select, vertex.tempByte, mode);
      break;
    case MMS_FULLFACE:
      for (const face of mesh.faces) {
        const match = face.vertices.every(index => mesh.vertices[index].tempByte);
        face.select = selectLogic(face.select, match, mode);
      }
      break;
    case MMS_PARTFACE:
      for (const face of mesh.faces) {
        const match = face.vertices.some(index => mesh.vertices[index].tempByte);
        face.select = selectLogic(face.select, match, mode);
      }
      break;
  }
  return mesh.invalidate();
}

function MinMesh_SelectLogic(input, mode, owned = false) {
  const mesh = writableMinMesh(input, owned, (mode | 0) !== 0); mode |= 0;
  if (mode === 0) for (const face of mesh.faces) face.select ^= 1;
  else if (mode === 1) for (const vertex of mesh.vertices) vertex.select ^= 1;
  else if (mode === 2) {
    for (const vertex of mesh.vertices) vertex.select = 0;
    for (const face of mesh.faces) if (face.select) for (const index of face.vertices) mesh.vertices[index].select = 1;
  }
  return mesh;
}

function MinMesh_DeleteFaces(input, owned = false) {
  const mesh = writableMinMesh(input, owned);
  for (const face of mesh.faces) if (face.select) face.cluster = 0;
  return mesh.invalidate();
}

function MinMesh_Invert(input, owned = false) { return writableMinMesh(input, owned).invert(); }

function MinMesh_TransformEx(input, mask, ...srtValues) {
  let owned = false;
  if (typeof srtValues[srtValues.length - 1] === 'boolean') owned = srtValues.pop();
  const mesh = writableMinMesh(input, owned, true);
  return mesh.transform(mask & 3, matrixSRT(srtValues.slice(0, 9)), (mask >>> 2) & 7, (mask >>> 5) & 7, (mask >>> 8) & 3);
}

function MinMesh_Displace(input, bitmap, mask, ax, ay, az, owned = false) {
  const mesh = writableMinMesh(input, owned, true);
  if (!bitmap?.data || !bitmap.width || !bitmap.height) return mesh;
  mesh.calcNormals();
  const context = new BilinearContext(bitmap, 0);
  const sample = new Uint16Array(4);
  for (const vertex of mesh.vertices) if ((mask | 0) === 0 || mask === 1 && vertex.select || mask === 2 && !vertex.select) {
    context.sample(Math.trunc(vertex.uv[0][0] * bitmap.width * 65536), Math.trunc(vertex.uv[0][1] * bitmap.height * 65536), sample);
    const height = (sample[0] - 16384) / 32767;
    vertex.position[0] = f32(vertex.position[0] + vertex.normal[0] * height * ax);
    vertex.position[1] = f32(vertex.position[1] + vertex.normal[1] * height * ay);
    vertex.position[2] = f32(vertex.position[2] + vertex.normal[2] * height * az);
  }
  return mesh.invalidate();
}

let perlinState = null;
function sourcePerlinState() {
  if (perlinState) return perlinState;
  const random = new Random(); random.setSeed(1);
  const ranks = new Uint32Array(256), permutation = new Uint8Array(512);
  for (let index = 0; index < 256; index++) { ranks[index] = random.int(0x10000); permutation[index] = index; }
  for (let i = 0; i < 255; i++) for (let j = i + 1; j < 256; j++) if (ranks[i] > ranks[j]) {
    [ranks[i], ranks[j]] = [ranks[j], ranks[i]]; [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
  }
  permutation.set(permutation.subarray(0, 256), 256);
  const source = [
    1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0,
    1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0,
    0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0,
    1, 1, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 0,
  ];
  const values = [0, 1, -1], gradients = new Float32Array(source.map(index => values[index]));
  return (perlinState = { permutation, gradients });
}

function lerpVector(a, b, amount, out = vector3()) {
  out[0] = f32(a[0] + (b[0] - a[0]) * amount); out[1] = f32(a[1] + (b[1] - a[1]) * amount); out[2] = f32(a[2] + (b[2] - a[2]) * amount);
  return out;
}

function sourcePerlin3(position, out = vector3()) {
  const { permutation: p, gradients: g } = sourcePerlinState();
  const integer = [0, 0, 0], fade = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    integer[axis] = Math.floor(position[axis]);
    const fraction = f32(position[axis] - integer[axis]);
    integer[axis] &= 255;
    fade[axis] = f32(fraction * fraction * fraction * (10 + fraction * (6 * fraction - 15)));
  }
  const gradient = (x, y, z) => g.subarray((p[p[p[x] + y] + z] & 15) * 4, (p[p[p[x] + y] + z] & 15) * 4 + 3);
  const [x, y, z] = integer;
  const a = lerpVector(gradient(x, y, z), gradient(x + 1, y, z), fade[0]);
  const b = lerpVector(gradient(x, y + 1, z), gradient(x + 1, y + 1, z), fade[0]);
  lerpVector(a, b, fade[1], a);
  const c = lerpVector(gradient(x, y, z + 1), gradient(x + 1, y, z + 1), fade[0]);
  const d = lerpVector(gradient(x, y + 1, z + 1), gradient(x + 1, y + 1, z + 1), fade[0]);
  lerpVector(c, d, fade[1], c);
  return lerpVector(a, c, fade[2], out);
}

function MinMesh_Perlin(input, mask, ...values) {
  let owned = false;
  if (typeof values[values.length - 1] === 'boolean') owned = values.pop();
  const mesh = writableMinMesh(input, owned, true), matrix = matrixSRT(values.slice(0, 9)), amplitude = values.slice(9, 12);
  for (const vertex of mesh.vertices) if ((mask | 0) === 0 || mask === 1 && vertex.select || mask === 2 && !vertex.select) {
    const sample = sourcePerlin3(transformPoint(matrix, vertex.position));
    vertex.position[0] = f32(vertex.position[0] + sample[0] * amplitude[0]);
    vertex.position[1] = f32(vertex.position[1] + sample[1] * amplitude[1]);
    vertex.position[2] = f32(vertex.position[2] + sample[2] * amplitude[2]);
  }
  return mesh.invalidate();
}

function MinMesh_ExtrudeNormal(input, mask, distance, owned = false) {
  const mesh = writableMinMesh(input, owned, true); mesh.calcNormals();
  for (const vertex of mesh.vertices) if ((mask | 0) === 0 || mask === 1 && vertex.select || mask === 2 && !vertex.select) add3(vertex.position, vertex.normal, distance);
  return mesh.invalidate();
}

function MinMesh_Bend2(input, cx, cy, cz, rx, ry, rz, length, angle, owned = false) {
  const mesh = writableMinMesh(input, owned, true);
  const forward = mat4EulerTurns([rx, ry, rz]);
  forward[12] = f32(-cx); forward[13] = f32(-cy); forward[14] = f32(-cz);
  const backward = rigidInverse(forward); angle = f32(angle * TAU);
  for (const vertex of mesh.vertices) {
    const value = transformPoint(forward, vertex.position), originalY = value[1];
    if (originalY >= 0) value[1] = f32(value[1] - Math.min(originalY, length));
    const phase = f32(Math.max(0, Math.min(1, f32(originalY / length))) * angle);
    const sine = f32(Math.sin(phase)), cosine = f32(Math.cos(phase)), x = value[0], y = value[1];
    value[0] = f32(cosine * x - sine * y); value[1] = f32(sine * x + cosine * y);
    transformPoint(backward, value, vertex.position);
  }
  return mesh.invalidate();
}

function MinMesh_Triangulate(input, owned = false) { return writableMinMesh(input, owned).triangulate(); }

function randomSRT(base, random) {
  const values = Array.from(base);
  // InitRandomSRT uses each component as the range around identity/zero.
  for (let i = 0; i < 3; i++) values[i] = 1 + random.float() * (base[i] - 1);
  for (let i = 3; i < 9; i++) values[i] = (random.float(2) - 1) * base[i];
  return matrixSRT(values);
}

function MinMesh_Multiply(input, srt, count, mode = 0, tu = 0, tv = 0, localRotation = [0, 0, 0], extrude = 0, owned = false) {
  void extrude;
  const source = requireMinMesh(input);
  let transform = identityMatrix(), local = identityMatrix();
  const step = matrixSRT(srt), localStep = mat4EulerTurns(localRotation);
  const random = new Random(); random.setSeed(count | 0);
  count = Math.max(0, count | 0);
  if (owned && count > 0) {
    const original = count > 1 ? source.clone() : null;
    const output = source;
    // Native Multiply always creates a fresh GenMinMesh. Add() carries the
    // clusters' animation declarations but not the source skeleton itself.
    output.animation = null;
    output.completelyRigid = false;
    output.stripped = false;
    output.detachVertices();
    for (let copy = 0; copy < count; copy++) {
      let start = 0;
      if (copy > 0) { start = output.vertices.length; output.add(original); }
      const combined = matrixMul(transform, local);
      for (let index = start; index < output.vertices.length; index++) {
        transformPoint(combined, output.vertices[index].position, output.vertices[index].position);
        if (mode & 1) { output.vertices[index].uv[0][0] += copy * tu; output.vertices[index].uv[0][1] += copy * tv; }
      }
      transform = mode & 2 ? randomSRT(srt, random) : matrixMul(step, transform);
      local = matrixMul(localStep, local);
    }
    return output.mergeClusters();
  }
  const output = new MinMesh();
  for (let copy = 0; copy < count; copy++) {
    const start = output.vertices.length; output.add(source);
    const combined = matrixMul(transform, local);
    for (let index = start; index < output.vertices.length; index++) {
      transformPoint(combined, output.vertices[index].position, output.vertices[index].position);
      if (mode & 1) { output.vertices[index].uv[0][0] += copy * tu; output.vertices[index].uv[0][1] += copy * tv; }
    }
    transform = mode & 2 ? randomSRT(srt, random) : matrixMul(step, transform);
    local = matrixMul(localStep, local);
  }
  return output.mergeClusters();
}

function MinMesh_Center(input, which, owned = false) {
  const mesh = writableMinMesh(input, owned, true), { minimum, maximum } = mesh.bounds();
  const offset = [
    (which & 1) ? f32((maximum[0] + minimum[0]) * 0.5) : 0,
    (which & 2) ? f32((maximum[1] + minimum[1]) * 0.5) : 0,
    (which & 4) ? f32((maximum[2] + minimum[2]) * 0.5) : 0,
  ];
  for (const vertex of mesh.vertices) {
    vertex.position[0] = f32(vertex.position[0] - offset[0]);
    vertex.position[1] = f32(vertex.position[1] - offset[1]);
    vertex.position[2] = f32(vertex.position[2] - offset[2]);
  }
  return mesh.invalidate();
}

function MinMesh_Normals(input, owned = false) { const mesh = writableMinMesh(input, owned, true); mesh.calcNormals(); return mesh; }

function classifyMinVector(value) {
  const ax = Math.abs(value[0]), ay = Math.abs(value[1]), az = Math.abs(value[2]);
  if (ax > ay && ax > az) return value[0] > 0 ? 1 : 0;
  if (ay > az) return value[1] > 0 ? 3 : 2;
  return value[2] > 0 ? 5 : 4;
}

function classifyMatrix(id) {
  const directions = [
    [-1, 0, 0], [1, 0, 0], [0, -1, 0],
    [0, 1, 0], [0, 0, -1], [0, 0, 1],
  ];
  const k = directions[id];
  const j = id === 2 || id === 3 ? [0, 0, 1] : [0, 1, 0];
  const i = [
    j[1] * k[2] - j[2] * k[1],
    j[2] * k[0] - j[0] * k[2],
    j[0] * k[1] - j[1] * k[0],
  ];
  const matrix = identityMatrix();
  matrix[0] = i[0]; matrix[1] = i[1]; matrix[2] = i[2];
  matrix[4] = j[0]; matrix[5] = j[1]; matrix[6] = j[2];
  matrix[8] = k[0]; matrix[9] = k[1]; matrix[10] = k[2];
  return matrix;
}

function MinMesh_AutoMap(input, flags = 0x3f, owned = false) {
  const mesh = writableMinMesh(input, owned, true); mesh.calcNormals();
  const { minimum, maximum } = mesh.bounds();
  const fields = new Int32Array(6); fields.fill(-1);
  let fieldCount = 0;
  for (let index = 0; index < 6; index++) if (flags & (1 << index)) fields[index] = fieldCount++;

  // Literal fallback chain from MinMesh_RenderAutoMap. Disabled directions
  // share an enabled atlas field rather than falling back to an unpacked
  // projection of their own.
  if (fields[0] === -1 && fields[1] >= 0) fields[0] = fields[1];
  if (fields[0] === -1 && fields[4] >= 0) fields[0] = fields[4];
  if (fields[0] === -1 && fields[5] >= 0) fields[0] = fields[5];
  if (fields[0] === -1 && fields[2] >= 0) fields[0] = fields[2];
  if (fields[0] === -1 && fields[3] >= 0) fields[0] = fields[3];
  if (fields[1] === -1) fields[1] = fields[0];
  if (fields[4] === -1 && fields[5] >= 0) fields[4] = fields[5];
  if (fields[4] === -1) fields[4] = fields[0];
  if (fields[5] === -1) fields[5] = fields[4];
  if (fields[2] === -1 && fields[3] >= 0) fields[2] = fields[3];
  if (fields[2] === -1) fields[2] = fields[0];
  if (fields[3] === -1) fields[3] = fields[2];

  for (const vertex of mesh.vertices) vertex.tempByte = -1;
  for (const face of mesh.faces) {
    face.temp = classifyMinVector(face.normal) ^ 1;
    for (const index of face.vertices) mesh.vertices[index].tempByte = face.temp;
  }

  const maps = Array.from({ length: 6 }, () => new Float32Array(6));
  const offsets = Array.from({ length: 6 }, () => new Float32Array(2));
  const inverseFieldCount = f32(1 / fieldCount);
  for (let index = 0; index < 6; index++) {
    const matrix = classifyMatrix(index);
    const transformedMinimum = transformPoint(matrix, minimum);
    const transformedMaximum = transformPoint(matrix, maximum);
    const xRange = Math.abs(f32(transformedMaximum[0] - transformedMinimum[0]));
    const yRange = Math.abs(f32(transformedMaximum[1] - transformedMinimum[1]));
    const map = maps[index];
    map[0] = f32(f32(matrix[0] / xRange) * inverseFieldCount);
    map[1] = f32(f32(matrix[1] / xRange) * inverseFieldCount);
    map[2] = f32(f32(matrix[2] / xRange) * inverseFieldCount);
    map[3] = f32(matrix[4] / yRange);
    map[4] = f32(matrix[5] / yRange);
    map[5] = f32(matrix[6] / yRange);
    offsets[index][0] = f32(-(
      map[0] * transformedMinimum[0] + map[1] * transformedMinimum[1] +
      map[2] * transformedMinimum[2]
    ) + f32(fields[index] / fieldCount));
    offsets[index][1] = f32(-(
      map[3] * transformedMinimum[0] + map[4] * transformedMinimum[1] +
      map[5] * transformedMinimum[2]
    ));
  }

  for (const vertex of mesh.vertices) {
    const direction = vertex.tempByte;
    if (direction < 0 || direction >= 6) continue;
    const map = maps[direction], offset = offsets[direction], position = vertex.position;
    vertex.uv[0][0] = f32(
      map[0] * position[0] + map[1] * position[1] + map[2] * position[2] + offset[0],
    );
    vertex.uv[0][1] = f32(1 - f32(
      map[3] * position[0] + map[4] * position[1] + map[5] * position[2] + offset[1],
    ));
  }
  mesh._prepared = null;
  return mesh;
}

// A conservative vertex chamfer used by import adapters. The released Debris
// graph does not dispatch a MinMesh chamfer operator, but exposing it keeps
// the direct port useful to the old-Mesh conversion path.
function MinMesh_Chamfer(input, amount = 0.01, owned = false) {
  const mesh = writableMinMesh(input, owned, true); mesh.calcNormals();
  for (const vertex of mesh.vertices) add3(vertex.position, vertex.normal, amount);
  return mesh.invalidate();
}

function MinMesh_BoneChain(input, p0x, p0y, p0z, p1x, p1y, p1z, count, flags, owned = false) {
  const mesh = writableMinMesh(input, owned, true); count = Math.max(1, count | 0);
  for (let index = 1; index < mesh.clusters.length; index++) mesh.clusters[index].animType = 2;
  let p0 = vector3(p0x, p0y, p0z), p1 = vector3(p1x, p1y, p1z);
  if (flags & 1) { const bounds = mesh.bounds(); p0 = vector3(0, 0, bounds.minimum[2]); p1 = vector3(0, 0, bounds.maximum[2]); }
  const difference = sub3(p1, p0), denominator = dot3(difference, difference) || 1;
  const direction = scale3(cloneVector(difference), 1 / denominator);
  const matrices = [];
  for (let index = 0; index < count; index++) {
    const phase = count > 1 ? index / (count - 1) : 0;
    const position = vector3(p0[0] + difference[0] * phase, p0[1] + difference[1] * phase, p0[2] + difference[2] * phase);
    const noAnimation = translationMatrix(position[0], position[1], position[2]);
    matrices.push({ basePose: translationMatrix(-position[0], -position[1], -position[2]), parent: -1, factor: 1, spread: 0, offset: 0, noAnimation, spline: null });
  }
  mesh.animation = { matrices };
  for (const vertex of mesh.vertices) {
    // Preserve the released source typo: the y component starts from Pos.x.
    const delta = vector3(vertex.position[0] - p0[0], vertex.position[0] - p0[1], vertex.position[2] - p0[2]);
    let fade = dot3(direction, delta) * (count - 1);
    const fixed = Math.max(0, Math.min(count * 1024 - 1, Math.trunc(fade * 1024)));
    const index = Math.trunc(fixed / 1024); fade = (fixed & 1023) / 1024;
    vertex.weights.fill(0); vertex.matrices.fill(0);
    if (index + 1 < count) {
      const f2 = fade * fade, f3 = f2 * fade;
      vertex.boneCount = 4;
      vertex.matrices.set([Math.max(0, index - 1), index, Math.min(count - 1, index + 1), Math.min(count - 1, index + 2)]);
      vertex.weights.set([-0.5 * f3 + f2 - 0.5 * fade, 1.5 * f3 - 2.5 * f2 + 1, -1.5 * f3 + 2 * f2 + 0.5 * fade, 0.5 * f3 - 0.5 * f2]);
    } else { vertex.boneCount = 1; vertex.matrices[0] = count - 1; vertex.weights[0] = 1; }
  }
  mesh._prepared = null;
  return mesh;
}

function MinMesh_BoneTrain(input, spline, delta, mode, offset, owned = false) {
  const mesh = writableMinMesh(input, owned);
  if (!spline || !mesh.animation) return mesh;
  const count = mesh.animation.matrices.length;
  for (let index = 0; index < count; index++) {
    const record = mesh.animation.matrices[index];
    const time = f32(f32(f32(index * delta) / f32(count - 1)) + offset);
    record.spline = spline;
    if ((mode & 3) === 0) { record.offset = time; record.factor = 1; record.spread = 0; }
    else if ((mode & 3) === 1) { record.offset = 0; record.factor = time; record.spread = 0; }
    else if ((mode & 3) === 2) { record.offset = 0; record.factor = 0; record.spread = 1; }
  }
  return mesh;
}

class KineticSpline {
  constructor(position, speed, gravity, rotation, axis = null, angle = 0) {
    this.position = cloneVector(position); this.speed = cloneVector(speed); this.gravity = cloneVector(gravity);
    this.rotation = cloneVector(rotation); this.axis = axis && cloneVector(axis); this.angle = f32(angle);
  }
  eval(time) {
    time = f32(Math.max(0, time)); let matrix;
    if (this.axis) {
      const axis = normalize3(cloneVector(this.axis));
      const radians = f32(f32(this.angle * time) * f32(TAU));
      const c = f32(Math.cos(radians)), s = f32(Math.sin(radians));
      const t = f32(1 - c), x = axis[0], y = axis[1], z = axis[2];
      matrix = new Float32Array([
        f32(f32(f32(t*x)*x)+c), f32(f32(f32(t*x)*y)+f32(s*z)), f32(f32(f32(t*x)*z)-f32(s*y)), 0,
        f32(f32(f32(t*x)*y)-f32(s*z)), f32(f32(f32(t*y)*y)+c), f32(f32(f32(t*y)*z)+f32(s*x)), 0,
        f32(f32(f32(t*x)*z)+f32(s*y)), f32(f32(f32(t*y)*z)-f32(s*x)), f32(f32(f32(t*z)*z)+c), 0,
        0, 0, 0, 1,
      ]);
    } else matrix = mat4Euler(
      f32(f32(this.rotation[0] * f32(TAU)) * time),
      f32(f32(this.rotation[1] * f32(TAU)) * time),
      f32(f32(this.rotation[2] * f32(TAU)) * time),
    );
    const time2 = f32(time * time);
    matrix[12] = f32(f32(this.position[0] + f32(this.speed[0] * time)) + f32(this.gravity[0] * time2));
    matrix[13] = f32(f32(this.position[1] + f32(this.speed[1] * time)) + f32(this.gravity[1] * time2));
    matrix[14] = f32(f32(this.position[2] + f32(this.speed[2] * time)) + f32(this.gravity[2] * time2));
    return { matrix, zoom: 0 };
  }
}

function faceCenter(mesh, face) {
  const center = vector3(); for (const index of face.vertices) add3(center, mesh.vertices[index].position);
  return scale3(center, 1 / face.count);
}

function explodeTriangleArea(a, b, c) {
  return cross3(sub3(a, b), sub3(b, c));
}

function explodeFaceIndices(source, face, mode) {
  const indices = face.vertices.slice();
  if (indices.length !== 4 || !(mode & 2)) return indices;
  const p0 = source.vertices[indices[0]].position;
  const p1 = source.vertices[indices[1]].position;
  const p2 = source.vertices[indices[2]].position;
  const p3 = source.vertices[indices[3]].position;
  const a1 = explodeTriangleArea(p0, p1, p2);
  const a2 = explodeTriangleArea(p1, p2, p3);
  const a3 = explodeTriangleArea(p2, p3, p0);
  const a0 = explodeTriangleArea(p3, p0, p1);
  if (dot3(a0, a2) > dot3(a1, a3)) indices.push(indices.shift());
  return indices;
}

function MinMesh_Explode(input, bitmap, ...p) {
  const source = requireMinMesh(input); source.calcNormals();
  const center = vector3(p[0], p[1], p[2]), extrude = p[3], speedNormal = p[4], speedCenter = p[5];
  const speedGravity = p[6], speedRandom = p[7], rotationSpeed = vector3(p[8], p[9], p[10]);
  const toDistance = p[11], toPower = p[12], toRandom = p[13], toConstant = p[14], mode = p[15] | 0;
  const towards = vector3(p[16] - center[0], p[17] - center[1], p[18] - center[2]);
  const towardsDiv = dot3(towards, towards);
  const towardsScaled = vector3(
    towards[0] / towardsDiv, towards[1] / towardsDiv, towards[2] / towardsDiv,
  );
  const extrudeFlat = 1 - p[19], tensorRandom = p[20];
  const output = new MinMesh();
  const liveFaces = source.faces.map((face, faceIndex) => ({ face, faceIndex }))
    .filter(({ face }) => face.count >= 3 && face.cluster > 0);
  if (!liveFaces.length || liveFaces.length > 0xffff) return output;
  const random = new Random(); random.setSeed(1);
  const matrixRecords = [];
  const bitmapSampler = bitmap?.data ? new BilinearContext(bitmap, 0) : null;
  const bitmapSample = bitmapSampler ? new Uint16Array(4) : null;
  let shardIndex = 0;
  for (const { face, faceIndex } of liveFaces) {
    const centerOfFace = faceCenter(source, face);
    const indices = explodeFaceIndices(source, face, mode);
    const top = [], bottom = [], mergeTag = faceIndex & 255;
    for (const sourceIndex of indices) {
      const vertex = makeVertex(source.vertices[sourceIndex]);
      vertex.mergeTag = mergeTag; vertex.boneCount = 1;
      vertex.matrices[0] = shardIndex; vertex.weights[0] = 1;
      top.push(output.vertices.push(vertex) - 1);
    }
    for (const sourceIndex of indices) {
      const original = source.vertices[sourceIndex], vertex = makeVertex(original);
      vertex.mergeTag = mergeTag; vertex.boneCount = 1;
      vertex.matrices[0] = shardIndex; vertex.weights[0] = 1;
      const difference = sub3(vertex.position, centerOfFace);
      scale3(difference, extrudeFlat); copy3(vertex.position, centerOfFace);
      add3(vertex.position, difference); add3(vertex.position, original.normal, -extrude);
      bottom.push(output.vertices.push(vertex) - 1);
    }
    for (let index = 2; index < top.length; index++) output.faces.push(makeFace([top[0], top[index - 1], top[index]], 1));
    for (let index = 2; index < bottom.length; index++) output.faces.push(makeFace([bottom[index], bottom[index - 1], bottom[0]], 1));
    for (let index = 0; index < indices.length; index++) {
      const a = source.vertices[indices[index]], b = source.vertices[indices[(index + 1) % indices.length]];
      const side = [makeVertex(a), makeVertex(b), makeVertex(a), makeVertex(b)];
      for (const vertex of side) { vertex.mergeTag = mergeTag; vertex.boneCount = 1; vertex.matrices[0] = shardIndex; vertex.weights[0] = 1; }
      for (let edge = 2; edge < 4; edge++) {
        const difference = sub3(side[edge].position, centerOfFace); scale3(difference, extrudeFlat); copy3(side[edge].position, centerOfFace); add3(side[edge].position, difference); add3(side[edge].position, edge === 2 ? a.normal : b.normal, -extrude);
      }
      const base = output.vertices.length; output.vertices.push(...side);
      output.faces.push(makeFace([base + 2, base + 3, base + 1, base], 1, 1));
    }
    const radial = sub3(centerOfFace, center), speed = vector3();
    add3(speed, radial, speedCenter); add3(speed, face.normal, speedNormal);
    speed[0] += (random.float(2) - 1) * speedRandom; speed[1] += (random.float(2) - 1) * speedRandom; speed[2] += (random.float(2) - 1) * speedRandom;
    let delay;
    if (bitmapSampler) {
      let u = 0, v = 0;
      for (const index of face.vertices) {
        u = f32(u + source.vertices[index].uv[0][0]);
        v = f32(v + source.vertices[index].uv[0][1]);
      }
      u = f32(u / face.count); v = f32(v / face.count);
      bitmapSampler.sample(
        Math.trunc(u * 0x10000 * bitmap.width),
        Math.trunc(v * 0x10000 * bitmap.width),
        bitmapSample,
      );
      const sample = bitmapSample[0];
      delay = sample ? -toDistance * Math.pow(1 - sample / 32767, toPower) : -9999;
    } else if (mode & 1) {
      const projection = dot3(radial, towardsScaled);
      const offAxis = sub3(radial, scale3(cloneVector(towards), projection));
      let distance = Math.sqrt(dot3(offAxis, offAxis)) - toPower;
      if (distance < 0) { distance = distance < -toPower ? 0 : toPower + distance; delay = projection - Math.sqrt(toPower * toPower - distance * distance) / Math.sqrt(towardsDiv); distance = 0; }
      else delay = projection;
      delay = -toDistance * distance - delay;
    } else delay = -toDistance * Math.pow(dot3(radial, radial), toPower);
    delay = f32(delay - toConstant - random.float() * toRandom);
    const noAnimation = translationMatrix(centerOfFace[0], centerOfFace[1], centerOfFace[2]);
    let kinetic;
    if (mode & 4) {
      const axis = vector3(random.float(2) - 1, random.float(2) - 1, random.float(2) - 1); normalize3(axis);
      kinetic = new KineticSpline(centerOfFace, speed, [0, speedGravity, 0], [0, 0, 0], axis,
        (random.float(2) - 1) * tensorRandom);
    } else {
      const rotation = vector3((random.float(2) - 1) * rotationSpeed[0],
        (random.float(2) - 1) * rotationSpeed[1], (random.float(2) - 1) * rotationSpeed[2]);
      kinetic = new KineticSpline(centerOfFace, speed, [0, speedGravity, 0], rotation);
    }
    matrixRecords.push({
      basePose: translationMatrix(-centerOfFace[0], -centerOfFace[1], -centerOfFace[2]), parent: -1,
      factor: 1, spread: 0, offset: delay, noAnimation,
      spline: kinetic,
    });
    shardIndex++;
  }
  output.animation = { matrices: matrixRecords }; output.clusters[1].animType = 2; output.completelyRigid = true;
  output.calcNormals();
  return output;
}

let fontAdapter = null;
function setMinMeshFontAdapter(adapter) { fontAdapter = adapter; }

// Cover a binary glyph mask with non-overlapping rectangles.  This remains a
// useful small pure helper (and documents the first Canvas Font3D fallback),
// but the actual glyph builder below no longer extrudes these rectangles one
// by one: doing so left internal walls through every letter.
function fontMaskRectangles(mask, width, height, limit = 4096) {
  width = Math.max(0, width | 0); height = Math.max(0, height | 0);
  const active = new Uint8Array(width * height);
  for (let index = 0; index < active.length; index++) active[index] = mask?.[index] ? 1 : 0;
  const rectangles = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const first = y * width + x;
    if (!active[first]) continue;
    let rectangleWidth = 1;
    while (x + rectangleWidth < width && active[first + rectangleWidth]) rectangleWidth++;
    let rectangleHeight = 1;
    rows: while (y + rectangleHeight < height) {
      const row = (y + rectangleHeight) * width + x;
      for (let dx = 0; dx < rectangleWidth; dx++) if (!active[row + dx]) break rows;
      rectangleHeight++;
    }
    rectangles.push([x, y, x + rectangleWidth, y + rectangleHeight]);
    if (rectangles.length > limit) return null;
    for (let dy = 0; dy < rectangleHeight; dy++) {
      active.fill(0, (y + dy) * width + x, (y + dy) * width + x + rectangleWidth);
    }
  }
  return rectangles;
}

const FONT_MASK_PIXEL_LIMIT = 2 * 1024 * 1024;
const FONT_MASK_BOUNDARY_LIMIT = 32768;
const FONT_MASK_CONTOUR_LIMIT = 4096;
const FONT_MASK_RECTANGLE_LIMIT = 8192;
const FONT_MASK_TRIANGLE_LIMIT = 32768;

function fontContourArea(contour) {
  let area = 0;
  for (let index = 0; index < contour.length; index++) {
    const current = contour[index], next = contour[(index + 1) % contour.length];
    area += current[0] * next[1] - next[0] * current[1];
  }
  return area * 0.5;
}

function simplifyFontContour(contour) {
  if (contour.length < 3) return [];
  const result = [];
  for (let index = 0; index < contour.length; index++) {
    const previous = contour[(index + contour.length - 1) % contour.length];
    const current = contour[index], next = contour[(index + 1) % contour.length];
    const ax = current[0] - previous[0], ay = current[1] - previous[1];
    const bx = next[0] - current[0], by = next[1] - current[1];
    // All source edges are unit grid edges.  A zero cross product therefore
    // means this point only subdivides a straight outline segment.
    if (ax * by - ay * bx || ax * bx + ay * by <= 0) result.push(current);
  }
  return result;
}

// Trace the exact rectilinear boundary of a binary mask.  Edges are directed
// with filled pixels on their right in Canvas coordinates: outer contours
// have positive signed area and holes have negative signed area.  A single
// byte per grid vertex stores the four possible outgoing edges, which also
// handles diagonally touching components without object-heavy edge maps.
function fontMaskContours(mask, width, height,
  boundaryLimit = FONT_MASK_BOUNDARY_LIMIT, contourLimit = FONT_MASK_CONTOUR_LIMIT) {
  width = Math.max(0, width | 0); height = Math.max(0, height | 0);
  boundaryLimit = Math.max(0, boundaryLimit | 0); contourLimit = Math.max(0, contourLimit | 0);
  const size = width * height;
  if (!mask || mask.length < size || size > FONT_MASK_PIXEL_LIMIT) return null;
  if (!size) return [];
  const filled = (x, y) => x >= 0 && x < width && y >= 0 && y < height && Boolean(mask[y * width + x]);
  let edgeCount = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (filled(x, y)) {
    if (!filled(x, y - 1)) edgeCount++;
    if (!filled(x + 1, y)) edgeCount++;
    if (!filled(x, y + 1)) edgeCount++;
    if (!filled(x - 1, y)) edgeCount++;
    if (edgeCount > boundaryLimit) return null;
  }
  if (!edgeCount) return [];

  const stride = width + 1;
  const outgoing = new Uint8Array(stride * (height + 1));
  const addEdge = (x, y, direction) => { outgoing[y * stride + x] |= 1 << direction; };
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (filled(x, y)) {
    if (!filled(x, y - 1)) addEdge(x, y, 0);          // east
    if (!filled(x + 1, y)) addEdge(x + 1, y, 1);      // south
    if (!filled(x, y + 1)) addEdge(x + 1, y + 1, 2); // west
    if (!filled(x - 1, y)) addEdge(x, y + 1, 3);      // north
  }

  const dx = [1, 0, -1, 0], dy = [0, 1, 0, -1];
  const firstDirection = bits => bits & 1 ? 0 : bits & 2 ? 1 : bits & 4 ? 2 : bits & 8 ? 3 : -1;
  const continuation = (bits, incoming) => {
    // Keeping the filled cell on the right splits corner-touching pixels into
    // separate contours instead of manufacturing a self-intersection.
    const candidates = [(incoming + 1) & 3, incoming, (incoming + 3) & 3, (incoming + 2) & 3];
    for (const direction of candidates) if (bits & (1 << direction)) return direction;
    return -1;
  };
  const contours = [];
  let remaining = edgeCount;
  for (let startY = 0; startY <= height; startY++) for (let startX = 0; startX <= width; startX++) {
    let startBits = outgoing[startY * stride + startX];
    while (startBits) {
      if (contours.length >= contourLimit) return null;
      const initialDirection = firstDirection(startBits);
      const contour = [];
      let x = startX, y = startY, incoming = -1, steps = 0;
      while (true) {
        contour.push([x, y]);
        const offset = y * stride + x, bits = outgoing[offset];
        const direction = incoming < 0 ? initialDirection : continuation(bits, incoming);
        if (direction < 0 || !(bits & (1 << direction))) return null;
        outgoing[offset] = bits & ~(1 << direction);
        remaining--; steps++;
        x += dx[direction]; y += dy[direction]; incoming = direction;
        if (x === startX && y === startY) break;
        if (steps > edgeCount) return null;
      }
      const simplified = simplifyFontContour(contour);
      if (simplified.length >= 3 && fontContourArea(simplified)) contours.push(simplified);
      startBits = outgoing[startY * stride + startX];
    }
  }
  return remaining === 0 ? contours : null;
}

function pointInFontContour(point, contour) {
  let inside = false;
  for (let index = 0, previous = contour.length - 1; index < contour.length; previous = index++) {
    const a = contour[index], b = contour[previous];
    if ((a[1] > point[1]) !== (b[1] > point[1]) &&
      point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

// Group contours into the same outer-plus-holes structure GLU receives in the
// native Font3D operator.  The raster triangulator does not need fragile hole
// bridges, but exposing this classification makes its hole semantics explicit
// and independently testable.
function fontMaskPolygons(contours) {
  if (!Array.isArray(contours)) return null;
  const regions = [];
  const holes = [];
  for (const contour of contours) {
    const area = fontContourArea(contour);
    if (area > 0) regions.push({ outer: contour, holes: [], area });
    else if (area < 0) holes.push({ contour, area });
  }
  for (const hole of holes) {
    let owner = null;
    for (const region of regions) if (pointInFontContour(hole.contour[0], region.outer) &&
      (!owner || region.area < owner.area)) owner = region;
    if (!owner) return null;
    owner.holes.push(hole.contour);
  }
  return regions;
}

// Merge identical horizontal mask runs into maximal rectangles.  These are a
// robust constrained triangulation of the rectilinear polygon-with-holes, not
// independently extruded pieces: vertices are shared, and only true exterior
// triangle edges receive side walls below.
function fontMaskBands(mask, width, height, limit = FONT_MASK_RECTANGLE_LIMIT) {
  const rectangles = [];
  let open = new Map();
  for (let y = 0; y <= height; y++) {
    const runs = [];
    if (y < height) for (let x = 0; x < width;) {
      while (x < width && !mask[y * width + x]) x++;
      if (x >= width) break;
      const x0 = x;
      while (x < width && mask[y * width + x]) x++;
      runs.push([x0, x]);
    }
    const next = new Map();
    for (const [x0, x1] of runs) {
      const key = `${x0},${x1}`;
      const rectangle = open.get(key) || { x0, x1, y0: y, y1: y };
      rectangle.y1 = y + 1;
      next.set(key, rectangle);
    }
    for (const [key, rectangle] of open) if (!next.has(key)) {
      rectangles.push(rectangle);
      if (rectangles.length > limit) return null;
    }
    open = next;
  }
  return rectangles;
}

function fontMaskToMinMesh(mask, width, height, depth = 0, options = null) {
  width = Math.max(0, width | 0); height = Math.max(0, height | 0);
  const size = width * height;
  if (!mask || mask.length < size) return null;
  options ||= {};
  const contours = fontMaskContours(mask, width, height,
    options.boundaryLimit ?? FONT_MASK_BOUNDARY_LIMIT,
    options.contourLimit ?? FONT_MASK_CONTOUR_LIMIT);
  if (!contours || !fontMaskPolygons(contours)) return null;
  const rectangles = fontMaskBands(mask, width, height,
    options.rectangleLimit ?? FONT_MASK_RECTANGLE_LIMIT);
  if (!rectangles) return null;

  // Add every run endpoint at a shared horizontal boundary to both incident
  // rectangles.  This prevents T-junctions when a stroke widens, narrows, or
  // splits around a hole.
  const boundaryBreaks = new Map();
  const addBreak = (y, x) => {
    let values = boundaryBreaks.get(y);
    if (!values) boundaryBreaks.set(y, values = new Set());
    const oldSize = values.size; values.add(x);
    return values.size !== oldSize;
  };
  for (const rectangle of rectangles) {
    addBreak(rectangle.y0, rectangle.x0); addBreak(rectangle.y0, rectangle.x1);
    addBreak(rectangle.y1, rectangle.x0); addBreak(rectangle.y1, rectangle.x1);
  }
  // A split introduced at one end of a rectangle has to continue to its
  // other end.  Propagate until stable so a chain of widening/narrowing runs
  // cannot leave a T-junction one row farther away.
  let changed = true, propagatedBreaks = 0, propagationPasses = 0, propagationWork = 0;
  while (changed) {
    if (++propagationPasses > 256) return null;
    changed = false;
    for (const rectangle of rectangles) {
      const candidates = new Set();
      for (const y of [rectangle.y0, rectangle.y1]) for (const x of boundaryBreaks.get(y) || []) {
        if (++propagationWork > (options.triangleLimit ?? FONT_MASK_TRIANGLE_LIMIT) * 128) return null;
        if (x > rectangle.x0 && x < rectangle.x1) candidates.add(x);
      }
      for (const x of candidates) {
        if (addBreak(rectangle.y0, x)) changed = true, propagatedBreaks++;
        if (addBreak(rectangle.y1, x)) changed = true, propagatedBreaks++;
        if (propagatedBreaks > (options.triangleLimit ?? FONT_MASK_TRIANGLE_LIMIT) * 2) return null;
      }
    }
  }
  for (const [y, values] of boundaryBreaks) {
    boundaryBreaks.set(y, Array.from(values).sort((a, b) => a - b));
  }

  let frontTriangleCount = 0;
  for (const rectangle of rectangles) {
    const splitSet = new Set([rectangle.x0, rectangle.x1]);
    for (const y of [rectangle.y0, rectangle.y1]) {
      for (const x of boundaryBreaks.get(y) || []) if (x > rectangle.x0 && x < rectangle.x1) splitSet.add(x);
    }
    rectangle.splits = Array.from(splitSet).sort((a, b) => a - b);
    frontTriangleCount += (rectangle.splits.length - 1) * 2;
    if (frontTriangleCount > (options.triangleLimit ?? FONT_MASK_TRIANGLE_LIMIT)) return null;
  }

  const mesh = new MinMesh();
  const scale = Number.isFinite(options.scale) ? options.scale : 1;
  const offsetX = Number.isFinite(options.offsetX) ? options.offsetX : 0;
  const baseline = Number.isFinite(options.baseline) ? options.baseline : height;
  const cornerInset = Math.min(0.25, Math.max(0, Number.isFinite(options.cornerInset) ? options.cornerInset : 0.125));
  depth = Number.isFinite(depth) ? depth : 0;
  const frontVertices = new Map();
  const ambiguousCorner = (x, y) => {
    const topLeft = x > 0 && y > 0 && Boolean(mask[(y - 1) * width + x - 1]);
    const topRight = x < width && y > 0 && Boolean(mask[(y - 1) * width + x]);
    const bottomLeft = x > 0 && y < height && Boolean(mask[y * width + x - 1]);
    const bottomRight = x < width && y < height && Boolean(mask[y * width + x]);
    return topLeft && bottomRight && !topRight && !bottomLeft ||
      topRight && bottomLeft && !topLeft && !bottomRight;
  };
  // sector is the filled quadrant adjacent to this rectangle corner:
  // southeast, southwest, northwest, northeast respectively.
  const sectorOffsets = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
  const frontVertex = (x, y, sector) => {
    const separate = cornerInset > 0 && ambiguousCorner(x, y);
    const key = separate ? `${x},${y},${sector}` : `${x},${y}`;
    let index = frontVertices.get(key);
    if (index !== undefined) return index;
    const vertex = makeVertex();
    const offset = separate ? sectorOffsets[sector] : [0, 0];
    // A binary threshold cannot decide how two diagonal pixels connected in
    // the original vector outline.  Insetting both singular corners by 1/8px
    // keeps the components visibly unchanged while avoiding a coincident
    // vertical extrusion edge with four incident faces.
    const px = (x + offset[0] * cornerInset + offsetX) * scale;
    const py = (baseline - y - offset[1] * cornerInset) * scale;
    vertex.position.set([px, py, 0]); vertex.uv[0].set([px, py]);
    index = mesh.vertices.length; mesh.vertices.push(vertex); frontVertices.set(key, index);
    return index;
  };
  for (const rectangle of rectangles) for (let split = 1; split < rectangle.splits.length; split++) {
    const x0 = rectangle.splits[split - 1], x1 = rectangle.splits[split];
    const topLeft = frontVertex(x0, rectangle.y0, 0), topRight = frontVertex(x1, rectangle.y0, 1);
    const bottomRight = frontVertex(x1, rectangle.y1, 2), bottomLeft = frontVertex(x0, rectangle.y1, 3);
    // Canvas Y is reflected into the glyph's Y-up coordinates, so this order
    // is clockwise and faces -Z just like gluTessNormal(0,0,-1).
    mesh.faces.push(makeFace([topLeft, topRight, bottomRight], 1));
    mesh.faces.push(makeFace([topLeft, bottomRight, bottomLeft], 1));
  }

  if (depth !== 0 && mesh.faces.length) {
    const frontVertexCount = mesh.vertices.length, frontFaceCount = mesh.faces.length;
    for (let index = 0; index < frontVertexCount; index++) {
      const vertex = makeVertex(mesh.vertices[index]); vertex.position[2] = depth; mesh.vertices.push(vertex);
    }
    for (let index = 0; index < frontFaceCount; index++) {
      const face = mesh.faces[index];
      mesh.faces.push(makeFace([
        face.vertices[2] + frontVertexCount,
        face.vertices[1] + frontVertexCount,
        face.vertices[0] + frontVertexCount,
      ], 1));
    }

    // Find the true boundary of the shared front triangulation.  Side walls
    // are generated only there, so holes are open and no internal rectangle
    // seams become hidden prisms.
    const edges = new Map();
    for (let index = 0; index < frontFaceCount; index++) {
      const vertices = mesh.faces[index].vertices;
      for (let edge = 0; edge < 3; edge++) {
        const a = vertices[edge], b = vertices[(edge + 1) % 3];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        const record = edges.get(key);
        if (record) record.count++;
        else edges.set(key, { a, b, count: 1 });
      }
    }
    const sideVertices = new Map();
    const sidePair = frontIndex => {
      let pair = sideVertices.get(frontIndex);
      if (pair) return pair;
      const front = makeVertex(mesh.vertices[frontIndex]);
      const back = makeVertex(mesh.vertices[frontIndex + frontVertexCount]);
      pair = [mesh.vertices.length, mesh.vertices.length + 1];
      mesh.vertices.push(front, back); sideVertices.set(frontIndex, pair);
      return pair;
    };
    for (const record of edges.values()) {
      if (record.count === 2) continue;
      if (record.count !== 1) return null;
      const a = sidePair(record.a), b = sidePair(record.b);
      // Reverse the directed front boundary edge.  With CW front faces this
      // points outer walls away from the solid and hole walls into the hole.
      mesh.faces.push(makeFace([b[0], a[0], a[1], b[1]], 1));
    }
  }
  return mesh.invalidate();
}

// Douglas-Peucker on a closed mask contour. Font3D's native quadratic
// flattener stops when the curve midpoint is within half of its authored
// control-point tolerance. Keeping the same geometric error here collapses
// raster stair steps into smooth polygon edges without inventing extra detail.
function fontPointSegmentDistanceSq(point, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (!lengthSq) {
    const px = point[0] - a[0], py = point[1] - a[1];
    return px * px + py * py;
  }
  const amount = Math.max(0, Math.min(1,
    ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSq));
  const px = point[0] - (a[0] + dx * amount);
  const py = point[1] - (a[1] + dy * amount);
  return px * px + py * py;
}

function simplifyOpenFontContour(points, toleranceSq) {
  if (points.length <= 2) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop();
    let farthest = -1, distanceSq = toleranceSq;
    for (let index = first + 1; index < last; index++) {
      const candidate = fontPointSegmentDistanceSq(points[index], points[first], points[last]);
      if (candidate > distanceSq) { distanceSq = candidate; farthest = index; }
    }
    if (farthest >= 0) {
      keep[farthest] = 1;
      stack.push([first, farthest], [farthest, last]);
    }
  }
  return points.filter((_, index) => keep[index]);
}

function simplifyClosedFontContour(contour, tolerance = 0) {
  const source = simplifyFontContour(contour || []);
  if (source.length <= 3 || !(tolerance > 0)) return source;
  let anchor = 0;
  for (let index = 1; index < source.length; index++) {
    if (source[index][0] < source[anchor][0] ||
        source[index][0] === source[anchor][0] && source[index][1] < source[anchor][1]) {
      anchor = index;
    }
  }
  const rotated = source.slice(anchor).concat(source.slice(0, anchor));
  let opposite = 1, farthestSq = -1;
  for (let index = 1; index < rotated.length; index++) {
    const dx = rotated[index][0] - rotated[0][0];
    const dy = rotated[index][1] - rotated[0][1];
    const distanceSq = dx * dx + dy * dy;
    if (distanceSq > farthestSq) { farthestSq = distanceSq; opposite = index; }
  }
  if (opposite <= 0 || opposite >= rotated.length) return source;
  const first = simplifyOpenFontContour(rotated.slice(0, opposite + 1), tolerance * tolerance);
  const second = simplifyOpenFontContour(
    rotated.slice(opposite).concat([rotated[0]]), tolerance * tolerance,
  );
  const result = simplifyFontContour(first.slice(0, -1).concat(second.slice(0, -1)));
  if (result.length < 3 || !fontContourArea(result) ||
      Math.sign(fontContourArea(result)) !== Math.sign(fontContourArea(source))) return source;
  return result;
}

function insetSharedFontContourCorners(contours, inset = 0.125) {
  if (!(inset > 0)) return contours.map(contour => contour.map(point => point.slice()));
  const occurrences = new Map();
  for (const contour of contours) for (const point of contour) {
    const key = `${point[0]},${point[1]}`;
    occurrences.set(key, (occurrences.get(key) || 0) + 1);
  }
  const unit = (dx, dy) => {
    const length = Math.hypot(dx, dy);
    return length ? [dx / length, dy / length] : [0, 0];
  };
  return contours.map(contour => contour.map((point, index) => {
    if ((occurrences.get(`${point[0]},${point[1]}`) || 0) < 2) return point.slice();
    const previous = contour[(index + contour.length - 1) % contour.length];
    const next = contour[(index + 1) % contour.length];
    const incoming = unit(point[0] - previous[0], point[1] - previous[1]);
    const outgoing = unit(next[0] - point[0], next[1] - point[1]);
    // fontMaskContours directs every edge with filled pixels on its visual
    // right. Canvas Y grows downward, so (-dy, dx) is the inward normal.
    let ix = -incoming[1] - outgoing[1];
    let iy = incoming[0] + outgoing[0];
    const length = Math.hypot(ix, iy);
    if (!length) return point.slice();
    ix *= inset / length; iy *= inset / length;
    return [point[0] + ix, point[1] + iy];
  }));
}

function fontSegmentsIntersect(a, b, c, d, epsilon = 1e-9) {
  const cross = (p, q, r) =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const onSegment = (p, q, r) => Math.abs(cross(p, q, r)) <= epsilon &&
    r[0] >= Math.min(p[0], q[0]) - epsilon && r[0] <= Math.max(p[0], q[0]) + epsilon &&
    r[1] >= Math.min(p[1], q[1]) - epsilon && r[1] <= Math.max(p[1], q[1]) + epsilon;
  const abC = cross(a, b, c), abD = cross(a, b, d);
  const cdA = cross(c, d, a), cdB = cross(c, d, b);
  if ((abC > epsilon && abD < -epsilon || abC < -epsilon && abD > epsilon) &&
      (cdA > epsilon && cdB < -epsilon || cdA < -epsilon && cdB > epsilon)) return true;
  return onSegment(a, b, c) || onSegment(a, b, d) ||
    onSegment(c, d, a) || onSegment(c, d, b);
}

function fontContoursAreSimple(contours) {
  const edges = [];
  for (let contourIndex = 0; contourIndex < contours.length; contourIndex++) {
    const contour = contours[contourIndex];
    if (contour.length < 3 || !fontContourArea(contour)) return false;
    const points = new Set();
    for (const point of contour) {
      const key = `${point[0]},${point[1]}`;
      if (points.has(key)) return false;
      points.add(key);
    }
    for (let index = 0; index < contour.length; index++) {
      edges.push({ contourIndex, index, count: contour.length,
        a: contour[index], b: contour[(index + 1) % contour.length] });
    }
  }
  for (let first = 0; first < edges.length; first++) for (let second = first + 1;
    second < edges.length; second++) {
    const a = edges[first], b = edges[second];
    if (a.contourIndex === b.contourIndex) {
      const adjacent = a.index === b.index ||
        (a.index + 1) % a.count === b.index || (b.index + 1) % b.count === a.index;
      if (adjacent) continue;
    }
    if (fontSegmentsIntersect(a.a, a.b, b.a, b.b)) return false;
  }
  return true;
}

// GLU's edge-flag callback forces triangle output instead of strips/fans. The
// released Font3D operator registers that callback even though its body is a
// no-op, so retain the same detail here. libtess.js is a direct JavaScript port
// of the SGI GLU tessellator and preserves its face traversal order.
function createFontGLUTessellator(mesh, triangleLimit = FONT_MASK_TRIANGLE_LIMIT) {
  if (!libtess?.GluTesselator || !libtess.gluEnum || !libtess.primitiveType) return null;
  triangleLimit = Math.max(1, Math.floor(triangleLimit));
  const tessellator = new libtess.GluTesselator();
  const enumeration = libtess.gluEnum;
  tessellator.gluTessCallback(enumeration.GLU_TESS_BEGIN_DATA, (type, state) => {
    state.pending.length = 0;
    if (type !== libtess.primitiveType.GL_TRIANGLES) state.failed = true;
  });
  tessellator.gluTessCallback(enumeration.GLU_TESS_VERTEX_DATA, (index, state) => {
    if (!Number.isInteger(index) || index < 0 || index >= mesh.vertices.length) {
      state.failed = true; return;
    }
    state.pending.push(index);
    if (state.pending.length !== 3) return;
    if (mesh.faces.length >= triangleLimit) state.failed = true;
    else mesh.faces.push(makeFace(state.pending, 1));
    state.pending = [];
  });
  tessellator.gluTessCallback(enumeration.GLU_TESS_END_DATA, state => {
    if (state.pending.length) state.failed = true;
    state.pending.length = 0;
  });
  tessellator.gluTessCallback(enumeration.GLU_TESS_ERROR_DATA, (_error, state) => {
    state.failed = true;
  });
  tessellator.gluTessCallback(enumeration.GLU_TESS_COMBINE_DATA,
    (coordinates, _sources, _weights, state) => {
      const vertex = makeVertex();
      if (!coordinates || coordinates.length < 3 ||
          !coordinates.slice(0, 3).every(Number.isFinite)) state.failed = true;
      vertex.position.set([
        Number(coordinates?.[0]) || 0,
        Number(coordinates?.[1]) || 0,
        Number(coordinates?.[2]) || 0,
      ]);
      const index = mesh.vertices.length;
      mesh.vertices.push(vertex);
      return index;
    });
  tessellator.gluTessCallback(enumeration.GLU_TESS_EDGE_FLAG_DATA, () => {});
  tessellator.gluTessNormal(0, 0, -1);

  return {
    append(contours, options = null) {
      options ||= {};
      const scale = Number.isFinite(options.scale) ? options.scale : 1;
      const offsetX = Number.isFinite(options.offsetX) ? options.offsetX : 0;
      const offsetY = Number.isFinite(options.offsetY) ? options.offsetY : 0;
      const state = { failed: false, pending: [] };
      try {
        tessellator.gluTessBeginPolygon(state);
        for (const contour of contours || []) {
          if (!Array.isArray(contour) || contour.length < 3) continue;
          tessellator.gluTessBeginContour();
          for (const point of contour) {
            const x = offsetX + Number(point?.[0]) * scale;
            const y = offsetY + Number(point?.[1]) * scale;
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
              state.failed = true; continue;
            }
            const vertex = makeVertex();
            vertex.position.set([x, y, 0]);
            // font3DAddPoint initializes native outline normals to -Z. The
            // ordinary MinMesh normal pass later replaces them for playback.
            vertex.normal[2] = -1;
            const index = mesh.vertices.length;
            mesh.vertices.push(vertex);
            // font3DAddPoint stores the GenMinVector (sF32) first, then widens
            // that stored value to GLU's double coordinate. Never let a JS
            // double bypass the same Float32 boundary.
            tessellator.gluTessVertex([
              vertex.position[0], vertex.position[1], vertex.position[2],
            ], index);
          }
          tessellator.gluTessEndContour();
        }
        tessellator.gluTessEndPolygon();
      } catch (_) {
        state.failed = true;
      }
      return !state.failed;
    },
    destroy() { tessellator.gluDeleteTess?.(); },
  };
}

function appendFontGLUContours(mesh, contours, options = null, shared = null) {
  const tessellator = shared || createFontGLUTessellator(
    mesh, options?.triangleLimit ?? FONT_MASK_TRIANGLE_LIMIT,
  );
  if (!tessellator) return false;
  const result = tessellator.append(contours, options);
  if (!shared) tessellator.destroy();
  return result;
}

function calculateFontTriangleAdjacency(mesh, faceCount = mesh.faces.length) {
  const canonical = new Map();
  const remap = new Int32Array(mesh.vertices.length);
  for (let index = 0; index < mesh.vertices.length; index++) {
    const position = mesh.vertices[index].position;
    const key = `${position[0]},${position[1]},${position[2]}`;
    let first = canonical.get(key);
    if (first === undefined) canonical.set(key, first = index);
    remap[index] = first;
  }
  const edges = new Map();
  for (let faceIndex = 0; faceIndex < faceCount; faceIndex++) {
    const face = mesh.faces[faceIndex];
    if (face.count !== 3) return false;
    face.adjacent = [-1, -1, -1];
    for (let edge = 0; edge < 3; edge++) {
      let a = remap[face.vertices[edge]];
      let b = remap[face.vertices[(edge + 1) % 3]];
      if (a > b) [a, b] = [b, a];
      const key = `${a},${b}`;
      let records = edges.get(key);
      if (!records) edges.set(key, records = []);
      records.push((faceIndex << 3) | edge);
      if (records.length > 2) return false;
    }
  }
  for (const records of edges.values()) if (records.length === 2) {
    const first = records[0], second = records[1];
    mesh.faces[first >> 3].adjacent[first & 7] = second;
    mesh.faces[second >> 3].adjacent[second & 7] = first;
  }
  return true;
}

function flipFontTriangleEdge(mesh, faceIndex, edge) {
  const firstFace = mesh.faces[faceIndex];
  const opposite = firstFace?.adjacent?.[edge] ?? -1;
  if (opposite < 0) return false;
  const secondFaceIndex = opposite >> 3, secondEdge = opposite & 7;
  const secondFace = mesh.faces[secondFaceIndex];
  if (firstFace.count !== 3 || secondFace?.count !== 3) return false;
  const va = firstFace.vertices[edge];
  const vb = secondFace.vertices[(secondEdge + 2) % 3];
  const vc = secondFace.vertices[secondEdge];
  const vd = firstFace.vertices[(edge + 2) % 3];
  const outerAB = secondFace.adjacent[(secondEdge + 1) % 3];
  const outerBC = secondFace.adjacent[(secondEdge + 2) % 3];
  const outerCD = firstFace.adjacent[(edge + 1) % 3];
  const outerDA = firstFace.adjacent[(edge + 2) % 3];
  firstFace.vertices = [vb, vd, va]; firstFace.count = 3;
  secondFace.vertices = [vd, vb, vc]; secondFace.count = 3;
  firstFace.adjacent = [secondFaceIndex << 3, outerDA, outerAB];
  secondFace.adjacent = [faceIndex << 3, outerBC, outerCD];
  const redirect = (tag, replacement) => {
    if (tag >= 0) mesh.faces[tag >> 3].adjacent[tag & 7] = replacement;
  };
  redirect(outerAB, (faceIndex << 3) | 2);
  redirect(outerBC, (secondFaceIndex << 3) | 1);
  redirect(outerCD, (secondFaceIndex << 3) | 2);
  redirect(outerDA, (faceIndex << 3) | 1);
  return true;
}

// Match Font3D's post-GLU cleanup: build position-welded adjacency, then give
// every near-degenerate or reversed triangle one chance to flip its first
// internal edge. This is intentionally not a general Delaunay cleanup; face
// order and the single-pass behavior feed the authored Explode RNG.
function cleanFontGLUTriangles(mesh) {
  if (!calculateFontTriangleAdjacency(mesh)) return false;
  for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex++) {
    const face = mesh.faces[faceIndex];
    const p0 = mesh.vertices[face.vertices[0]].position;
    const p1 = mesh.vertices[face.vertices[1]].position;
    const p2 = mesh.vertices[face.vertices[2]].position;
    const d1x = f32(p1[0] - p0[0]), d1y = f32(p1[1] - p0[1]);
    const d2x = f32(p2[0] - p0[0]), d2y = f32(p2[1] - p0[1]);
    const normalZ = f32(f32(d2x * d1y) - f32(d2y * d1x));
    if (normalZ >= 1e-6) continue;
    for (let edge = 0; edge < 3; edge++) if (face.adjacent[edge] >= 0) {
      flipFontTriangleEdge(mesh, faceIndex, edge); break;
    }
  }
  return true;
}

function extrudeFontSurface(mesh, depth) {
  if (!depth || !mesh.faces.length) return true;
  const frontVertexCount = mesh.vertices.length, frontFaceCount = mesh.faces.length;
  for (let index = 0; index < frontVertexCount; index++) {
    const vertex = makeVertex(mesh.vertices[index]);
    vertex.position[2] = depth; mesh.vertices.push(vertex);
  }
  for (let index = 0; index < frontFaceCount; index++) {
    const face = mesh.faces[index];
    mesh.faces.push(makeFace([
      face.vertices[2] + frontVertexCount,
      face.vertices[1] + frontVertexCount,
      face.vertices[0] + frontVertexCount,
    ], 1));
  }
  // Native reserves two complete sharp-edge copies after the front/back
  // surfaces, even when a tessellated vertex is not on a boundary. Keep that
  // exact layout: Font3D output can feed topology-sensitive operators later.
  for (let index = 0; index < frontVertexCount * 2; index++) {
    mesh.vertices.push(makeVertex(mesh.vertices[index]));
  }
  // CalcAdjacency welds coincident positions before Font3D's cleanup pass.
  // Consult that retained adjacency rather than rebuilding it from raw ids.
  for (let faceIndex = 0; faceIndex < frontFaceCount; faceIndex++) {
    const face = mesh.faces[faceIndex];
    for (let edge = 0; edge < 3; edge++) {
      if ((face.adjacent?.[edge] ?? -1) !== -1) continue;
      const a = face.vertices[edge], b = face.vertices[(edge + 1) % 3];
      mesh.faces.push(makeFace([
        b + frontVertexCount * 2,
        a + frontVertexCount * 2,
        a + frontVertexCount * 3,
        b + frontVertexCount * 3,
      ], 1));
    }
  }
  return true;
}

function projectFontUVs(mesh) {
  // Font3D finishes with Transform(all, mtx, position, UV0), where mtx is
  // identity except for k.x=1. Geometry is unchanged; UV0 becomes (x+z,y).
  for (const vertex of mesh.vertices) {
    vertex.uv[0][0] = f32(vertex.position[0] + vertex.position[2]);
    vertex.uv[0][1] = vertex.position[1];
  }
}

function fontPolygonsToMinMesh(regions, depth = 0, options = null) {
  const mesh = new MinMesh();
  const contours = [];
  for (const region of regions || []) {
    if (Array.isArray(region?.outer)) contours.push(region.outer);
    for (const hole of region?.holes || []) if (Array.isArray(hole)) contours.push(hole);
  }
  if (!appendFontGLUContours(mesh, contours, options) ||
      !cleanFontGLUTriangles(mesh) || !extrudeFontSurface(mesh, depth)) return null;
  projectFontUVs(mesh);
  return mesh.invalidate();
}

function canvasForFont3D(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  return canvas;
}

const FONT3D_GLYPH_PIXEL_LIMIT = 1024 * 1024;
const FONT3D_GLYPH_CACHE_LIMIT = 128;
const FONT3D_GLYPH_CACHE_POINT_LIMIT = 32768;
const font3DGlyphCache = new Map();
let font3DGlyphCachePoints = 0;
const font3DVectorGlyphCache = new Map();
let font3DVectorGlyphCachePoints = 0;

function font3DFamily(font) {
  // CreateFontA receives the production names verbatim, including a few
  // accidental trailing periods. The expected GDI fallback resolves "arial."
  // to Arial; normalize only that authored alias and retain the Arial/Georgia
  // distinction.
  return String(font || 'Arial').trim().replace(/\.+$/, '') || 'Arial';
}

const FONT3D_POINTFX_SCALE = 65536;
const FONT3D_NATIVE_SCALE = 128;

function roundFont3DFixed(value) {
  return value < 0 ? Math.ceil(value - 0.5) : Math.floor(value + 0.5);
}

// GGO_UNHINTED returns FreeType/GDI native outline points on a 26.6 grid.
// Wine expands the six fractional bits into POINTFX 16.16 by repetition, just
// as Windows does, before the original operator divides the result by 128.
function font3DPointFXCoordinate(value, unitsPerEm, ppem) {
  if (!(unitsPerEm > 0) || !(ppem > 0)) return 0;
  const fixed26_6 = roundFont3DFixed(Number(value) * ppem * 64 / unitsPerEm);
  const integer = fixed26_6 >> 6;
  let fraction = (fixed26_6 & 0x3f) << 10;
  fraction |= (fraction >> 6) | (fraction >> 12);
  const full = integer * FONT3D_POINTFX_SCALE + fraction;
  const logical = f32(f32(full) / f32(FONT3D_POINTFX_SCALE));
  return f32(logical / f32(FONT3D_NATIVE_SCALE));
}

function font3DF32Point(x, y) { return [f32(x), f32(y)]; }

function font3DF32Lerp(a, b) {
  // kdoc selects x87 single precision before evaluating GenMinVector::Lin3.
  const x = f32(a[0] + f32(f32(0.5) * f32(b[0] - a[0])));
  const y = f32(a[1] + f32(f32(0.5) * f32(b[1] - a[1])));
  return [x, y];
}

function font3DToleranceSq(height, maxError) {
  // Native uses the untruncated operator height here even though CreateFontA
  // receives trunc(height*128). Keep each assignment's sF32 rounding visible.
  const tolerance = f32(f32(f32(height) * f32(0.1)) * f32(maxError));
  return f32(tolerance * tolerance);
}

function appendFont3DQuadratic(output, start, control, end, toleranceSq, depth = 0) {
  const chordMiddle = font3DF32Lerp(start, end);
  const dx = f32(chordMiddle[0] - control[0]);
  const dy = f32(chordMiddle[1] - control[1]);
  const distanceSq = f32(f32(dx * dx) + f32(dy * dy));
  if (depth >= 12 || distanceSq <= toleranceSq) {
    if (output.length >= FONT3D_GLYPH_CACHE_POINT_LIMIT) return false;
    output.push(font3DF32Point(end[0], end[1]));
    return true;
  }
  const left = font3DF32Lerp(start, control);
  const right = font3DF32Lerp(control, end);
  const middle = font3DF32Lerp(left, right);
  return appendFont3DQuadratic(output, start, left, middle, toleranceSq, depth + 1) &&
    appendFont3DQuadratic(output, middle, right, end, toleranceSq, depth + 1);
}

// Decode a cyclic TrueType point stream and flatten its quadratic segments
// with the exact midpoint test used by font3DAddCurve in genminmesh.cpp.
function flattenFont3DContour(encoded, unitsPerEm, height, maxError, ppem = null) {
  if (!Array.isArray(encoded) || encoded.length < 9 || encoded.length % 3) return [];
  height = Math.max(0, f32(Number(height) || 0));
  const logicalHeight = Math.trunc(height * FONT3D_NATIVE_SCALE);
  const coordinatePpem = ppem == null ? logicalHeight : Number(ppem);
  const points = [];
  for (let index = 0; index < encoded.length; index += 3) {
    points.push([
      font3DPointFXCoordinate(encoded[index], unitsPerEm, coordinatePpem),
      font3DPointFXCoordinate(encoded[index + 1], unitsPerEm, coordinatePpem),
      Boolean(encoded[index + 2]),
    ]);
  }
  const first = points[0], last = points[points.length - 1];
  let start, sequence;
  if (first[2]) {
    start = first.slice(0, 2); sequence = points.slice(1);
  } else if (last[2]) {
    start = last.slice(0, 2); sequence = points.slice(0, -1);
  } else {
    start = font3DF32Lerp(first, last);
    sequence = points.slice();
  }
  sequence.push([start[0], start[1], true]);
  const contour = [font3DF32Point(start[0], start[1])];
  const toleranceSq = font3DToleranceSq(height, Math.max(0, Number(maxError) || 0));
  let current = start;
  for (let index = 0; index < sequence.length; index++) {
    const point = sequence[index];
    if (point[2]) {
      const end = point.slice(0, 2);
      if (end[0] !== current[0] || end[1] !== current[1]) contour.push(end);
      current = end;
      continue;
    }
    const following = sequence[index + 1];
    const end = following?.[2]
      ? following.slice(0, 2)
      : font3DF32Lerp(point, following);
    if (!appendFont3DQuadratic(contour, current, point, end, toleranceSq)) return null;
    current = end;
    if (following?.[2]) index++;
  }
  const final = contour[contour.length - 1];
  if (contour.length > 1 && final[0] === contour[0][0] && final[1] === contour[0][1]) contour.pop();
  // GGO_NATIVE submits every line endpoint and every recursively flattened
  // curve endpoint. Do not run the raster-mask collinearity simplifier over
  // this native-style vector stream; point order feeds GLU and Explode RNG.
  return contour;
}

function font3DVectorGlyph(family, height, maxError, character) {
  const familyKey = String(family || '').toLowerCase();
  const source = FONT3D_FAMILIES[familyKey];
  const record = source?.glyphs?.[character];
  if (!record) return null;
  height = Math.max(0, f32(Number(height) || 0));
  const key = `${familyKey}\u0000${f32(height)}\u0000${Number(maxError) || 0}\u0000${character}`;
  const cached = font3DVectorGlyphCache.get(key);
  if (cached) {
    font3DVectorGlyphCache.delete(key); font3DVectorGlyphCache.set(key, cached);
    return cached;
  }
  const logicalHeight = Math.trunc(height * FONT3D_NATIVE_SCALE);
  const selectedPpem = Number(source.ppemByLogicalHeight?.[logicalHeight]);
  const gdiSized = selectedPpem > 0;
  // The three authored family/height pairs use the exact integer ppem chosen
  // by Arial 2.82/Georgia 2.05's VDMX tables. Retain continuous cell-height
  // scaling only as a deterministic fallback for unauthored sizes.
  const coordinateUnits = gdiSized ? source.unitsPerEm : source.unitsPerCell;
  const coordinatePpem = gdiSized ? selectedPpem : logicalHeight;
  const contours = [];
  for (const contour of record[1]) {
    const flattened = flattenFont3DContour(
      contour, coordinateUnits, height, maxError, coordinatePpem,
    );
    if (!flattened) throw new Error(`Font3D deterministic ${familyKey} glyph exceeds curve limit`);
    if (flattened.length >= 3) contours.push(flattened);
  }
  const glyph = {
    advance: record[0] / source.unitsPerCell,
    advanceUnits: record[0],
    unitsPerEm: source.unitsPerEm,
    unitsPerCell: source.unitsPerCell,
    coordinateUnits,
    coordinatePpem,
    gdiSized,
    ppem: gdiSized ? selectedPpem : null,
    referenceBounds: source.referenceBounds?.[character] || null,
    contours,
    deterministic: true,
  };
  glyph.pointCount = contours.reduce((sum, contour) => sum + contour.length, 0);
  if (glyph.pointCount <= FONT3D_GLYPH_CACHE_POINT_LIMIT) {
    font3DVectorGlyphCache.set(key, glyph);
    font3DVectorGlyphCachePoints += glyph.pointCount;
  }
  while (font3DVectorGlyphCache.size > FONT3D_GLYPH_CACHE_LIMIT ||
      font3DVectorGlyphCachePoints > FONT3D_GLYPH_CACHE_POINT_LIMIT) {
    const oldestKey = font3DVectorGlyphCache.keys().next().value;
    const oldest = font3DVectorGlyphCache.get(oldestKey);
    font3DVectorGlyphCache.delete(oldestKey);
    font3DVectorGlyphCachePoints -= oldest?.pointCount || 0;
  }
  return glyph;
}

function font3DGlyphAdvance(glyph, height) {
  height = Math.max(0, f32(Number(height) || 0));
  if (glyph?.deterministic) {
    // GLYPHMETRICS.gmCellIncX is an integer logical-pixel value. The native
    // operator creates a height*128 font and divides the accumulated xPos by
    // 128, so retain that 1/128-unit quantization instead of accumulating
    // floating Canvas widths.
    if (glyph.gdiSized) {
      // FreeType first rounds the unhinted design advance onto its 26.6 grid;
      // GetGlyphOutline then rounds that positive value up to gmCellIncX.
      const advance26_6 = roundFont3DFixed(
        glyph.advanceUnits * glyph.ppem * 64 / glyph.unitsPerEm,
      );
      return Math.ceil(advance26_6 / 64) / FONT3D_NATIVE_SCALE;
    }
    const logicalHeight = Math.trunc(height * FONT3D_NATIVE_SCALE);
    return Math.round(glyph.advanceUnits * logicalHeight / glyph.unitsPerCell) /
      FONT3D_NATIVE_SCALE;
  }
  return (Number(glyph?.advance) || 0) * height;
}

function fitFont3DGlyphVertexRange(mesh, fit) {
  const {
    firstVertex, endVertex, offsetX, referenceBounds, coordinateUnits, coordinatePpem,
  } = fit;
  if (!(endVertex > firstVertex) || !Array.isArray(referenceBounds) ||
      referenceBounds.length !== 4 || !(coordinateUnits > 0) || !(coordinatePpem > 0)) return false;
  let sourceMinX = Infinity, sourceMinY = Infinity;
  let sourceMaxX = -Infinity, sourceMaxY = -Infinity;
  for (let index = firstVertex; index < endVertex; index++) {
    const position = mesh.vertices[index].position;
    sourceMinX = Math.min(sourceMinX, position[0]);
    sourceMinY = Math.min(sourceMinY, position[1]);
    sourceMaxX = Math.max(sourceMaxX, position[0]);
    sourceMaxY = Math.max(sourceMaxY, position[1]);
  }
  if (!(sourceMaxX > sourceMinX) || !(sourceMaxY > sourceMinY)) return false;
  const targetMinX = f32(offsetX +
    font3DPointFXCoordinate(referenceBounds[0], coordinateUnits, coordinatePpem));
  const targetMinY = font3DPointFXCoordinate(referenceBounds[1], coordinateUnits, coordinatePpem);
  const targetMaxX = f32(offsetX +
    font3DPointFXCoordinate(referenceBounds[2], coordinateUnits, coordinatePpem));
  const targetMaxY = font3DPointFXCoordinate(referenceBounds[3], coordinateUnits, coordinatePpem);
  const mapCoordinate = (value, sourceMin, sourceMax, targetMin, targetMax) => {
    if (value === sourceMin) return targetMin;
    if (value === sourceMax) return targetMax;
    return f32(targetMin + (value - sourceMin) *
      (targetMax - targetMin) / (sourceMax - sourceMin));
  };
  for (let index = firstVertex; index < endVertex; index++) {
    const position = mesh.vertices[index].position;
    position[0] = mapCoordinate(position[0], sourceMinX, sourceMaxX, targetMinX, targetMaxX);
    position[1] = mapCoordinate(position[1], sourceMinY, sourceMaxY, targetMinY, targetMaxY);
  }
  return true;
}

function font3DResolution(maxError) {
  // Oversample beyond the native quadratic midpoint error before applying
  // that error during contour simplification. Merely fitting a pixel's half
  // diagonal inside the tolerance (283 px/em for production) leaves weakly
  // collinear threshold staircases which can change sign after Float32
  // storage. At production's maxError ~= 0.05 this deliberately reaches the
  // bounded 512 px/em ceiling; simplification still sets the authored visible
  // detail while the denser raster makes its topology stable.
  const error = Math.max(1e-6, Number(maxError) || 0);
  return Math.max(128, Math.min(512, Math.ceil(Math.SQRT2 * 20 / error)));
}

function configureFont3DContext(context, cssFont) {
  context.font = cssFont;
  context.textAlign = 'left';
  context.textBaseline = 'alphabetic';
  context.fillStyle = '#fff';
  if ('fontKerning' in context) context.fontKerning = 'none';
  if ('textRendering' in context) context.textRendering = 'geometricPrecision';
}

function canvasFont3DGlyph(family, resolution, maxError, character) {
  const key = `${family}\u0000${resolution}\u0000${Number(maxError) || 0}\u0000${character}`;
  const cached = font3DGlyphCache.get(key);
  if (cached) {
    // Bounded LRU: production uses only a small ASCII subset, but a hostile
    // string must not pin 128 maximum-complexity glyph object graphs.
    font3DGlyphCache.delete(key); font3DGlyphCache.set(key, cached);
    return cached;
  }
  let canvas = canvasForFont3D(1, 1);
  let context = canvas?.getContext?.('2d', { willReadFrequently: true });
  if (!context) return null;
  const cssFont = `500 ${resolution}px "${family.replaceAll('"', '')}", Arial, sans-serif`;
  configureFont3DContext(context, cssFont);
  const metrics = context.measureText(character);
  const advance = Math.max(0, Number(metrics.width) || 0) / resolution;
  let glyph = { advance, contours: [] };
  if (!/\s/.test(character)) {
    const left = Math.max(0, Math.ceil(Number(metrics.actualBoundingBoxLeft) || 0));
    const right = Math.max(1, Math.ceil(Number(metrics.actualBoundingBoxRight) ||
      Number(metrics.width) || resolution * 0.6));
    const ascent = Math.max(1, Math.ceil(Number(metrics.actualBoundingBoxAscent) || resolution * 0.8));
    const descent = Math.max(1, Math.ceil(Number(metrics.actualBoundingBoxDescent) || resolution * 0.25));
    const padding = 3;
    const rasterWidth = Math.max(1, left + right + padding * 2);
    const rasterHeight = Math.max(1, ascent + descent + padding * 2);
    if (rasterWidth * rasterHeight > FONT3D_GLYPH_PIXEL_LIMIT) return null;
    canvas = canvasForFont3D(rasterWidth, rasterHeight);
    context = canvas?.getContext?.('2d', { willReadFrequently: true });
    if (!context) return null;
    configureFont3DContext(context, cssFont);
    context.clearRect(0, 0, rasterWidth, rasterHeight);
    const originX = padding + left, baseline = padding + ascent;
    context.fillText(character, originX, baseline);
    const pixels = context.getImageData(0, 0, rasterWidth, rasterHeight).data;
    const mask = new Uint8Array(rasterWidth * rasterHeight);
    for (let index = 0, pixel = 0; index < mask.length; index++, pixel += 4) {
      mask[index] = pixels[pixel + 3] >= 128 ? 1 : 0;
    }
    const tracedContours = fontMaskContours(mask, rasterWidth, rasterHeight);
    if (!tracedContours) return null;
    const contours = insetSharedFontContourCorners(tracedContours);
    let tolerancePixels = resolution * 0.05 * Math.max(0, Number(maxError) || 0);
    let simplified = null;
    for (let attempt = 0; attempt < 8; attempt++) {
      const candidate = contours.map(contour =>
        simplifyClosedFontContour(contour, tolerancePixels)).filter(contour => contour.length >= 3);
      if (fontContoursAreSimple(candidate)) { simplified = candidate; break; }
      tolerancePixels *= 0.5;
    }
    if (!simplified) simplified = contours;
    // Validate contour nesting even though GLU itself applies odd winding.
    // An orphaned hole here signals a tracing error rather than useful input.
    if (!fontMaskPolygons(simplified)) return null;
    const normalizeRing = ring => ring.map(point => [
      (point[0] - originX) / resolution,
      (baseline - point[1]) / resolution,
    ]);
    glyph = {
      advance,
      // The native operator submits every glyph contour directly to one GLU
      // polygon. Preserve that contour stream instead of pre-bridging holes
      // or independently triangulating classified regions.
      contours: simplified.map(normalizeRing),
    };
  }
  const pointCount = glyph.contours.reduce((sum, contour) => sum + contour.length, 0);
  glyph.pointCount = pointCount;
  if (pointCount <= FONT3D_GLYPH_CACHE_POINT_LIMIT) {
    font3DGlyphCache.set(key, glyph); font3DGlyphCachePoints += pointCount;
  }
  while (font3DGlyphCache.size > FONT3D_GLYPH_CACHE_LIMIT ||
      font3DGlyphCachePoints > FONT3D_GLYPH_CACHE_POINT_LIMIT) {
    const oldestKey = font3DGlyphCache.keys().next().value;
    const oldest = font3DGlyphCache.get(oldestKey);
    font3DGlyphCache.delete(oldestKey);
    font3DGlyphCachePoints -= oldest?.pointCount || 0;
  }
  return glyph;
}

function canvasFont3D(height, extrude, maxError, text, font) {
  if (!libtess?.GluTesselator) return null;
  const family = font3DFamily(font), resolution = font3DResolution(maxError);
  const deterministicFamily = FONT3D_FAMILIES[family.toLowerCase()] || null;
  const mesh = new MinMesh();
  const deterministicFits = [];
  const tessellator = createFontGLUTessellator(mesh, FONT_MASK_TRIANGLE_LIMIT);
  if (!tessellator) return null;
  let x = 0;
  for (const character of text) {
    // Every glyph authored by Debris resolves here without Canvas. Arimo and
    // Gelasio are OFL substitutes with the exact Arial/Georgia advances; the
    // source outlines are flattened directly, so browser font availability,
    // hinting, and rasterization cannot change the production geometry.
    const glyph = font3DVectorGlyph(family, height, maxError, character);
    if (!glyph && deterministicFamily) {
      tessellator.destroy();
      throw new Error(`Font3D deterministic ${family} subset has no glyph ${JSON.stringify(character)}`);
    }
    const resolvedGlyph = glyph || canvasFont3DGlyph(family, resolution, maxError, character);
    if (!resolvedGlyph) { tessellator.destroy(); return null; }
    const firstVertex = mesh.vertices.length;
    if (resolvedGlyph.contours.length && !appendFontGLUContours(mesh, resolvedGlyph.contours, {
      scale: resolvedGlyph.deterministic ? 1 : height,
      offsetX: x, triangleLimit: FONT_MASK_TRIANGLE_LIMIT,
    }, tessellator)) {
      tessellator.destroy(); return null;
    }
    if (resolvedGlyph.deterministic && resolvedGlyph.contours.length) {
      deterministicFits.push({
        firstVertex,
        endVertex: mesh.vertices.length,
        offsetX: x,
        referenceBounds: resolvedGlyph.referenceBounds,
        coordinateUnits: resolvedGlyph.coordinateUnits,
        coordinatePpem: resolvedGlyph.coordinatePpem,
      });
    }
    x += font3DGlyphAdvance(resolvedGlyph, height);
  }
  tessellator.destroy();
  if (!cleanFontGLUTriangles(mesh)) return null;
  // Bounds-only affine fitting retains the libre contours while matching the
  // Arial/Georgia bearings and visible size used by the Windows production.
  // Keep this after GLU and the native edge-flip pass so vertex/face ordering
  // (and therefore Explode's RNG assignment) remains bit-identical.
  for (const fit of deterministicFits) if (!fitFont3DGlyphVertexRange(mesh, fit)) return null;
  if (!extrudeFontSurface(mesh, extrude)) return null;
  projectFontUVs(mesh);
  return mesh.invalidate();
}

function MinMesh_Font3D(height, extrude, maxError, text, font) {
  height = Math.max(height, 8 / 128); text = String(text || ''); font = String(font || 'sans-serif');
  const adapter = fontAdapter;
  if (adapter) {
    const result = adapter({ height, extrude, maxError, text, font, MinMesh, makeVertex, makeFace });
    if (result instanceof MinMesh) return result;
  }
  const canvasResult = canvasFont3D(height, extrude, maxError, text, font);
  if (canvasResult) return canvasResult;
  throw new Error('Font3D requires Canvas outline extraction or an explicit font adapter');
}

function pipeKeyMatrix(key, out = new Float32Array(16)) {
  // BlobSpline mode 4 stores a quaternion as Zoom,w followed by rx/ry/rz.
  // MinMesh_Pipe converts each authored key directly instead of evaluating
  // the Hermite spline, so do not renormalize or interpolate it here.
  const w = key.zoom, x = key.rx, y = key.ry, z = key.rz;
  const xx = f32(2 * x * x), xy = f32(2 * x * y), xz = f32(2 * x * z);
  const yy = f32(2 * y * y), yz = f32(2 * y * z), zz = f32(2 * z * z);
  const xw = f32(2 * x * w), yw = f32(2 * y * w), zw = f32(2 * z * w);
  out[0] = f32(1 - yy - zz); out[1] = f32(xy - zw); out[2] = f32(xz + yw); out[3] = 0;
  out[4] = f32(xy + zw); out[5] = f32(1 - xx - zz); out[6] = f32(yz - xw); out[7] = 0;
  out[8] = f32(xz - yw); out[9] = f32(yz + xw); out[10] = f32(1 - xx - yy); out[11] = 0;
  out[12] = key.px; out[13] = key.py; out[14] = key.pz; out[15] = 1;
  return out;
}

function pipeAxisAngleMatrix(unitAxis, angle, out = new Float32Array(16)) {
  // sMatrix::InitRot uses scalar sin/cos after UnitSafe3 normalization.
  const cosine = f32(Math.cos(angle)), sine = f32(Math.sin(angle));
  const x = unitAxis[0], y = unitAxis[1], z = unitAxis[2];
  out[0] = f32(x * x + cosine * (1 - x * x));
  out[1] = f32(x * y + cosine * (-x * y) + sine * z);
  out[2] = f32(x * z + cosine * (-x * z) - sine * y); out[3] = 0;
  out[4] = f32(y * x + cosine * (-y * x) - sine * z);
  out[5] = f32(y * y + cosine * (1 - y * y));
  out[6] = f32(y * z + cosine * (-y * z) + sine * x); out[7] = 0;
  out[8] = f32(z * x + cosine * (-z * x) + sine * y);
  out[9] = f32(z * y + cosine * (-z * y) - sine * x);
  out[10] = f32(z * z + cosine * (1 - z * z)); out[11] = 0;
  out[12] = out[13] = out[14] = 0; out[15] = 1;
  return out;
}

function addPipeStraight(output, source, matrix, difference, zMinimum, zRange,
  copy, copies, flags, textureZoom, distance, absoluteDistance) {
  const first = output.vertices.length; output.add(source);
  for (let index = first; index < output.vertices.length; index++) {
    const vertex = output.vertices[index];
    let fraction = f32((vertex.position[2] - zMinimum) / zRange);
    fraction = f32((fraction + copy) / copies);
    vertex.position[2] = 0;
    transformPoint(matrix, vertex.position, vertex.position);
    add3(vertex.position, difference, fraction);
    if (flags & 1) {
      vertex.uv[0][1] = f32(fraction * textureZoom * distance + absoluteDistance);
    }
  }
}

function addPipeCurve(output, source, previousMatrix, center, axis, gamma,
  zMinimum, zRange, flags, textureZoom, distance, absoluteDistance) {
  const first = output.vertices.length; output.add(source);
  const rotation = new Float32Array(16);
  const unitAxis = normalize3(axis, vector3());
  for (let index = first; index < output.vertices.length; index++) {
    const vertex = output.vertices[index];
    const fraction = f32((vertex.position[2] - zMinimum) / zRange);
    pipeAxisAngleMatrix(unitAxis, f32(-fraction * (Math.PI - gamma)), rotation);
    vertex.position[2] = 0;
    transformPoint(previousMatrix, vertex.position, vertex.position);
    sub3(vertex.position, center, vertex.position);
    transformPoint(rotation, vertex.position, vertex.position);
    add3(vertex.position, center);
    if ((flags & 17) === 17) {
      vertex.uv[0][1] = f32(fraction * textureZoom * distance + absoluteDistance);
    }
  }
}

function addPipeTransformed(output, source, matrix) {
  const first = output.vertices.length; output.add(source);
  for (let index = first; index < output.vertices.length; index++) {
    transformPoint(matrix, output.vertices[index].position, output.vertices[index].position);
  }
}

function MinMesh_Pipe(spline, mesh0, mesh1, mesh2, flags, textureZoom, ringDistance, objectDistance) {
  const pipe = spline?.pipe;
  if (spline?.mode !== 4 || !pipe || !(mesh0 instanceof MinMesh)) return null;
  const keys = spline.keys || [], segmentCount = pipe.count | 0;
  if (segmentCount < 1 || keys.length !== segmentCount * 2 ||
      !Array.isArray(pipe.keys) || pipe.keys.length !== segmentCount) return null;
  const output = new MinMesh();
  const mesh0Bounds = mesh0.bounds();
  const mesh0ZMinimum = mesh0Bounds.minimum[2];
  const mesh0ZRange = f32(mesh0Bounds.maximum[2] - mesh0ZMinimum);
  const curveSource = mesh1 instanceof MinMesh ? mesh1 : mesh0;
  const curveBounds = curveSource.bounds();
  const curveZMinimum = curveBounds.minimum[2];
  const curveZRange = f32(curveBounds.maximum[2] - curveZMinimum);
  let absoluteDistance = 0, previousMatrix = mat4Identity();
  for (let segment = 0; segment < segmentCount; segment++) {
    if (segment > 0) {
      const corner = pipe.keys[segment - 1];
      const previous = keys[segment * 2 - 1], current = keys[segment * 2];
      const cornerPosition = vector3(corner.x, corner.y, corner.z);
      const direction0 = normalize3(sub3(
        vector3(previous.px, previous.py, previous.pz), cornerPosition,
      ));
      const direction1 = normalize3(sub3(
        vector3(current.px, current.py, current.pz), cornerPosition,
      ));
      const axis = cross3(direction0, direction1);
      const gamma = f32(Math.acos(dot3(direction0, direction1)));
      const middleDirection = normalize3(add3(cloneVector(direction0), direction1));
      const center = cloneVector(cornerPosition);
      add3(center, middleDirection, f32(corner.radius / Math.cos(gamma / 2)));
      const curveDistance = f32(corner.radius * gamma);
      addPipeCurve(output, curveSource, previousMatrix, center, axis, gamma,
        curveZMinimum, curveZRange, flags, textureZoom, curveDistance, absoluteDistance);
      if ((flags & 24) === 24) {
        absoluteDistance = f32(absoluteDistance + curveDistance * textureZoom);
      }
    }
    const firstMatrix = pipeKeyMatrix(keys[segment * 2]);
    const secondMatrix = pipeKeyMatrix(keys[segment * 2 + 1]);
    const difference = sub3(
      vector3(secondMatrix[12], secondMatrix[13], secondMatrix[14]),
      vector3(firstMatrix[12], firstMatrix[13], firstMatrix[14]),
    );
    const distance = f32(Math.sqrt(dot3(difference, difference)));
    const copies = flags & 4 ? Math.max(1, Math.min(64, Math.trunc(distance / objectDistance + 0.5))) : 1;
    for (let copy = 0; copy < copies; copy++) {
      addPipeStraight(output, mesh0, firstMatrix, difference, mesh0ZMinimum, mesh0ZRange,
        copy, copies, flags, textureZoom, distance, absoluteDistance);
    }
    if (mesh2 instanceof MinMesh) {
      addPipeTransformed(output, mesh2, firstMatrix);
      addPipeTransformed(output, mesh2, secondMatrix);
      if ((flags & 2) && ringDistance > 0.1) {
        const rings = Math.trunc(distance / ringDistance - 0.5);
        for (let ring = 0; ring < rings; ring++) {
          const matrix = new Float32Array(firstMatrix);
          const phase = f32((ring + 1) / (rings + 1));
          matrix[12] = f32(matrix[12] + difference[0] * phase);
          matrix[13] = f32(matrix[13] + difference[1] * phase);
          matrix[14] = f32(matrix[14] + difference[2] * phase);
          addPipeTransformed(output, mesh2, matrix);
        }
      }
    }
    if (flags & 8) absoluteDistance = f32(absoluteDistance + distance * textureZoom);
    previousMatrix = secondMatrix;
  }
  return output.mergeClusters();
}

function meshToMinCompact(result, view, oldMeshHasColor, oldMeshHasUV0, oldMeshHasUV1,
  packOldColorComponents) {
  const storage = view.storage;
  const vertexMap = new Int32Array(storage.vertexCount);
  vertexMap.fill(-1);

  const physicalVertex = oldIndex => {
    const first = storage.vertexInts[oldIndex * 5 + 1];
    return first >= 0 && first < storage.vertexCount ? first : oldIndex;
  };
  const addOldVertex = oldIndex => {
    const mapped = vertexMap[oldIndex];
    if (mapped >= 0) return mapped;
    const vertex = makeVertex();
    const positionOffset = physicalVertex(oldIndex) * 4;
    vertex.position[0] = f32(storage.vertexPositions[positionOffset]);
    vertex.position[1] = f32(storage.vertexPositions[positionOffset + 1]);
    vertex.position[2] = f32(storage.vertexPositions[positionOffset + 2]);
    const attributeOffset = oldIndex * 4;
    if (oldMeshHasUV0 && storage.vertexUVs) {
      vertex.uv[0][0] = storage.vertexUVs[attributeOffset];
      vertex.uv[0][1] = storage.vertexUVs[attributeOffset + 1];
    }
    if (oldMeshHasUV1 && storage.vertexUV1s) {
      vertex.uv[1][0] = storage.vertexUV1s[attributeOffset];
      vertex.uv[1][1] = storage.vertexUV1s[attributeOffset + 1];
    }
    const colors = storage.vertexColors;
    vertex.color = oldMeshHasColor
      ? packOldColorComponents(
        colors ? colors[attributeOffset] : 0,
        colors ? colors[attributeOffset + 1] : 0,
        colors ? colors[attributeOffset + 2] : 0,
        colors ? colors[attributeOffset + 3] : 0,
      )
      : 0xffffffff;
    const index = result.vertices.push(vertex) - 1;
    vertexMap[oldIndex] = index;
    return index;
  };

  const addCentroid = oldIndices => {
    const position = new Float32Array(3);
    const color = new Float32Array(4);
    const uv = new Float32Array(2);
    const uv1 = new Float32Array(2);
    const inverse = f32(1 / oldIndices.length);
    for (const oldIndex of oldIndices) {
      const positionOffset = physicalVertex(oldIndex) * 4;
      for (let component = 0; component < 3; component++) {
        position[component] = f32(
          position[component] + storage.vertexPositions[positionOffset + component] * inverse,
        );
      }
      const attributeOffset = oldIndex * 4;
      if (storage.vertexColors) {
        for (let component = 0; component < 4; component++) {
          color[component] = f32(
            color[component] + storage.vertexColors[attributeOffset + component] * inverse,
          );
        }
      }
      if (oldMeshHasUV0 && storage.vertexUVs) {
        for (let component = 0; component < 2; component++) {
          uv[component] = f32(uv[component] + storage.vertexUVs[attributeOffset + component] * inverse);
        }
      }
      if (oldMeshHasUV1 && storage.vertexUV1s) {
        for (let component = 0; component < 2; component++) {
          uv1[component] = f32(
            uv1[component] + storage.vertexUV1s[attributeOffset + component] * inverse,
          );
        }
      }
    }
    const vertex = makeVertex();
    copy3(vertex.position, position);
    vertex.color = oldMeshHasColor
      ? packOldColorComponents(color[0], color[1], color[2], color[3])
      : 0xffffffff;
    if (oldMeshHasUV0) vertex.uv[0].set(uv);
    if (oldMeshHasUV1) vertex.uv[1].set(uv1);
    const index = result.vertices.length;
    result.vertices.push(vertex);
    return index;
  };

  for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
    const material = storage.faceInts[faceIndex * 5] | 0;
    if (!material) continue;
    const start = storage.faceInts[faceIndex * 5 + 1];
    const oldIndices = [];
    if (start >= 0) {
      let halfedge = start;
      const limit = storage.edgeCount * 2 + 1;
      do {
        const edgeOffset = (halfedge >> 1) * 11;
        const side = halfedge & 1;
        oldIndices.push(storage.edgeInts[edgeOffset + 6 + side]);
        halfedge = storage.edgeInts[edgeOffset + side];
        if (oldIndices.length > limit) throw new Error('broken GenMesh face loop');
      } while (halfedge !== start);
    }
    if (oldIndices.length < 3) continue;
    const vertices = oldIndices.map(addOldVertex);
    const used = storage.faceBytes[faceIndex * 4 + 3] !== 0;
    if (vertices.length >= 8) {
      const centroid = addCentroid(oldIndices);
      for (let index = 0; index < vertices.length; index++) {
        const converted = makeFace([
          centroid, vertices[index], vertices[(index + 1) % vertices.length],
        ], material);
        converted.flags = used ? 0 : 1;
        result.faces.push(converted);
      }
    } else {
      const converted = makeFace(vertices, material);
      converted.flags = used ? 0 : 1;
      result.faces.push(converted);
    }
  }
  return result.invalidate();
}

function meshToMin(source) {
  if (!source) return null;
  if (source instanceof MinMesh) return source.clone();
  const result = new MinMesh();
  const slots = source.materials || source.Mtrl || [];
  if (slots.length) {
    result.clusters = Array.from(slots, (slot, index) => makeCluster(slot?.material || null, slot?.pass || 0, index));
    if (!result.clusters.length || result.clusters[0].material) result.clusters.unshift(makeCluster(null));
    result.Clusters = result.clusters;
  }
  // Mesh_ToMin uses white only when COLOR0 is absent from the old mesh's
  // vertex format. A present sVector is packed with GetColor(), including the
  // all-zero default created by GenMesh::Init. Treating that zero vector as a
  // missing color turns Material11's `Color0 SET + Vertex ADD` base phase
  // white before any lighting (most visibly on Debris' instanced particles).
  const oldMeshHasColor = source.vertexMask === undefined || Boolean(source.vertexMask & (1 << 3));
  const oldMeshHasUV0 = source.vertexMask === undefined || Boolean(source.vertexMask & (1 << 5));
  const oldMeshHasUV1 = source.vertexMask === undefined || Boolean(source.vertexMask & (1 << 6));
  const byte = component => Math.max(0, Math.min(255, Math.trunc(Number(component) * 255)));
  const packOldColorComponents = (red, green, blue, alpha) =>
    ((byte(alpha) << 24) | (byte(red) << 16) | (byte(green) << 8) | byte(blue)) >>> 0;
  const packOldColor = value => {
    if (typeof value === 'number') return value >>> 0;
    if (!oldMeshHasColor || !value || value.length < 4) return 0xffffffff;
    return packOldColorComponents(value[0], value[1], value[2], value[3]);
  };
  const compactView = source.compactMeshConversionView?.();
  if (compactView) {
    return meshToMinCompact(result, compactView, oldMeshHasColor, oldMeshHasUV0,
      oldMeshHasUV1, packOldColorComponents);
  }
  const oldVertices = source.vertices || source.Vert || [];
  const vertexMap = new Map();
  const addOldVertex = oldIndex => {
    if (vertexMap.has(oldIndex)) return vertexMap.get(oldIndex);
    const old = oldVertices[oldIndex], vertex = makeVertex();
    const first = Number.isInteger(old?.first) ? old.first | 0 : -1;
    const position = oldVertices[first]?.position || old?.position || old?.values?.[0] || [0, 0, 0];
    const uv = old?.uv || old?.values?.[4] || [0, 0];
    const uv1 = old?.uv1 || old?.values?.[5] || [0, 0];
    copy3(vertex.position, position);
    if (oldMeshHasUV0) vertex.uv[0].set(uv.subarray ? uv.subarray(0, 2) : uv.slice(0, 2));
    if (oldMeshHasUV1) vertex.uv[1].set(uv1.subarray ? uv1.subarray(0, 2) : uv1.slice(0, 2));
    vertex.color = packOldColor(old?.color);
    const index = result.vertices.push(vertex) - 1; vertexMap.set(oldIndex, index); return index;
  };
  const oldFaces = source.faces || source.Face || [];
  if (typeof source.faceEdges === 'function' && typeof source.getVertId === 'function') {
    const addCentroid = oldIndices => {
      const synthetic = {
        first: -1,
        values: Array.from({ length: 6 }, () => new Float32Array(4)),
      };
      synthetic.position = synthetic.values[0]; synthetic.normal = synthetic.values[1];
      synthetic.tangent = synthetic.values[2]; synthetic.color = synthetic.values[3];
      synthetic.uv = synthetic.values[4]; synthetic.uv1 = synthetic.values[5];
      const inverse = f32(1 / oldIndices.length);
      for (const oldIndex of oldIndices) {
        const old = oldVertices[oldIndex];
        const first = Number.isInteger(old?.first) ? old.first | 0 : -1;
        const attributes = [
          oldVertices[first]?.position || old?.position || old?.values?.[0],
          old?.normal || old?.values?.[1], old?.tangent || old?.values?.[2],
          old?.color || old?.values?.[3], old?.uv || old?.values?.[4],
          old?.uv1 || old?.values?.[5],
        ];
        for (let attribute = 0; attribute < synthetic.values.length; attribute++) {
          const value = attributes[attribute];
          if (!value) continue;
          for (let component = 0; component < 4; component++) {
            synthetic.values[attribute][component] = f32(
              synthetic.values[attribute][component] + value[component] * inverse,
            );
          }
        }
      }
      const index = result.vertices.length;
      const vertex = makeVertex();
      copy3(vertex.position, synthetic.position);
      vertex.color = packOldColor(synthetic.color);
      if (oldMeshHasUV0) vertex.uv[0].set(synthetic.uv.subarray(0, 2));
      if (oldMeshHasUV1) vertex.uv[1].set(synthetic.uv1.subarray(0, 2));
      result.vertices.push(vertex);
      return index;
    };
    for (let faceIndex = 0; faceIndex < oldFaces.length; faceIndex++) {
      const face = oldFaces[faceIndex], material = face.material | 0; if (!material) continue;
      const oldIndices = Array.from(source.faceEdges(faceIndex), edge => source.getVertId(edge));
      if (oldIndices.length < 3) continue;
      const vertices = oldIndices.map(addOldVertex);
      if (vertices.length >= 8) {
        const centroid = addCentroid(oldIndices);
        for (let index = 0; index < vertices.length; index++) {
          const converted = makeFace([
            centroid, vertices[index], vertices[(index + 1) % vertices.length],
          ], material);
          converted.flags = face.used ? 0 : 1; result.faces.push(converted);
        }
      } else {
        const converted = makeFace(vertices, material);
        converted.flags = face.used ? 0 : 1; result.faces.push(converted);
      }
    }
  } else {
    let prepared = null;
    if (typeof source.prepare === 'function') prepared = source.prepare();
    const positions = prepared?.positions || source.positions, indices = prepared?.indices || source.indices;
    if (positions && indices) {
      result.vertices.length = 0;
      for (let index = 0; index < positions.length / 3; index++) {
        const vertex = makeVertex(); vertex.position.set(positions.subarray(index * 3, index * 3 + 3));
        if (prepared?.normals) vertex.normal.set(prepared.normals.subarray(index * 3, index * 3 + 3));
        const uvs = prepared?.uv0 || prepared?.uvs; if (uvs) vertex.uv[0].set(uvs.subarray(index * 2, index * 2 + 2));
        const uv1s = prepared?.uv1 || prepared?.uv1s; if (uv1s) vertex.uv[1].set(uv1s.subarray(index * 2, index * 2 + 2));
        result.vertices.push(vertex);
      }
      for (let index = 0; index < indices.length; index += 3) {
        const material = prepared?.triangleMaterials?.[index / 3] ?? 1;
        if (material) result.faces.push(makeFace([indices[index], indices[index + 1], indices[index + 2]], material));
      }
      if (prepared?.materials?.length) {
        result.clusters = [makeCluster(null), ...prepared.materials.map((material, index) => makeCluster(material, 0, index + 1))]; result.Clusters = result.clusters;
      }
    }
  }
  return result.invalidate();
}

function createMeshToMinHandler(meshInputIsUniquelyConsumed) {
  return function handleMeshToMin(call) {
    const source = call.inputs[0];
    const result = meshToMin(source);
    // Old GenMesh operators use runtime consumer counts to transfer unique
    // ownership down linear chains. Conversion is a terminal seam: once all
    // records have been copied into MinMesh form, the much heavier half-edge
    // storage can be released immediately when this was its sole consumer.
    if (result && meshInputIsUniquelyConsumed?.(call, 0)) source?.releaseStorage?.();
    return finalizeMinMeshOutput(call, result);
  };
}

const minMeshRuntimeConsumerCounts = new WeakMap();
const minMeshRuntimeConsumptionState = new WeakMap();
const minMeshRuntimeConsumers = new WeakMap();

function minMeshConsumerCounts(runtime) {
  if (!runtime || !Array.isArray(runtime.operations)) return null;
  let counts = minMeshRuntimeConsumerCounts.get(runtime);
  if (counts) return counts;
  counts = new Map();
  const consumers = new Map();
  const add = op => { if (op) counts.set(op, (counts.get(op) || 0) + 1); };
  for (const op of runtime.operations) {
    for (const input of op.inputs || []) {
      add(input);
      if (input) {
        let list = consumers.get(input);
        if (!list) consumers.set(input, list = []);
        list.push(op);
      }
    }
    for (const link of op.links || []) add(link);
  }
  for (const root of runtime.roots || []) add(root);
  for (const event of runtime.events || []) add(event.op);
  minMeshRuntimeConsumerCounts.set(runtime, counts);
  minMeshRuntimeConsumers.set(runtime, consumers);
  return counts;
}

function minMeshInputIsUniquelyConsumed(call, inputIndex = 0) {
  const inputOp = call?.op?.inputs?.[inputIndex];
  const counts = minMeshConsumerCounts(call?.runtime);
  if (!inputOp || !counts || !(call.inputs?.[inputIndex] instanceof MinMesh)) return false;
  let state = minMeshRuntimeConsumptionState.get(call.runtime);
  if (!state) {
    state = { remaining: new Map(counts), claimed: new WeakMap() };
    minMeshRuntimeConsumptionState.set(call.runtime, state);
  }
  let claimed = state.claimed.get(call.op);
  if (!claimed) state.claimed.set(call.op, claimed = new Set());
  if (!claimed.has(inputIndex)) {
    claimed.add(inputIndex);
    state.remaining.set(inputOp, Math.max(0, (state.remaining.get(inputOp) || 0) - 1));
  }
  return state.remaining.get(inputOp) === 0;
}

function minMeshInputOwnership(call) {
  return call.inputs.map((value, index) => value instanceof MinMesh && minMeshInputIsUniquelyConsumed(call, index));
}

const MINMESH_MODIFIER_IDS = new Set([
  0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x117,
  0x120, 0x121, 0x122, 0x123, 0x124, 0x125, 0x128, 0x129,
  0x12a, 0x12c, 0x12e, 0x12f, 0x131, 0x132, 0x134,
]);

function finalizeMinMeshOutput(call, result) {
  if (!(result instanceof MinMesh) || !call?.runtime || !call.op) return result;
  const counts = minMeshConsumerCounts(call.runtime);
  const consumers = minMeshRuntimeConsumers.get(call.runtime)?.get(call.op) || [];
  const linearModifierChain = (counts.get(call.op) || 0) === 1 && consumers.length === 1 &&
    MINMESH_MODIFIER_IDS.has(consumers[0].classId);
  if (!linearModifierChain) result.compact();
  return result;
}

function finishMinMeshInputs(call, ownership, result) {
  const sources = new Map();
  for (let index = 0; index < ownership.length; index++) {
    const input = call.inputs[index];
    if (!(input instanceof MinMesh) || input === result) continue;
    sources.set(input, Boolean(sources.get(input) || ownership[index]));
  }
  for (const [input, owned] of sources) {
    if (owned) input.releaseStorage();
    else input.compact();
  }
  return finalizeMinMeshOutput(call, result);
}

function minMeshStorageStats(runtime) {
  const meshes = new Set();
  let references = 0;
  for (const op of runtime?.operations || []) if (op.cache instanceof MinMesh) {
    references++;
    meshes.add(op.cache);
  }
  let released = 0, compact = 0, expanded = 0, preparedOnly = 0;
  let vertices = 0, faces = 0, bones = 0, compactBytes = 0, preparedBytes = 0;
  const compactBuffers = new Set();
  const preparedBuffers = new Set();
  const countCompactBuffers = value => {
    if (ArrayBuffer.isView(value)) {
      if (!compactBuffers.has(value.buffer)) {
        compactBuffers.add(value.buffer);
        compactBytes += value.buffer.byteLength;
      }
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const nested of Object.values(value)) countCompactBuffers(nested);
    }
  };
  for (const mesh of meshes) {
    const storage = mesh.storageSummary();
    vertices += storage.vertices;
    faces += storage.faces;
    bones += storage.bones;
    countCompactBuffers(mesh._compact);
    for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(mesh._prepared || {}))) {
      if (!Object.prototype.hasOwnProperty.call(descriptor, 'value')) continue;
      const value = descriptor.value;
      if (ArrayBuffer.isView(value) && value.buffer && !preparedBuffers.has(value.buffer)) {
        preparedBuffers.add(value.buffer);
        preparedBytes += value.buffer.byteLength;
      }
    }
    if (storage.released) released++;
    else if (storage.compact) compact++;
    else if (storage.preparedOnly) preparedOnly++;
    else expanded++;
  }
  return {
    references, identities: meshes.size, released, compact, expanded, preparedOnly,
    vertices, faces, bones, compactBytes, preparedBytes,
  };
}

const minMeshHandlers = {
  0x0101: call => finalizeMinMeshOutput(call, MinMesh_Grid(call.parameters[0], call.parameters[1], call.parameters[2])),
  0x0102: call => finalizeMinMeshOutput(call, MinMesh_Cube(call.parameters[0], call.parameters[1], call.parameters[2], call.parameters[3], call.parameters.slice(4, 13))),
  0x0104: call => finalizeMinMeshOutput(call, MinMesh_Sphere(call.parameters[0], call.parameters[1])),
  0x0105: call => finalizeMinMeshOutput(call, MinMesh_Cylinder(call.parameters[0], call.parameters[1], call.parameters[2], call.parameters[3], call.parameters[4])),
  0x0110: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_MatLink(call.inputs[0], call.links[0], call.parameters[0], call.parameters[1], ownership[0]));
  },
  0x0111: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Add(call.inputs, ownership));
  },
  0x0112: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_SelectAll(call.inputs[0], call.parameters[0], ownership[0]));
  },
  0x0113: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_SelectCube(call.inputs[0], ...call.parameters, ownership[0]));
  },
  0x0114: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_DeleteFaces(call.inputs[0], ownership[0]));
  },
  0x0115: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Invert(call.inputs[0], ownership[0]));
  },
  0x0117: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_MatLink(call.inputs[0], call.inputs[1], call.parameters[0], call.parameters[1], ownership[0]));
  },
  0x0120: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_TransformEx(call.inputs[0], call.parameters[0], ...call.parameters.slice(1, 10), ownership[0]));
  },
  0x0121: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_TransformEx(call.inputs[0], call.parameters[0], ...call.parameters.slice(1, 10), ownership[0]));
  },
  0x0122: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_ExtrudeNormal(call.inputs[0], call.parameters[0], call.parameters[1], ownership[0]));
  },
  0x0123: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Displace(call.inputs[0], call.inputs[1], call.parameters[0], call.parameters[1], call.parameters[2], call.parameters[3], ownership[0]));
  },
  0x0124: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Perlin(call.inputs[0], call.parameters[0], ...call.parameters.slice(1), ownership[0]));
  },
  0x0125: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Bend2(call.inputs[0], ...call.parameters, ownership[0]));
  },
  0x0128: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_BoneChain(call.inputs[0], ...call.parameters, ownership[0]));
  },
  0x0129: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_BoneTrain(call.inputs[0], call.inputs[1], call.parameters[0], call.parameters[1], call.parameters[2], ownership[0]));
  },
  0x012a: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Triangulate(call.inputs[0], ownership[0]));
  },
  0x012c: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Pipe(call.inputs[0], call.inputs[1], call.inputs[2], call.inputs[3], ...call.parameters));
  },
  0x012e: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_AutoMap(call.inputs[0], call.parameters[0], ownership[0]));
  },
  0x012f: call => {
    const p = call.parameters, ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Multiply(call.inputs[0], p.slice(0, 9), p[9], p[10], p[11], p[12], p.slice(13, 16), p[16], ownership[0]));
  },
  0x0131: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Center(call.inputs[0], call.parameters[0], ownership[0]));
  },
  0x0132: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_SelectLogic(call.inputs[0], call.parameters[0], ownership[0]));
  },
  0x0133: call => finalizeMinMeshOutput(call, MinMesh_Font3D(call.parameters[0], call.parameters[1], call.parameters[2], call.strings[0], call.strings[1])),
  0x0134: call => {
    const ownership = minMeshInputOwnership(call);
    return finishMinMeshInputs(call, ownership, MinMesh_Explode(call.inputs[0], call.inputs[1], ...call.parameters));
  },
};

export {
  MinMesh,
  MINMESH_SELECT,
  MinMesh_Add, MinMesh_AutoMap, MinMesh_Bend2, MinMesh_BoneChain, MinMesh_BoneTrain,
  MinMesh_Center, MinMesh_Chamfer, MinMesh_Cube, MinMesh_Cylinder, MinMesh_DeleteFaces,
  MinMesh_Displace, MinMesh_Explode, MinMesh_ExtrudeNormal, MinMesh_Font3D, MinMesh_Grid,
  MinMesh_Invert, MinMesh_MatLink, MinMesh_Multiply, MinMesh_Normals, MinMesh_Perlin,
  MinMesh_Pipe, MinMesh_SelectAll, MinMesh_SelectCube, MinMesh_SelectLogic,
  MinMesh_Sphere, MinMesh_TransformEx, MinMesh_Triangulate,
  fontMaskContours, fontMaskPolygons, fontMaskRectangles, fontMaskToMinMesh,
  fontPolygonsToMinMesh, simplifyClosedFontContour,
  canvasFont3DGlyph, flattenFont3DContour, font3DGlyphAdvance, font3DPointFXCoordinate,
  font3DResolution, font3DToleranceSq, font3DVectorGlyph,
  makeFace as makeMinMeshFace, makeVertex as makeMinMeshVertex,
  meshToMin, meshToMin as meshToMinMesh, minMeshInputIsUniquelyConsumed, minMeshStorageStats,
  setMinMeshFontAdapter, sourcePerlin3,
  createMeshToMinHandler,
  minMeshHandlers,
};
