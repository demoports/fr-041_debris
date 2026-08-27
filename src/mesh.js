import {
  Random,
  f32ToBits,
  mat4Euler,
  mat4EulerTurns,
  mat4Identity,
  mat4Mul,
  mat4SRT,
} from './core.js';

  // Plain JavaScript port of the old (pre GenMinMesh) werkkzeug3 GenMesh.
  // The mutable records intentionally retain the source topology: an edge is
  // one record with two directed half-edges, encoded as edge * 2 + side.
  const f32 = Math.fround;
  const TAU = Math.PI * 2;
  const PI = Math.PI;
  const DEFAULT_VERTEX_MASK = 0x2f; // pos, normal, tangent, color0, uv0
  const ATTR = Object.freeze({ POS: 0, NORMAL: 1, TANGENT: 2, COLOR0: 3, UV0: 5, UV1: 6 });
  const FEATURE = Object.freeze({
    POS: 1 << 0, NORMAL: 1 << 1, TANGENT: 1 << 2,
    COLOR0: 1 << 3, UV0: 1 << 5, UV1: 1 << 6,
  });
  const SELECT = Object.freeze({ ADD: 0, SUB: 1, SET: 2, SETNOT: 3 });
  const ALL = Object.freeze({ EDGE: 1, FACE: 2, VERT: 4 });

  const builtinDefaultMeshMaterial = Object.freeze({
    kind: 'material', system: 'default', passes: [],
  });
  let defaultMeshMaterial = builtinDefaultMeshMaterial;

  function getDefaultMeshMaterial() {
    return defaultMeshMaterial;
  }

  function setDefaultMeshMaterial(material) {
    defaultMeshMaterial = material || builtinDefaultMeshMaterial;
    return defaultMeshMaterial;
  }

  function elem() {
    return { mask: 0, id: 0, select: false, used: true };
  }

  function newVertex() {
    const values = Array.from({ length: 6 }, () => new Float32Array(4));
    return {
      ...elem(),
      next: -1, first: -1, temp: -1, temp2: -1, reIndex: -1,
      weights: new Uint8Array(4), matrices: new Uint8Array([255, 255, 255, 255]),
      values,
      position: values[0], normal: values[1], tangent: values[2],
      color: values[3], uv: values[4], uv1: values[5],
    };
  }

  function cloneVertex(source) {
    const vertex = newVertex();
    vertex.mask = source.mask & 255;
    vertex.id = source.id & 255;
    vertex.select = !!source.select;
    vertex.used = !!source.used;
    vertex.next = source.next | 0;
    vertex.first = source.first | 0;
    vertex.temp = source.temp | 0;
    vertex.temp2 = source.temp2 | 0;
    vertex.reIndex = source.reIndex | 0;
    vertex.weights.set(source.weights);
    vertex.matrices.set(source.matrices);
    for (let i = 0; i < vertex.values.length; i++) vertex.values[i].set(source.values[i]);
    return vertex;
  }

  function newEdge() {
    return {
      ...elem(),
      next: [-1, -1], prev: [-1, -1], face: [-1, -1], vert: [-1, -1],
      temp: [-1, -1], crease: 0,
    };
  }

  function cloneEdge(source) {
    return {
      mask: source.mask & 255, id: source.id & 255,
      select: !!source.select, used: !!source.used,
      next: source.next.slice(), prev: source.prev.slice(),
      face: source.face.slice(), vert: source.vert.slice(),
      temp: source.temp.slice(), crease: source.crease | 0,
    };
  }

  function newFace() {
    return {
      ...elem(),
      material: 1, edge: -1, temp: -1, temp2: 0, temp3: 0,
    };
  }

  function cloneFace(source) {
    return {
      mask: source.mask & 255, id: source.id & 255,
      select: !!source.select, used: !!source.used,
      material: source.material | 0, edge: source.edge | 0,
      temp: source.temp | 0, temp2: source.temp2 | 0, temp3: source.temp3 | 0,
    };
  }

  function cloneCollision(source, vertexOffset = 0) {
    const collision = { ...source };
    for (const key of ['vert', 'Vert']) {
      if (!source?.[key]) continue;
      const values = Array.from(source[key], value => (value | 0) + vertexOffset);
      collision[key] = ArrayBuffer.isView(source[key])
        ? new source[key].constructor(values) : values;
    }
    return collision;
  }

  function materialSlot(material, pass = 0) {
    return { material, pass: pass | 0, jobIds: new Int32Array(16), remap: 0 };
  }

  function cloneSlot(source) {
    const slot = materialSlot(source.material, source.pass);
    slot.jobIds.set(source.jobIds || []);
    slot.remap = source.remap | 0;
    return slot;
  }

  function selElem(record, mask, state, mode) {
    mask &= 255;
    let value = record.mask & 255;
    switch (mode & 3) {
      case SELECT.ADD: if (state) value |= mask; break;
      case SELECT.SUB: if (state) value &= ~mask; break;
      case SELECT.SET: value = state ? value | mask : value & ~mask; break;
      case SELECT.SETNOT: value = state ? value & ~mask : value | mask; break;
    }
    record.mask = value & 255;
  }

  function selLogic(record, source1, source2, destination, mode) {
    let s1 = (record.mask & source1) !== 0;
    let s2 = (record.mask & source2) !== 0;
    if (mode & 4) s1 = !s1;
    if (mode & 8) s2 = !s2;
    switch (mode & 3) {
      case 0: s1 = s1 || s2; break;
      case 1: s1 = s1 && s2; break;
      case 2: s1 = s1 !== s2; break;
      // Source case 3 deliberately leaves s1 unchanged.
    }
    if (mode & 16) s1 = !s1;
    record.mask = (s1 ? record.mask | destination : record.mask & ~destination) & 255;
  }

  function hashWord(hash, value) {
    value >>>= 0;
    hash = Math.imul(hash ^ (value & 255), 0x01000193) >>> 0;
    hash = Math.imul(hash ^ ((value >>> 8) & 255), 0x01000193) >>> 0;
    hash = Math.imul(hash ^ ((value >>> 16) & 255), 0x01000193) >>> 0;
    return Math.imul(hash ^ (value >>> 24), 0x01000193) >>> 0;
  }

  function floatBits(value) {
    return f32ToBits(value);
  }

  function clamp01(value) { return value < 0 ? 0 : value > 1 ? 1 : value; }

  function vector4(x = 0, y = 0, z = 0, w = 0) {
    return new Float32Array([f32(x), f32(y), f32(z), f32(w)]);
  }

  function transformXYZ(matrix, value, out = new Float32Array(4), w = undefined) {
    const x = value[0], y = value[1], z = value[2];
    if (w === undefined) w = value.length > 3 ? value[3] : 1;
    out[0] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w);
    out[1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w);
    out[2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w);
    out[3] = f32(matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w);
    return out;
  }

  function matrixIdentity() { return mat4Identity(new Float32Array(16)); }

  // sMatrix::PivotTransform composes M*T(-pivot), then restores the pivot in
  // the resulting translation. Keep this local to the old GenMesh path: the
  // pivot may refer to any stored attribute, not just physical position.
  function pivotTransform(mesh, matrix, attribute = ATTR.POS) {
    const pivot = mesh.pivot | 0;
    const attributeIndex = mesh.attributeMap(attribute);
    if (pivot < 0 || pivot >= mesh.vertices.length || attributeIndex < 0) return matrix;
    const point = mesh.vertices[pivot].values[attributeIndex];
    const translate = matrixIdentity();
    translate[12] = f32(-point[0]);
    translate[13] = f32(-point[1]);
    translate[14] = f32(-point[2]);
    const result = mat4Mul(matrix, translate, new Float32Array(16));
    result[12] = f32(result[12] + point[0]);
    result[13] = f32(result[13] + point[1]);
    result[14] = f32(result[14] + point[2]);
    return result;
  }

  function compactFaceVertices(storage, faceIndex) {
    const start = storage.faceInts[faceIndex * 5 + 1];
    if (start < 0) return [];
    const result = [];
    let halfedge = start;
    const limit = storage.edgeCount * 2 + 1;
    do {
      const edgeOffset = (halfedge >> 1) * 11;
      const side = halfedge & 1;
      result.push(storage.edgeInts[edgeOffset + 6 + side]);
      halfedge = storage.edgeInts[edgeOffset + side];
      if (result.length > limit) throw new Error('broken GenMesh face loop');
    } while (halfedge !== start);
    return result;
  }

  // GenMesh::CalcNormals(0,7), expressed directly over dormant flat storage.
  // The released bump shaders consume the UV-derived (N, N x T, T) frame, so
  // substituting an arbitrary perpendicular tangent changes the lighting.
  function calculateCompactNormals(storage) {
    const directions = new Float32Array(storage.vertexCount * 8);
    const positions = storage.vertexPositions;
    const uvs = storage.vertexUVs;
    const edges = storage.edgeInts;
    const vertexEdges = new Int32Array(storage.vertexCount);
    vertexEdges.fill(-1);

    const getVertId = halfedge => edges[(halfedge >> 1) * 11 + 6 + (halfedge & 1)];
    const nextFaceEdge = halfedge => edges[(halfedge >> 1) * 11 + (halfedge & 1)];
    const prevFaceEdge = halfedge => edges[(halfedge >> 1) * 11 + 2 + (halfedge & 1)];
    const nextVertEdge = halfedge => prevFaceEdge(halfedge) ^ 1;
    const prevVertEdge = halfedge => edges[(halfedge >> 1) * 11 + ((halfedge & 1) ^ 1)];
    const crease = halfedge => edges[(halfedge >> 1) * 11 + 10];

    for (let halfedge = 0; halfedge < storage.edgeCount * 2; halfedge++) {
      const vertex = getVertId(halfedge);
      if (vertex >= 0 && vertex < storage.vertexCount) vertexEdges[vertex] = halfedge;
    }

    const traversalLimit = storage.edgeCount * 2 + 2;
    for (let component = 0; component < 2; component++) {
      const creaseMask = component ? FEATURE.UV0 : FEATURE.NORMAL;
      const directionOffset = component ? 4 : 0;
      for (let vertex = 0; vertex < storage.vertexCount; vertex++) {
        let edge = vertexEdges[vertex];
        if (edge < 0) continue;
        let end = edge;
        let guard = 0;
        do {
          if (crease(edge) & creaseMask) end = edge;
          else edge = prevVertEdge(edge);
          if (++guard > traversalLimit) throw new Error('broken GenMesh vertex loop');
        } while (edge !== end);

        end = edge;
        const p0 = getVertId(edge) * 4;
        let p2 = getVertId(nextFaceEdge(edge)) * 4;
        let t2x = positions[p2] - positions[p0];
        let t2y = positions[p2 + 1] - positions[p0 + 1];
        let t2z = positions[p2 + 2] - positions[p0 + 2];
        let t2Length2 = t2x * t2x + t2y * t2y + t2z * t2z;
        let accuX = 0, accuY = 0, accuZ = 0;
        let exit = false, wasExit = false;
        guard = 0;
        do {
          wasExit = exit;
          const t1x = t2x, t1y = t2y, t1z = t2z;
          const t1Length2 = t2Length2;
          const p1 = p2;
          p2 = getVertId(wasExit
            ? prevFaceEdge(prevVertEdge(edge))
            : nextFaceEdge(edge)) * 4;
          t2x = positions[p2] - positions[p0];
          t2y = positions[p2 + 1] - positions[p0 + 1];
          t2z = positions[p2 + 2] - positions[p0 + 2];
          t2Length2 = t2x * t2x + t2y * t2y + t2z * t2z;

          if (component === 0) {
            if (t1Length2 * t2Length2 > 1e-20) {
              const scale = 1 / (t1Length2 * t2Length2);
              accuX += (t1y * t2z - t1z * t2y) * scale;
              accuY += (t1z * t2x - t1x * t2z) * scale;
              accuZ += (t1x * t2y - t1y * t2x) * scale;
            }
          } else {
            const c1 = (uvs ? uvs[p0 + 1] : 0) - (uvs ? uvs[p1 + 1] : 0);
            const c2 = (uvs ? uvs[p2 + 1] : 0) - (uvs ? uvs[p0 + 1] : 0);
            let determinant = ((uvs ? uvs[p1] : 0) - (uvs ? uvs[p0] : 0)) * c2 +
              ((uvs ? uvs[p2] : 0) - (uvs ? uvs[p0] : 0)) * c1;
            if (Math.abs(determinant) > 1e-20) {
              determinant = 1 / determinant;
              let tx = t1x * c2 * determinant + t2x * c1 * determinant;
              let ty = t1y * c2 * determinant + t2y * c1 * determinant;
              let tz = t1z * c2 * determinant + t2z * c1 * determinant;
              const length2 = tx * tx + ty * ty + tz * tz;
              if (length2 > 1e-20) {
                const inverseLength = 1 / Math.sqrt(length2);
                tx *= inverseLength; ty *= inverseLength; tz *= inverseLength;
                accuX += tx; accuY += ty; accuZ += tz;
              }
            }
          }

          edge = nextVertEdge(edge);
          exit = edge === end || (crease(edge) & creaseMask) !== 0;
          if (++guard > traversalLimit) throw new Error('broken GenMesh vertex loop');
        } while (!wasExit);

        const output = vertex * 8 + directionOffset;
        if (component) {
          const normal = vertex * 8;
          const parallel = accuX * directions[normal] +
            accuY * directions[normal + 1] + accuZ * directions[normal + 2];
          accuX -= directions[normal] * parallel;
          accuY -= directions[normal + 1] * parallel;
          accuZ -= directions[normal + 2] * parallel;
        }
        const length2 = accuX * accuX + accuY * accuY + accuZ * accuZ;
        if (length2 > 1e-20) {
          const inverseLength = 1 / Math.sqrt(length2);
          directions[output] = f32(accuX * inverseLength);
          directions[output + 1] = f32(accuY * inverseLength);
          directions[output + 2] = f32(accuZ * inverseLength);
        } else {
          directions[output] = 1;
          directions[output + 1] = directions[output + 2] = 0;
        }
        directions[output + 3] = 0;
      }
    }
    return directions;
  }

  function compactBounds(storage, activeOnly = true) {
    const referenced = activeOnly ? new Uint8Array(storage.vertexCount) : null;
    if (referenced) {
      for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
        if (!storage.faceInts[faceIndex * 5]) continue;
        for (const id of compactFaceVertices(storage, faceIndex)) referenced[id] = 1;
      }
    }
    const min = new Float32Array([Infinity, Infinity, Infinity]);
    const max = new Float32Array([-Infinity, -Infinity, -Infinity]);
    for (let index = 0; index < storage.vertexCount; index++) {
      if (referenced && !referenced[index]) continue;
      const first = storage.vertexInts[index * 5 + 1];
      const positionIndex = first >= 0 && first < storage.vertexCount ? first : index;
      const positionOffset = positionIndex * 4;
      for (let axis = 0; axis < 3; axis++) {
        const value = storage.vertexPositions[positionOffset + axis];
        if (value < min[axis]) min[axis] = value;
        if (value > max[axis]) max[axis] = value;
      }
    }
    if (!Number.isFinite(min[0])) min.fill(0), max.fill(0);
    return { min, max };
  }

  function prepareCompactMesh(mesh, storage) {
    const count = storage.vertexCount;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    const tangents = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);
    const uvs = new Float32Array(count * 2);
    const uv1s = mesh.vertexMask & FEATURE.UV1 ? new Float32Array(count * 2) : null;
    const shadowVertexMap = new Uint32Array(count);
    const directions = storage.vertexDirections;
    for (let index = 0; index < count; index++) {
      const firstValue = storage.vertexInts[index * 5 + 1];
      const first = firstValue >= 0 && firstValue < count ? firstValue : index;
      const positionOffset = first * 4;
      const positionTarget = index * 3;
      positions[positionTarget] = storage.vertexPositions[positionOffset];
      positions[positionTarget + 1] = storage.vertexPositions[positionOffset + 1];
      positions[positionTarget + 2] = storage.vertexPositions[positionOffset + 2];
      if (directions) {
        const directionOffset = index * 8;
        normals[positionTarget] = directions[directionOffset];
        normals[positionTarget + 1] = directions[directionOffset + 1];
        normals[positionTarget + 2] = directions[directionOffset + 2];
        const tangentTarget = index * 4;
        tangents[tangentTarget] = directions[directionOffset + 4];
        tangents[tangentTarget + 1] = directions[directionOffset + 5];
        tangents[tangentTarget + 2] = directions[directionOffset + 6];
        tangents[tangentTarget + 3] = directions[directionOffset + 7];
      }
      if (storage.vertexColors) {
        const source = index * 4;
        colors[source] = storage.vertexColors[source];
        colors[source + 1] = storage.vertexColors[source + 1];
        colors[source + 2] = storage.vertexColors[source + 2];
        colors[source + 3] = storage.vertexColors[source + 3];
      }
      if (storage.vertexUVs) {
        const source = index * 4;
        const target = index * 2;
        uvs[target] = storage.vertexUVs[source];
        uvs[target + 1] = storage.vertexUVs[source + 1];
      }
      if (uv1s && storage.vertexUV1s) {
        const source = index * 4;
        const target = index * 2;
        uv1s[target] = storage.vertexUV1s[source];
        uv1s[target + 1] = storage.vertexUV1s[source + 1];
      }
      shadowVertexMap[index] = first;
    }

    const perMaterial = new Map();
    const perMaterialShadow = new Map();
    for (let faceIndex = 0; faceIndex < storage.faceCount; faceIndex++) {
      const faceOffset = faceIndex * 5;
      const materialIndex = storage.faceInts[faceOffset];
      if (!materialIndex || storage.faceInts[faceOffset + 1] < 0) continue;
      const ids = compactFaceVertices(storage, faceIndex);
      if (ids.length < 3) continue;
      let list = perMaterial.get(materialIndex);
      if (!list) perMaterial.set(materialIndex, list = []);
      let shadow = perMaterialShadow.get(materialIndex);
      if (!shadow) perMaterialShadow.set(materialIndex, shadow = []);
      const used = storage.faceBytes[faceIndex * 4 + 3] !== 0;
      for (let i = 1; i + 1 < ids.length; i++) {
        list.push(ids[0], ids[i], ids[i + 1]);
        shadow.push(used ? 1 : 0);
      }
    }
    const indexValues = [];
    const triangleMaterials = [];
    const shadowTriangleValues = [];
    const groups = [];
    for (const [materialIndex, list] of perMaterial) {
      const start = indexValues.length;
      indexValues.push(...list);
      shadowTriangleValues.push(...(perMaterialShadow.get(materialIndex) || []));
      for (let i = 0; i < list.length / 3; i++) triangleMaterials.push(materialIndex);
      const slot = mesh.materials[materialIndex] || mesh.materials[0];
      groups.push({
        material: slot.material, materialIndex, pass: slot.pass | 0,
        start, count: list.length,
      });
    }
    return {
      kind: 'mesh-buffer', positions, normals, tangents, colors, uvs, uv1s,
      indices: new Uint32Array(indexValues),
      triangleMaterials: new Uint16Array(triangleMaterials),
      shadowVertexMap, shadowTriangleMask: new Uint8Array(shadowTriangleValues),
      groups, materials: mesh.materials, bounds: compactBounds(storage, true),
    };
  }

  class Mesh {
    constructor(vertexMask = DEFAULT_VERTEX_MASK) {
      this.kind = 'mesh';
      this.vertexMask = vertexMask >>> 0;
      this._compact = null;
      this._vertices = [];
      this._edges = [];
      this._faces = [];
      this.materials = [materialSlot(null, 0), materialSlot(defaultMeshMaterial, 0)];
      this.parts = [];
      this.collisions = [];
      this.lights = [];
      this.pivot = -1;
      this.gotNormals = false;
      this.stripped = false;
      this._prepared = null;
      this.topologyReleasedForPlayback = false;
      this._syncAliases();
    }

    get vertices() { this.ensureExpanded(); return this._vertices; }
    set vertices(value) { this._compact = null; this._vertices = value; }
    get edges() { this.ensureExpanded(); return this._edges; }
    set edges(value) { this._compact = null; this._edges = value; }
    get faces() { this.ensureExpanded(); return this._faces; }
    set faces(value) { this._compact = null; this._faces = value; }

    get Vert() { return this.vertices; }
    set Vert(value) { this.vertices = value; }
    get Edge() { return this.edges; }
    set Edge(value) { this.edges = value; }
    get Face() { return this.faces; }
    set Face(value) { this.faces = value; }
    get Mtrl() { return this.materials; }
    set Mtrl(value) { this.materials = value; }
    get Parts() { return this.parts; }
    set Parts(value) { this.parts = value; }
    get Coll() { return this.collisions; }
    set Coll(value) { this.collisions = value; }
    get Lgts() { return this.lights; }
    set Lgts(value) { this.lights = value; }

    _syncAliases() {
      // Classic names are prototype accessors so a compact dormant cache is
      // expanded only when its records are actually requested.
      return this;
    }

    // Old GenMesh records are convenient while an operator is executing but
    // prohibitively expensive as a retained JS graph cache. A compact cache
    // has the exact same f32/int data in a handful of flat typed arrays and is
    // expanded lazily before a later consumer or public record access.
    compact() {
      if (this._compact || this.released || this.topologyReleasedForPlayback) return this;
      const vertices = this._vertices;
      const edges = this._edges;
      const faces = this._faces;
      const vertexPositions = new Float32Array(vertices.length * 4);
      const vertexInts = new Int32Array(vertices.length * 5);
      const vertexBytes = new Uint8Array(vertices.length * 4);
      let hasDirections = false;
      let hasColors = false;
      let hasUVs = false;
      let hasUV1s = false;
      let hasWeights = false;
      let hasMatrices = false;
      for (let index = 0; index < vertices.length; index++) {
        const vertex = vertices[index];
        const vectorOffset = index * 4;
        vertexPositions.set(vertex.position, vectorOffset);
        const intOffset = index * 5;
        vertexInts[intOffset] = vertex.next | 0;
        vertexInts[intOffset + 1] = vertex.first | 0;
        vertexInts[intOffset + 2] = vertex.temp | 0;
        vertexInts[intOffset + 3] = vertex.temp2 | 0;
        vertexInts[intOffset + 4] = vertex.reIndex | 0;
        const byteOffset = index * 4;
        vertexBytes[byteOffset] = vertex.mask & 255;
        vertexBytes[byteOffset + 1] = vertex.id & 255;
        vertexBytes[byteOffset + 2] = vertex.select ? 1 : 0;
        vertexBytes[byteOffset + 3] = vertex.used ? 1 : 0;
        for (let component = 0; component < 4; component++) {
          if (!Object.is(vertex.normal[component], 0) || !Object.is(vertex.tangent[component], 0)) {
            hasDirections = true;
          }
          if (!Object.is(vertex.color[component], 0)) hasColors = true;
          if (!Object.is(vertex.uv[component], 0)) hasUVs = true;
          if (!Object.is(vertex.uv1[component], 0)) hasUV1s = true;
          if (vertex.weights[component] !== 0) hasWeights = true;
          if (vertex.matrices[component] !== 255) hasMatrices = true;
        }
      }

      const vertexDirections = hasDirections ? new Float32Array(vertices.length * 8) : null;
      const vertexColors = hasColors ? new Float32Array(vertices.length * 4) : null;
      const vertexUVs = hasUVs ? new Float32Array(vertices.length * 4) : null;
      const vertexUV1s = hasUV1s ? new Float32Array(vertices.length * 4) : null;
      const vertexWeights = hasWeights ? new Uint8Array(vertices.length * 4) : null;
      const vertexMatrices = hasMatrices ? new Uint8Array(vertices.length * 4) : null;
      if (vertexMatrices) vertexMatrices.fill(255);
      if (vertexDirections || vertexColors || vertexUVs || vertexUV1s || vertexWeights || vertexMatrices) {
        for (let index = 0; index < vertices.length; index++) {
          const vertex = vertices[index];
          const vectorOffset = index * 4;
          if (vertexDirections) {
            vertexDirections.set(vertex.normal, index * 8);
            vertexDirections.set(vertex.tangent, index * 8 + 4);
          }
          if (vertexColors) vertexColors.set(vertex.color, vectorOffset);
          if (vertexUVs) vertexUVs.set(vertex.uv, vectorOffset);
          if (vertexUV1s) vertexUV1s.set(vertex.uv1, vectorOffset);
          if (vertexWeights) vertexWeights.set(vertex.weights, vectorOffset);
          if (vertexMatrices) vertexMatrices.set(vertex.matrices, vectorOffset);
        }
      }

      const edgeInts = new Int32Array(edges.length * 11);
      const edgeBytes = new Uint8Array(edges.length * 4);
      for (let index = 0; index < edges.length; index++) {
        const edge = edges[index];
        const intOffset = index * 11;
        edgeInts[intOffset] = edge.next[0] | 0;
        edgeInts[intOffset + 1] = edge.next[1] | 0;
        edgeInts[intOffset + 2] = edge.prev[0] | 0;
        edgeInts[intOffset + 3] = edge.prev[1] | 0;
        edgeInts[intOffset + 4] = edge.face[0] | 0;
        edgeInts[intOffset + 5] = edge.face[1] | 0;
        edgeInts[intOffset + 6] = edge.vert[0] | 0;
        edgeInts[intOffset + 7] = edge.vert[1] | 0;
        edgeInts[intOffset + 8] = edge.temp[0] | 0;
        edgeInts[intOffset + 9] = edge.temp[1] | 0;
        edgeInts[intOffset + 10] = edge.crease | 0;
        const byteOffset = index * 4;
        edgeBytes[byteOffset] = edge.mask & 255;
        edgeBytes[byteOffset + 1] = edge.id & 255;
        edgeBytes[byteOffset + 2] = edge.select ? 1 : 0;
        edgeBytes[byteOffset + 3] = edge.used ? 1 : 0;
      }

      const faceInts = new Int32Array(faces.length * 5);
      const faceBytes = new Uint8Array(faces.length * 4);
      for (let index = 0; index < faces.length; index++) {
        const face = faces[index];
        const intOffset = index * 5;
        faceInts[intOffset] = face.material | 0;
        faceInts[intOffset + 1] = face.edge | 0;
        faceInts[intOffset + 2] = face.temp | 0;
        faceInts[intOffset + 3] = face.temp2 | 0;
        faceInts[intOffset + 4] = face.temp3 | 0;
        const byteOffset = index * 4;
        faceBytes[byteOffset] = face.mask & 255;
        faceBytes[byteOffset + 1] = face.id & 255;
        faceBytes[byteOffset + 2] = face.select ? 1 : 0;
        faceBytes[byteOffset + 3] = face.used ? 1 : 0;
      }

      this._compact = {
        vertexCount: vertices.length, vertexPositions, vertexDirections,
        vertexColors, vertexUVs, vertexUV1s, vertexInts, vertexBytes,
        vertexWeights, vertexMatrices,
        edgeCount: edges.length, edgeInts, edgeBytes,
        faceCount: faces.length, faceInts, faceBytes,
      };
      this._vertices = [];
      this._edges = [];
      this._faces = [];
      return this;
    }

    ensureExpanded() {
      if (this.topologyReleasedForPlayback) {
        throw new Error('GenMesh topology was released for immutable playback');
      }
      const storage = this._compact;
      if (!storage) return this;
      // Clear first so assignments and record helpers cannot recursively enter
      // expansion through the public accessors.
      this._compact = null;
      const vertices = new Array(storage.vertexCount);
      for (let index = 0; index < vertices.length; index++) {
        const vertex = newVertex();
        const vectorOffset = index * 4;
        for (let component = 0; component < 4; component++) {
          vertex.position[component] = storage.vertexPositions[vectorOffset + component];
        }
        if (storage.vertexDirections) {
          const directionOffset = index * 8;
          for (let component = 0; component < 4; component++) {
            vertex.normal[component] = storage.vertexDirections[directionOffset + component];
            vertex.tangent[component] = storage.vertexDirections[directionOffset + 4 + component];
          }
        }
        if (storage.vertexColors) {
          for (let component = 0; component < 4; component++) {
            vertex.color[component] = storage.vertexColors[vectorOffset + component];
          }
        }
        if (storage.vertexUVs) {
          for (let component = 0; component < 4; component++) {
            vertex.uv[component] = storage.vertexUVs[vectorOffset + component];
          }
        }
        if (storage.vertexUV1s) {
          for (let component = 0; component < 4; component++) {
            vertex.uv1[component] = storage.vertexUV1s[vectorOffset + component];
          }
        }
        const intOffset = index * 5;
        vertex.next = storage.vertexInts[intOffset];
        vertex.first = storage.vertexInts[intOffset + 1];
        vertex.temp = storage.vertexInts[intOffset + 2];
        vertex.temp2 = storage.vertexInts[intOffset + 3];
        vertex.reIndex = storage.vertexInts[intOffset + 4];
        const byteOffset = index * 4;
        vertex.mask = storage.vertexBytes[byteOffset];
        vertex.id = storage.vertexBytes[byteOffset + 1];
        vertex.select = storage.vertexBytes[byteOffset + 2] !== 0;
        vertex.used = storage.vertexBytes[byteOffset + 3] !== 0;
        for (let component = 0; component < 4; component++) {
          if (storage.vertexWeights) vertex.weights[component] = storage.vertexWeights[vectorOffset + component];
          if (storage.vertexMatrices) vertex.matrices[component] = storage.vertexMatrices[vectorOffset + component];
        }
        vertices[index] = vertex;
      }

      const edges = new Array(storage.edgeCount);
      for (let index = 0; index < edges.length; index++) {
        const edge = newEdge();
        const intOffset = index * 11;
        edge.next[0] = storage.edgeInts[intOffset];
        edge.next[1] = storage.edgeInts[intOffset + 1];
        edge.prev[0] = storage.edgeInts[intOffset + 2];
        edge.prev[1] = storage.edgeInts[intOffset + 3];
        edge.face[0] = storage.edgeInts[intOffset + 4];
        edge.face[1] = storage.edgeInts[intOffset + 5];
        edge.vert[0] = storage.edgeInts[intOffset + 6];
        edge.vert[1] = storage.edgeInts[intOffset + 7];
        edge.temp[0] = storage.edgeInts[intOffset + 8];
        edge.temp[1] = storage.edgeInts[intOffset + 9];
        edge.crease = storage.edgeInts[intOffset + 10];
        const byteOffset = index * 4;
        edge.mask = storage.edgeBytes[byteOffset];
        edge.id = storage.edgeBytes[byteOffset + 1];
        edge.select = storage.edgeBytes[byteOffset + 2] !== 0;
        edge.used = storage.edgeBytes[byteOffset + 3] !== 0;
        edges[index] = edge;
      }

      const faces = new Array(storage.faceCount);
      for (let index = 0; index < faces.length; index++) {
        const face = newFace();
        const intOffset = index * 5;
        face.material = storage.faceInts[intOffset];
        face.edge = storage.faceInts[intOffset + 1];
        face.temp = storage.faceInts[intOffset + 2];
        face.temp2 = storage.faceInts[intOffset + 3];
        face.temp3 = storage.faceInts[intOffset + 4];
        const byteOffset = index * 4;
        face.mask = storage.faceBytes[byteOffset];
        face.id = storage.faceBytes[byteOffset + 1];
        face.select = storage.faceBytes[byteOffset + 2] !== 0;
        face.used = storage.faceBytes[byteOffset + 3] !== 0;
        faces[index] = face;
      }
      this._vertices = vertices;
      this._edges = edges;
      this._faces = faces;
      return this;
    }

    storageSummary() {
      const storage = this._compact;
      if (this.topologyReleasedForPlayback) {
        return {
          compact: false, released: false, preparedOnly: true,
          vertices: this._prepared?.positions?.length / 3 || 0,
          edges: 0,
          faces: this._prepared?.triangleMaterials?.length || 0,
        };
      }
      return {
        compact: !!storage,
        released: !!this.released,
        vertices: storage ? storage.vertexCount : this._vertices.length,
        edges: storage ? storage.edgeCount : this._edges.length,
        faces: storage ? storage.faceCount : this._faces.length,
      };
    }

    _touch() {
      if (this.topologyReleasedForPlayback) {
        throw new Error('cannot mutate GenMesh after playback topology release');
      }
      this._prepared = null;
      return this;
    }

    get vertexCount() { return this.vertices.length; }
    get edgeCount() { return this.edges.length; }
    get faceCount() { return this.faces.length; }
    get VertCount() { return this.vertices.length; }

    attributeMap(attribute) {
      const id = attribute & 15;
      if (!(this.vertexMask & (1 << id))) return -1;
      switch (id) {
        case ATTR.POS: return 0;
        case ATTR.NORMAL: return 1;
        case ATTR.TANGENT: return 2;
        case ATTR.COLOR0: return 3;
        case ATTR.UV0: return 4;
        case ATTR.UV1: return 5;
        default: return -1;
      }
    }

    addVertex(position = null, uv = null, mask = 0, first = -1) {
      const index = this.vertices.length;
      const vertex = newVertex();
      vertex.next = vertex.reIndex = index;
      vertex.first = first < 0 ? index : first | 0;
      vertex.mask = mask & 255;
      if (position) {
        vertex.position[0] = f32(position[0]);
        vertex.position[1] = f32(position[1]);
        vertex.position[2] = f32(position[2]);
        vertex.position[3] = position.length > 3 ? f32(position[3]) : 1;
      }
      if (uv) {
        vertex.uv[0] = f32(uv[0]);
        vertex.uv[1] = f32(uv[1]);
        vertex.uv[2] = uv.length > 2 ? f32(uv[2]) : 0;
        vertex.uv[3] = uv.length > 3 ? f32(uv[3]) : 1;
      }
      this.vertices.push(vertex);
      return index;
    }

    addCopiedVertex(index) {
      const result = cloneVertex(this.vertices[index]);
      const id = this.vertices.length;
      result.reIndex = id;
      this.vertices.push(result);
      return id;
    }

    linkVertexCopies() {
      const groups = new Map();
      for (let i = 0; i < this.vertices.length; i++) {
        const vertex = this.vertices[i];
        let first = vertex.first;
        if (first < 0 || first >= this.vertices.length) first = vertex.first = i;
        const list = groups.get(first);
        if (list) list.push(i); else groups.set(first, [i]);
      }
      for (const [first, list] of groups) {
        for (let i = 0; i < list.length; i++) {
          const vertex = this.vertices[list[i]];
          vertex.first = first;
          vertex.next = list[(i + 1) % list.length];
          vertex.reIndex = list[i];
        }
      }
    }

    // Rebuild the edge records from oriented polygon loops. Vertex records are
    // retained, so per-corner attributes and First/Next crease cycles survive.
    setPolygons(polygons) {
      this.edges = [];
      this.faces = [];
      const buckets = new Map();

      for (const polygon of polygons) {
        const verts = polygon.verts || polygon;
        if (!verts || verts.length < 3) continue;
        const faceIndex = this.faces.length;
        const face = newFace();
        face.mask = (polygon.mask || 0) & 255;
        face.id = (polygon.id || 0) & 255;
        face.select = !!polygon.select;
        face.used = polygon.used === undefined ? true : !!polygon.used;
        face.material = polygon.material === undefined ? 1 : polygon.material | 0;
        this.faces.push(face);
        const halfedges = [];

        for (let i = 0; i < verts.length; i++) {
          const a = verts[i] | 0;
          const b = verts[(i + 1) % verts.length] | 0;
          const ca = this.vertices[a].first | 0;
          const cb = this.vertices[b].first | 0;
          if (ca === cb) continue;
          const lo = Math.min(ca, cb), hi = Math.max(ca, cb);
          const side = ca === lo ? 0 : 1;
          const key = `${lo}:${hi}`;
          let choices = buckets.get(key);
          if (!choices) buckets.set(key, choices = []);
          let edgeIndex = -1;
          for (const candidate of choices) {
            if (this.edges[candidate].face[side] === -1) { edgeIndex = candidate; break; }
          }
          if (edgeIndex < 0) {
            edgeIndex = this.edges.length;
            const edge = newEdge();
            edge._end = [-1, -1];
            this.edges.push(edge);
            choices.push(edgeIndex);
          }
          const edge = this.edges[edgeIndex];
          edge.face[side] = faceIndex;
          edge.vert[side] = a;
          edge._end[side] = b;
          halfedges.push(edgeIndex * 2 + side);
        }

        if (halfedges.length < 3) {
          this.faces.pop();
          continue;
        }
        face.edge = halfedges[0];
        for (let i = 0; i < halfedges.length; i++) {
          const current = halfedges[i], side = current & 1;
          const edge = this.edges[current >> 1];
          edge.prev[side] = halfedges[(i + halfedges.length - 1) % halfedges.length];
          edge.next[side] = halfedges[(i + 1) % halfedges.length];
        }
      }

      // Source meshes keep deleted opposite faces rather than naked boundary
      // pointers. A one-edge deleted loop gives the same safe adjacency for a
      // genuinely open polygon assembled by a browser-side adapter.
      for (let edgeIndex = 0; edgeIndex < this.edges.length; edgeIndex++) {
        const edge = this.edges[edgeIndex];
        for (let side = 0; side < 2; side++) {
          if (edge.face[side] !== -1) continue;
          const face = newFace();
          face.material = 0;
          face.edge = edgeIndex * 2 + side;
          const faceIndex = this.faces.length;
          this.faces.push(face);
          edge.face[side] = faceIndex;
          edge.vert[side] = edge._end[side ^ 1];
          edge.next[side] = edge.prev[side] = face.edge;
        }
        delete edge._end;
      }
      this._syncAliases();
      return this._touch();
    }

    getFaceId(halfedge) { return this.edges[halfedge >> 1].face[halfedge & 1]; }
    getVertId(halfedge) { return this.edges[halfedge >> 1].vert[halfedge & 1]; }
    getFace(halfedge) { return this.faces[this.getFaceId(halfedge)]; }
    getFaceI(halfedge) { return this.faces[this.edges[halfedge >> 1].face[(halfedge ^ 1) & 1]]; }
    getVert(halfedge) { return this.vertices[this.getVertId(halfedge)]; }
    nextFaceEdge(halfedge) { return this.edges[halfedge >> 1].next[halfedge & 1]; }
    prevFaceEdge(halfedge) { return this.edges[halfedge >> 1].prev[halfedge & 1]; }
    nextVertEdge(halfedge) { return this.prevFaceEdge(halfedge) ^ 1; }
    prevVertEdge(halfedge) { return this.edges[halfedge >> 1].next[(halfedge ^ 1) & 1]; }

    faceEdges(faceOrIndex) {
      const face = typeof faceOrIndex === 'number' ? this.faces[faceOrIndex] : faceOrIndex;
      if (!face || face.edge < 0) return [];
      const result = [];
      let edge = face.edge;
      const limit = this.edges.length * 2 + 1;
      do {
        result.push(edge);
        edge = this.nextFaceEdge(edge);
        if (result.length > limit) throw new Error('broken GenMesh face loop');
      } while (edge !== face.edge);
      return result;
    }

    faceVertices(faceOrIndex) {
      return this.faceEdges(faceOrIndex).map(edge => this.getVertId(edge));
    }

    clone() {
      this.ensureExpanded();
      const mesh = new Mesh(this.vertexMask);
      mesh.vertices = this.vertices.map(cloneVertex);
      mesh.edges = this.edges.map(cloneEdge);
      mesh.faces = this.faces.map(cloneFace);
      mesh.materials = this.materials.map(cloneSlot);
      mesh.parts = this.parts.slice();
      mesh.collisions = this.collisions.map(value => cloneCollision(value));
      mesh.lights = this.lights.slice();
      mesh.pivot = this.pivot | 0;
      mesh.gotNormals = !!this.gotNormals;
      mesh.stripped = !!this.stripped;
      mesh._syncAliases();
      return mesh;
    }

    cloneForMaterial() {
      const storage = this._compact;
      if (!storage) return this.clone();
      const mesh = new Mesh(this.vertexMask);
      // MatLink changes only selection bytes, face material indices and the
      // material table. Share all immutable geometry/attribute buffers across
      // compact graph branches and make only those small mutable channels
      // private. Expanding either branch still reconstructs private records.
      mesh._compact = {
        ...storage,
        vertexBytes: storage.vertexBytes.slice(),
        edgeBytes: storage.edgeBytes.slice(),
        faceInts: storage.faceInts.slice(),
        faceBytes: storage.faceBytes.slice(),
      };
      mesh.materials = this.materials.map(cloneSlot);
      mesh.parts = this.parts.slice();
      mesh.collisions = this.collisions.map(value => cloneCollision(value));
      mesh.lights = this.lights.slice();
      mesh.pivot = this.pivot | 0;
      mesh.gotNormals = !!this.gotNormals;
      mesh.stripped = !!this.stripped;
      mesh._syncAliases();
      return mesh;
    }

    cloneForPositionTransform() {
      const storage = this._compact;
      if (!storage) return this.clone();
      const mesh = new Mesh(this.vertexMask);
      // A mask-zero Transform changes positions and resets all three selection
      // domains. Topology and the remaining authored attributes stay immutable,
      // so forked compact branches only need private copies of those channels.
      mesh._compact = {
        ...storage,
        vertexPositions: storage.vertexPositions.slice(),
        vertexBytes: storage.vertexBytes.slice(),
        edgeBytes: storage.edgeBytes.slice(),
        faceBytes: storage.faceBytes.slice(),
      };
      mesh.materials = this.materials.map(cloneSlot);
      mesh.parts = this.parts.slice();
      mesh.collisions = this.collisions.map(value => cloneCollision(value));
      mesh.lights = this.lights.slice();
      mesh.pivot = this.pivot | 0;
      mesh.gotNormals = !!this.gotNormals;
      mesh.stripped = !!this.stripped;
      mesh._syncAliases();
      return mesh;
    }

    compactMeshConversionView() {
      if (!this._compact || this.released || this.topologyReleasedForPlayback) return null;
      // Mesh_ToMin consumes this synchronously and read-only. Keeping the old
      // compact layout behind a narrow view avoids a mesh/minmesh import cycle.
      return { vertexMask: this.vertexMask, materials: this.materials, storage: this._compact };
    }

    copy() { return this.clone(); }

    // Runtime precalc keeps every operator cache alive. The native graph used
    // reference-counted arguments, so a uniquely consumed input was normally
    // mutated/released instead of retaining a complete deep-copy at every
    // node. adopt() gives the JS dispatch layer the same safe ownership move.
    adopt(source) {
      if (!(source instanceof Mesh) || source === this) return this;
      if (source.topologyReleasedForPlayback) {
        throw new Error('cannot adopt GenMesh after playback topology release');
      }
      this.vertexMask = source.vertexMask;
      this._compact = source._compact;
      this._vertices = source._vertices;
      this._edges = source._edges;
      this._faces = source._faces;
      this.materials = source.materials;
      this.parts = source.parts;
      this.collisions = source.collisions;
      this.lights = source.lights;
      this.pivot = source.pivot;
      this.gotNormals = source.gotNormals;
      this.stripped = source.stripped;
      this._prepared = source._prepared;
      this.topologyReleasedForPlayback = false;
      this.released = false;
      this._syncAliases();
      return this;
    }

    releaseStorage() {
      this._compact = null;
      this._vertices = [];
      this._edges = [];
      this._faces = [];
      this.materials = [];
      this.parts = [];
      this.collisions = [];
      this.lights = [];
      this._prepared = null;
      this.topologyReleasedForPlayback = false;
      this.released = true;
      this._syncAliases();
    }

    maskToSelection(mask) {
      mask >>>= 0;
      if (mask & 0x00ff0000) {
        const bits = (mask >>> 16) & 255;
        for (const vertex of this.vertices) vertex.select = (vertex.mask & bits) !== 0;
      }
      if (mask & 0x0000ff00) {
        const bits = (mask >>> 8) & 255;
        for (const face of this.faces) face.select = (face.mask & bits) !== 0;
      }
      if (mask & 255) {
        const bits = mask & 255;
        for (const edge of this.edges) edge.select = (edge.mask & bits) !== 0;
      }
      return this;
    }

    allToSelection(selected = true, domains = 7) {
      if (domains & ALL.VERT) for (const vertex of this.vertices) vertex.select = !!selected;
      if (domains & ALL.FACE) {
        for (const face of this.faces) face.select = !!selected && face.material !== 0;
      }
      if (domains & ALL.EDGE) for (const edge of this.edges) edge.select = !!selected;
      return this;
    }

    selectionToMask(mask, mode) {
      mask >>>= 0;
      if (mask & 0x00ff0000) {
        const bits = mask >>> 16;
        for (const vertex of this.vertices) selElem(vertex, bits, vertex.select, mode);
      }
      if (mask & 0x0000ff00) {
        const bits = mask >>> 8;
        for (const face of this.faces) selElem(face, bits, face.select, mode);
      }
      if (mask & 255) for (const edge of this.edges) selElem(edge, mask, edge.select, mode);
      return this._touch();
    }

    allToMask(mask, mode = SELECT.ADD) {
      mask >>>= 0;
      for (const vertex of this.vertices) {
        if (mask & 0x00ff0000) selElem(vertex, mask >>> 16, true, mode);
      }
      for (const face of this.faces) {
        if (face.material && (mask & 0x0000ff00)) selElem(face, mask >>> 8, true, mode);
      }
      for (const edge of this.edges) {
        if (mask & 255) selElem(edge, mask, true, mode);
      }
      return this._touch();
    }

    faceToVertex() {
      this.allToSelection(false, ALL.VERT);
      for (const face of this.faces) {
        if (!face.select || face.edge < 0) continue;
        for (const edge of this.faceEdges(face)) this.getVert(edge).select = true;
      }
      return this;
    }

    vertexToFaceEdge() {
      for (let i = 0; i < this.edges.length; i++) {
        this.edges[i].select = this.getVert(i * 2).select && this.getVert(i * 2 + 1).select;
      }
      for (const face of this.faces) {
        if (!face.material) continue;
        face.select = this.faceEdges(face).every(edge => this.getVert(edge).select);
      }
      return this;
    }

    transformVertices(matrix, sourceAttribute = ATTR.POS, destinationAttribute = ATTR.POS,
      start = 0, end = undefined) {
      const sourceIndex = this.attributeMap(sourceAttribute);
      const destinationIndex = this.attributeMap(destinationAttribute & 15);
      if (sourceIndex < 0 || destinationIndex < 0) return this;
      const additive = (destinationAttribute & 0x10) !== 0;
      const colorIndex = this.attributeMap(ATTR.COLOR0);
      const vertices = this.vertices;
      start = Math.max(0, start | 0);
      end = end === undefined ? vertices.length : Math.min(vertices.length, Math.max(start, end | 0));
      // A transform can read and write the same attribute, so retain one
      // call-local result vector. Allocating it inside the vertex loop created
      // millions of short-lived typed arrays during Multiply/Multiply2.
      const transformed = new Float32Array(4);
      for (let index = start; index < end; index++) {
        const vertex = vertices[index];
        if (!vertex.select) continue;
        const source = vertex.values[sourceIndex];
        const destination = vertex.values[destinationIndex];
        const sw = sourceIndex === colorIndex ? 1 : source[3];
        transformXYZ(matrix, source, transformed, sw);
        if (additive) {
          destination[0] = f32(destination[0] + transformed[0]);
          destination[1] = f32(destination[1] + transformed[1]);
          destination[2] = f32(destination[2] + transformed[2]);
        } else {
          destination[0] = transformed[0];
          destination[1] = transformed[1];
          destination[2] = transformed[2];
          destination[3] = source[3];
        }
      }
      return this._touch();
    }

    append(other) {
      if (!(other instanceof Mesh) || other.vertexMask !== this.vertexMask) return false;
      const vertexOffset = this.vertices.length;
      const edgeOffset = this.edges.length;
      const faceOffset = this.faces.length;
      const remap = new Int32Array(other.materials.length);
      remap[0] = 0;
      for (let i = 1; i < other.materials.length; i++) {
        const source = other.materials[i];
        let found = -1;
        for (let j = 1; j < this.materials.length; j++) {
          const target = this.materials[j];
          if (target.material === source.material && target.pass === source.pass) { found = j; break; }
        }
        if (found < 0) {
          found = this.materials.length;
          this.materials.push(cloneSlot(source));
        }
        remap[i] = found;
      }
      for (const source of other.vertices) {
        const vertex = cloneVertex(source);
        if (vertex.next >= 0) vertex.next += vertexOffset;
        if (vertex.first >= 0) vertex.first += vertexOffset;
        if (vertex.reIndex >= 0) vertex.reIndex += vertexOffset;
        this.vertices.push(vertex);
      }
      for (const source of other.edges) {
        const edge = cloneEdge(source);
        for (let side = 0; side < 2; side++) {
          if (edge.next[side] >= 0) edge.next[side] += edgeOffset * 2;
          if (edge.prev[side] >= 0) edge.prev[side] += edgeOffset * 2;
          if (edge.face[side] >= 0) edge.face[side] += faceOffset;
          if (edge.vert[side] >= 0) edge.vert[side] += vertexOffset;
        }
        this.edges.push(edge);
      }
      for (const source of other.faces) {
        const face = cloneFace(source);
        if (face.edge >= 0) face.edge += edgeOffset * 2;
        face.material = remap[face.material] ?? 0;
        this.faces.push(face);
      }
      for (const source of other.collisions) {
        this.collisions.push(cloneCollision(source, vertexOffset));
      }
      for (const light of other.lights) this.lights.push((light | 0) + vertexOffset);
      for (const part of other.parts) this.parts.push((part | 0) + faceOffset);
      this.pivot = -1;
      this.gotNormals = false;
      this._syncAliases();
      this._touch();
      return true;
    }

    deleteSelectedFaces() {
      for (const face of this.faces) {
        if (face.select) {
          face.material = 0;
          face.select = false;
          face.mask = 0;
        }
      }
      return this._touch();
    }

    calculateNormals() {
      const vertices = this.vertices;
      const edges = this.edges;
      const vertexEdges = new Int32Array(vertices.length);
      vertexEdges.fill(-1);
      for (let halfedge = 0; halfedge < edges.length * 2; halfedge++) {
        const vertex = this.getVertId(halfedge);
        if (vertex >= 0 && vertex < vertices.length) vertexEdges[vertex] = halfedge;
      }

      const traversalLimit = edges.length * 2 + 2;
      for (let component = 0; component < 2; component++) {
        const creaseMask = component ? FEATURE.UV0 : FEATURE.NORMAL;
        for (let vertex = 0; vertex < vertices.length; vertex++) {
          let edge = vertexEdges[vertex];
          if (edge < 0) continue;
          let end = edge;
          let guard = 0;
          do {
            if (edges[edge >> 1].crease & creaseMask) end = edge;
            else edge = this.prevVertEdge(edge);
            if (++guard > traversalLimit) throw new Error('broken GenMesh vertex loop');
          } while (edge !== end);

          end = edge;
          const v0 = vertices[this.getVertId(edge)];
          let v2 = vertices[this.getVertId(this.nextFaceEdge(edge))];
          let t2x = v2.position[0] - v0.position[0];
          let t2y = v2.position[1] - v0.position[1];
          let t2z = v2.position[2] - v0.position[2];
          let t2Length2 = t2x * t2x + t2y * t2y + t2z * t2z;
          let accuX = 0, accuY = 0, accuZ = 0;
          let exit = false, wasExit = false;
          guard = 0;
          do {
            wasExit = exit;
            const t1x = t2x, t1y = t2y, t1z = t2z;
            const t1Length2 = t2Length2;
            const v1 = v2;
            v2 = vertices[this.getVertId(wasExit
              ? this.prevFaceEdge(this.prevVertEdge(edge))
              : this.nextFaceEdge(edge))];
            t2x = v2.position[0] - v0.position[0];
            t2y = v2.position[1] - v0.position[1];
            t2z = v2.position[2] - v0.position[2];
            t2Length2 = t2x * t2x + t2y * t2y + t2z * t2z;

            if (component === 0) {
              if (t1Length2 * t2Length2 > 1e-20) {
                const scale = 1 / (t1Length2 * t2Length2);
                accuX += (t1y * t2z - t1z * t2y) * scale;
                accuY += (t1z * t2x - t1x * t2z) * scale;
                accuZ += (t1x * t2y - t1y * t2x) * scale;
              }
            } else {
              const c1 = v0.uv[1] - v1.uv[1];
              const c2 = v2.uv[1] - v0.uv[1];
              let determinant = (v1.uv[0] - v0.uv[0]) * c2 +
                (v2.uv[0] - v0.uv[0]) * c1;
              if (Math.abs(determinant) > 1e-20) {
                determinant = 1 / determinant;
                let tx = t1x * c2 * determinant + t2x * c1 * determinant;
                let ty = t1y * c2 * determinant + t2y * c1 * determinant;
                let tz = t1z * c2 * determinant + t2z * c1 * determinant;
                const length2 = tx * tx + ty * ty + tz * tz;
                if (length2 > 1e-20) {
                  const inverseLength = 1 / Math.sqrt(length2);
                  tx *= inverseLength; ty *= inverseLength; tz *= inverseLength;
                  accuX += tx; accuY += ty; accuZ += tz;
                }
              }
            }

            edge = this.nextVertEdge(edge);
            exit = edge === end || (edges[edge >> 1].crease & creaseMask) !== 0;
            if (++guard > traversalLimit) throw new Error('broken GenMesh vertex loop');
          } while (!wasExit);

          const output = component ? vertices[vertex].tangent : vertices[vertex].normal;
          if (component) {
            const normal = vertices[vertex].normal;
            const parallel = accuX * normal[0] + accuY * normal[1] + accuZ * normal[2];
            accuX -= normal[0] * parallel;
            accuY -= normal[1] * parallel;
            accuZ -= normal[2] * parallel;
          }
          const length2 = accuX * accuX + accuY * accuY + accuZ * accuZ;
          if (length2 > 1e-20) {
            const inverseLength = 1 / Math.sqrt(length2);
            output[0] = f32(accuX * inverseLength);
            output[1] = f32(accuY * inverseLength);
            output[2] = f32(accuZ * inverseLength);
          } else {
            output[0] = 1;
            output[1] = output[2] = 0;
          }
          output[3] = 0;
        }
      }
      this.gotNormals = true;
      this._prepared = null;
      return this;
    }

    needNormals() { return this.gotNormals ? this : this.calculateNormals(); }

    bounds(activeOnly = true) {
      const referenced = activeOnly ? new Uint8Array(this.vertices.length) : null;
      if (referenced) {
        for (const face of this.faces) {
          if (!face.material) continue;
          for (const id of this.faceVertices(face)) referenced[id] = 1;
        }
      }
      const min = new Float32Array([Infinity, Infinity, Infinity]);
      const max = new Float32Array([-Infinity, -Infinity, -Infinity]);
      for (let i = 0; i < this.vertices.length; i++) {
        if (referenced && !referenced[i]) continue;
        const p = this.vertices[this.vertices[i].first]?.position || this.vertices[i].position;
        for (let axis = 0; axis < 3; axis++) {
          if (p[axis] < min[axis]) min[axis] = p[axis];
          if (p[axis] > max[axis]) max[axis] = p[axis];
        }
      }
      if (!Number.isFinite(min[0])) min.fill(0), max.fill(0);
      return { min, max };
    }

    releaseTopologyForPlayback() {
      if (!this._prepared) throw new Error('GenMesh must be prepared before releasing topology');
      const deferredRelease = this._playbackTopologyRelease || null;
      this._playbackTopologyRelease = null;
      this._compact = null;
      this._vertices = [];
      this._edges = [];
      this._faces = [];
      this.topologyReleasedForPlayback = true;
      this.released = false;
      deferredRelease?.release?.(this._prepared);
      return this;
    }

    prepare(options = {}) {
      const releaseTopology = Boolean(options.releaseTopology || this._playbackTopologyRelease);
      if (this._prepared) {
        if (releaseTopology) this.releaseTopologyForPlayback();
        return this._prepared;
      }
      if (this._compact) {
        if (!this.gotNormals) {
          // Always allocate the derived direction channel. Material-only
          // compact branches can share an older channel with their source;
          // replacing it here preserves that copy-on-write boundary.
          this._compact.vertexDirections = calculateCompactNormals(this._compact);
          this.gotNormals = true;
        }
        this._prepared = prepareCompactMesh(this, this._compact);
        if (releaseTopology) this.releaseTopologyForPlayback();
        return this._prepared;
      }
      this.needNormals();
      const count = this.vertices.length;
      const positions = new Float32Array(count * 3);
      const normals = new Float32Array(count * 3);
      const tangents = new Float32Array(count * 4);
      const colors = new Float32Array(count * 4);
      const uvs = new Float32Array(count * 2);
      const uv1s = this.vertexMask & FEATURE.UV1 ? new Float32Array(count * 2) : null;
      const shadowVertexMap = new Uint32Array(count);
      for (let i = 0; i < count; i++) {
        const vertex = this.vertices[i];
        const first = vertex.first >= 0 && vertex.first < count ? vertex.first : i;
        const position = this.vertices[first]?.position || vertex.position;
        positions.set(position.subarray(0, 3), i * 3);
        normals.set(vertex.normal.subarray(0, 3), i * 3);
        tangents.set(vertex.tangent, i * 4);
        colors.set(vertex.color, i * 4);
        uvs.set(vertex.uv.subarray(0, 2), i * 2);
        if (uv1s) uv1s.set(vertex.uv1.subarray(0, 2), i * 2);
        shadowVertexMap[i] = first;
      }
      const perMaterial = new Map();
      const perMaterialShadow = new Map();
      for (const face of this.faces) {
        if (!face.material || face.edge < 0) continue;
        const ids = this.faceVertices(face);
        if (ids.length < 3) continue;
        let list = perMaterial.get(face.material);
        if (!list) perMaterial.set(face.material, list = []);
        let shadow = perMaterialShadow.get(face.material);
        if (!shadow) perMaterialShadow.set(face.material, shadow = []);
        for (let i = 1; i + 1 < ids.length; i++) {
          list.push(ids[0], ids[i], ids[i + 1]);
          shadow.push(face.used ? 1 : 0);
        }
      }
      const indexValues = [];
      const triangleMaterials = [];
      const shadowTriangleValues = [];
      const groups = [];
      for (const [materialIndex, list] of perMaterial) {
        const start = indexValues.length;
        indexValues.push(...list);
        shadowTriangleValues.push(...(perMaterialShadow.get(materialIndex) || []));
        for (let i = 0; i < list.length / 3; i++) triangleMaterials.push(materialIndex);
        const slot = this.materials[materialIndex] || this.materials[0];
        groups.push({
          material: slot.material, materialIndex, pass: slot.pass | 0,
          start, count: list.length,
        });
      }
      const prepared = this._prepared = {
        kind: 'mesh-buffer', positions, normals, tangents, colors, uvs, uv1s,
        indices: new Uint32Array(indexValues),
        triangleMaterials: new Uint16Array(triangleMaterials),
        shadowVertexMap, shadowTriangleMask: new Uint8Array(shadowTriangleValues),
        groups, materials: this.materials, bounds: this.bounds(true),
      };
      // Renderer buffers are now self-contained. Return topology to dormant
      // flat storage so preparing many scene meshes does not accumulate the
      // expanded record representation alongside their GPU-ready arrays.
      if (releaseTopology) this.releaseTopologyForPlayback();
      else this.compact();
      return prepared;
    }

    summary() {
      let topologyHash = 0x811c9dc5;
      let vertexHash = 0x811c9dc5;
      const attributeCount = this.vertexMask & FEATURE.UV1 ? 6 : 5;
      for (const vertex of this.vertices) {
        topologyHash = hashWord(topologyHash,
          (vertex.mask & 255) | ((vertex.id & 255) << 8) |
          ((vertex.select ? 1 : 0) << 16) | ((vertex.used ? 1 : 0) << 17));
        topologyHash = hashWord(topologyHash, vertex.first);
        topologyHash = hashWord(topologyHash, vertex.next);
        for (let attribute = 0; attribute < attributeCount; attribute++) {
          const value = vertex.values[attribute];
          for (let i = 0; i < 4; i++) vertexHash = hashWord(vertexHash, floatBits(value[i]));
        }
      }
      for (const edge of this.edges) {
        topologyHash = hashWord(topologyHash, edge.mask | (edge.crease << 8));
        for (let side = 0; side < 2; side++) {
          topologyHash = hashWord(topologyHash, edge.next[side]);
          topologyHash = hashWord(topologyHash, edge.prev[side]);
          topologyHash = hashWord(topologyHash, edge.face[side]);
          topologyHash = hashWord(topologyHash, edge.vert[side]);
        }
      }
      let activeFaces = 0;
      for (const face of this.faces) {
        if (face.material) activeFaces++;
        topologyHash = hashWord(topologyHash,
          face.mask | (face.material << 8) | ((face.used ? 1 : 0) << 24));
        topologyHash = hashWord(topologyHash, face.edge);
      }
      const box = this.bounds();
      return {
        kind: this.kind,
        vertices: this.vertices.length,
        edges: this.edges.length,
        faces: this.faces.length,
        activeFaces,
        materials: this.materials.length,
        parts: this.parts.length,
        bounds: { min: Array.from(box.min), max: Array.from(box.max) },
        topologyHash: topologyHash >>> 0,
        vertexHash: vertexHash >>> 0,
      };
    }
  }

  function requireMesh(value) {
    return value instanceof Mesh ? value : null;
  }

  function checkedCopy(value, mask = 0, owned = false) {
    const source = requireMesh(value);
    if (!source) return null;
    const mesh = owned ? source : source.clone();
    mask >>>= 0;
    if (!(mask & 0x80000000)) {
      if (mask) mesh.maskToSelection(mask);
      else mesh.allToSelection(true);
    }
    return mesh;
  }

  const runtimeConsumerCounts = new WeakMap();
  const runtimeConsumers = new WeakMap();
  const runtimeConsumptionState = new WeakMap();

  function consumerCounts(runtime) {
    if (!runtime || !Array.isArray(runtime.operations)) return null;
    let counts = runtimeConsumerCounts.get(runtime);
    if (counts) return counts;
    counts = new Map();
    const consumers = new Map();
    const add = (producer, consumer = null) => {
      if (!producer) return;
      counts.set(producer, (counts.get(producer) || 0) + 1);
      let list = consumers.get(producer);
      if (!list) consumers.set(producer, list = []);
      list.push(consumer);
    };
    for (const op of runtime.operations) {
      for (const input of op.inputs || []) add(input, op);
      for (const link of op.links || []) add(link, op);
    }
    for (const root of runtime.roots || []) add(root);
    for (const event of runtime.events || []) add(event.op);
    runtimeConsumerCounts.set(runtime, counts);
    runtimeConsumers.set(runtime, consumers);
    return counts;
  }

  function meshInputIsUniquelyConsumed(call, inputIndex = 0) {
    const inputOp = call?.op?.inputs?.[inputIndex];
    const counts = consumerCounts(call?.runtime);
    if (!inputOp || !counts) return false;
    let state = runtimeConsumptionState.get(call.runtime);
    if (!state) {
      state = { remaining: new Map(counts), claimed: new WeakMap() };
      runtimeConsumptionState.set(call.runtime, state);
    }
    let claimed = state.claimed.get(call.op);
    if (!claimed) state.claimed.set(call.op, claimed = new Set());
    if (!claimed.has(inputIndex)) {
      claimed.add(inputIndex);
      state.remaining.set(inputOp, Math.max(0, (state.remaining.get(inputOp) || 0) - 1));
    }
    // Ownership is safe for the final graph consumer, not merely for a node
    // whose static fan-out was one. This mirrors the native Release() order.
    return state.remaining.get(inputOp) === 0;
  }

  function inputOwnership(call) {
    return call.inputs.map((value, index) => value instanceof Mesh && meshInputIsUniquelyConsumed(call, index));
  }

  const OLD_MESH_HANDLER_IDS = new Set([
    0x81, 0x82, 0x83, 0x84, 0x86, 0x87, 0x88, 0x89, 0x92, 0x93,
    0x95, 0x96, 0x9d, 0xa0, 0xa1, 0xa3, 0xa5, 0xa6, 0xab, 0xb0, 0xb4,
  ]);

  function finalizeMeshOutput(call, result) {
    if (!(result instanceof Mesh) || !call?.runtime || !call.op) return result;
    consumerCounts(call.runtime);
    const consumers = runtimeConsumers.get(call.runtime)?.get(call.op) || [];
    // Keep the hot representation only along a single linear old-Mesh chain.
    // Forks and terminal scene/minmesh inputs otherwise sit dormant until a
    // later consumer, so their flat representation is decisively cheaper.
    const linearOldMeshChain = consumers.length === 1 &&
      OLD_MESH_HANDLER_IDS.has(consumers[0]?.classId);
    if (!linearOldMeshChain) result.compact();
    return result;
  }

  function dispatchMeshInputs(call, callback, expandInputs = true) {
    const ownership = inputOwnership(call);
    const sources = [];
    const seen = new Set();
    for (const input of call.inputs) {
      if (!(input instanceof Mesh) || seen.has(input)) continue;
      seen.add(input);
      if (expandInputs) input.ensureExpanded();
      sources.push(input);
    }
    const result = callback(ownership);
    // A non-result input cache remains observable through its operator, but
    // none of its object-form records are needed until another consumer asks
    // for them. Compacting here avoids retaining fork snapshots at full JS
    // object cost while preserving the exact cache identity and public API.
    for (const source of sources) {
      if (source !== result && !source.released) source.compact();
    }
    return finalizeMeshOutput(call, result);
  }

  function meshStorageStats(runtime) {
    const meshes = new Set();
    for (const op of runtime?.operations || []) {
      if (op.cache instanceof Mesh) meshes.add(op.cache);
    }
    let released = 0, compact = 0, expanded = 0, preparedOnly = 0;
    let vertices = 0, edges = 0, faces = 0, compactBytes = 0, preparedBytes = 0;
    const compactBuffers = new Set();
    const preparedBuffers = new Set();
    for (const mesh of meshes) {
      const storage = mesh.storageSummary();
      vertices += storage.vertices;
      edges += storage.edges;
      faces += storage.faces;
      for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(mesh._prepared || {}))) {
        if (!Object.prototype.hasOwnProperty.call(descriptor, 'value')) continue;
        const value = descriptor.value;
        if (ArrayBuffer.isView(value) && value.buffer && !preparedBuffers.has(value.buffer)) {
          preparedBuffers.add(value.buffer);
          preparedBytes += value.buffer.byteLength;
        }
      }
      if (storage.released) released++;
      else if (storage.compact) {
        compact++;
        for (const value of Object.values(mesh._compact)) {
          if (ArrayBuffer.isView(value) && !compactBuffers.has(value.buffer)) {
            compactBuffers.add(value.buffer);
            compactBytes += value.buffer.byteLength;
          }
        }
      } else if (storage.preparedOnly) preparedOnly++;
      else expanded++;
    }
    return {
      references: (runtime?.operations || []).reduce((total, op) =>
        total + (op.cache instanceof Mesh ? 1 : 0), 0),
      identities: meshes.size, released, compact, expanded, preparedOnly,
      vertices, edges, faces, compactBytes, preparedBytes,
    };
  }

  function physicalKey(position) {
    return `${f32(position[0])},${f32(position[1])},${f32(position[2])}`;
  }

  function cornerAdder(mesh) {
    const firstByKey = new Map();
    return function add(position, uv, mask = 0, key = physicalKey(position)) {
      const first = firstByKey.get(key);
      const index = mesh.addVertex(position, uv, mask, first === undefined ? -1 : first);
      if (first === undefined) firstByKey.set(key, index);
      return index;
    };
  }

  // Crease() stores a separate attribute wedge on each side of a UV seam.
  // Procedural constructors below already emit those wedges directly, so the
  // remaining native side effect is the UV crease bit on their shared edge.
  function markUVDiscontinuities(mesh) {
    for (let edgeIndex = 0; edgeIndex < mesh.edges.length; edgeIndex++) {
      const edge = mesh.edges[edgeIndex];
      const halfedge0 = edgeIndex * 2;
      const halfedge1 = halfedge0 + 1;
      const start0 = mesh.vertices[mesh.getVertId(halfedge0)]?.uv;
      const end0 = mesh.vertices[mesh.getVertId(mesh.nextFaceEdge(halfedge0))]?.uv;
      const start1 = mesh.vertices[mesh.getVertId(halfedge1)]?.uv;
      const end1 = mesh.vertices[mesh.getVertId(mesh.nextFaceEdge(halfedge1))]?.uv;
      if (!start0 || !end0 || !start1 || !end1) continue;
      // Opposite halfedges run in opposite directions: start0 corresponds to
      // end1 physically, and end0 corresponds to start1.
      const firstEndpointDiffers = start0[0] !== end1[0] || start0[1] !== end1[1];
      const secondEndpointDiffers = end0[0] !== start1[0] || end0[1] !== start1[1];
      // A crease splits the whole edge. A difference at just one endpoint is
      // the wedge split created by another incident crease (not a second edge
      // crease), most visibly in the triangle fans at the sphere poles.
      if (firstEndpointDiffers && secondEndpointDiffers) {
        edge.crease |= FEATURE.UV0;
      }
    }
  }

  function orient(mesh, ids, target) {
    const a = mesh.vertices[ids[0]].position;
    const b = mesh.vertices[ids[1]].position;
    const c = mesh.vertices[ids[2]].position;
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    if (nx * target[0] + ny * target[1] + nz * target[2] < 0) {
      return [ids[0], ...ids.slice(1).reverse()];
    }
    return ids;
  }

  function applySRT(mesh, srt) {
    const matrix = mat4SRT(new Float32Array(srt), new Float32Array(16));
    // Mesh_Cube finishes with All2Sel(1) before its optional bottom shift and
    // authored SRT, leaving every native topology domain selected.
    mesh.allToSelection(true, ALL.EDGE | ALL.FACE | ALL.VERT);
    mesh.transformVertices(matrix);
    mesh.gotNormals = false;
    return mesh;
  }

  function Mesh_Cube(tx, ty, tz, flags, srt) {
    tx = Math.max(1, tx | 0); ty = Math.max(1, ty | 0); tz = Math.max(1, tz | 0);
    flags |= 0;
    srt = Array.from(srt || [1, 1, 1, 0, 0, 0, 0, 0, 0], f32);
    const mesh = new Mesh();
    const add = cornerAdder(mesh);
    const polygons = [];
    const uvScale = flags & 8 ? [srt[0], srt[1], srt[2]] : [1, 1, 1];

    for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
      const axis = faceIndex >> 1;
      const sign = faceIndex & 1 ? 1 : -1;
      let uSegments, vSegments;
      if (axis === 0) { uSegments = ty; vSegments = tz; }
      else if (axis === 1) { uSegments = tx; vSegments = tz; }
      else { uSegments = tx; vSegments = ty; }
      const grid = Array.from({ length: uSegments + 1 }, () => new Int32Array(vSegments + 1));
      for (let u = 0; u <= uSegments; u++) for (let v = 0; v <= vSegments; v++) {
        const position = [0, 0, 0];
        if (axis === 0) { position[0] = sign * 0.5; position[1] = u / uSegments - 0.5; position[2] = v / vSegments - 0.5; }
        else if (axis === 1) { position[0] = u / uSegments - 0.5; position[1] = sign * 0.5; position[2] = v / vSegments - 0.5; }
        else { position[0] = u / uSegments - 0.5; position[1] = v / vSegments - 0.5; position[2] = sign * 0.5; }
        if (flags & 4) position[1] += 0.5;
        let uv;
        if (axis === 0) uv = [sign * position[2] * uvScale[2] + 0.5 * uvScale[2], -position[1] * uvScale[1] + (flags & 4 ? uvScale[1] : 0.5 * uvScale[1])];
        else if (axis === 1) uv = [position[0] * uvScale[0] + 0.5 * uvScale[0], -sign * position[2] * uvScale[2] + 0.5 * uvScale[2]];
        else uv = [-sign * position[0] * uvScale[0] + 0.5 * uvScale[0], -position[1] * uvScale[1] + (flags & 4 ? uvScale[1] : 0.5 * uvScale[1])];
        let uOffset = 0;
        if (flags & 2) {
          if (faceIndex === 4) uOffset = uvScale[2];
          else if (faceIndex === 1) uOffset = uvScale[2] + uvScale[0];
          else if (faceIndex === 5) uOffset = uvScale[2] * 2 + uvScale[0];
        }
        uv[0] += uOffset;
        grid[u][v] = add(position, uv, 1 << faceIndex);
      }
      const target = [0, 0, 0]; target[axis] = sign;
      for (let u = 0; u < uSegments; u++) for (let v = 0; v < vSegments; v++) {
        const ids = [grid[u][v], grid[u + 1][v], grid[u + 1][v + 1], grid[u][v + 1]];
        polygons.push({ verts: orient(mesh, ids, target), mask: 1 << faceIndex, material: 1 });
      }
    }
    mesh.linkVertexCopies();
    mesh.setPolygons(polygons);
    for (const edge of mesh.edges) {
      const face0 = mesh.faces[edge.face[0]], face1 = mesh.faces[edge.face[1]];
      // Each of the six SelectCube/Vert2FaceEdge passes stores its face bit on
      // every coplanar edge. Boundary edges therefore retain both incident
      // face bits, while tessellation edges retain their one plane bit.
      edge.mask = ((face0?.material ? face0.mask : 0) |
        (face1?.material ? face1.mask : 0)) & 255;
      if (face0 && face1 && face0.mask !== face1.mask) {
        edge.crease = flags & 1 ? 0x1fe : FEATURE.UV0;
      }
    }
    mesh.parts.push(0);
    return applySRT(mesh, srt);
  }

  function Mesh_Cylinder(tx, ty, mode, tz, arc) {
    tx = Math.max(3, tx | 0);
    ty = Math.max(1, ty | 0);
    tz = Math.max(1, tz | 0);
    mode |= 0;
    arc = Math.max(0, Math.min(tx - 1, arc | 0));
    const mesh = new Mesh();
    const add = cornerAdder(mesh);
    const polygons = [];
    const yOffset = mode & 2 ? 0.5 : 0;

    if (arc === 0) {
      const side = Array.from({ length: ty + 1 }, () => new Int32Array(tx + 1));
      for (let level = 0; level <= ty; level++) for (let j = 0; j <= tx; j++) {
        const angle = TAU * (j % tx) / tx;
        const position = [0.5 * Math.sin(angle), 0.5 - level / ty + yOffset, -0.5 * Math.cos(angle)];
        side[level][j] = add(position, [j / tx, level / ty], 0, `c:${level}:${j % tx}`);
      }
      for (let level = 0; level < ty; level++) for (let j = 0; j < tx; j++) {
        const ids = [side[level][j], side[level + 1][j], side[level + 1][j + 1], side[level][j + 1]];
        const angle = TAU * (j + 0.5) / tx;
        polygons.push({ verts: orient(mesh, ids, [Math.sin(angle), 0, -Math.cos(angle)]), material: 1 });
      }

      for (const cap of [0, 1]) {
        const sign = cap === 0 ? 1 : -1;
        const y = sign * 0.5 + yOffset;
        const center = add([0, y, 0], [0.5, 0.5], 1, `cc:${cap}:0`);
        let previous = null;
        for (let ring = 1; ring <= tz; ring++) {
          const current = new Int32Array(tx + 1);
          const radius = 0.5 * ring / tz;
          for (let j = 0; j <= tx; j++) {
            const angle = TAU * (j % tx) / tx;
            const position = [radius * Math.sin(angle), y, -radius * Math.cos(angle)];
            const uv = [position[0] + 0.5, -sign * position[2] + 0.5];
            const key = ring === tz ? `c:${cap === 0 ? 0 : ty}:${j % tx}` : `cc:${cap}:${ring}:${j % tx}`;
            current[j] = add(position, uv, 1, key);
          }
          if (ring === 1) {
            for (let j = 0; j < tx; j++) {
              const ids = [center, current[j], current[j + 1]];
              polygons.push({ verts: orient(mesh, ids, [0, sign, 0]), mask: 1, material: mode & 1 ? 0 : 1 });
            }
          } else {
            for (let j = 0; j < tx; j++) {
              const ids = [previous[j], current[j], current[j + 1], previous[j + 1]];
              polygons.push({ verts: orient(mesh, ids, [0, sign, 0]), mask: 1, material: mode & 1 ? 0 : 1 });
            }
          }
          previous = current;
        }
      }
    } else {
      // Ring's arc mode is a circular sector: retained perimeter samples plus
      // one center point. Extruding that polygon exactly closes both radial walls.
      const outerCount = tx - arc + 1;
      const base = [];
      for (let j = 0; j < outerCount; j++) {
        const angle = TAU * j / tx;
        base.push([0.5 * Math.sin(angle), -0.5 * Math.cos(angle)]);
      }
      base.push([0, 0]);
      const rings = Array.from({ length: ty + 1 }, () => new Int32Array(base.length));
      for (let level = 0; level <= ty; level++) for (let j = 0; j < base.length; j++) {
        const position = [base[j][0], 0.5 - level / ty + yOffset, base[j][1]];
        rings[level][j] = add(position, [j / base.length, level / ty], 0, `ca:${level}:${j}`);
      }
      for (let level = 0; level < ty; level++) for (let j = 0; j < base.length; j++) {
        const next = (j + 1) % base.length;
        const ids = [rings[level][j], rings[level + 1][j], rings[level + 1][next], rings[level][next]];
        const mx = base[j][0] + base[next][0], mz = base[j][1] + base[next][1];
        polygons.push({ verts: orient(mesh, ids, [mx, 0, mz]), material: 1 });
      }
      for (const cap of [0, 1]) {
        const sign = cap === 0 ? 1 : -1;
        const loop = Array.from(rings[cap === 0 ? 0 : ty]);
        const centerIndex = loop.length - 1;
        for (let j = 0; j + 1 < centerIndex; j++) {
          const ids = [loop[centerIndex], loop[j], loop[j + 1]];
          polygons.push({ verts: orient(mesh, ids, [0, sign, 0]), mask: 1, material: mode & 1 ? 0 : 1 });
        }
      }
    }

    mesh.linkVertexCopies();
    mesh.setPolygons(polygons);
    for (const edge of mesh.edges) {
      const face0 = mesh.faces[edge.face[0]], face1 = mesh.faces[edge.face[1]];
      if (face0 && face1 && ((face0.mask ^ face1.mask) & 1)) edge.crease |= 0xfe;
      const a = mesh.vertices[edge.vert[0]]?.position;
      const b = mesh.vertices[edge.vert[1]]?.position;
      if (a && b && Math.abs(a[0]) < 1e-6 && Math.abs(b[0]) < 1e-6 && a[2] < -0.49 && b[2] < -0.49) {
        edge.crease |= FEATURE.UV0;
      }
    }
    mesh.parts.push(0);
    mesh.gotNormals = false;
    return mesh;
  }

  function Mesh_Torus(tx, ty, ro, ri, phase, arcLength, flags) {
    tx = Math.max(3, tx | 0); ty = Math.max(3, ty | 0);
    ro = f32(ro); ri = f32(ri); phase = f32(phase); arcLength = f32(arcLength); flags |= 0;
    if (flags & 1) {
      ri = f32((ro - ri) * 0.5);
      ro = f32(ro - ri);
    }
    const closed = arcLength === 1;
    const mesh = new Mesh();
    const add = cornerAdder(mesh);
    const polygons = [];
    const grid = Array.from({ length: tx + 1 }, () => new Int32Array(ty + 1));
    for (let major = 0; major <= tx; major++) for (let minor = 0; minor <= ty; minor++) {
      const majorStep = closed ? major % tx : major;
      const a = arcLength * TAU * majorStep / tx;
      const b = phase * TAU / ty + TAU * (minor % ty) / ty;
      const radial = f32(Math.sin(b) * ri - ro);
      const position = [radial * Math.cos(a), -Math.cos(b) * ri + (flags & 2 ? ri : 0), -radial * Math.sin(a)];
      const majorKey = closed ? major % tx : major;
      grid[major][minor] = add(position, [minor / ty, major / tx], 0, `t:${majorKey}:${minor % ty}`);
    }
    const majorCells = closed ? tx : tx;
    for (let major = 0; major < majorCells; major++) for (let minor = 0; minor < ty; minor++) {
      const ids = [grid[major][minor], grid[major][minor + 1], grid[major + 1][minor + 1], grid[major + 1][minor]];
      const a = arcLength * TAU * (major + 0.5) / tx;
      const b = phase * TAU / ty + TAU * (minor + 0.5) / ty;
      const target = [Math.sin(b) * Math.cos(a), -Math.cos(b), -Math.sin(b) * Math.sin(a)];
      polygons.push({ verts: orient(mesh, ids, target), material: 1 });
    }
    if (!closed) {
      for (const end of [0, tx]) {
        // Native Crease(UV0) splits both cap rims from the swept surface even
        // though the inherited cap UV values initially agree. Emit that split
        // up front while retaining one physical First cycle for shadowing.
        const ids = [];
        for (let minor = 0; minor < ty; minor++) {
          const source = mesh.vertices[grid[end][minor]];
          ids.push(add(source.position, source.uv, 1, `t:${end}:${minor}`));
        }
        const a = arcLength * TAU * end / tx;
        const tangent = [-Math.sin(a), 0, -Math.cos(a)];
        const sign = end === 0 ? 1 : -1;
        polygons.push({ verts: orient(mesh, ids, tangent.map(value => value * sign)), mask: 1, material: 1 });
      }
    }
    mesh.linkVertexCopies();
    mesh.setPolygons(polygons);
    for (const edge of mesh.edges) {
      const face0 = mesh.faces[edge.face[0]], face1 = mesh.faces[edge.face[1]];
      if (face0 && face1 && face0.mask !== face1.mask) edge.crease |= FEATURE.UV0;
    }
    markUVDiscontinuities(mesh);
    mesh.parts.push(0);
    mesh.allToMask(0xffffffff, SELECT.SETNOT);
    mesh.gotNormals = false;
    return mesh;
  }

  function Mesh_Sphere(tx, ty) {
    tx = Math.max(3, tx | 0); ty = Math.max(1, ty | 0);
    const mesh = new Mesh();
    const add = cornerAdder(mesh);
    const polygons = [];
    const rings = Array.from({ length: ty + 1 }, () => new Int32Array(tx + 1));
    for (let latitude = 0; latitude <= ty; latitude++) {
      const phi = (latitude + 0.5) * PI / (ty + 1);
      const radius = 0.5 * Math.sin(phi);
      const y = -0.5 * Math.cos(phi);
      for (let longitude = 0; longitude <= tx; longitude++) {
        const theta = TAU * (longitude % tx) / tx;
        const position = [radius * Math.sin(theta), y, -radius * Math.cos(theta)];
        rings[latitude][longitude] = add(position, [longitude / tx, 1 - latitude / ty], 0,
          `s:${latitude}:${longitude % tx}`);
      }
    }
    for (let latitude = 0; latitude < ty; latitude++) for (let longitude = 0; longitude < tx; longitude++) {
      const ids = [rings[latitude][longitude], rings[latitude + 1][longitude],
        rings[latitude + 1][longitude + 1], rings[latitude][longitude + 1]];
      const p = mesh.vertices[ids[0]].position;
      polygons.push({ verts: orient(mesh, ids, p), material: 1 });
    }
    const south = add([0, -0.5, 0], [0, 1], 0, 's:south');
    const north = add([0, 0.5, 0], [0, 0], 0, 's:north');
    // The native radial UV crease also splits each pole. The final longitude
    // wedge is translated by +1 together with the duplicated ring seam.
    const southSeam = add([0, -0.5, 0], [1, 1], 0, 's:south');
    const northSeam = add([0, 0.5, 0], [1, 0], 0, 's:north');
    for (let longitude = 0; longitude < tx; longitude++) {
      const southPole = longitude + 1 === tx ? southSeam : south;
      const northPole = longitude + 1 === tx ? northSeam : north;
      let ids = [southPole, rings[0][longitude + 1], rings[0][longitude]];
      polygons.push({ verts: orient(mesh, ids, [0, -1, 0]), mask: 1, material: 1 });
      ids = [northPole, rings[ty][longitude], rings[ty][longitude + 1]];
      polygons.push({ verts: orient(mesh, ids, [0, 1, 0]), mask: 1, material: 1 });
    }
    mesh.linkVertexCopies();
    mesh.setPolygons(polygons);
    markUVDiscontinuities(mesh);
    mesh.parts.push(0);
    mesh.allToMask(0x00fffeff, SELECT.SUB);
    mesh.gotNormals = false;
    return mesh;
  }

  function Mesh_Grid(mode, tessU, tessV) {
    mode |= 0; tessU = Math.max(1, tessU | 0); tessV = Math.max(1, tessV | 0);
    const mesh = new Mesh();
    const add = cornerAdder(mesh);
    const polygons = [];
    for (let side = 0; side < 2; side++) {
      const grid = Array.from({ length: tessU + 1 }, () => new Int32Array(tessV + 1));
      for (let u = 0; u <= tessU; u++) for (let v = 0; v <= tessV; v++) {
        // Native Grid starts as an XY ring with V running downward, then
        // rotates it +90 degrees around X: spatial +Z therefore corresponds
        // to V=0. Keeping that position/UV relation matters for asymmetric
        // alpha-cutout planes such as the bridge supports in Debris.
        const position = [u / tessU - 0.5, 0, 0.5 - v / tessV];
        const sideMask = side === 0 ? 1 : 2;
        let mask = sideMask;
        if (v === 0 || v === tessV) mask |= 4; else mask |= 8;
        if (u === 0 || u === tessU) mask |= 16; else mask |= 32;
        if (mask & (4 | 16)) mask |= 64; else mask |= 128;
        // CalcNormals in the native constructor mirrors U only on the -Y
        // face. Together with the Z/V relation above this preserves its
        // original texture orientation and tangent basis on both sides.
        const uv = side === 0 ? [1 - u / tessU, v / tessV] : [u / tessU, v / tessV];
        // Native Crease/SplitBridge keeps the original front/back First link
        // along the perimeter but creates separate physical interior vertices
        // for each side. Welding every coincident interior wedge makes their
        // opposite normals cancel to the +X zero-length fallback.
        const perimeter = u === 0 || u === tessU || v === 0 || v === tessV;
        const physicalKey = perimeter ? `g:${u}:${v}` : `g:${side}:${u}:${v}`;
        grid[u][v] = add(position, uv, mask, physicalKey);
      }
      const target = [0, side === 0 ? -1 : 1, 0];
      for (let u = 0; u < tessU; u++) for (let v = 0; v < tessV; v++) {
        // Ring()'s two native seed faces start at opposite halfedges. Keeping
        // that rotated start makes the back face fan use the other cell
        // diagonal; merely reversing the front loop creates coincident
        // opposite triangles and a four-face non-manifold shadow diagonal.
        const ids = side === 0
          ? [grid[u][v], grid[u + 1][v], grid[u + 1][v + 1], grid[u][v + 1]]
          : [grid[u + 1][v], grid[u][v], grid[u][v + 1], grid[u + 1][v + 1]];
        let mask = side === 0 ? 1 : 2;
        if (v === 0 || v + 1 === tessV) mask |= 4; else mask |= 8;
        if (u === 0 || u + 1 === tessU) mask |= 16; else mask |= 32;
        if (mask & (4 | 16)) mask |= 64; else mask |= 128;
        polygons.push({ verts: orient(mesh, ids, target), mask, material: side === 1 && !(mode & 1) ? 0 : 1 });
      }
    }
    mesh.linkVertexCopies();
    mesh.setPolygons(polygons);
    for (const edge of mesh.edges) {
      const face0 = mesh.faces[edge.face[0]], face1 = mesh.faces[edge.face[1]];
      if (face0 && face1 && ((face0.mask ^ face1.mask) & 3) === 3) edge.crease = 0xfe;
    }
    mesh.parts.push(0);
    mesh.gotNormals = false;
    return mesh;
  }

  function Mesh_Transform(mesh, mask, srt, owned = false) {
    const source = requireMesh(mesh);
    if (source?._compact && (mask | 0) === 0) {
      const result = owned ? source : source.cloneForPositionTransform();
      const storage = result._compact;
      if (owned) {
        // Mesh ownership does not imply unique typed-buffer ownership: compact
        // material branches may share their otherwise immutable positions.
        storage.vertexPositions = storage.vertexPositions.slice();
      }
      for (let index = 0; index < storage.vertexCount; index++) storage.vertexBytes[index * 4 + 2] = 1;
      for (let index = 0; index < storage.edgeCount; index++) storage.edgeBytes[index * 4 + 2] = 1;
      for (let index = 0; index < storage.faceCount; index++) {
        storage.faceBytes[index * 4 + 2] = storage.faceInts[index * 5] !== 0 ? 1 : 0;
      }

      if (result.attributeMap(ATTR.POS) >= 0) {
        let matrix = mat4SRT(new Float32Array(srt), new Float32Array(16));
        const pivot = result.pivot | 0;
        if (pivot >= 0 && pivot < storage.vertexCount) {
          const offset = pivot * 4;
          const px = storage.vertexPositions[offset];
          const py = storage.vertexPositions[offset + 1];
          const pz = storage.vertexPositions[offset + 2];
          const translate = matrixIdentity();
          translate[12] = f32(-px); translate[13] = f32(-py); translate[14] = f32(-pz);
          matrix = mat4Mul(matrix, translate, new Float32Array(16));
          matrix[12] = f32(matrix[12] + px);
          matrix[13] = f32(matrix[13] + py);
          matrix[14] = f32(matrix[14] + pz);
        }
        const positions = storage.vertexPositions;
        for (let index = 0; index < storage.vertexCount; index++) {
          const offset = index * 4;
          const x = positions[offset], y = positions[offset + 1];
          const z = positions[offset + 2], w = positions[offset + 3];
          positions[offset] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w);
          positions[offset + 1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w);
          positions[offset + 2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w);
          positions[offset + 3] = w;
        }
      }
      result.gotNormals = false;
      return result._touch();
    }
    const result = checkedCopy(mesh, (mask << 16) >>> 0, owned);
    if (!result) return null;
    const matrix = pivotTransform(result,
      mat4SRT(new Float32Array(srt), new Float32Array(16)), ATTR.POS);
    result.transformVertices(matrix);
    result.gotNormals = false;
    return result;
  }

  function Mesh_TransformEx(mesh, mask, srt, sourceAttribute, destinationAttribute, owned = false) {
    const result = checkedCopy(mesh, (mask << 16) >>> 0, owned);
    if (!result) return null;
    if (sourceAttribute === ATTR.NORMAL || sourceAttribute === ATTR.TANGENT) result.needNormals();
    const matrix = pivotTransform(result,
      mat4SRT(new Float32Array(srt), new Float32Array(16)), sourceAttribute);
    result.transformVertices(matrix, sourceAttribute, destinationAttribute);
    if (destinationAttribute === ATTR.POS || destinationAttribute === ATTR.NORMAL || destinationAttribute === ATTR.TANGENT) {
      result.gotNormals = false;
    }
    return result;
  }

  function Mesh_Add(inputs, ownership = []) {
    const firstIndex = inputs.findIndex(value => value instanceof Mesh);
    const first = inputs[firstIndex];
    if (!first) return null;
    const result = checkedCopy(first, 0, !!ownership[firstIndex]);
    const adoptedSnapshot = result === first && inputs.some((value, index) => index !== firstIndex && value === result)
      ? result.clone() : null;
    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
      const input = inputs[inputIndex];
      if (!(input instanceof Mesh)) continue;
      if (inputIndex === firstIndex) continue;
      result.append(input === result ? adoptedSnapshot : input);
      if (ownership[inputIndex] && input !== result) input.releaseStorage();
    }
    result.gotNormals = false;
    return result;
  }

  function Mesh_MatLink(mesh, material, mask, pass, owned = false) {
    if (!material) return null;
    const source = requireMesh(mesh);
    if (!source) return null;
    const selectionMask = (mask << 8) >>> 0;
    const result = source._compact
      ? (owned ? source : source.cloneForMaterial())
      : checkedCopy(source, selectionMask, owned);
    if (!result) return null;
    const slotIndex = result.materials.length;
    result.materials.push(materialSlot(material, pass));
    const storage = result._compact;
    if (storage) {
      if (owned) {
        // Compact Transform branches may share topology/material indices until
        // an operator actually edits them. Detach this mutable channel before
        // an owned MatLink changes the transferred mesh in place.
        storage.faceInts = storage.faceInts.slice();
      }
      if (!(selectionMask & 0x80000000)) {
        if (selectionMask) {
          if (selectionMask & 0x00ff0000) {
            const bits = (selectionMask >>> 16) & 255;
            for (let index = 0; index < storage.vertexCount; index++) {
              const offset = index * 4;
              storage.vertexBytes[offset + 2] = storage.vertexBytes[offset] & bits ? 1 : 0;
            }
          }
          if (selectionMask & 0x0000ff00) {
            const bits = (selectionMask >>> 8) & 255;
            for (let index = 0; index < storage.faceCount; index++) {
              const offset = index * 4;
              storage.faceBytes[offset + 2] = storage.faceBytes[offset] & bits ? 1 : 0;
            }
          }
          if (selectionMask & 255) {
            const bits = selectionMask & 255;
            for (let index = 0; index < storage.edgeCount; index++) {
              const offset = index * 4;
              storage.edgeBytes[offset + 2] = storage.edgeBytes[offset] & bits ? 1 : 0;
            }
          }
        } else {
          for (let index = 0; index < storage.vertexCount; index++) storage.vertexBytes[index * 4 + 2] = 1;
          for (let index = 0; index < storage.edgeCount; index++) storage.edgeBytes[index * 4 + 2] = 1;
          for (let index = 0; index < storage.faceCount; index++) {
            const offset = index * 4;
            storage.faceBytes[offset + 2] = storage.faceInts[index * 5] !== 0 ? 1 : 0;
          }
        }
      }
      for (let index = 0; index < storage.faceCount; index++) {
        if (storage.faceBytes[index * 4 + 2]) storage.faceInts[index * 5] = slotIndex;
      }
      result._syncAliases();
      return result._touch();
    }
    for (const face of result.faces) if (face.select) face.material = slotIndex;
    result._syncAliases();
    return result._touch();
  }

  function Mesh_SelectCube(mesh, destinationMask, mode, center, size, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    const half = [f32(size[0] * 0.5), f32(size[1] * 0.5), f32(size[2] * 0.5)];
    for (const vertex of result.vertices) {
      const position = result.vertices[vertex.first]?.position || vertex.position;
      vertex.select = Math.abs(position[0] - center[0]) <= half[0] &&
        Math.abs(position[1] - center[1]) <= half[1] &&
        Math.abs(position[2] - center[2]) <= half[2];
    }
    result.vertexToFaceEdge();
    if (mode & 4) result.faceToVertex();
    result.selectionToMask(destinationMask >>> 0, mode & 3);
    return result;
  }

  function Mesh_DeleteFaces(mesh, mask, owned = false) {
    const result = checkedCopy(mesh, (mask << 8) >>> 0, owned);
    return result ? result.deleteSelectedFaces() : null;
  }

  function Mesh_Invert(mesh, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    for (const edge of result.edges) {
      edge.temp[0] = result.getVertId(edge.next[0]);
      edge.temp[1] = result.getVertId(edge.next[1]);
    }
    for (const edge of result.edges) {
      let swap = edge.next[0]; edge.next[0] = edge.prev[0]; edge.prev[0] = swap;
      swap = edge.next[1]; edge.next[1] = edge.prev[1]; edge.prev[1] = swap;
      edge.vert[0] = edge.temp[0]; edge.vert[1] = edge.temp[1];
    }
    result.gotNormals = false;
    return result._touch();
  }

  function faceNormal(mesh, face) {
    const ids = mesh.faceVertices(face);
    if (ids.length < 3) return vector4(1, 0, 0, 0);
    const a = mesh.vertices[ids[ids.length - 1]].position;
    const b = mesh.vertices[ids[0]].position;
    const c = mesh.vertices[ids[1]].position;
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    return vector4(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx, 0);
  }

  // GenMesh::Crease(0,...): split a vertex wedge when a newly marked
  // selected/unselected face boundary meets an older crease. Running this for
  // each cube direction prevents later projections overwriting a shared UV.
  function creaseSelectedFaceBoundaries(mesh, feature) {
    const halfedgeCount = mesh.edges.length * 2;
    for (let halfedge = 0; halfedge < halfedgeCount; halfedge++) {
      const edge = mesh.edges[halfedge >> 1];
      const face = mesh.faces[edge.face[halfedge & 1]];
      const opposite = mesh.faces[edge.face[(halfedge & 1) ^ 1]];
      if (!face?.select || opposite?.select) continue;
      if (!edge.crease) {
        for (let endpoint = 0; endpoint < 2; endpoint++) {
          let wedge = halfedge ^ endpoint;
          const sourceId = mesh.getVertId(wedge);
          const first = mesh.vertices[sourceId].first;
          let existing = 0;
          let around = wedge;
          let guard = 0;
          do {
            existing |= mesh.edges[around >> 1].crease;
            around = mesh.nextVertEdge(around);
            if (++guard > halfedgeCount + 1) throw new Error('broken GenMesh vertex loop');
          } while (around !== wedge);

          if (existing) {
            const copyId = mesh.addCopiedVertex(sourceId);
            mesh.vertices[copyId].first = first;
            mesh.vertices[sourceId].next = copyId;
            guard = 0;
            do {
              mesh.edges[wedge >> 1].vert[wedge & 1] = copyId;
              wedge = mesh.nextVertEdge(wedge);
              if (++guard > halfedgeCount + 1) throw new Error('broken GenMesh vertex loop');
            } while (!(mesh.edges[wedge >> 1].crease));
          }
        }
      }
      edge.crease |= feature;
    }
  }

  function Mesh_UVProjection(mesh, mask, srt, type, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    const matrix = mat4SRT(new Float32Array(srt), new Float32Array(16));
    type |= 0; mask |= 0;
    if (type === 0 || type === 3) {
      const real = type === 0;
      for (let axisCode = 0; axisCode < 6; axisCode++) {
        const sign = real ? (axisCode & 1 ? 1 : -1) : 1;
        result.allToSelection(false, ALL.FACE | ALL.VERT);
        for (const face of result.faces) {
          if (mask && !(face.mask & mask)) continue;
          const normal = faceNormal(result, face);
          let maximum = normal[0], axis = 0;
          if (Math.abs(normal[1]) > Math.abs(maximum)) { maximum = normal[1]; axis = 2; }
          if (Math.abs(normal[2]) > Math.abs(maximum)) { maximum = normal[2]; axis = 4; }
          if (maximum > 0) axis++;
          face.select = axis === axisCode;
        }
        creaseSelectedFaceBoundaries(result, FEATURE.UV0);
        result.faceToVertex();
        for (const vertex of result.vertices) {
          if (!vertex.select) continue;
          const point = transformXYZ(matrix, vertex.position);
          if ((axisCode >> 1) === 0) {
            vertex.uv[0] = f32(point[2] * sign);
            vertex.uv[1] = f32(-point[1]);
          } else if ((axisCode >> 1) === 1) {
            vertex.uv[0] = point[0];
            vertex.uv[1] = f32(-point[2] * sign);
          } else {
            vertex.uv[0] = f32(-point[0] * sign);
            vertex.uv[1] = f32(-point[1]);
          }
          vertex.uv[2] = 0;
          vertex.uv[3] = vertex.position[3];
        }
      }
    } else {
      for (const vertex of result.vertices) {
        const point = transformXYZ(matrix, vertex.position);
        const xzSquared = point[0] * point[0] + point[2] * point[2];
        vertex.uv[0] = f32(0.5 - Math.atan2(point[0], point[2]) / TAU);
        vertex.uv[1] = type === 1
          ? f32(-point[1])
          : f32(0.5 - Math.atan2(point[1], Math.sqrt(xzSquared)) / PI);
      }
    }
    return result._touch();
  }

  function Mesh_Center(mesh, mask, which, owned = false) {
    const result = checkedCopy(mesh, (mask << 16) >>> 0, owned);
    if (!result) return null;
    // CalcBBox itself selects every live face and propagates that selection to
    // vertices. These observable selection side effects survive the operator.
    result.allToSelection(true, ALL.FACE);
    result.faceToVertex();
    const box = result.bounds(true);
    result.allToSelection(true, ALL.VERT);
    const matrix = matrixIdentity();
    if (which & 1) matrix[12] = f32(-0.5 * (box.min[0] + box.max[0]));
    if (which & 2) matrix[13] = f32(which & 128 ? -box.min[1] : -0.5 * (box.min[1] + box.max[1]));
    if (which & 4) matrix[14] = f32(-0.5 * (box.min[2] + box.max[2]));
    result.transformVertices(matrix);
    return result;
  }

  function Mesh_SelectLogic(mesh, sourceMask1, sourceMask2, destinationMask, mode, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    sourceMask1 >>>= 0; sourceMask2 >>>= 0; destinationMask >>>= 0; mode |= 0;
    const sources = sourceMask1 | sourceMask2;
    if ((sources & 255) && (destinationMask & 255)) {
      for (const edge of result.edges) selLogic(edge, sourceMask1 & 255, sourceMask2 & 255, destinationMask & 255, mode & 31);
    }
    if ((sources & 0xff00) && (destinationMask & 0xff00)) {
      for (const face of result.faces) selLogic(face, (sourceMask1 >>> 8) & 255,
        (sourceMask2 >>> 8) & 255, (destinationMask >>> 8) & 255, mode & 31);
    }
    if ((sources & 0xff0000) && (destinationMask & 0xff0000)) {
      for (const vertex of result.vertices) selLogic(vertex, (sourceMask1 >>> 16) & 255,
        (sourceMask2 >>> 16) & 255, (destinationMask >>> 16) & 255, mode & 31);
    }
    if ((destinationMask & 0xff0000) && (destinationMask & 0xff00) && (mode & 32)) {
      result.maskToSelection(destinationMask & 0xff00);
      result.faceToVertex();
      result.selectionToMask(destinationMask & 0xff0000, SELECT.SET);
    }
    return result._touch();
  }

  function Mesh_ShadowEnable(mesh, enabled, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    for (const face of result.faces) face.used = !!enabled;
    return result._touch();
  }

  function Mesh_Bend(mesh, mask, srt1, srt2, direction, yRange, mode, owned = false) {
    const result = checkedCopy(mesh, (mask << 16) >>> 0, owned);
    if (!result) return null;
    const matrix1 = pivotTransform(result,
      mat4SRT(new Float32Array(srt1), new Float32Array(16)), ATTR.POS);
    const matrix2 = pivotTransform(result,
      mat4SRT(new Float32Array(srt2), new Float32Array(16)), ATTR.POS);
    const measuring = mat4Euler(f32(direction[0] * TAU), 0, f32(direction[1] * TAU), new Float32Array(16));
    let yMin, yMax;
    if (mode & 2) {
      yMin = 1e30; yMax = -1e30;
      for (const vertex of result.vertices) if (vertex.select) {
        const p = vertex.position;
        const value = p[0] * measuring[1] + p[1] * measuring[5] + p[2] * measuring[9];
        if (value < yMin) yMin = value;
        if (value > yMax) yMax = value;
      }
    } else {
      yMin = yRange[0]; yMax = yRange[1];
    }
    const yScale = 1 / (yMax - yMin);
    for (const vertex of result.vertices) {
      if (!vertex.select) continue;
      const p = vertex.position;
      let t = (p[0] * measuring[1] + p[1] * measuring[5] + p[2] * measuring[9] - yMin) * yScale;
      t = clamp01(t);
      switch (mode & 5) {
        case 1: t = t * t * (3 - 2 * t); break;
        case 4: t = 1 - Math.abs(t - 0.5) * 2; break;
        case 5: t = t * t * (16 + t * (16 * t - 32)); break;
      }
      const p1 = transformXYZ(matrix1, p);
      const p2 = transformXYZ(matrix2, p);
      p[0] = f32(p1[0] + (p2[0] - p1[0]) * t);
      p[1] = f32(p1[1] + (p2[1] - p1[1]) * t);
      p[2] = f32(p1[2] + (p2[2] - p1[2]) * t);
    }
    // The released function does not clear GotNormals here.
    return result._touch();
  }

  function rigidInverse(matrix) {
    const out = matrixIdentity();
    out[0] = matrix[0]; out[1] = matrix[4]; out[2] = matrix[8];
    out[4] = matrix[1]; out[5] = matrix[5]; out[6] = matrix[9];
    out[8] = matrix[2]; out[9] = matrix[6]; out[10] = matrix[10];
    const tx = matrix[12], ty = matrix[13], tz = matrix[14];
    out[12] = f32(-(out[0] * tx + out[4] * ty + out[8] * tz));
    out[13] = f32(-(out[1] * tx + out[5] * ty + out[9] * tz));
    out[14] = f32(-(out[2] * tx + out[6] * ty + out[10] * tz));
    return out;
  }

  function Mesh_Bend2(mesh, center, rotate, length, angle, owned = false) {
    const result = checkedCopy(mesh, 0, owned);
    if (!result) return null;
    const top = mat4EulerTurns(new Float32Array(rotate), new Float32Array(16));
    top[12] = f32(-center[0]); top[13] = f32(-center[1]); top[14] = f32(-center[2]);
    const bottom = rigidInverse(top);
    angle = f32(angle * TAU);
    for (const vertex of result.vertices) {
      const local = transformXYZ(top, vertex.position);
      let t = local[1];
      if (t >= 0) local[1] = f32(local[1] - Math.min(t, length));
      t = f32(clamp01(f32(t / length)) * angle);
      const sine = f32(Math.sin(t)), cosine = f32(Math.cos(t));
      const x = local[0], y = local[1];
      local[0] = f32(cosine * x - sine * y);
      local[1] = f32(sine * x + cosine * y);
      transformXYZ(bottom, local, vertex.position);
    }
    result.gotNormals = false;
    return result._touch();
  }

  function randomSRT(random, source) {
    const values = new Float32Array(9);
    for (let i = 0; i < 3; i++) values[i] = f32((source[i] - 1) * random.float() + 1);
    for (let i = 3; i < 9; i++) values[i] = f32(source[i] * (random.float() - 0.5) * 2);
    return mat4SRT(values, new Float32Array(16));
  }

  function Mesh_Multiply(mesh, srt, count, mode, translateU, translateV, localRotate, extrude, owned = false) {
    const source = checkedCopy(mesh, 0, owned);
    if (!source) return null;
    count |= 0; mode |= 0;
    const step = mat4SRT(new Float32Array(srt), new Float32Array(16));
    const localStep = mat4EulerTurns(new Float32Array(localRotate), new Float32Array(16));
    let transform = matrixIdentity();
    let localTransform = matrixIdentity();
    const output = new Mesh(source.vertexMask);
    const random = new Random();
    random.setSeed(count);
    if (extrude) source.needNormals();
    let previousStart = 0, previousEnd = 0;
    for (let iteration = 0; iteration < count; iteration++) {
      const start = output.vertices.length;
      output.append(source);
      const end = output.vertices.length;
      // Only the preceding copy can still be selected. Native selection state
      // is unchanged, but avoiding the complete accumulated prefix turns the
      // instancing loop back into linear work.
      for (let i = previousStart; i < previousEnd; i++) output.vertices[i].select = false;
      for (let i = start; i < end; i++) output.vertices[i].select = true;
      if (extrude) {
        const matrix = matrixIdentity();
        matrix[0] = matrix[5] = matrix[10] = f32(iteration * extrude);
        output.transformVertices(matrix, ATTR.NORMAL, ATTR.POS | 0x10, start, end);
      }
      output.transformVertices(
        mat4Mul(transform, localTransform, new Float32Array(16)), ATTR.POS, ATTR.POS, start, end,
      );
      transform = mode & 2
        ? randomSRT(random, srt)
        : mat4Mul(step, transform, new Float32Array(16));
      localTransform = mat4Mul(localStep, localTransform, new Float32Array(16));
      if (mode & 1) {
        const matrix = matrixIdentity();
        matrix[12] = f32(iteration * translateU);
        matrix[13] = f32(iteration * translateV);
        output.transformVertices(matrix, ATTR.UV0, ATTR.UV0, start, end);
      }
      previousStart = start;
      previousEnd = end;
    }
    output.gotNormals = false;
    output._touch();
    return owned ? source.adopt(output) : output;
  }

  function Mesh_Multiply2(inputs, seed, count1, translate1, count2, translate2, randomCount, count3, translate3,
    ownership = []) {
    const valid = inputs.map((mesh, index) => ({ mesh, index })).filter(value => value.mesh instanceof Mesh);
    if (!valid.length) return null;
    count1 = count1.map(value => value | 0);
    count2 = count2.map(value => value | 0);
    count3 = count3.map(value => value | 0);
    let product = 1;
    for (const count of [...count3, ...count2, ...count1]) product = Math.imul(product, count) | 0;
    if (product > 1024) count1 = count2 = count3 = [1, 1, 1];
    const result = new Mesh();
    const random = new Random();
    random.setSeed(seed | 0);
    let lastStart = 0, start = 0;
    const matrix = matrixIdentity();
    for (let z3 = 0; z3 < count3[2]; z3++) for (let y3 = 0; y3 < count3[1]; y3++) for (let x3 = 0; x3 < count3[0]; x3++) {
      for (let z2 = 0; z2 < count2[2]; z2++) for (let y2 = 0; y2 < count2[1]; y2++) for (let x2 = 0; x2 < count2[0]; x2++) {
        for (let z1 = 0; z1 < count1[2]; z1++) for (let y1 = 0; y1 < count1[1]; y1++) for (let x1 = 0; x1 < count1[0]; x1++) {
          lastStart = start;
          start = result.vertices.length;
          const choice = Math.min(random.int(valid.length + (randomCount | 0)), valid.length - 1);
          result.append(valid[choice].mesh);
          for (let i = lastStart; i < start; i++) result.vertices[i].select = false;
          for (let i = start; i < result.vertices.length; i++) result.vertices[i].select = true;
          matrix[12] = f32(x1 * translate1[0] + x2 * translate2[0] + x3 * translate3[0]);
          matrix[13] = f32(y1 * translate1[1] + y2 * translate2[1] + y3 * translate3[1]);
          matrix[14] = f32(z1 * translate1[2] + z2 * translate2[2] + z3 * translate3[2]);
          result.transformVertices(matrix, ATTR.POS, ATTR.POS, start, result.vertices.length);
        }
      }
    }
    result.gotNormals = false;
    result._touch();
    const adoption = valid.find(value => ownership[value.index]);
    for (const value of valid) {
      if (ownership[value.index] && value !== adoption) value.mesh.releaseStorage();
    }
    return adoption ? adoption.mesh.adopt(result) : result;
  }

  function weightedValues(destination, terms) {
    for (let attribute = 0; attribute < destination.values.length; attribute++) {
      const target = destination.values[attribute];
      for (let component = 0; component < 4; component++) {
        let value = 0;
        for (const [vertex, weight] of terms) value += vertex.values[attribute][component] * weight;
        target[component] = f32(value);
      }
    }
  }

  const ATTRIBUTE_FEATURES = Object.freeze([
    FEATURE.POS, FEATURE.NORMAL, FEATURE.TANGENT,
    FEATURE.COLOR0, FEATURE.UV0, FEATURE.UV1,
  ]);

  function subdivideOnce(source, alpha) {
    const output = new Mesh(source.vertexMask);
    output.vertices = source.vertices.map(cloneVertex);
    output.materials = source.materials.map(cloneSlot);
    output.parts = source.parts.slice();
    output.collisions = source.collisions.map(value => cloneCollision(value));
    output.lights = source.lights.slice();
    output.pivot = source.pivot | 0;
    output.stripped = !!source.stripped;
    output._syncAliases();

    const faceLoops = source.faces.map(face => face.edge >= 0 ? source.faceEdges(face) : []);
    const incidentFaces = Array.from({ length: source.vertices.length }, () => []);
    const neighbors = Array.from({ length: source.vertices.length }, () => new Set());
    const boundaryNeighbors = Array.from({ length: source.vertices.length }, () => new Set());
    const incidentCreases = new Uint32Array(source.vertices.length);
    for (let faceIndex = 0; faceIndex < source.faces.length; faceIndex++) {
      const face = source.faces[faceIndex];
      if (!face.select) continue;
      for (const halfedge of faceLoops[faceIndex]) incidentFaces[source.getVertId(halfedge)].push(faceIndex);
    }
    for (let edgeIndex = 0; edgeIndex < source.edges.length; edgeIndex++) {
      const edge = source.edges[edgeIndex];
      const a = edge.vert[0], b = edge.vert[1];
      if (a < 0 || b < 0) continue;
      incidentCreases[a] |= edge.crease >>> 0;
      incidentCreases[b] |= edge.crease >>> 0;
      const selected0 = !!source.faces[edge.face[0]]?.select;
      const selected1 = !!source.faces[edge.face[1]]?.select;
      if (!(selected0 || selected1)) continue;
      neighbors[a].add(b); neighbors[b].add(a);
      if (!(selected0 && selected1)) {
        boundaryNeighbors[a].add(b); boundaryNeighbors[b].add(a);
      }
    }

    // Position is feature zero and is never creased. Work out its even-vertex
    // rule over First (the physical vertex), then copy the result to every
    // per-corner record in that crease cycle.
    const physicalGroups = new Map();
    const physicalFaces = new Map();
    const physicalNeighbors = new Map();
    const physicalBoundary = new Map();
    for (let i = 0; i < source.vertices.length; i++) {
      const first = source.vertices[i].first | 0;
      const group = physicalGroups.get(first);
      if (group) group.push(i); else physicalGroups.set(first, [i]);
    }
    for (let faceIndex = 0; faceIndex < source.faces.length; faceIndex++) {
      if (!source.faces[faceIndex].select) continue;
      for (const halfedge of faceLoops[faceIndex]) {
        const first = source.vertices[source.getVertId(halfedge)].first | 0;
        let set = physicalFaces.get(first);
        if (!set) physicalFaces.set(first, set = new Set());
        set.add(faceIndex);
      }
    }
    for (const edge of source.edges) {
      if (edge.vert[0] < 0 || edge.vert[1] < 0) continue;
      const selected0 = !!source.faces[edge.face[0]]?.select;
      const selected1 = !!source.faces[edge.face[1]]?.select;
      if (!(selected0 || selected1)) continue;
      const a = source.vertices[edge.vert[0]].first | 0;
      const b = source.vertices[edge.vert[1]].first | 0;
      if (a === b) continue;
      let setA = physicalNeighbors.get(a), setB = physicalNeighbors.get(b);
      if (!setA) physicalNeighbors.set(a, setA = new Set());
      if (!setB) physicalNeighbors.set(b, setB = new Set());
      setA.add(b); setB.add(a);
      if (!(selected0 && selected1)) {
        let boundaryA = physicalBoundary.get(a), boundaryB = physicalBoundary.get(b);
        if (!boundaryA) physicalBoundary.set(a, boundaryA = new Set());
        if (!boundaryB) physicalBoundary.set(b, boundaryB = new Set());
        boundaryA.add(b); boundaryB.add(a);
      }
    }

    const faceCenters = new Map();
    for (let faceIndex = 0; faceIndex < source.faces.length; faceIndex++) {
      const face = source.faces[faceIndex];
      if (!face.select || !faceLoops[faceIndex].length) continue;
      const ids = faceLoops[faceIndex].map(edge => source.getVertId(edge));
      const center = newVertex();
      center.mask = face.mask & 255;
      center.select = !!face.select;
      const weight = 1 / ids.length;
      weightedValues(center, ids.map(id => [source.vertices[id], weight]));
      const index = output.vertices.length;
      center.first = center.next = center.reIndex = index;
      output.vertices.push(center);
      faceCenters.set(faceIndex, index);
    }

    const physicalEvenPositions = new Map();
    for (const [first] of physicalGroups) {
      const faces = [...(physicalFaces.get(first) || [])];
      if (!faces.length) continue;
      const original = source.vertices[first].position;
      const boundary = [...(physicalBoundary.get(first) || [])];
      const value = new Float32Array(4);
      if (boundary.length) {
        const w1 = alpha * 0.125;
        const a = source.vertices[boundary[0]].position;
        const b = source.vertices[boundary[boundary.length - 1]].position;
        for (let component = 0; component < 4; component++) {
          value[component] = f32(original[component] * (1 - 2 * w1) + (a[component] + b[component]) * w1);
        }
      } else {
        const n = faces.length;
        const ringWeight = alpha / (n * n);
        for (let component = 0; component < 4; component++) {
          let sum = original[component] * (1 - alpha * 2 / n);
          for (const neighbor of physicalNeighbors.get(first) || []) {
            sum += source.vertices[neighbor].position[component] * ringWeight;
          }
          for (const faceIndex of faces) sum += output.vertices[faceCenters.get(faceIndex)].position[component] * ringWeight;
          value[component] = f32(sum);
        }
      }
      physicalEvenPositions.set(first, value);
    }

    // Even vertices. The source's k contains alternating edge-neighbor and
    // face-center entries, yielding this alpha/n^2 form on a closed one-ring.
    for (let vertexIndex = 0; vertexIndex < source.vertices.length; vertexIndex++) {
      const faces = incidentFaces[vertexIndex];
      if (!faces.length) continue;
      const original = source.vertices[vertexIndex];
      const boundary = [...boundaryNeighbors[vertexIndex]];
      if (boundary.length) {
        const w1 = alpha * 0.125;
        const terms = [[original, 1 - 2 * w1]];
        terms.push([source.vertices[boundary[0]], w1]);
        terms.push([source.vertices[boundary[boundary.length - 1]], w1]);
        weightedValues(output.vertices[vertexIndex], terms);
      } else {
        const n = faces.length;
        const terms = [[original, 1 - alpha * 2 / n]];
        const ringWeight = alpha / (n * n);
        for (const neighbor of neighbors[vertexIndex]) terms.push([source.vertices[neighbor], ringWeight]);
        for (const faceIndex of faces) terms.push([output.vertices[faceCenters.get(faceIndex)], ringWeight]);
        weightedValues(output.vertices[vertexIndex], terms);
        // MakeSubdMask maps the sparse vertex feature bits onto the packed
        // attribute vectors. A crease freezes only the affected attributes;
        // position can remain Catmull-Clark smooth across a UV/normal seam.
        for (let attribute = 0; attribute < ATTRIBUTE_FEATURES.length; attribute++) {
          if (incidentCreases[vertexIndex] & ATTRIBUTE_FEATURES[attribute]) {
            output.vertices[vertexIndex].values[attribute].set(original.values[attribute]);
          }
        }
      }
    }
    for (const [first, value] of physicalEvenPositions) {
      for (const vertexIndex of physicalGroups.get(first)) output.vertices[vertexIndex].position.set(value);
    }

    const edgePoints = new Map();
    const edgePointSource = new Map();
    for (let edgeIndex = 0; edgeIndex < source.edges.length; edgeIndex++) {
      const edge = source.edges[edgeIndex];
      const face0 = source.faces[edge.face[0]], face1 = source.faces[edge.face[1]];
      const selected0 = !!face0?.select, selected1 = !!face1?.select;
      if (!(selected0 || selected1) || edge.vert[0] < 0 || edge.vert[1] < 0) continue;
      const sideVertices = [];
      for (let side = 0; side < 2; side++) {
        const halfedge = edgeIndex * 2 + side;
        sideVertices.push([
          source.vertices[source.getVertId(halfedge)],
          source.vertices[source.getVertId(source.nextFaceEdge(halfedge))],
        ]);
      }
      const split = edge.crease !== 0;
      const points = [-1, -1];
      for (let side = 0; side < (split ? 2 : 1); side++) {
        const [a, b] = sideVertices[side];
        const point = newVertex();
        point.mask = (a.mask | b.mask) & 255;
        point.select = true;
        const smooth = selected0 && selected1;
        const w1 = alpha * 0.25, w2 = 0.5 - w1;
        for (let attribute = 0; attribute < point.values.length; attribute++) {
          const feature = ATTRIBUTE_FEATURES[attribute];
          const target = point.values[attribute];
          const va = a.values[attribute], vb = b.values[attribute];
          for (let component = 0; component < 4; component++) {
            if (smooth && !(edge.crease & feature)) {
              const center0 = output.vertices[faceCenters.get(edge.face[0])].values[attribute];
              const center1 = output.vertices[faceCenters.get(edge.face[1])].values[attribute];
              target[component] = f32((va[component] + vb[component]) * w2 +
                (center0[component] + center1[component]) * w1);
            } else {
              target[component] = f32((va[component] + vb[component]) * 0.5);
            }
          }
        }
        const index = output.vertices.length;
        point.first = side ? points[0] : index;
        point.next = point.reIndex = index;
        output.vertices.push(point);
        edgePointSource.set(index, edgeIndex);
        points[side] = index;
      }
      if (!split) points[1] = points[0];
      edgePoints.set(edgeIndex, points);
    }

    const polygons = [];
    for (let faceIndex = 0; faceIndex < source.faces.length; faceIndex++) {
      const face = source.faces[faceIndex];
      const loop = faceLoops[faceIndex];
      if (loop.length < 3) continue;
      if (face.select) {
        const center = faceCenters.get(faceIndex);
        for (let i = 0; i < loop.length; i++) {
          const current = source.getVertId(loop[i]);
          const outgoingHalfedge = loop[i];
          const incomingHalfedge = loop[(i + loop.length - 1) % loop.length];
          const outgoing = edgePoints.get(outgoingHalfedge >> 1)?.[outgoingHalfedge & 1];
          const incoming = edgePoints.get(incomingHalfedge >> 1)?.[incomingHalfedge & 1];
          polygons.push({
            verts: [current, outgoing, center, incoming], mask: face.mask,
            id: face.id, select: true, used: face.used, material: face.material,
          });
        }
      } else {
        const ids = [];
        for (const halfedge of loop) {
          ids.push(source.getVertId(halfedge));
          const point = edgePoints.get(halfedge >> 1)?.[halfedge & 1];
          if (point !== undefined) ids.push(point);
        }
        polygons.push({
          verts: ids, mask: face.mask, id: face.id, select: false,
          used: face.used, material: face.material,
        });
      }
    }
    output.linkVertexCopies();
    output.setPolygons(polygons);
    // SplitBridge retains the source edge's flags on both child halves. The
    // center spokes are new topology and intentionally remain uncreased.
    for (const edge of output.edges) {
      let sourceEdge = -1;
      for (let side = 0; side < 2; side++) {
        const pointSource = edgePointSource.get(edge.vert[side]);
        if (pointSource !== undefined) sourceEdge = pointSource;
      }
      if (sourceEdge < 0) continue;
      const otherIsOldVertex = edge.vert.some(vertex =>
        vertex < source.vertices.length && !edgePointSource.has(vertex));
      if (!otherIsOldVertex) continue;
      const oldEdge = source.edges[sourceEdge];
      edge.mask = oldEdge.mask;
      edge.id = oldEdge.id;
      edge.select = oldEdge.select;
      edge.used = oldEdge.used;
      edge.crease = oldEdge.crease;
    }
    output.gotNormals = false;
    return output;
  }

  function Mesh_Subdivide(mesh, mask, alpha, count, owned = false) {
    const ownedTarget = owned ? mesh : null;
    let result = checkedCopy(mesh, (mask << 8) >>> 0, owned);
    if (!result) return null;
    count |= 0; alpha = f32(alpha);
    while (count-- > 0) {
      const next = subdivideOnce(result, alpha);
      result = ownedTarget ? ownedTarget.adopt(next) : next;
    }
    result.gotNormals = false;
    return result;
  }

  function copyMeshForConversion(mesh) {
    return checkedCopy(mesh);
  }

  const meshHandlers = {
    0x81: call => finalizeMeshOutput(call, Mesh_Cube(call.parameters[0], call.parameters[1],
      call.parameters[2], call.parameters[3], call.parameters.slice(4, 13))),
    0x96: call => dispatchMeshInputs(call, ownership => Mesh_MatLink(call.inputs[0], call.links[0],
      call.parameters[0], call.parameters[1], ownership[0])),
    0x89: call => dispatchMeshInputs(call, ownership => Mesh_TransformEx(call.inputs[0], call.parameters[0],
      call.parameters.slice(1, 10), call.parameters[10], call.parameters[11], ownership[0])),
    0x92: call => dispatchMeshInputs(call, ownership => Mesh_Add(call.inputs, ownership)),
    0x88: call => dispatchMeshInputs(
      call,
      ownership => Mesh_Transform(call.inputs[0], call.parameters[0],
        call.parameters.slice(1, 10), ownership[0]),
      !((call.parameters[0] | 0) === 0 && call.inputs[0] instanceof Mesh && call.inputs[0]._compact),
    ),
    0x9d: call => finalizeMeshOutput(call,
      Mesh_Grid(call.parameters[0], call.parameters[1], call.parameters[2])),
    0xa5: call => dispatchMeshInputs(call, ownership => Mesh_UVProjection(call.inputs[0], call.parameters[0],
      call.parameters.slice(1, 10), call.parameters[10], ownership[0])),
    0x95: call => dispatchMeshInputs(call, ownership => {
      // The stream packs 18 fields but convention 0x...11 exposes precisely
      // words 0..16. Word 17 is the historical alias padding and is ignored.
      const p = call.parameters.slice(0, 17);
      return Mesh_Multiply(call.inputs[0], p.slice(0, 9), p[9], p[10], p[11], p[12],
        p.slice(13, 16), p[16], ownership[0]);
    }),
    0xa6: call => dispatchMeshInputs(call, ownership => Mesh_Center(call.inputs[0], call.parameters[0],
      call.parameters[1], ownership[0])),
    0xb0: call => dispatchMeshInputs(call, ownership => Mesh_ShadowEnable(call.inputs[0], call.parameters[0],
      ownership[0])),
    0x86: call => dispatchMeshInputs(call, ownership => Mesh_SelectCube(call.inputs[0], call.parameters[0],
      call.parameters[1], call.parameters.slice(2, 5), call.parameters.slice(5, 8), ownership[0])),
    0x82: call => finalizeMeshOutput(call, Mesh_Cylinder(call.parameters[0], call.parameters[1],
      call.parameters[2], call.parameters[3], call.parameters[4])),
    0x93: call => dispatchMeshInputs(call, ownership => Mesh_DeleteFaces(call.inputs[0], call.parameters[0],
      ownership[0])),
    0xa3: call => dispatchMeshInputs(call, ownership => Mesh_Invert(call.inputs[0], ownership[0])),
    0xb4: call => dispatchMeshInputs(call, ownership => Mesh_Multiply2(call.inputs, call.parameters[0], call.parameters.slice(1, 4),
      call.parameters.slice(4, 7), call.parameters.slice(7, 10), call.parameters.slice(10, 13),
      call.parameters[13], call.parameters.slice(14, 17), call.parameters.slice(17, 20), ownership)),
    0xa0: call => dispatchMeshInputs(call, ownership => Mesh_Bend(call.inputs[0], call.parameters[0], call.parameters.slice(1, 10),
      call.parameters.slice(10, 19), call.parameters.slice(19, 21), call.parameters.slice(21, 23),
      call.parameters[23], ownership[0])),
    0xab: call => dispatchMeshInputs(call, ownership => Mesh_Bend2(call.inputs[0], call.parameters.slice(0, 3),
      call.parameters.slice(3, 6), call.parameters[6], call.parameters[7], ownership[0])),
    0x83: call => finalizeMeshOutput(call, Mesh_Torus(call.parameters[0], call.parameters[1],
      call.parameters[2], call.parameters[3], call.parameters[4], call.parameters[5], call.parameters[6])),
    0xa1: call => dispatchMeshInputs(call, ownership => Mesh_SelectLogic(call.inputs[0], call.parameters[0],
      call.parameters[1], call.parameters[2], call.parameters[3], ownership[0])),
    0x84: call => finalizeMeshOutput(call, Mesh_Sphere(call.parameters[0], call.parameters[1])),
    0x87: call => dispatchMeshInputs(call, ownership => Mesh_Subdivide(call.inputs[0], call.parameters[0],
      call.parameters[1], call.parameters[2], ownership[0])),
  };

  export {
    Mesh,
    ALL as MESH_ALL,
    ATTR as MESH_ATTR,
    DEFAULT_VERTEX_MASK as MESH_DEFAULT_VERTEX_MASK,
    FEATURE as MESH_FEATURE,
    SELECT as MESH_SELECT,
    Mesh_Add,
    Mesh_Bend,
    Mesh_Bend2,
    Mesh_Center,
    Mesh_Cube,
    Mesh_Cylinder,
    Mesh_DeleteFaces,
    Mesh_Grid,
    Mesh_Invert,
    Mesh_MatLink,
    Mesh_Multiply,
    Mesh_Multiply2,
    Mesh_SelectCube,
    Mesh_SelectLogic,
    Mesh_ShadowEnable,
    Mesh_Sphere,
    Mesh_Subdivide,
    Mesh_Torus,
    Mesh_Transform,
    Mesh_TransformEx,
    Mesh_UVProjection,
    copyMeshForConversion,
    defaultMeshMaterial,
    getDefaultMeshMaterial,
    setDefaultMeshMaterial,
    meshInputIsUniquelyConsumed,
    meshStorageStats,
    meshHandlers,
  };
