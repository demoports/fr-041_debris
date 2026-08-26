// Neutral bridge for the one released operator that crosses the old GenMesh
// and compact GenMinMesh subsystems. Keeping it here leaves both geometry
// modules independently importable and removes their classic-script cycle.
import { copyMeshForConversion, meshInputIsUniquelyConsumed } from './mesh.js';
import { createMeshToMinHandler, meshToMin } from './minmesh.js';

function Mesh_ToMin(mesh) {
  const source = copyMeshForConversion(mesh);
  return source ? meshToMin(source) : null;
}

const meshToMinHandlers = {
  0x00b5: createMeshToMinHandler(meshInputIsUniquelyConsumed),
};

export { Mesh_ToMin, meshToMinHandlers };
