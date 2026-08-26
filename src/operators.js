// The native player assembled one KHandlers table in player_demo/demo_oplist.cpp.
// Keep the same family order here while letting each released subsystem own its
// implementation. A duplicate class id is always a porting error.
import { runtimeHandlers } from './runtime.js';
import { bitmapHandlers } from './bitmap.js';
import { effectHandlers } from './effects.js';
import { meshHandlers } from './mesh.js';
import { meshToMinHandlers } from './mesh_to_min.js';
import { sceneHandlers } from './scene.js';
import { materialHandlers } from './material.js';
import { overlayHandlers } from './overlay.js';
import { minMeshHandlers } from './minmesh.js';

const OPERATOR_FAMILIES = Object.freeze([
  Object.freeze(['kdoc / genblobspline', runtimeHandlers]),
  Object.freeze(['genbitmap', bitmapHandlers]),
  Object.freeze(['geneffectdebris / geneffectipp', effectHandlers]),
  Object.freeze(['genmesh', meshHandlers]),
  Object.freeze(['genmesh Mesh_ToMin bridge', meshToMinHandlers]),
  Object.freeze(['genscene', sceneHandlers]),
  Object.freeze(['genmaterial', materialHandlers]),
  Object.freeze(['genoverlay / geneffectipp', overlayHandlers]),
  Object.freeze(['genminmesh', minMeshHandlers]),
]);

function createOperatorHandlers(families = OPERATOR_FAMILIES) {
  const handlers = new Map();
  const owners = new Map();
  for (const [owner, family] of families) {
    for (const [rawId, handler] of Object.entries(family)) {
      const id = Number(rawId);
      if (!Number.isInteger(id) || id < 0) {
        throw new TypeError(`${owner} exported invalid operator id ${rawId}`);
      }
      if (handlers.has(id)) {
        throw new Error(
          `operator 0x${id.toString(16)} is owned by both ${owners.get(id)} and ${owner}`,
        );
      }
      handlers.set(id, handler);
      owners.set(id, owner);
    }
  }
  return handlers;
}

export { OPERATOR_FAMILIES, createOperatorHandlers };
