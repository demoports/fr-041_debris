import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const toolsDirectory = dirname(fileURLToPath(import.meta.url));
const rootDirectory = resolve(toolsDirectory, '..');
const sourceDirectory = resolve(rootDirectory, 'src');

// Snapshot the browser namespace before loading any of the real application
// graph. Native modules must not recreate the old script-bundle singleton.
assert.equal(
  Object.prototype.hasOwnProperty.call(globalThis, 'Debris'),
  false,
  'test environment unexpectedly already has globalThis.Debris',
);

const [
  { OPERATOR_FAMILIES, createOperatorHandlers },
  { loadProductionData },
  { parseKX, summarizeKX },
  ,
  ,
  { CLASS_REGISTRY },
  ABI,
  { Runtime },
] =
  await Promise.all([
    import('../src/operators.js'),
    import('../src/data.js'),
    import('../src/kx.js'),
    import('../src/app.js'),
    import('../src/audio_worker_core.js'),
    import('../src/classes.js'),
    import('../src/abi.js'),
    import('../src/runtime.js'),
  ]);

assert.equal(
  Object.prototype.hasOwnProperty.call(globalThis, 'Debris'),
  false,
  'importing the application graph created globalThis.Debris',
);

const handlers = createOperatorHandlers();
const familyIds = OPERATOR_FAMILIES.flatMap(([, family]) =>
  Object.keys(family).map(Number));
assert.equal(familyIds.length, 101,
  'operator families expose 90 production ids and 11 source-backed generic ids');
assert.equal(new Set(familyIds).size, familyIds.length, 'operator families contain duplicate ids');
assert.equal(handlers.size, 101, 'composed operator table must contain 101 unique ids');
for (const [id, handler] of handlers) {
  assert.ok(
    typeof handler === 'function' || (handler && typeof handler === 'object'),
    `operator 0x${id.toString(16)} has no handler implementation`,
  );
}

// Parse the embedded production rather than executing its operators. This is
// enough to prove that every class used by the shipped KX can be dispatched,
// while keeping the test comfortably inside the memory cap.
const { kx, loaderSong } = await loadProductionData();
const [
  assetKX, assetLoaderSong, dispatchSource, partyInventorySource, classMapSource,
  nativeHandlerSource, nativeAbiSource,
] = await Promise.all([
  readFile(resolve(rootDirectory, 'assets/debris_party.kx')),
  readFile(resolve(rootDirectory, 'assets/debris_loader.v2m')),
  readFile(resolve(rootDirectory, 'notes/debris_js_dispatch.json'), 'utf8'),
  readFile(resolve(rootDirectory, 'notes/debris_party_kx.json'), 'utf8'),
  readFile(resolve(rootDirectory, 'notes/debris_class_map.json'), 'utf8'),
  readFile(resolve(rootDirectory, 'vendor/wz3/player_demo/demo_oplist.cpp'), 'utf8'),
  readFile(resolve(rootDirectory, 'vendor/wz3/kdoc.hpp'), 'utf8'),
]);
assert.deepEqual(kx, new Uint8Array(assetKX),
  'the embedded production expands to the exact extracted party KX');
assert.deepEqual(loaderSong, new Uint8Array(assetLoaderSong),
  'the embedded loader expands to the exact extracted loader V2M');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const partyInventory = JSON.parse(partyInventorySource);
assert.equal(sha256(kx), partyInventory.sha256);
assert.equal(sha256(loaderSong),
  'd0afcd84ac58991d6b0f8505161be7212b70200a084574eeb8584e4862bcadb5');
const document = parseKX(kx);
const productionSummary = summarizeKX(document);
assert.deepEqual({
  size: productionSummary.size,
  flags: productionSummary.flags,
  operationCount: productionSummary.operationCount,
  eventCount: productionSummary.eventCount,
  splineCount: productionSummary.splineCount,
  animationBytes: productionSummary.animationBytes,
  blobCount: productionSummary.blobCount,
  blobBytes: productionSummary.blobBytes,
}, {
  size: partyInventory.size,
  flags: partyInventory.flags,
  operationCount: partyInventory.operation_count,
  eventCount: partyInventory.event_count,
  splineCount: partyInventory.spline_count,
  animationBytes: partyInventory.animation_bytes,
  blobCount: partyInventory.blob_count,
  blobBytes: partyInventory.blob_bytes,
}, 'the JavaScript reader reproduces the exact extraction inventory');
assert.deepEqual(productionSummary.sections, partyInventory.sections);
assert.deepEqual(document.roots, partyInventory.roots);
assert.deepEqual(productionSummary.classes.map((value, index) => ({
  index,
  id: value.id,
  id_hex: `0x${value.id.toString(16).padStart(4, '0')}`,
  convention_hex: `0x${value.convention.toString(16).padStart(8, '0')}`,
  packing: value.packing,
  operation_count: value.operationCount,
})), partyInventory.classes,
'all 90 class schemas and counts reproduce the extraction inventory');

const dispatchRows = JSON.parse(dispatchSource);
const classMapRows = JSON.parse(classMapSource);
const dispatchRowsById = new Map(dispatchRows.map(row => [row.id, row]));
assert.equal(dispatchRows.length, 90);
assert.equal(classMapRows.length, dispatchRows.length);

// Check the checked-in released source itself, rather than trusting only the
// generated correspondence notes. Conditional and /**/ annotated entries are
// still members of the player's KHandlers table; line comments are not.
const nativeHandlers = new Map();
for (const match of nativeHandlerSource.matchAll(
  /^[ \t]*(?!\/\/)(?:\/\*\*\/\s*)?(0x[0-9a-f]+)\s*,\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,/gim,
)) {
  const id = Number.parseInt(match[1], 16);
  assert.equal(nativeHandlers.has(id), false,
    `released KHandlers contains duplicate id 0x${id.toString(16)}`);
  nativeHandlers.set(id, { init: match[2], exec: match[3] });
}

assert.deepEqual(Object.keys(CLASS_REGISTRY).map(Number).sort((a, b) => a - b),
  dispatchRows.map(row => row.id).sort((a, b) => a - b));
const documentClassesById = new Map(document.classes.map(value => [value.id, value]));
for (const row of dispatchRows) {
  assert.deepEqual(CLASS_REGISTRY[row.id], {
    id: row.id,
    name: row.name,
    convention: Number.parseInt(row.convention, 16) >>> 0,
    dataWords: row.dataWords,
    inputSlots: row.inputSlots,
    linkSlots: row.linkSlots,
    stringSlots: row.stringSlots,
    splineSlots: row.splineSlots,
    packingOverrunWords: row.packingOverrunWords,
    outputClass: row.outputClass,
    init: row.initHandler,
    exec: row.execHandler,
    dynamic: row.dynamic,
  }, `generated class metadata drifted for ${row.idHex}`);
  const documentClass = documentClassesById.get(row.id);
  assert.equal(documentClass?.convention, Number.parseInt(row.convention, 16) >>> 0,
    `party convention drifted for ${row.idHex}`);
  assert.equal(documentClass?.packing, row.packing,
    `party packing drifted for ${row.idHex}`);
  const sourceRow = classMapRows[row.classIndex];
  assert.deepEqual(nativeHandlers.get(row.id), {
    init: row.initHandler,
    exec: row.execHandler,
  }, `released player KHandlers drifted for ${row.idHex}`);
  assert.deepEqual({
    classIndex: sourceRow.class_index,
    id: sourceRow.id,
    convention: sourceRow.kx_convention,
    packing: sourceRow.kx_packing,
    operationCount: sourceRow.operation_count,
    outputClass: sourceRow.output_class,
    initHandler: sourceRow.init_handler,
    execHandler: sourceRow.exec_handler,
    packingOverrunWords: sourceRow.packing_overrun_words,
    runtimeDispatchMatches: sourceRow.runtime_dispatch_matches,
  }, {
    classIndex: row.classIndex,
    id: row.id,
    convention: row.convention,
    packing: row.packing,
    operationCount: row.operationCount,
    outputClass: row.outputClass,
    initHandler: row.initHandler,
    execHandler: row.execHandler,
    packingOverrunWords: row.packingOverrunWords,
    runtimeDispatchMatches: true,
  }, `released-source class map drifted for ${row.idHex}`);
}

// The class map deliberately records editor-source versus compact-party
// differences instead of normalising them away. Eighteen conventions add the
// export-only OPC_FLEXINPUT bit; Material 2.0 also exports four fewer words.
// Five bitmap classes use the serialized scalar encodings observed in the
// party payload rather than the later released editor declarations.
assert.deepEqual(classMapRows
  .filter(row => row.source_convention !== row.kx_convention)
  .map(row => row.id_hex), [
    '0x00d3', '0x00c0', '0x0024', '0x00d0', '0x0092', '0x00b4',
    '0x00c1', '0x0182', '0x0019', '0x00f0', '0x00c5', '0x0134',
    '0x0133', '0x000d', '0x0111', '0x0129', '0x0073', '0x012c',
    '0x00e9',
  ]);
assert.deepEqual(classMapRows
  .filter(row => row.source_packing !== row.kx_packing)
  .map(row => row.id_hex), [
    '0x00d3', '0x002e', '0x0030', '0x0031', '0x0032', '0x0034',
  ]);
for (const row of classMapRows) {
  assert.equal(typeof row.source_convention, 'string');
  assert.equal(typeof row.source_packing, 'string');
  if (row.id !== 0xd3 && row.source_convention !== row.kx_convention) {
    assert.equal(
      (Number.parseInt(row.source_convention, 16) ^
        Number.parseInt(row.kx_convention, 16)) >>> 0,
      ABI.OPC_FLEXINPUT,
      `${row.id_hex} has an unexplained source/export convention delta`,
    );
  }
}
assert.deepEqual(Object.fromEntries(Object.entries(ABI).sort()), {
  KC_ANY: 255,
  KC_BITMAP: 1,
  KC_DEMO: 11,
  KC_EFFECT: 9,
  KC_IPP: 8,
  KC_MATERIAL: 5,
  KC_MESH: 6,
  KC_MINMESH: 2,
  KC_SCENE: 3,
  KC_SPLINE: 13,
  OPC_ALTEXEC: 0x04000000,
  OPC_ALTINIT: 0x20000000,
  OPC_BLOB: 0x00080000,
  OPC_DONTCALLLINK: 0x10000000,
  OPC_FLEXINPUT: 0x80000000,
  OPC_KENV: 0x00800000,
  OPC_KOP: 0x40000000,
  OPC_SKIPEXEC: 0x02000000,
  OPC_STRIPPEDIN: 0x08000000,
  OPC_VARIABLEINPUT: 0x01000000,
}, 'the shared JavaScript ABI must stay byte-for-byte aligned with kdoc.hpp');
for (const [name, value] of Object.entries(ABI)) {
  const match = nativeAbiSource.match(new RegExp(
    `^#define\\s+${name}\\s+(0x[0-9a-f]+|[0-9]+)\\b`, 'im'));
  assert.ok(match, `released kdoc.hpp no longer defines ${name}`);
  assert.equal(Number.parseInt(match[1], 0), value,
    `${name} drifted from released kdoc.hpp`);
}
const productionClassIds = new Set(document.classes.map(opClass => opClass.id));
assert.equal(productionClassIds.size, 90, 'embedded production must retain its 90 class ids');
const missingClassIds = [...productionClassIds]
  .filter(id => !handlers.has(id))
  .sort((a, b) => a - b);
assert.deepEqual(
  missingClassIds,
  [],
  `production classes without handlers: ${missingClassIds
    .map(id => `0x${id.toString(16)}`)
    .join(', ')}`,
);

function callbackIsEmpty(callback) {
  const source = Function.prototype.toString.call(callback);
  const block = source.match(/\{([\s\S]*)\}\s*$/);
  return block ? block[1].trim() === '' : /=>\s*(?:undefined|null)\s*$/.test(source);
}

const emptyExecIds = [];
for (const opClass of document.classes) {
  const metadata = CLASS_REGISTRY[opClass.id];
  const handler = handlers.get(opClass.id);
  const init = typeof handler === 'function' ? handler : handler?.init;
  assert.equal(typeof init, 'function',
    `production class 0x${opClass.id.toString(16)} has no init callback`);
  assert.equal(callbackIsEmpty(init), false,
    `production class 0x${opClass.id.toString(16)} has an empty init callback`);
  if (metadata.exec !== 'Exec_Misc_Nop') {
    assert.equal(typeof handler?.exec, 'function',
      `dynamic class 0x${opClass.id.toString(16)} has no exec callback`);
  }
  if (typeof handler?.exec === 'function' && callbackIsEmpty(handler.exec)) {
    emptyExecIds.push(opClass.id);
  }
}
assert.deepEqual(emptyExecIds, [0x3f],
  'only native empty Exec_Bitmap_Render may have an empty callback');

// Build the real Runtime objects without evaluating procedural handlers. This
// independently exercises all convention/output-class wiring and checks every
// one of the 16,478 operation records, including the two packing overruns.
const dispatchRuntime = new Runtime(document, { handlers });
assert.equal(dispatchRuntime.strictHandlers, true);
const expectedOperationCounts = new Map();
for (const raw of document.operations) {
  expectedOperationCounts.set(raw.classId,
    (expectedOperationCounts.get(raw.classId) || 0) + 1);
  const op = dispatchRuntime.operations[raw.id];
  const metadata = CLASS_REGISTRY[raw.classId];
  assert.ok(metadata, `operator ${raw.id} has no class metadata`);
  assert.ok(handlers.has(raw.classId), `operator ${raw.id} has no handler`);
  assert.equal(op.classInfo, metadata, `operator ${raw.id} resolved the wrong class metadata`);
  assert.equal(op.convention, document.classes[raw.classIndex].convention >>> 0,
    `operator ${raw.id} convention drifted`);
  assert.equal(raw.parameters.length, document.classes[raw.classIndex].packing.length,
    `operator ${raw.id} packing width drifted`);
  assert.equal(op.animParameters.length, metadata.dataWords,
    `operator ${raw.id} exposed packing overrun words to its handler`);
  const packing = document.classes[raw.classIndex].packing;
  for (let index = 0; index < metadata.dataWords; index++) {
    const kind = (packing[index] || '-').toLowerCase();
    const rawValue = raw.parameters[index] ?? 0;
    const expected = kind === 'g' || kind === 'f' || kind === 'e'
      ? Math.fround(rawValue)
      : kind === 'i' || kind === 's' ? rawValue | 0 : rawValue >>> 0;
    assert.equal(op.animParameters[index], expected,
      `operator ${raw.id} parameter ${index} (${kind}) crossed the ABI incorrectly`);
  }
}
assert.equal([...expectedOperationCounts.values()].reduce((sum, count) => sum + count, 0),
  16478);
for (const opClass of document.classes) {
  assert.equal(expectedOperationCounts.get(opClass.id),
    dispatchRowsById.get(opClass.id).operationCount,
    `class 0x${opClass.id.toString(16)} operation count drifted`);
}

// Exec_Misc_Nop is the native forwarding implementation, not a missing/stub
// callback. Resolve it through Runtime for every non-bitmap production class
// that names it, with graph edges temporarily detached so no generation runs.
const resolvedNopClasses = [];
for (const opClass of document.classes) {
  const metadata = CLASS_REGISTRY[opClass.id];
  if (metadata.exec !== 'Exec_Misc_Nop' || metadata.outputClass === 'KC_BITMAP') continue;
  const op = dispatchRuntime.operations.find(candidate => candidate.classId === opClass.id);
  const inputs = op.inputs, links = op.links;
  let resolved = null;
  op.inputs = [];
  op.links = [];
  dispatchRuntime.onHandlerCall = (call, phase, callback) => {
    if (call.op === op && phase === 'exec') resolved = callback;
  };
  try {
    dispatchRuntime._invokeHandler('exec', op, dispatchRuntime.environment);
  } finally {
    op.inputs = inputs;
    op.links = links;
  }
  assert.equal(typeof resolved, 'function',
    `class 0x${opClass.id.toString(16)} did not resolve Exec_Misc_Nop`);
  assert.match(Function.prototype.toString.call(resolved), /execInputs/,
    `class 0x${opClass.id.toString(16)} resolved a non-forwarding nop`);
  resolvedNopClasses.push(opClass.id);
}
dispatchRuntime.onHandlerCall = null;
assert.equal(resolvedNopClasses.length, 45);

// Bitmap Exec is unconditionally skipped by KOp::Exec in both the native and
// JavaScript runtimes. This includes the deliberately empty Render callback.
for (const opClass of document.classes) {
  const metadata = CLASS_REGISTRY[opClass.id];
  if (metadata.outputClass !== 'KC_BITMAP') continue;
  const op = dispatchRuntime.operations.find(candidate => candidate.classId === opClass.id);
  const calls = dispatchRuntime.handlerCallCount;
  op.exec(dispatchRuntime.environment);
  assert.equal(dispatchRuntime.handlerCallCount, calls,
    `bitmap class 0x${opClass.id.toString(16)} unexpectedly entered Exec`);
}

const genericHandlerIds = [...handlers.keys()]
  .filter(id => !productionClassIds.has(id))
  .sort((a, b) => a - b);
assert.deepEqual(genericHandlerIds, [
  0x102, 0x105, 0x112, 0x115, 0x122, 0x123, 0x12e, 0x132,
  0x180, 0x181, 0x183,
], 'the composed table has exactly the 11 documented generic source handlers');
for (const id of genericHandlerIds) {
  assert.ok(nativeHandlers.has(id), `generic handler 0x${id.toString(16)} is not source-backed`);
  assert.equal(document.operations.some(operation => operation.classId === id), false,
    `generic handler 0x${id.toString(16)} is unexpectedly used by Debris`);
}

// Model callback reachability over the whole authored timeline. The rules are
// the small cross-subsystem ownership seams in the exec handlers: Demo/Event,
// IPP, scene cache dispatch, materials, and deferred effects. This proves that
// neither a generic handler, Exec_Misc_Nop, MissingCache, nor the empty bitmap
// Render exec is on a production playback path.
const eventOperators = new Set(document.events.map(event => event.operation));
assert.deepEqual([...new Set(document.events.map(event =>
  document.operations[event.operation].classId))], [0x06]);
function effectiveOutputClass(operation, seen = new Set()) {
  if (!operation || seen.has(operation.id)) return null;
  seen.add(operation.id);
  if ((operation.classId === 0x06 || operation.classId === 0x16) && operation.inputs.length) {
    return effectiveOutputClass(document.operations[operation.inputs[0]], seen);
  }
  return CLASS_REGISTRY[operation.classId]?.outputClass || null;
}
function calledInputsAndLinks(operation) {
  const ids = operation.inputs.slice();
  if (!(document.classes[operation.classIndex].convention & ABI.OPC_DONTCALLLINK)) {
    ids.push(...operation.links.filter(Number.isInteger));
  }
  return ids.map(id => document.operations[id]);
}
function calledSceneInputs(operation, indexes = operation.inputs) {
  return indexes.map(id => document.operations[id]).filter(input => {
    const output = effectiveOutputClass(input);
    return output === 'KC_SCENE' || output === 'KC_EFFECT';
  });
}
const execReachable = new Set();
const execPending = [document.operations[document.roots[0]]];
while (execPending.length) {
  const operation = execPending.pop();
  if (!operation || execReachable.has(operation.id)) continue;
  const metadata = CLASS_REGISTRY[operation.classId];
  if (metadata.outputClass === 'KC_BITMAP') continue;
  execReachable.add(operation.id);
  let next;
  switch (operation.classId) {
    case 0x0d:
      next = operation.inputs.map(id => document.operations[id]).filter(input =>
        ['KC_DEMO', 'KC_IPP'].includes(effectiveOutputClass(input)));
      break;
    case 0x06:
      next = eventOperators.has(operation.id) && operation.inputs.length
        ? [document.operations[operation.inputs[0]]] : [];
      break;
    case 0x16:
      next = operation.inputs.length ? [document.operations[operation.inputs[0]]] : [];
      break;
    case 0xf0:
      next = operation.inputs.length ? [document.operations[operation.inputs[0]]] : [];
      break;
    case 0xe9:
    case 0xd2:
    case 0xd0:
    case 0xd3:
      next = calledInputsAndLinks(operation);
      break;
    case 0xc0:
    case 0xc1:
    case 0xc2:
    case 0xc3:
    case 0x182:
      next = calledSceneInputs(operation);
      break;
    case 0xc5:
      next = calledSceneInputs(operation, operation.inputs.slice(0, 1));
      break;
    case 0xc4:
    case 0x184:
    case 0x6b:
    case 0x73:
    case 0x74:
    case 0x75:
      next = [];
      break;
    default:
      if (metadata.exec === 'Exec_Misc_Nop') next = calledInputsAndLinks(operation);
      else assert.fail(`unmodelled exec class 0x${operation.classId.toString(16)}`);
  }
  execPending.push(...next);
}
const execReachableClassIds = [...new Set([...execReachable].map(id =>
  document.operations[id].classId))].sort((a, b) => a - b);
assert.equal(execReachable.size, 1846);
assert.deepEqual(execReachableClassIds, [
  0x06, 0x0d, 0x6b, 0x73, 0x74, 0x75,
  0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5,
  0xd0, 0xd2, 0xd3, 0xe9, 0xf0, 0x182, 0x184,
]);
assert.deepEqual(execReachableClassIds.filter(id =>
  CLASS_REGISTRY[id].exec === 'Exec_Misc_Nop'), []);
assert.deepEqual(execReachableClassIds.filter(id => genericHandlerIds.includes(id)), []);
assert.equal(execReachableClassIds.includes(0x3f), false,
  'the native empty bitmap Render exec is not exercised');

// Cable visibility is authored by which ChainLine scene nodes are reachable
// from each Event viewport; markers only place their endpoints. In particular,
// the city shot ending at beat 360 contains the 26 static cables, while the
// immediately following 1:50.204--1:57.551 shot deliberately contains none.
// This guards against carrying deferred cable jobs into a later viewport.
function reachableOperations(rootId) {
  const result = new Set();
  const pending = [rootId];
  while (pending.length) {
    const id = pending.pop();
    if (result.has(id)) continue;
    result.add(id);
    pending.push(...document.operations[id].inputs);
  }
  return [...result];
}
function reachableByClass(rootId, classId) {
  return reachableOperations(rootId)
    .filter(id => document.operations[id].classId === classId)
    .sort((a, b) => a - b);
}
assert.deepEqual(reachableByClass(3343, 0x6b),
  Array.from({ length: 26 }, (_, index) => 2750 + index));
assert.deepEqual(reachableByClass(3343, 0x182), [1513, 1855, 2566]);
assert.deepEqual(reachableByClass(14526, 0x6b),
  Array.from({ length: 26 }, (_, index) => 7511 + index));
assert.deepEqual(reachableByClass(12267, 0x6b), [],
  'the viewport beginning at beat 360 must not inherit the prior city cables');
assert.deepEqual(reachableByClass(12267, 0x182), []);
assert.deepEqual(reachableByClass(13803, 0x6b), [13779]);
assert.deepEqual(reachableByClass(13803, 0x182), [13774, 13776]);
assert.ok(document.events.some(event => event.operation === 14526 &&
  event.start === 336 * 65536 && event.end === 360 * 65536));
assert.ok(document.events.some(event => event.operation === 12267 &&
  event.start === 360 * 65536 && event.end === 384 * 65536));

const sourceNames = (await readdir(sourceDirectory))
  .filter(name => name.endsWith('.js'))
  .sort();
const sourcePaths = sourceNames.map(name => resolve(sourceDirectory, name));
const sourceSet = new Set(sourcePaths);
const sources = new Map(await Promise.all(sourcePaths.map(async path =>
  [path, await readFile(path, 'utf8')])));

function localModuleSpecifiers(source) {
  const specifiers = [];
  const patterns = [
    /\b(?:import|export)\s+(?:[\s\S]*?\s+from\s*)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) {
      if (match[1].startsWith('.')) specifiers.push(match[1]);
    }
  }
  return specifiers;
}

const graph = new Map();
for (const path of sourcePaths) {
  const dependencies = localModuleSpecifiers(sources.get(path)).map(specifier => {
    const dependency = resolve(dirname(path), specifier);
    assert.ok(
      sourceSet.has(dependency),
      `${relative(rootDirectory, path)} imports missing local module ${specifier}`,
    );
    return dependency;
  });
  graph.set(path, dependencies);
}

// Fail with the concrete cycle, rather than relying on an engine-dependent
// temporal-dead-zone failure during import.
const visiting = new Set();
const visited = new Set();
const stack = [];
function visit(path) {
  if (visited.has(path)) return;
  if (visiting.has(path)) {
    const cycleStart = stack.indexOf(path);
    const cycle = [...stack.slice(cycleStart), path]
      .map(item => relative(sourceDirectory, item))
      .join(' -> ');
    assert.fail(`source import graph contains a cycle: ${cycle}`);
  }
  visiting.add(path);
  stack.push(path);
  for (const dependency of graph.get(path)) visit(dependency);
  stack.pop();
  visiting.delete(path);
  visited.add(path);
}
for (const path of sourcePaths) visit(path);

// Import every host-independent source module directly as well as through the
// application graph. The three environment entry modules execute browser,
// Worker, or AudioWorklet APIs at top level and are covered by graph/static
// checks instead.
const environmentEntries = new Set(['main.js', 'audio_worker.js', 'audio_worklet.js']);
await Promise.all(sourcePaths
  .filter(path => !environmentEntries.has(relative(sourceDirectory, path)))
  .map(path => import(pathToFileURL(path).href)));

const forbiddenPatterns = [
  [/\bglobalThis\s*\.\s*Debris\b/, 'globalThis.Debris namespace'],
  [/\b(?:window|self)\s*\.\s*Debris\b/, 'browser Debris namespace'],
  [/\bWebAssembly\b/, 'WebAssembly dependency'],
  [/\.wasm(?:\b|[?#])/, 'Wasm module reference'],
  [/\bregisterHandlers\s*\(/, 'classic registerHandlers registry'],
  [/\bimportScripts\s*\(/, 'classic worker importScripts'],
  [/\bURL\s*\.\s*createObjectURL\s*\(/, 'executable object-URL module'],
  [/new\s+Blob\s*\([\s\S]{0,240}?\b(?:javascript|ecmascript)\b/i, 'executable Blob module'],
];
for (const [path, source] of sources) {
  for (const [pattern, label] of forbiddenPatterns) {
    assert.doesNotMatch(source, pattern, `${relative(rootDirectory, path)} contains ${label}`);
  }
}

const indexSource = await readFile(resolve(rootDirectory, 'index.html'), 'utf8');
for (const [pattern, label] of forbiddenPatterns) {
  assert.doesNotMatch(indexSource, pattern, `index.html contains ${label}`);
}
const scripts = [...indexSource.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
assert.equal(scripts.length, 1, 'index.html must contain exactly one launcher script');
assert.match(
  scripts[0][1],
  /\btype\s*=\s*["']module["']/i,
  'the index launcher must be a native module',
);
assert.deepEqual((await readdir(resolve(rootDirectory, 'assets'))).sort(), [
  'ATTRIBUTION.md', 'debris_loader.v2m', 'debris_party.kx', 'launcher-background.jpg',
], 'shipping assets contain only approved inputs/presentation art, not generated texture/geometry caches');

// Ordinary clean-checkout tests may consume tracked assets/vendor references,
// but must not silently rely on ignored extraction or browser-output trees.
// test_v2.mjs owns a separate audio-oracle audit and is checked by that suite.
const ordinaryTestNames = (await readdir(toolsDirectory))
  .filter(name => /^test_.*\.(?:mjs|py)$/.test(name) &&
    name !== 'test_modules.mjs' && name !== 'test_v2.mjs')
  .sort();
for (const name of ordinaryTestNames) {
  const testSource = await readFile(resolve(toolsDirectory, name), 'utf8');
  assert.doesNotMatch(testSource, /(?:\.\.\/)?(?:work|artifacts)\//,
    `${name} depends on an ignored work/artifacts path`);
}

console.log(JSON.stringify({
  sourceModules: sourcePaths.length,
  importEdges: [...graph.values()].reduce((sum, dependencies) => sum + dependencies.length, 0),
  operatorHandlers: handlers.size,
  productionClasses: productionClassIds.size,
  productionOperations: document.operations.length,
  playbackExecClasses: execReachableClassIds.length,
  playbackExecOperations: execReachable.size,
  genericProductionOperations: 0,
  cleanCheckoutTests: ordinaryTestNames.length,
  launcherScripts: scripts.length,
}, null, 2));
