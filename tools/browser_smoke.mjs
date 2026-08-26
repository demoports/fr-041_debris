#!/usr/bin/env node
// Dependency-free end-to-end smoke test for the production browser player.
//
// Production precalc is memory intensive. This runner permits one isolated
// Chrome tree, watches aggregate RSS from outside Chrome, and owns cleanup of
// the complete detached process group.

import { execFile, spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { createServer } from 'node:http';
import {
  access, mkdir, mkdtemp, readFile, rename, rm, stat, writeFile,
} from 'node:fs/promises';
import { constants } from 'node:fs';
import { tmpdir, totalmem } from 'node:os';
import { extname, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const MEBIBYTE = 1024 * 1024;
const MAX_PROFILE_TIMES = 32;
const MAX_PROFILE_SECONDS = 3600;
const MAX_PROFILE_FRAMES = 8;
const MAX_PROFILE_SCREENSHOTS = 8;
const MAX_PROFILE_SCREENSHOT_PIXELS = 2_000_000;
const MAX_SMOKE_DIMENSION = 4096;
const MAX_SMOKE_PIXELS = 3840 * 2160;
const projectRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));
const artifacts = resolve(projectRoot, 'artifacts');
const width = parseBoundedInteger(
  'DEBRIS_SMOKE_WIDTH', process.env.DEBRIS_SMOKE_WIDTH, 320, 64, MAX_SMOKE_DIMENSION,
);
const height = parseBoundedInteger(
  'DEBRIS_SMOKE_HEIGHT', process.env.DEBRIS_SMOKE_HEIGHT, 180, 64, MAX_SMOKE_DIMENSION,
);
if (width * height > MAX_SMOKE_PIXELS) {
  throw new Error(`smoke-test dimensions may not exceed ${MAX_SMOKE_PIXELS} pixels`);
}
const time = parseBoundedNumber(
  'DEBRIS_SMOKE_TIME', process.env.DEBRIS_SMOKE_TIME, 0, 0, MAX_PROFILE_SECONDS,
);
// A fixed-time page still supplies the one retained screenshot. Profiling
// then walks these sorted times inside that same guarded Chrome instance.
const profileTimes = parseProfileTimes(process.env.DEBRIS_SMOKE_PROFILE_TIMES);
const profileFrames = parseBoundedInteger(
  'DEBRIS_SMOKE_PROFILE_FRAMES', process.env.DEBRIS_SMOKE_PROFILE_FRAMES,
  3, 2, MAX_PROFILE_FRAMES,
);
const captureProfileScreenshots = /^(?:1|true|yes)$/i.test(
  process.env.DEBRIS_SMOKE_PROFILE_SCREENSHOTS || '',
);
const artifactLabel = String(process.env.DEBRIS_SMOKE_LABEL || '').trim();
if (artifactLabel && (artifactLabel.length > 32 || !/^[a-z0-9][a-z0-9_-]*$/i.test(artifactLabel))) {
  throw new Error('DEBRIS_SMOKE_LABEL must be 1-32 ASCII letters, digits, underscores, or hyphens');
}
const textureQuality = String(process.env.DEBRIS_SMOKE_TEXTURE || 'low').toLowerCase();
if (!['high', 'medium', 'low', 'normal', 'reduced', 'ultra', 'full'].includes(textureQuality)) {
  throw new Error('DEBRIS_SMOKE_TEXTURE must be high, medium, or low');
}
const dxt5Mode = String(process.env.DEBRIS_SMOKE_DXT5 || 'auto').toLowerCase();
if (!['auto', 's3tc', 'rgba8'].includes(dxt5Mode)) {
  throw new Error('DEBRIS_SMOKE_DXT5 must be auto, s3tc, or rgba8');
}
const diagnostics = /^(?:1|true|yes)$/i.test(process.env.DEBRIS_SMOKE_DIAGNOSTICS || '');
const showStats = !/^(?:0|false|no)$/i.test(process.env.DEBRIS_SMOKE_SHOW_STATS || '');
// Diagnostic-only phase isolation. The production page has no corresponding
// query option: this patch is injected after its fixed frame has rendered, so
// a comparison capture can bypass IPP without changing normal playback.
const disablePostEffects = /^(?:1|true|yes)$/i.test(
  process.env.DEBRIS_SMOKE_DISABLE_POST || '',
);
// Diagnostic-only conservative shadow isolation. Clearing cached geometry
// bounds makes the renderer retain every light/caster pair and choose z-fail,
// without changing the production page or its normal culling path.
const forceAllCasters = /^(?:1|true|yes)$/i.test(
  process.env.DEBRIS_SMOKE_FORCE_ALL_CASTERS || '',
);
const warmFrames = parseBoundedInteger(
  'DEBRIS_SMOKE_WARM_FRAMES', process.env.DEBRIS_SMOKE_WARM_FRAMES, 0, 0, 30,
);
const warmFrameRate = parseBoundedNumber(
  'DEBRIS_SMOKE_WARM_FPS', process.env.DEBRIS_SMOKE_WARM_FPS, 60, 1, 240,
);
const inspectOperationIds = String(process.env.DEBRIS_SMOKE_INSPECT_OPS || '')
  .split(',').map(value => value.trim()).filter(Boolean)
  .map(Number).filter(Number.isSafeInteger).slice(0, 32);
const requestedAngleBackend = String(process.env.DEBRIS_SMOKE_ANGLE || 'swiftshader').toLowerCase();
const angleBackend = ['swiftshader', 'metal', 'gl', 'default'].includes(requestedAngleBackend)
  ? requestedAngleBackend : 'swiftshader';
const timeoutMilliseconds = parseBoundedInteger(
  'DEBRIS_SMOKE_TIMEOUT_MS', process.env.DEBRIS_SMOKE_TIMEOUT_MS, 90000, 10000, 3600000,
);
const rssSampleMilliseconds = parseBoundedInteger(
  'DEBRIS_SMOKE_RSS_SAMPLE_MS', process.env.DEBRIS_SMOKE_RSS_SAMPLE_MS, 100, 100, 500,
);
const requestedRSSLimitMB = Number(
  process.env.DEBRIS_SMOKE_RSS_LIMIT_MB ?? process.env.DEBRIS_SMOKE_MAX_RSS_MB,
);
const defaultRSSLimitBytes = Math.min(1700 * MEBIBYTE, Math.floor(totalmem() / 4));
const rssLimitBytes = Number.isFinite(requestedRSSLimitMB) && requestedRSSLimitMB > 0
  // The environment may lower the safety ceiling, never raise it. A larger
  // value would silently defeat the host-relative hard cap documented below.
  ? Math.min(defaultRSSLimitBytes,
    Math.max(64 * MEBIBYTE, Math.floor(requestedRSSLimitMB * MEBIBYTE)))
  : defaultRSSLimitBytes;
const requestedV8HeapLimitMB = Number(process.env.DEBRIS_SMOKE_V8_HEAP_MB);
// Retained production captures currently occupy about 0.7 GiB of JS heap.
// An 896 MiB old-space ceiling leaves playback headroom while making V8 collect
// dead expanded mesh records during precalc instead of allowing Chrome to
// approach the external aggregate-RSS guard. Like the RSS setting, callers
// may lower this ceiling but cannot raise it.
const v8HeapLimitMB = Number.isFinite(requestedV8HeapLimitMB) && requestedV8HeapLimitMB > 0
  ? Math.min(896, Math.max(128, Math.floor(requestedV8HeapLimitMB)))
  : 896;
const stdoutLimit = Math.max(64 * 1024,
  Math.floor((Number(process.env.DEBRIS_SMOKE_STDOUT_KIB) || 2048) * 1024));
const stderrLimit = Math.max(32 * 1024,
  Math.floor((Number(process.env.DEBRIS_SMOKE_STDERR_KIB) || 256) * 1024));
const cdpTimeoutMilliseconds = Math.max(2000, Number(process.env.DEBRIS_SMOKE_CDP_TIMEOUT_MS) || 30000);
const gcAfterRender = process.env.DEBRIS_SMOKE_GC_AFTER_RENDER !== '0';
const gcWaitMilliseconds = Math.max(0, Number(process.env.DEBRIS_SMOKE_GC_WAIT_MS) || 250);
const terminationGraceMilliseconds = 3000;
const killVerificationMilliseconds = 2000;
const lockDirectory = resolve(tmpdir(), 'fr-041-debris-browser-smoke.lock');
const lockRetryMilliseconds = 100;
const requestedPath = process.env.DEBRIS_SMOKE_PATH || '/';
const smokePath = requestedPath.startsWith('/') ? requestedPath : `/${requestedPath}`;
if (smokePath.includes('..') || smokePath.includes('?') || smokePath.includes('#')) {
  throw new Error('DEBRIS_SMOKE_PATH must be a local absolute path without query or fragment');
}
const productionPage = smokePath === '/' || smokePath === '/index.html';
const audioPage = smokePath === '/tools/audio_smoke.html' ||
  smokePath === '/tools/audio_worker_smoke.html';
const liveProductionRequested = /^(?:1|true|yes)$/i.test(
  process.env.DEBRIS_SMOKE_LIVE || '',
);
const liveProduction = productionPage && liveProductionRequested;
if (liveProductionRequested && !productionPage) {
  throw new Error('DEBRIS_SMOKE_LIVE is only valid for the production page');
}
if (liveProduction && warmFrames) {
  throw new Error('DEBRIS_SMOKE_WARM_FRAMES cannot be combined with DEBRIS_SMOKE_LIVE');
}
if (profileTimes.length && !productionPage) {
  throw new Error('DEBRIS_SMOKE_PROFILE_TIMES is only valid for the production page');
}
if (profileTimes.length && liveProduction) {
  throw new Error('DEBRIS_SMOKE_PROFILE_TIMES cannot be combined with DEBRIS_SMOKE_LIVE');
}
if (profileTimes.length && warmFrames) {
  throw new Error('DEBRIS_SMOKE_PROFILE_TIMES cannot be combined with DEBRIS_SMOKE_WARM_FRAMES');
}
if (captureProfileScreenshots && !profileTimes.length) {
  throw new Error('DEBRIS_SMOKE_PROFILE_SCREENSHOTS requires DEBRIS_SMOKE_PROFILE_TIMES');
}
if (captureProfileScreenshots && profileTimes.length > MAX_PROFILE_SCREENSHOTS) {
  throw new Error(
    `DEBRIS_SMOKE_PROFILE_SCREENSHOTS accepts at most ${MAX_PROFILE_SCREENSHOTS} profile times`,
  );
}
if (captureProfileScreenshots && width * height * profileTimes.length >
    MAX_PROFILE_SCREENSHOT_PIXELS) {
  throw new Error(
    `profile screenshots may contain at most ${MAX_PROFILE_SCREENSHOT_PIXELS} total pixels`,
  );
}
const chromeCandidates = [
  process.env.CHROME_BIN,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

function parseBoundedInteger(name, rawValue, fallback, minimum, maximum) {
  if (rawValue === undefined || String(rawValue).trim() === '') return fallback;
  const value = Number(rawValue);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function parseBoundedNumber(name, rawValue, fallback, minimum, maximum) {
  if (rawValue === undefined || String(rawValue).trim() === '') return fallback;
  const value = Number(rawValue);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a finite number from ${minimum} through ${maximum}`);
  }
  return value;
}

function parseProfileTimes(rawValue) {
  if (rawValue === undefined || String(rawValue).trim() === '') return [];
  const source = String(rawValue);
  if (source.length > 1024) {
    throw new Error('DEBRIS_SMOKE_PROFILE_TIMES is too long');
  }
  const tokens = source.split(',');
  if (tokens.length > MAX_PROFILE_TIMES) {
    throw new Error(`DEBRIS_SMOKE_PROFILE_TIMES accepts at most ${MAX_PROFILE_TIMES} times`);
  }
  const times = tokens.map((token, index) => {
    const trimmed = token.trim();
    if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(trimmed)) {
      throw new Error(`invalid DEBRIS_SMOKE_PROFILE_TIMES value at position ${index + 1}`);
    }
    const seconds = Number(trimmed);
    if (!Number.isFinite(seconds) || seconds < 0 || seconds > MAX_PROFILE_SECONDS) {
      throw new Error(
        `DEBRIS_SMOKE_PROFILE_TIMES values must be between 0 and ${MAX_PROFILE_SECONDS} seconds`,
      );
    }
    return seconds;
  });
  for (let index = 1; index < times.length; index++) {
    if (times[index] <= times[index - 1]) {
      throw new Error('DEBRIS_SMOKE_PROFILE_TIMES values must be strictly increasing');
    }
  }
  return times;
}

let chrome = null;
for (const candidate of chromeCandidates) {
  try { await access(candidate, constants.X_OK); chrome = candidate; break; } catch (_) { /* try next */ }
}
if (!chrome) throw new Error('Chrome or Chromium was not found; set CHROME_BIN');

class BoundedCapture {
  constructor(limit) {
    this.headLimit = Math.ceil(limit / 2);
    this.tailLimit = limit - this.headLimit;
    this.head = '';
    this.tail = '';
    this.totalCharacters = 0;
    this.droppedCharacters = 0;
  }

  append(value) {
    let chunk = String(value);
    this.totalCharacters += chunk.length;
    if (this.head.length < this.headLimit) {
      const count = Math.min(chunk.length, this.headLimit - this.head.length);
      this.head += chunk.slice(0, count);
      chunk = chunk.slice(count);
    }
    if (!chunk) return;
    this.tail += chunk;
    if (this.tail.length > this.tailLimit) {
      const drop = this.tail.length - this.tailLimit;
      this.tail = this.tail.slice(drop);
      this.droppedCharacters += drop;
    }
  }

  value() {
    if (!this.droppedCharacters) return this.head + this.tail;
    return `${this.head}\n<!-- browser-smoke output truncated: ${this.droppedCharacters} characters -->\n${this.tail}`;
  }

  get truncated() { return this.droppedCharacters > 0; }
}

function processExists(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return error?.code === 'EPERM'; }
}

async function quarantineStaleLock(attempt) {
  const quarantine = `${lockDirectory}.stale-${process.pid}-${Date.now()}-${attempt}`;
  try {
    // Renaming is atomic. If several contenders discover the same stale lock,
    // only one can move it; the others must retry instead of deleting a new
    // owner's directory by path.
    await rename(lockDirectory, quarantine);
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
  await rm(quarantine, { recursive: true, force: true });
  return true;
}

async function acquireRunLock() {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      await mkdir(lockDirectory);
      try {
        await writeFile(resolve(lockDirectory, 'owner.json'), JSON.stringify({
          pid: process.pid, started: new Date().toISOString(), projectRoot,
        }));
      } catch (error) {
        await rm(lockDirectory, { recursive: true, force: true }).catch(() => {});
        throw error;
      }
      return true;
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      let owner = null;
      try { owner = JSON.parse(await readFile(resolve(lockDirectory, 'owner.json'), 'utf8')); }
      catch (_) { /* an interrupted writer is treated as stale */ }
      if (processExists(Number(owner?.pid))) {
        throw new Error(`another Debris browser smoke is already running (pid ${owner.pid})`);
      }

      if (!owner) {
        // Fail closed. This can be the tiny mkdir -> owner.json initialization
        // window, and an age-based reclaim would let a stalled creator wake up
        // and overwrite a newer owner's record. An ownerless crash residue is
        // intentionally left for explicit cleanup after checking processes.
        throw new Error(
          `Debris browser-smoke lock has no readable owner; verify no run is active, then remove ${lockDirectory}`,
        );
      }

      if (!await quarantineStaleLock(attempt)) {
        await delay(lockRetryMilliseconds);
      }
    }
  }
  throw new Error('could not acquire the Debris browser-smoke lock');
}

async function existingSmokeProcesses() {
  const { stdout } = await execFileAsync('ps', ['-axo', 'pid=,command='], {
    encoding: 'utf8', maxBuffer: 4 * MEBIBYTE,
  });
  const result = [];
  for (const line of stdout.split('\n')) {
    const match = /^\s*(\d+)\s+(.*)$/.exec(line);
    if (!match || !match[2].includes('--user-data-dir=') || !match[2].includes('debris-chrome-')) continue;
    result.push(Number(match[1]));
  }
  return result;
}

async function existingAutomatedChromeProcesses() {
  const { stdout } = await execFileAsync('ps', ['-axo', 'pid=,command='], {
    encoding: 'utf8', maxBuffer: 4 * MEBIBYTE,
  });
  const result = [];
  for (const line of stdout.split('\n')) {
    const match = /^\s*(\d+)\s+(.*)$/.exec(line);
    if (!match) continue;
    const command = match[2];
    // Match packaged Chrome/Chromium as well as Playwright's lower-case
    // chromium_headless_shell/chrome-headless-shell binaries. Missing the
    // latter would allow two memory-heavy automated browser trees to overlap.
    if (!/(?:google chrome|chromium|chrome-headless-shell)/i.test(command) ||
        !command.includes('--headless') ||
        !command.includes('--remote-debugging-port=')) continue;
    result.push(Number(match[1]));
  }
  return result;
}

async function chromeTreeMemory(rootPid) {
  const { stdout } = await execFileAsync('ps', ['-axo', 'pid=,ppid=,pgid=,rss='], {
    encoding: 'utf8', maxBuffer: 4 * MEBIBYTE,
  });
  const rows = [];
  const children = new Map();
  for (const line of stdout.split('\n')) {
    const match = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/.exec(line);
    if (!match) continue;
    const row = {
      pid: Number(match[1]), ppid: Number(match[2]),
      pgid: Number(match[3]), rssKiB: Number(match[4]),
    };
    rows.push(row);
    let list = children.get(row.ppid);
    if (!list) children.set(row.ppid, list = []);
    list.push(row.pid);
  }

  const pids = new Set([rootPid]);
  const queue = [rootPid];
  while (queue.length) {
    for (const pid of children.get(queue.shift()) || []) {
      if (pids.has(pid)) continue;
      pids.add(pid); queue.push(pid);
    }
  }
  // Helpers can be reparented while remaining in the detached group. Include
  // them in accounting and cleanup. Summed RSS is intentionally conservative
  // because shared pages may be counted in more than one Chrome process.
  for (const row of rows) if (row.pgid === rootPid) pids.add(row.pid);
  let rssKiB = 0;
  for (const row of rows) if (pids.has(row.pid)) rssKiB += row.rssKiB;
  return { rssBytes: rssKiB * 1024, pids: [...pids].filter(pid => pid !== process.pid) };
}

function signalChromeTree(child, pids, signal) {
  let sent = false;
  if (child?.pid && process.platform !== 'win32') {
    try { process.kill(-child.pid, signal); sent = true; } catch (_) { /* use individual pids below */ }
  }
  for (const pid of [...pids].sort((a, b) => b - a)) {
    if (!processExists(pid)) continue;
    try { process.kill(pid, signal); sent = true; } catch (_) { /* process may just have exited */ }
  }
  if (child && child.exitCode === null && child.signalCode === null) {
    try { sent = child.kill(signal) || sent; } catch (_) { /* already gone */ }
  }
  return sent;
}

function processGroupExists(pid) {
  if (!pid || process.platform === 'win32') return processExists(pid);
  try { process.kill(-pid, 0); return true; }
  catch (error) { return error?.code === 'EPERM'; }
}

function delay(milliseconds) {
  return new Promise(resolveDelay => setTimeout(resolveDelay, milliseconds));
}

async function terminateChromeTree(child, pids, graceMilliseconds = terminationGraceMilliseconds) {
  if (!child?.pid) return false;
  const groupPid = child.pid;
  const signalled = signalChromeTree(child, pids, 'SIGTERM');
  const deadline = performance.now() + graceMilliseconds;
  while (processGroupExists(groupPid) && performance.now() < deadline) await delay(100);
  if (processGroupExists(groupPid) || pids.some(processExists)) {
    signalChromeTree(child, pids, 'SIGKILL');
    const killDeadline = performance.now() + killVerificationMilliseconds;
    while ((processGroupExists(groupPid) || pids.some(processExists)) &&
      performance.now() < killDeadline) await delay(50);
  }
  if (processGroupExists(groupPid) || pids.some(processExists)) {
    throw new Error(`Chrome process group ${groupPid} survived SIGKILL cleanup`);
  }
  return signalled;
}

function decodeAttribute(value) {
  return value
    .replaceAll('&quot;', '"').replaceAll('&#39;', "'")
    .replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&');
}

function metricValue(value) {
  if (value === 'true') return true;
  if (value === 'false') return false;
  const number = Number(value);
  if (value !== '' && Number.isFinite(number)) return number;
  if (value.startsWith('{') || value.startsWith('[')) {
    try { return JSON.parse(value); } catch (_) { /* retain malformed diagnostic text */ }
  }
  return value;
}

function pageDatasets(html) {
  const tag = /<html\b[^>]*>/i.exec(html)?.[0] || '';
  const result = {};
  const expression = /\bdata-([a-z0-9-]+)="([^"]*)"/gi;
  let match;
  while ((match = expression.exec(tag))) result[match[1]] = metricValue(decodeAttribute(match[2]));
  return result;
}

function pickMetrics(source, names) {
  const result = {};
  for (const name of names) if (source[name] !== undefined) result[name] = source[name];
  return result;
}

class CDPClient {
  constructor(socket, commandTimeoutMilliseconds = 10000) {
    this.socket = socket;
    this.commandTimeoutMilliseconds = commandTimeoutMilliseconds;
    this.nextId = 1;
    this.pending = new Map();
    socket.addEventListener('message', event => {
      let message;
      try { message = JSON.parse(String(event.data)); }
      catch (_) { return; }
      if (!message.id) return;
      const request = this.pending.get(message.id);
      if (!request) return;
      this.pending.delete(message.id);
      clearTimeout(request.timer);
      if (message.error) request.reject(new Error(`${request.method}: ${message.error.message || 'CDP error'}`));
      else request.resolve(message.result || {});
    });
    socket.addEventListener('close', () => this.rejectPending(new Error('Chrome DevTools connection closed')));
    socket.addEventListener('error', () => this.rejectPending(new Error('Chrome DevTools connection failed')));
  }

  static connect(url, timeoutMilliseconds = 10000) {
    if (typeof globalThis.WebSocket !== 'function') {
      throw new Error('browser_smoke requires the global WebSocket available in Node 22 or newer');
    }
    return new Promise((resolveConnect, rejectConnect) => {
      const socket = new WebSocket(url);
      const timer = setTimeout(() => {
        socket.close(); rejectConnect(new Error('Chrome DevTools connection timed out'));
      }, timeoutMilliseconds);
      socket.addEventListener('open', () => {
        clearTimeout(timer); resolveConnect(new CDPClient(socket, timeoutMilliseconds));
      }, { once: true });
      socket.addEventListener('error', () => {
        clearTimeout(timer); rejectConnect(new Error('Chrome DevTools connection failed'));
      }, { once: true });
    });
  }

  rejectPending(error) {
    for (const request of this.pending.values()) {
      clearTimeout(request.timer); request.reject(error);
    }
    this.pending.clear();
  }

  send(method, params = {}, timeoutMilliseconds = this.commandTimeoutMilliseconds) {
    if (this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error(`cannot send ${method}: Chrome DevTools connection is not open`));
    }
    const commandTimeoutMilliseconds = Math.max(1,
      Number(timeoutMilliseconds) || this.commandTimeoutMilliseconds);
    const id = this.nextId++;
    return new Promise((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        rejectRequest(new Error(`${method} timed out after ${commandTimeoutMilliseconds} ms`));
      }, commandTimeoutMilliseconds);
      this.pending.set(id, { method, resolve: resolveRequest, reject: rejectRequest, timer });
      try { this.socket.send(JSON.stringify({ id, method, params })); }
      catch (error) {
        clearTimeout(timer); this.pending.delete(id); rejectRequest(error);
      }
    });
  }

  async evaluate(expression, timeoutMilliseconds = this.commandTimeoutMilliseconds) {
    const response = await this.send('Runtime.evaluate', {
      expression, returnByValue: true, awaitPromise: true,
    }, timeoutMilliseconds);
    if (response.exceptionDetails) {
      const description = response.exceptionDetails.exception?.description ||
        response.exceptionDetails.text || 'page evaluation failed';
      throw new Error(description);
    }
    return response.result?.value;
  }

  close() {
    try { this.socket.close(); } catch (_) { /* Chrome may already be gone */ }
  }
}

async function discoverPageWebSocket(browserWebSocketURL, expectedPath, timeoutMilliseconds = 10000) {
  const endpoint = new URL(browserWebSocketURL);
  endpoint.protocol = endpoint.protocol === 'wss:' ? 'https:' : 'http:';
  endpoint.pathname = '/json/list';
  endpoint.search = '';
  endpoint.hash = '';
  const deadline = performance.now() + timeoutMilliseconds;
  let lastError = null;
  while (performance.now() < deadline) {
    try {
      const response = await fetch(endpoint, { cache: 'no-store', signal: AbortSignal.timeout(2000) });
      if (!response.ok) throw new Error(`DevTools target list returned HTTP ${response.status}`);
      const targets = await response.json();
      const target = targets.find(entry => {
        if (entry.type !== 'page' || !entry.webSocketDebuggerUrl) return false;
        try { return new URL(entry.url).pathname === expectedPath; }
        catch (_) { return false; }
      });
      if (target) return target.webSocketDebuggerUrl;
    } catch (error) { lastError = error; }
    await delay(100);
  }
  throw new Error(`Chrome page target was not found${lastError ? `: ${lastError.message}` : ''}`);
}

async function waitForTerminalPage(client, production, audio, onProgress = null) {
  for (;;) {
    const progress = await client.evaluate(`(() => {
      const data = document.documentElement?.dataset || {};
      return {
        terminal: ${production
        ? "data.debrisState === 'rendered' || data.debrisState === 'error'"
        : audio
          ? "data.audioSmoke === 'ok' || data.audioSmoke === 'error'"
          : "data.webglSmoke === 'ok' || data.webglSmoke === 'error'"},
        state: data.debrisState || data.audioSmoke || data.webglSmoke || '',
        status: data.debrisStatus || data.audioErrors || data.webglError || '',
      };
    })()`);
    onProgress?.(progress);
    if (progress?.terminal) return;
    await delay(100);
  }
}

async function waitForLiveLauncher(client) {
  for (;;) {
    const state = await client.evaluate(`(async () => {
      const data = document.documentElement?.dataset || {};
      if (data.debrisState === 'error') {
        throw new Error(data.debrisStatus || 'Debris launcher failed to initialize');
      }
      const [{ AudioStream }, { DebrisApp }] = await Promise.all([
        import('/src/audio.js'),
        import('/src/app.js'),
      ]);
      return data.debrisState === 'launcher' &&
        typeof AudioStream === 'function' && typeof DebrisApp === 'function' &&
        Boolean(document.querySelector('#start'));
    })()`);
    if (state) return;
    await delay(100);
  }
}

async function startLiveProduction(client) {
  await waitForLiveLauncher(client);
  return client.evaluate(`(async () => {
    const [{ AudioStream: OriginalAudioStream }, { DebrisApp }] = await Promise.all([
      import('/src/audio.js'),
      import('/src/app.js'),
    ]);
    const button = document.querySelector('#start');
    if (typeof OriginalAudioStream !== 'function' || typeof DebrisApp !== 'function' || !button) {
      throw new Error('Debris live launcher is incomplete');
    }
    const trace = {
      protocol: location.protocol,
      events: [], streams: [], openStreams: 0, maxOpenStreams: 0,
    };
    const mark = (record, type, detail = {}) => {
      const event = {
        sequence: trace.events.length,
        streamIndex: record?.index ?? null,
        type,
        milliseconds: performance.now(),
        ...detail,
      };
      trace.events.push(event);
      return event.sequence;
    };
    const captureWorkerState = stream => ({
      producerBackend: stream.producerBackend || null,
      producerWorkerPresent: Boolean(stream.producerWorker),
      producerChannelPresent: Boolean(stream.producerChannel),
      workerPrepared: Boolean(stream.workerPrepared),
      workerPrimed: Boolean(stream.workerPrimed),
      workerError: stream.workerError
        ? String(stream.workerError?.message || stream.workerError) : null,
      workerFallbackReason: stream.workerFallbackReason
        ? String(stream.workerFallbackReason?.message || stream.workerFallbackReason) : null,
      workerSongPresent: Boolean(stream.workerSong),
      synthPresent: Boolean(stream.synth),
    });
    class SmokeAudioStream extends OriginalAudioStream {
      constructor(options = {}) {
        super(options);
        const record = {
          index: trace.streams.length,
          blockFrames: this.blockFrames,
          queueBlocks: this.queueBlocks,
          tailSeconds: this.tailSeconds,
          initCalls: 0,
          starts: 0,
          closes: 0,
          initialized: false,
          started: false,
          closed: false,
          error: null,
          workerRequested: Boolean(this.workerSong),
          reportClock: this.reportClock,
          synthPresentAtConstruct: Boolean(this.synth),
          workerPlayerCheckpointMemoryBytes:
            Number.isFinite(this.workerPlayerOptions?.checkpointMemoryBytes)
              ? this.workerPlayerOptions.checkpointMemoryBytes : null,
          workerPlayerCheckpointIntervalSamples:
            Number.isFinite(this.workerPlayerOptions?.checkpointIntervalSamples)
              ? this.workerPlayerOptions.checkpointIntervalSamples : null,
        };
        this.__debrisSmokeAudioRecord = record;
        trace.streams.push(record);
        trace.openStreams++;
        trace.maxOpenStreams = Math.max(trace.maxOpenStreams, trace.openStreams);
        record.constructSequence = mark(record, 'construct');
      }
      async init(player) {
        const record = this.__debrisSmokeAudioRecord;
        record.initCalls++;
        record.initPlayerPresent = player != null;
        mark(record, 'init-begin');
        try {
          const result = await super.init(player);
          record.initialized = true;
          record.sampleRate = this.sampleRate;
          record.durationSamples = Number.isFinite(this.durationSamples)
            ? this.durationSamples : null;
          record.producerBackendAfterInit = this.producerBackend || null;
          record.synthPresentAfterInit = Boolean(this.synth);
          record.workerStateAfterInit = captureWorkerState(this);
          mark(record, 'init-end');
          return result;
        } catch (error) {
          record.error = String(error?.message || error);
          mark(record, 'init-error', { error: record.error });
          throw error;
        }
      }
      async start() {
        const record = this.__debrisSmokeAudioRecord;
        record.starts++;
        mark(record, 'start-begin');
        try {
          const result = await super.start();
          record.started = true;
          record.workerStateAfterStart = captureWorkerState(this);
          mark(record, 'start-end');
          return result;
        } catch (error) {
          record.error = String(error?.message || error);
          mark(record, 'start-error', { error: record.error });
          throw error;
        }
      }
      async close() {
        const record = this.__debrisSmokeAudioRecord;
        const wasOpen = !record.closed && !record.closing;
        record.closing = true;
        record.closes++;
        record.producerBackendAtClose = this.producerBackend || null;
        record.synthPresentAtClose = Boolean(this.synth);
        record.workerStateAtClose = captureWorkerState(this);
        record.checkpointMemoryBytesAtClose = Number.isFinite(this.synth?.checkpointMemoryBytes)
          ? this.synth.checkpointMemoryBytes : null;
        record.checkpointIntervalSamplesAtClose = Number.isFinite(this.synth?.checkpointIntervalSamples)
          ? this.synth.checkpointIntervalSamples : null;
        const checkpoints = this.synth?.checkpointStats?.();
        record.checkpointBytesAtClose = Number.isFinite(checkpoints?.bytes)
          ? checkpoints.bytes : null;
        mark(record, 'close-begin');
        try {
          const result = await super.close();
          record.closed = true;
          record.workerStateAfterClose = captureWorkerState(this);
          record.closeSequence = mark(record, 'close-end');
          return result;
        } catch (error) {
          record.error = String(error?.message || error);
          mark(record, 'close-error', { error: record.error });
          throw error;
        } finally {
          record.closing = false;
          if (wasOpen) trace.openStreams = Math.max(0, trace.openStreams - 1);
        }
      }
    }
    const originalInit = DebrisApp.prototype.init;
    DebrisApp.prototype.init = async function(...args) {
      this.dependencies.AudioStream = SmokeAudioStream;
      try { return await originalInit.apply(this, args); }
      finally { DebrisApp.prototype.init = originalInit; }
    };
    globalThis.__debrisSmokeAudioLifecycle = trace;
    button.click();
    return true;
  })()`);
}

async function readLiveAudioLifecycle(client) {
  return client.evaluate(`(() => {
    const trace = globalThis.__debrisSmokeAudioLifecycle;
    const app = globalThis.__debris;
    if (!trace) return { valid: false, problems: ['lifecycle probe is missing'] };
    const streams = trace.streams || [];
    const loader = streams[0] || null;
    const main = streams[1] || null;
    const problems = [];
    const httpLiveProduction = trace.protocol === 'http:' || trace.protocol === 'https:';
    const event = (streamIndex, type) => trace.events.find(entry =>
      entry.streamIndex === streamIndex && entry.type === type);
    if (streams.length !== 2) problems.push('expected exactly loader and main streams');
    if (loader?.tailSeconds !== 0) problems.push('loader tail must be disabled');
    if (loader?.initCalls !== 1 || loader?.starts !== 1 || loader?.closes !== 1 || !loader?.closed) {
      problems.push('loader must initialize, start, and close exactly once');
    }
    if (loader?.workerRequested !== true) {
      problems.push('loader must request the Worker producer backend');
    }
    if (httpLiveProduction && loader?.producerBackendAfterInit !== 'worker') {
      problems.push('HTTP live loader did not initialize the Worker producer backend');
    }
    if (loader?.reportClock !== false) {
      problems.push('loader worklet clock reports must remain disabled');
    }
    if (loader?.initPlayerPresent !== false ||
        loader?.synthPresentAtConstruct !== false ||
        loader?.synthPresentAfterInit !== false ||
        loader?.workerStateAfterStart?.synthPresent !== false ||
        loader?.synthPresentAtClose !== false) {
      problems.push('loader must not construct or retain a main-thread synth');
    }
    if (loader?.workerPlayerCheckpointMemoryBytes !== 0 ||
        loader?.workerPlayerCheckpointIntervalSamples !== 0 ||
        loader?.checkpointMemoryBytesAtClose !== null ||
        loader?.checkpointIntervalSamplesAtClose !== null ||
        loader?.checkpointBytesAtClose !== null) {
      problems.push('loader Worker checkpoints must remain disabled with no main-thread checkpoint state');
    }
    const workerAtClose = loader?.workerStateAtClose;
    if (!workerAtClose?.producerWorkerPresent ||
        !workerAtClose?.producerChannelPresent ||
        !workerAtClose?.workerPrepared ||
        !workerAtClose?.workerPrimed ||
        workerAtClose?.workerError ||
        workerAtClose?.workerFallbackReason ||
        workerAtClose?.synthPresent) {
      problems.push('loader Worker was not healthy and exclusively owned at close');
    }
    const workerAfterClose = loader?.workerStateAfterClose;
    if (!workerAfterClose ||
        workerAfterClose.producerWorkerPresent ||
        workerAfterClose.producerChannelPresent ||
        workerAfterClose.workerPrepared ||
        workerAfterClose.workerPrimed ||
        workerAfterClose.workerSongPresent ||
        workerAfterClose.synthPresent) {
      problems.push('loader Worker resources remained reachable after close');
    }
    if (main?.initCalls !== 1 || main?.starts !== 1 || main?.closes !== 0 || !main?.started) {
      problems.push('main stream must initialize and start after precalc');
    }
    const loaderClose = event(0, 'close-end');
    const mainConstruct = event(1, 'construct');
    if (!loaderClose || !mainConstruct || loaderClose.sequence >= mainConstruct.sequence) {
      problems.push('main stream was constructed before loader shutdown completed');
    }
    if (trace.maxOpenStreams !== 1 || trace.openStreams !== 1) {
      problems.push('loader and main streams overlapped');
    }
    if (app?.loaderAudio != null) problems.push('app still retains loader AudioStream');
    if (!app?.audio) problems.push('app has no main AudioStream');
    for (const stream of streams) if (stream.error) problems.push(stream.error);
    return {
      valid: problems.length === 0,
      problems,
      loaderAudioPresent: app?.loaderAudio != null,
      mainAudioPresent: Boolean(app?.audio),
      appReady: Boolean(app?.ready),
      appRunning: Boolean(app?.running),
      protocol: trace.protocol,
      openStreams: trace.openStreams,
      maxOpenStreams: trace.maxOpenStreams,
      streams: streams.map((stream, index) => ({
        ...stream,
        role: index === 0 ? 'loader' : index === 1 ? 'main' : 'unexpected',
      })),
      events: trace.events,
    };
  })()`);
}

const mimeTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.kx', 'application/octet-stream'],
  ['.v2m', 'application/octet-stream'],
]);

const server = createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url || '/', 'http://127.0.0.1');
    const relative = decodeURIComponent(requestUrl.pathname === '/' ? '/index.html' : requestUrl.pathname);
    const pathname = resolve(projectRoot, `.${relative}`);
    if (pathname !== projectRoot && !pathname.startsWith(projectRoot + sep)) {
      response.writeHead(403).end('forbidden'); return;
    }
    const file = await stat(pathname);
    if (!file.isFile()) {
      response.writeHead(404).end('not found');
      return;
    }
    response.writeHead(200, {
      'content-type': mimeTypes.get(extname(pathname)) || 'application/octet-stream',
      'content-length': file.size,
      'cache-control': 'no-store',
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
      'cross-origin-resource-policy': 'same-origin',
    });
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    const body = createReadStream(pathname);
    body.once('error', error => response.destroy(error));
    response.once('close', () => {
      if (!response.writableEnded) body.destroy();
    });
    body.pipe(response);
  } catch (error) {
    if (!response.headersSent) {
      response.writeHead(error?.code === 'ENOENT' ? 404 : 500).end(String(error?.message || error));
    } else if (!response.destroyed) response.destroy(error);
  }
});

let lockHeld = false;
let listening = false;
let userData = null;
let child = null;
let monitorTimer = null;
let timeoutTimer = null;
let killPromise = null;
let lastTreePids = [];
let result = null;
let runnerError = null;
let cleanupError = null;
let interruptedSignal = null;
let signalHandler = null;
const stdoutCapture = new BoundedCapture(stdoutLimit);
const stderrCapture = new BoundedCapture(stderrLimit);
const started = performance.now();
let peakRSSBytes = 0;
let lastRSSBytes = 0;
let peakProcessCount = 0;
let rssSamples = 0;
let memoryExceeded = false;
let timedOut = false;
let domCompleted = false;
let monitorError = null;
let stopReason = null;
let sampling = false;
let cdpError = null;
let lastPageProgress = null;
let cdpHeapUsage = null;
let cdpPerformanceMetrics = null;
let pageMemoryMetrics = null;
let warmFrameMetrics = null;
let profileMetrics = null;
let profileScreenshotArtifacts = [];
let operationInspection = null;
let audioLifecycle = null;
let gcCalled = false;
let capturedDom = '';
let debugEndpointTail = '';
let resultArtifactPath = null;

try {
  lockHeld = await acquireRunLock();
  const existing = await existingSmokeProcesses();
  if (existing.length) throw new Error(`refusing to overlap existing Debris Chrome process(es): ${existing.join(', ')}`);
  const automatedChrome = await existingAutomatedChromeProcesses();
  if (automatedChrome.length) {
    throw new Error(
      `refusing to overlap existing automated Chrome process(es): ${automatedChrome.join(', ')}`,
    );
  }

  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  listening = true;
  await mkdir(artifacts, { recursive: true });
  userData = await mkdtemp(resolve(tmpdir(), 'debris-chrome-'));
  const artifactStem = productionPage
    ? `debris-${liveProduction ? 'live' : profileTimes.length
      ? `profile-${profileTimes.length}times`
      : `t${String(time).replace('.', '_')}`}` +
      `-${textureQuality.replace(/[^a-z0-9]+/gi, '-')}-${width}x${height}` +
      `${dxt5Mode === 'auto' ? '' : `-${dxt5Mode}`}` +
      `${diagnostics ? '-diag' : ''}` +
      `${warmFrames ? `-warm${warmFrames}` : ''}` +
      `${angleBackend === 'swiftshader' ? '' : `-${angleBackend}`}` +
      `${artifactLabel ? `-${artifactLabel}` : ''}`
    : `debris-${smokePath.replace(/^\/+|\.html$/g, '').replace(/[^a-z0-9]+/gi, '-') || 'page'}`;
  const screenshot = resolve(artifacts, `${artifactStem}.png`);
  const domPath = resolve(artifacts, `${artifactStem}.html`);
  resultArtifactPath = resolve(artifacts, `${artifactStem}.json`);
  const expectedProfileScreenshots = captureProfileScreenshots
    ? profileTimes.map(seconds => resolve(
      artifacts, `${artifactStem}-t${String(seconds).replace('.', '_')}.png`,
    )) : [];
  // Never let a failed run inherit success artifacts from an earlier frame.
  await rm(screenshot, { force: true });
  await rm(domPath, { force: true });
  await rm(resultArtifactPath, { force: true });
  for (const path of expectedProfileScreenshots) await rm(path, { force: true });
  const port = server.address().port;
  const origin = `http://127.0.0.1:${port}`;
  const productionQuery = `?${liveProduction ? '' : `t=${encodeURIComponent(time)}&`}` +
    `w=${width}&h=${height}&dpr=1` +
    `${showStats ? '&stats' : ''}&tex=${encodeURIComponent(textureQuality)}` +
    `&dxt5=${encodeURIComponent(dxt5Mode)}` +
    `${diagnostics ? '&diag' : ''}`;
  const url = `${origin}${smokePath}${productionPage ? productionQuery : ''}`;
  const argumentsList = [
    '--headless=new',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-extensions',
    '--disable-sync',
    ...(audioPage || liveProduction ? ['--autoplay-policy=no-user-gesture-required'] : []),
    '--disable-gpu-sandbox',
    '--disable-dev-shm-usage',
    '--enable-webgl',
    '--enable-precise-memory-info',
    ...(angleBackend === 'swiftshader' ? ['--enable-unsafe-swiftshader'] : []),
    '--ignore-gpu-blocklist',
    ...(angleBackend === 'default' ? [] : [`--use-angle=${angleBackend}`]),
    '--run-all-compositor-stages-before-draw',
    '--remote-debugging-address=127.0.0.1',
    '--remote-debugging-port=0',
    '--remote-allow-origins=*',
    `--js-flags=--expose-gc --max-old-space-size=${v8HeapLimitMB}`,
    `--window-size=${width},${height}`,
    `--user-data-dir=${userData}`,
    url,
  ];

  child = spawn(chrome, argumentsList, {
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');

  const requestStop = reason => {
    if (!stopReason) stopReason = reason;
    // Once the hard memory ceiling is crossed, do not grant the disposable
    // browser a graceful-allocation window: a renderer can allocate much
    // faster than the RSS polling cadence. terminateChromeTree still sends
    // SIGTERM first, but a zero deadline escalates to SIGKILL immediately.
    const grace = reason === 'rss-limit' ? 0 : terminationGraceMilliseconds;
    if (!killPromise) killPromise = terminateChromeTree(child, lastTreePids, grace);
  };
  signalHandler = signal => {
    interruptedSignal ||= signal;
    requestStop(`signal-${signal.toLowerCase()}`);
  };
  process.once('SIGINT', signalHandler);
  process.once('SIGTERM', signalHandler);

  child.stdout.on('data', chunk => stdoutCapture.append(chunk));
  let resolveDebugEndpoint;
  const debugEndpointPromise = new Promise(resolveEndpoint => { resolveDebugEndpoint = resolveEndpoint; });
  child.stderr.on('data', chunk => {
    stderrCapture.append(chunk);
    debugEndpointTail = (debugEndpointTail + chunk).slice(-2048);
    const match = /DevTools listening on (ws:\/\/[^\s]+)/.exec(debugEndpointTail);
    if (match && resolveDebugEndpoint) {
      const resolveEndpoint = resolveDebugEndpoint;
      resolveDebugEndpoint = null;
      resolveEndpoint(match[1]);
    }
  });

  const exitOutcomePromise = new Promise(resolveExit => {
    child.once('error', error => resolveExit({ code: null, signal: null, error }));
    child.once('exit', (code, signal) => resolveExit({ code, signal, error: null }));
  });

  const sampleMemory = async () => {
    if (sampling || !child?.pid) return;
    sampling = true;
    try {
      const sample = await chromeTreeMemory(child.pid);
      lastTreePids = sample.pids;
      lastRSSBytes = sample.rssBytes;
      peakRSSBytes = Math.max(peakRSSBytes, sample.rssBytes);
      peakProcessCount = Math.max(peakProcessCount, sample.pids.length);
      rssSamples++;
      if (sample.rssBytes >= rssLimitBytes && !memoryExceeded) {
        memoryExceeded = true;
        requestStop('rss-limit');
      }
    } catch (error) {
      monitorError ||= error;
      requestStop('rss-monitor-error');
    } finally {
      sampling = false;
    }
  };
  await sampleMemory();
  monitorTimer = setInterval(sampleMemory, rssSampleMilliseconds);
  timeoutTimer = setTimeout(() => { timedOut = true; requestStop('timeout'); }, timeoutMilliseconds);

  const exitedBeforeCDP = exitOutcomePromise.then(outcome => {
    const detail = outcome.error?.message || outcome.signal || outcome.code;
    throw new Error(`Chrome exited before CDP capture completed (${detail ?? 'unknown reason'})`);
  });
  let client = null;
  let screenshotCaptured = false;
  const capturePage = async () => {
    if (!client) return;
    if (!capturedDom) {
      capturedDom = await client.evaluate(`(() => {
        const type = document.doctype ? '<!doctype ' + document.doctype.name + '>\\n' : '';
        return type + document.documentElement.outerHTML;
      })()`);
      domCompleted = typeof capturedDom === 'string' && capturedDom.includes('</html>');
    }
    if (!screenshotCaptured) {
      const capture = await client.send('Page.captureScreenshot', {
        format: 'png', fromSurface: true, captureBeyondViewport: false,
      });
      if (!capture.data) throw new Error('Page.captureScreenshot returned no image data');
      await writeFile(screenshot, Buffer.from(capture.data, 'base64'));
      screenshotCaptured = true;
    }
  };

  try {
    const browserWebSocketURL = await Promise.race([debugEndpointPromise, exitedBeforeCDP]);
    const pageWebSocketURL = await Promise.race([
      discoverPageWebSocket(browserWebSocketURL, smokePath, cdpTimeoutMilliseconds),
      exitedBeforeCDP,
    ]);
    client = await CDPClient.connect(pageWebSocketURL, cdpTimeoutMilliseconds);
    await client.send('Runtime.enable');
    await client.send('Page.enable');
    // Headless Chrome imposes a platform window minimum (500x232 on macOS),
    // even when --window-size requests a smaller validation canvas. Override
    // the CSS viewport through CDP so screenshots preserve the exact authored
    // aspect ratio instead of stretching a correctly sized drawing buffer.
    await client.send('Emulation.setDeviceMetricsOverride', {
      width, height, deviceScaleFactor: 1, mobile: false,
      screenWidth: width, screenHeight: height,
    });
    await client.send('Performance.enable');
    if (liveProduction) {
      await Promise.race([startLiveProduction(client), exitedBeforeCDP]);
    }
    await Promise.race([
      waitForTerminalPage(client, productionPage, audioPage,
        progress => { lastPageProgress = progress; }),
      exitedBeforeCDP,
    ]);

    const terminalState = await client.evaluate(`(() => {
      const data = document.documentElement?.dataset || {};
      return ${productionPage ? 'data.debrisState' : audioPage ? 'data.audioSmoke' : 'data.webglSmoke'} || '';
    })()`);
    const rendered = productionPage ? terminalState === 'rendered' : terminalState === 'ok';
    if (liveProduction) audioLifecycle = await readLiveAudioLifecycle(client);
    if (rendered) {
      if (productionPage && forceAllCasters) {
        await client.evaluate(`(() => {
          const app = globalThis.__debris;
          const renderer = app?.renderer;
          if (!app?.ready || !renderer?.geometry?.allEntries ||
              typeof app.redraw !== 'function') {
            throw new Error('Debris renderer is not available for caster isolation');
          }
          for (const entry of renderer.geometry.allEntries) entry.bounds = null;
          app.redraw();
          return true;
        })()`);
      }
      if (productionPage && disablePostEffects) {
        await client.evaluate(`(() => {
          const app = globalThis.__debris;
          const renderer = app?.renderer;
          if (!app?.ready || !renderer || typeof app.redraw !== 'function' ||
              typeof renderer.drawFullscreen !== 'function') {
            throw new Error('Debris renderer is not available for post-effect isolation');
          }
          renderer.applyPost = function(image, _job, options = {}) {
            this.configureFullscreenState(null);
            this.drawFullscreen(image, 0, { uvRect: options.uvRect });
          };
          app.redraw();
          return true;
        })()`);
      }
      if (productionPage && profileTimes.length) {
        profileMetrics = await client.evaluate(`(async () => {
          const app = globalThis.__debris;
          if (!app?.ready || typeof app.renderSample !== 'function' ||
              typeof app.seekRuntime !== 'function' || typeof app.addSnapshot !== 'function' ||
              typeof app.runtime?.snapshot !== 'function') {
            throw new Error('Debris app is not available for transition profiling');
          }
          const requestedTimes = ${JSON.stringify(profileTimes)};
          const repetitions = ${profileFrames};
          const sampleRate = Number(app.sampleRate);
          const duration = Number(app.duration);
          if (!(sampleRate > 0) || !(duration >= 0)) {
            throw new Error('Debris app reported invalid timing metadata');
          }
          const durationSamples = Math.max(0, Math.floor(duration * sampleRate));
          const baselineSample = Math.floor(${JSON.stringify(time)} * sampleRate);
          let previousRequestedSample = -1;
          for (const requestedSeconds of requestedTimes) {
            if (requestedSeconds > duration + (1 / sampleRate)) {
              throw new Error(
                'profile time ' + requestedSeconds + ' exceeds demo duration ' + duration.toFixed(3),
              );
            }
            const requestedSample = Math.min(durationSamples,
              Math.max(0, Math.floor(requestedSeconds * sampleRate)));
            if (requestedSample === baselineSample) {
              throw new Error('profile time collides with the already-rendered baseline sample');
            }
            if (requestedSample <= previousRequestedSample) {
              throw new Error('profile times must resolve to strictly increasing sample positions');
            }
            previousRequestedSample = requestedSample;
          }
          const rounded = value => Number(Number(value || 0).toFixed(6));
          const timingSummary = (entries, key) => {
            const values = entries.map(entry => Number(entry[key]) || 0);
            if (!values.length) return null;
            return {
              mean: rounded(values.reduce((sum, value) => sum + value, 0) / values.length),
              minimum: rounded(Math.min(...values)),
              maximum: rounded(Math.max(...values)),
            };
          };
          const countSummary = (entries, key) => {
            const values = entries.map(entry => Math.max(0, Math.floor(Number(entry[key]) || 0)));
            if (!values.length) return null;
            return { minimum: Math.min(...values), maximum: Math.max(...values) };
          };
          const resourceSnapshot = () => {
            const resources = app.renderer?.resourceStats?.() || {};
            const geometry = resources.geometry || {};
            const textures = resources.textures || {};
            return {
              geometryEntries: Math.floor(Number(geometry.entries) || 0),
              geometryGPUBytes: Math.floor(Number(geometry.gpuBytes) || 0),
              shadowTopologies: Math.floor(Number(geometry.shadowTopologies) || 0),
              shadowTopologyBytes: Math.floor(Number(geometry.shadowBytes) || 0),
              textureEntries: Math.floor(Number(textures.bitmapTextures ?? textures.textures) || 0),
              textureGPUBytes: Math.floor(Number(textures.estimatedBytes) || 0),
              renderTargets: Math.floor(Number(resources.renderTargets) || 0),
              renderTargetBytes: Math.floor(Number(resources.renderTargetBytes) || 0),
            };
          };
          const resourceDelta = (before, after) => Object.fromEntries(
            Object.keys(before).map(key => [key, after[key] - before[key]]),
          );
          const compactFrame = frame => ({
            runtimeMilliseconds: rounded(frame.runtimeMilliseconds),
            renderMilliseconds: rounded(frame.renderMilliseconds),
            frameMilliseconds: rounded(frame.frameMilliseconds),
            gpuDrainMilliseconds: rounded(frame.gpuDrainMilliseconds),
            totalMilliseconds: rounded(frame.totalMilliseconds),
            drawCalls: frame.drawCalls,
            triangles: frame.triangles,
            resourceDelta: frame.resourceDelta,
          });
          const snapshotsBefore = {
            count: app.snapshots?.length || 0,
            bytes: Math.floor(Number(app.snapshotBytes) || 0),
          };
          let profileSnapshotsAdded = 0;
          const profiles = [];
          const captures = [];
          for (const requestedSeconds of requestedTimes) {
            if (requestedSeconds > duration + (1 / sampleRate)) {
              throw new Error(
                'profile time ' + requestedSeconds + ' exceeds demo duration ' + duration.toFixed(3),
              );
            }
            const sample = Math.min(durationSamples,
              Math.max(0, Math.floor(requestedSeconds * sampleRate)));
            const resets = [];
            const frames = [];
            let snapshotMilliseconds = 0;
            for (let repetition = 0; repetition < repetitions; repetition++) {
              const resetStarted = performance.now();
              await app.seekRuntime(sample, { yield: false });
              resets.push(performance.now() - resetStarted);
              // Start each observation with an empty GPU queue. The finish
              // after render then measures the work submitted by this frame,
              // rather than inheriting asynchronous work from the prior one.
              app.renderer?.gl?.finish?.();
              const resourcesBefore = resourceSnapshot();
              const frameStarted = performance.now();
              app.renderSample(sample);
              const submissionEnded = performance.now();
              app.renderer?.gl?.finish?.();
              const frameEnded = performance.now();
              const resourcesAfter = resourceSnapshot();
              const frame = {
                runtimeMilliseconds: app.stats.runtimeMilliseconds,
                renderMilliseconds: app.stats.renderMilliseconds,
                frameMilliseconds: app.stats.frameMilliseconds,
                gpuDrainMilliseconds: frameEnded - submissionEnded,
                totalMilliseconds: frameEnded - frameStarted,
                drawCalls: Math.max(0, Math.floor(Number(app.stats.drawCalls) || 0)),
                triangles: Math.max(0, Math.floor(Number(app.stats.triangles) || 0)),
                resourceDelta: resourceDelta(resourcesBefore, resourcesAfter),
              };
              for (const key of ['runtimeMilliseconds', 'renderMilliseconds',
                'frameMilliseconds', 'gpuDrainMilliseconds', 'totalMilliseconds']) {
                if (!Number.isFinite(frame[key]) || frame[key] < 0) {
                  throw new Error('invalid profile timing for ' + key);
                }
              }
              frames.push(frame);
              if (repetition === 0) {
                if (${captureProfileScreenshots}) {
                  const canvas = document.querySelector('canvas');
                  const dataURL = canvas?.toDataURL?.('image/png');
                  if (typeof dataURL !== 'string' ||
                      !dataURL.startsWith('data:image/png;base64,')) {
                    throw new Error('profile frame could not be captured as PNG');
                  }
                  captures.push({ requestedSeconds, sample, dataURL });
                }
                const snapshotStarted = performance.now();
                // addSnapshot owns the count and byte-budget eviction policy.
                // Retaining the newly reached state makes every later sorted
                // time incremental and gives same-sample warm repetitions an
                // exact restore point without accumulating unbounded states.
                app.addSnapshot(sample / sampleRate, sample, app.runtime.snapshot());
                snapshotMilliseconds = performance.now() - snapshotStarted;
                profileSnapshotsAdded++;
              }
              await new Promise(resolve => setTimeout(resolve, 0));
            }
            const cold = frames[0];
            const warmFrames = frames.slice(1);
            if (warmFrames.some(frame => frame.drawCalls !== cold.drawCalls ||
              frame.triangles !== cold.triangles)) {
              throw new Error('profile repetitions produced divergent draw or triangle counts');
            }
            profiles.push({
              requestedSeconds,
              actualSeconds: rounded(sample / sampleRate),
              sample,
              baselineAlreadyRendered:
                sample === Math.floor(${JSON.stringify(time)} * sampleRate),
              snapshotMilliseconds: rounded(snapshotMilliseconds),
              retainedSnapshots: {
                count: app.snapshots?.length || 0,
                bytes: Math.floor(Number(app.snapshotBytes) || 0),
              },
              resetMilliseconds: {
                cold: rounded(resets[0]),
                warm: timingSummary(resets.slice(1).map(value => ({ value })), 'value'),
              },
              cold: compactFrame(cold),
              warm: {
                repetitions: warmFrames.length,
                runtimeMilliseconds: timingSummary(warmFrames, 'runtimeMilliseconds'),
                renderMilliseconds: timingSummary(warmFrames, 'renderMilliseconds'),
                frameMilliseconds: timingSummary(warmFrames, 'frameMilliseconds'),
                gpuDrainMilliseconds: timingSummary(warmFrames, 'gpuDrainMilliseconds'),
                totalMilliseconds: timingSummary(warmFrames, 'totalMilliseconds'),
                drawCalls: countSummary(warmFrames, 'drawCalls'),
                triangles: countSummary(warmFrames, 'triangles'),
                resourceDeltas: warmFrames.map(frame => frame.resourceDelta),
              },
            });
          }
          return {
            requestedTimes, repetitions, sampleRate, duration,
            baselineSeconds: ${JSON.stringify(time)},
            baselineMatchesRequestedSample: requestedTimes.some(seconds =>
              Math.floor(seconds * sampleRate) === Math.floor(${JSON.stringify(time)} * sampleRate)),
            timingIncludesGLFinish: true,
            runtimeResetBeforeEachRepetition: true,
            strictlyIncreasingSamplesRequired: true,
            incrementalProfileSnapshots: true,
            profileSnapshotsAdded,
            snapshotLimit: app.snapshotLimit,
            snapshotBudgetBytes: app.snapshotBudgetBytes,
            snapshotsBefore,
            snapshotsAfter: {
              count: app.snapshots?.length || 0,
              bytes: Math.floor(Number(app.snapshotBytes) || 0),
            },
            profiles, captures,
          };
        })()`, Math.max(cdpTimeoutMilliseconds, timeoutMilliseconds));
        if (profileMetrics?.captures?.length) {
          profileScreenshotArtifacts = [];
          for (const capture of profileMetrics.captures) {
            const prefix = 'data:image/png;base64,';
            if (typeof capture?.dataURL !== 'string' || !capture.dataURL.startsWith(prefix)) {
              throw new Error('profile screenshot returned malformed PNG data');
            }
            const path = expectedProfileScreenshots[profileScreenshotArtifacts.length];
            if (!path) throw new Error('profile screenshot exceeded the bounded artifact list');
            await writeFile(path, Buffer.from(capture.dataURL.slice(prefix.length), 'base64'));
            profileScreenshotArtifacts.push({
              requestedSeconds: capture.requestedSeconds,
              sample: capture.sample,
              path,
            });
          }
          profileMetrics.captures = profileScreenshotArtifacts;
        }
      }
      if (productionPage && inspectOperationIds.length) {
        operationInspection = await client.evaluate(`(() => {
          const ids = ${JSON.stringify(inspectOperationIds)};
          const app = globalThis.__debris;
          const operations = app?.runtime?.operations || [];
          const requested = new Set(ids);
          const materialCaches = new Map();
          const summarizeComponents = (values, width) => {
            if (!values?.length || !(width > 0)) return null;
            const minimum = new Array(width).fill(Infinity);
            const maximum = new Array(width).fill(-Infinity);
            let invalid = 0;
            for (let offset = 0; offset + width <= values.length; offset += width) {
              for (let component = 0; component < width; component++) {
                const value = Number(values[offset + component]);
                if (!Number.isFinite(value)) { invalid++; continue; }
                minimum[component] = Math.min(minimum[component], value);
                maximum[component] = Math.max(maximum[component], value);
              }
            }
            return {
              count: Math.floor(values.length / width), invalid,
              minimum: minimum.map(value => Number.isFinite(value) ? value : null),
              maximum: maximum.map(value => Number.isFinite(value) ? value : null),
            };
          };
          const summarizeBitmap = bitmap => {
            const data = bitmap?.data;
            if (!data?.length) return null;
            let coloredPixels = 0, alphaPixels = 0;
            let minimum = 0xffff, maximum = 0;
            for (let offset = 0; offset + 3 < data.length; offset += 4) {
              const red = data[offset], green = data[offset + 1], blue = data[offset + 2];
              if (red || green || blue) coloredPixels++;
              if (data[offset + 3]) alphaPixels++;
              minimum = Math.min(minimum, red, green, blue, data[offset + 3]);
              maximum = Math.max(maximum, red, green, blue, data[offset + 3]);
            }
            return { width: bitmap.width, height: bitmap.height,
              words: data.length, coloredPixels, alphaPixels, minimum, maximum };
          };
          const ops = ids.map(id => {
            const operation = operations[id];
            const cache = operation?.cache;
            if (cache?.kind === 'material') materialCaches.set(cache, id);
            return {
              id, classId: operation?.classId ?? null,
              cacheKind: cache?.kind || cache?.type || cache?.constructor?.name || null,
              bitmap: summarizeBitmap(cache),
              material: cache?.kind === 'material' ? {
                system: cache.system,
                passes: cache.passes?.map(pass => ({ usage: pass.usage,
                  renderPass: pass.renderPass, state: pass.state })) || [],
                textureBitmaps: cache.textures?.map(summarizeBitmap) || [],
              } : null,
            };
          });
          const jobs = [];
          const seenNodes = new Set();
          const visit = node => {
            if (!node || typeof node !== 'object' || seenNodes.has(node)) return;
            seenNodes.add(node);
            for (const job of node.meshJobs || []) {
              const slots = job.mesh?.materials || job.mesh?.Mtrl || job.mesh?._prepared?.materials || [];
              for (const slot of slots) {
                const material = slot?.material ?? slot?.Material ?? slot;
                const materialOpId = materialCaches.get(material);
                if (materialOpId !== undefined || requested.has(job.opId)) {
                  const prepared = job.mesh?._prepared || null;
                  jobs.push({ jobOpId: job.opId ?? null, materialOpId: materialOpId ?? null,
                    slotPass: slot?.pass ?? slot?.Pass ?? null,
                    meshKind: job.mesh?.kind || job.mesh?.constructor?.name || null,
                    matrix: job.matrix?.length >= 16 ? Array.from(job.matrix) : null,
                    geometry: prepared ? {
                      positions: summarizeComponents(prepared.positions, 3),
                      normals: summarizeComponents(prepared.normals, 3),
                      uvs: summarizeComponents(prepared.uv0 || prepared.uvs, 2),
                      indices: prepared.indices?.length || 0,
                      bounds: prepared.bounds ? {
                        minimum: Array.from(prepared.bounds.minimum || prepared.bounds.min || []),
                        maximum: Array.from(prepared.bounds.maximum || prepared.bounds.max || []),
                      } : null,
                      groups: (prepared.groups || []).map(group => ({
                        materialOpId: materialCaches.get(group.material) ?? null,
                        materialIndex: group.materialIndex ?? null,
                        pass: group.pass ?? group.renderPass ?? null,
                        start: group.start ?? null, count: group.count ?? null,
                      })),
                    } : null });
                }
              }
            }
            visit(node.input); visit(node.output);
            for (const child of node.outputs || []) visit(child);
          };
          for (const output of app?.runtime?.environment?.frameOutputs || []) visit(output);
          visit(app?.runtime?.environment?.lastOutput);
          return { ops, jobs };
        })()`);
      }
      if (productionPage && warmFrames > 0) {
        warmFrameMetrics = await client.evaluate(`(async () => {
          const app = globalThis.__debris;
          if (!app?.ready || typeof app.renderSample !== 'function') {
            throw new Error('Debris app is not available for warm-frame measurement');
          }
          const count = ${warmFrames};
          const rate = ${warmFrameRate};
          const stepSamples = Math.max(1, Math.round(app.sampleRate / rate));
          const baseSample = app.currentSample;
          const heapUsedBefore = performance.memory?.usedJSHeapSize || 0;
          const resourceBytesBefore = app.resourceStats?.()?.totalEstimatedBytes || 0;
          const samples = [];
          for (let index = 0; index < count; index++) {
            const sample = baseSample + stepSamples * (index + 1);
            const synchronizedStart = performance.now();
            app.renderSample(sample);
            const submissionEnd = performance.now();
            app.renderer?.gl?.finish?.();
            const synchronizedEnd = performance.now();
            samples.push({
              sample,
              frameMilliseconds: app.stats.frameMilliseconds,
              runtimeMilliseconds: app.stats.runtimeMilliseconds,
              renderMilliseconds: app.stats.renderMilliseconds,
              synchronizedMilliseconds: synchronizedEnd - synchronizedStart,
              gpuDrainMilliseconds: synchronizedEnd - submissionEnd,
              drawCalls: app.stats.drawCalls,
              triangles: app.stats.triangles,
            });
            await new Promise(resolve => setTimeout(resolve, 0));
          }
          const resources = app.resourceStats?.() || null;
          if (resources) {
            app.stats.resources = resources;
            app.options?.onFrame?.(app.stats, app);
          }
          const heapUsedAfter = performance.memory?.usedJSHeapSize || 0;
          const ordered = samples.map(entry => entry.synchronizedMilliseconds).sort((a, b) => a - b);
          const percentile = value => ordered.length
            ? ordered[Math.min(ordered.length - 1, Math.floor((ordered.length - 1) * value))] : 0;
          return {
            count, rate, stepSamples, timingIncludesGLFinish: true,
            heapUsedBefore, heapUsedAfter,
            resourceBytesBefore,
            resourceBytesAfter: resources?.totalEstimatedBytes || 0,
            samples,
            p50Milliseconds: percentile(0.50),
            p95Milliseconds: percentile(0.95),
            maximumMilliseconds: ordered[ordered.length - 1] || 0,
          };
        })()`);
      }
      pageMemoryMetrics = await client.evaluate(`(async () => {
        let didGC = false;
        if (${gcAfterRender}) {
          if (typeof globalThis.gc === 'function') { globalThis.gc(); didGC = true; }
          await new Promise(resolve => setTimeout(resolve, ${gcWaitMilliseconds}));
        }
        const heap = performance.memory ? {
          jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
          totalJSHeapSize: performance.memory.totalJSHeapSize,
          usedJSHeapSize: performance.memory.usedJSHeapSize,
        } : null;
        let userAgentSpecificMemory = null;
        if (typeof performance.measureUserAgentSpecificMemory === 'function') {
          try {
            userAgentSpecificMemory = await Promise.race([
              performance.measureUserAgentSpecificMemory(),
              new Promise((_, reject) => setTimeout(() => reject(new Error('measurement timed out')), 5000)),
            ]);
          }
          catch (error) { userAgentSpecificMemory = { error: String(error?.message || error) }; }
        }
        return { didGC, crossOriginIsolated: globalThis.crossOriginIsolated, heap, userAgentSpecificMemory };
      })()`);
      gcCalled = Boolean(pageMemoryMetrics?.didGC);
      cdpHeapUsage = await client.send('Runtime.getHeapUsage');
      const retainedHeapBytes = Math.round(
        pageMemoryMetrics?.heap?.usedJSHeapSize ?? cdpHeapUsage?.usedSize ?? 0,
      );
      if (retainedHeapBytes > 0) {
        await client.evaluate(`document.documentElement.dataset.debrisHeapBytes = ${JSON.stringify(String(retainedHeapBytes))}`);
      }
    }
    const cdpPerformance = await client.send('Performance.getMetrics');
    cdpPerformanceMetrics = Object.fromEntries(
      (cdpPerformance.metrics || []).map(metric => [metric.name, metric.value]),
    );
    await client.send('Page.bringToFront');
    await capturePage();
  } catch (error) {
    cdpError = error;
    if (client) await capturePage().catch(() => {});
  } finally {
    client?.close();
  }

  requestStop(cdpError ? 'cdp-error' : 'cdp-complete');
  const exitOutcome = await exitOutcomePromise;
  if (killPromise) await killPromise;
  clearInterval(monitorTimer); monitorTimer = null;
  clearTimeout(timeoutTimer); timeoutTimer = null;

  const dom = capturedDom || stdoutCapture.value();
  await writeFile(domPath, dom);
  let screenshotExists = false;
  try { await access(screenshot, constants.R_OK); screenshotExists = true; } catch (_) { /* reported below */ }

  const datasets = pageDatasets(dom);
  const domMetrics = {};
  for (const [name, value] of Object.entries(datasets)) {
    if (name.startsWith('debris-')) domMetrics[name.slice('debris-'.length)] = value;
  }
  const state = String(domMetrics.state ?? lastPageProgress?.state ?? (productionPage
    ? '(missing)' : audioPage ? datasets['audio-smoke'] : datasets['webgl-smoke']) ?? '(missing)');
  const status = String(domMetrics.status ?? lastPageProgress?.status ?? (audioPage
    ? datasets['audio-errors'] : datasets['webgl-error']) ?? '');
  const draws = Number(domMetrics.draws ?? datasets['webgl-draw-calls'] ?? 0);
  const triangles = Number(domMetrics.triangles ?? datasets['webgl-triangles'] ?? 0);
  const elapsedMilliseconds = performance.now() - started;
  const pagePassed = productionPage
    ? state === 'rendered' && draws > 0 && triangles > 0 &&
      (!liveProduction || audioLifecycle?.valid === true)
    : audioPage ? datasets['audio-smoke'] === 'ok' : datasets['webgl-smoke'] === 'ok';
  result = {
    ok: !timedOut && !memoryExceeded && !monitorError && !cdpError && !interruptedSignal &&
      domCompleted && screenshotExists && pagePassed,
    timedOut, memoryExceeded, domCompleted, screenshotExists,
    exitCode: exitOutcome.code,
    exitSignal: exitOutcome.signal,
    interruptedSignal,
    stopReason,
    state, status, draws, triangles,
    path: smokePath,
    mode: productionPage ? (liveProduction ? 'live' : profileTimes.length
      ? 'profile' : 'fixed-time') : null,
    time: productionPage && !liveProduction && !profileTimes.length ? time : null,
    profileTimes: profileTimes.length ? profileTimes : null,
    profileFrames: profileTimes.length ? profileFrames : null,
    width, height,
    angleBackend,
    textureQuality: productionPage ? textureQuality : null,
    dxt5Mode: productionPage ? dxt5Mode : null,
    diagnostics: productionPage ? diagnostics : null,
    postEffects: productionPage ? !disablePostEffects : null,
    forceAllCasters: productionPage ? forceAllCasters : null,
    statsOverlay: productionPage ? showStats : null,
    screenshot, dom: domPath,
    elapsedMilliseconds: Number(elapsedMilliseconds.toFixed(3)),
    timeoutMilliseconds,
    performance: {
      ...pickMetrics(domMetrics, [
        'frame-ms', 'runtime-ms', 'render-ms', 'precalc-ms', 'seek-ms',
      ]),
      cdp: cdpPerformanceMetrics,
      warmFrames: warmFrameMetrics,
      profile: profileMetrics,
      operationInspection,
    },
    memory: {
      ...pickMetrics(domMetrics, [
        'heap-bytes', 'heap-total-bytes', 'snapshot-bytes', 'resource-bytes',
        'runtime-cache-bytes', 'bitmap-cache-bytes', 'mesh-cache-bytes',
        'prepared-mesh-cache-bytes',
        'cache-identities', 'pruned-cache-identities', 'pruned-cache-references',
        'pruned-cache-bytes', 'retained-cache-bytes',
        'playback-geometry-identities', 'playback-geometry-marked-identities',
        'playback-geometry-queued-identities', 'playback-geometry-pending-identities',
        'playback-geometry-released-identities', 'playback-geometry-alias-references',
        'playback-geometry-shared-identities', 'released-topology-bytes',
        'reclaimable-topology-bytes', 'pending-topology-bytes',
        'retained-shared-topology-bytes', 'prepared-geometry-bytes',
        'deferred-prepared-geometry-bytes', 'geometry-conversion-net-bytes',
        'playback-geometry-net-bytes',
        'uncaptured-geometry-identities', 'animated-geometry-identities',
        'geometry-cache-entries', 'animated-geometry-cache-entries',
        'geometry-gpu-bytes', 'geometry-cpu-bytes', 'shadow-topology-bytes',
        'texture-cache-entries', 'texture-gpu-bytes',
      ]),
      physicalMB: Number((totalmem() / MEBIBYTE).toFixed(1)),
      rssLimitMB: Number((rssLimitBytes / MEBIBYTE).toFixed(1)),
      v8HeapLimitMB,
      peakChromeTreeRSSMB: Number((peakRSSBytes / MEBIBYTE).toFixed(1)),
      lastChromeTreeRSSMB: Number((lastRSSBytes / MEBIBYTE).toFixed(1)),
      peakChromeProcessCount: peakProcessCount,
      rssSamples,
      sampleMilliseconds: rssSampleMilliseconds,
      aggregateRSSIsConservative: true,
      gcAfterRender,
      gcCalled,
      cdpHeapUsage,
      page: pageMemoryMetrics,
    },
    geometry: pickMetrics(domMetrics, [
      'degenerate-triangles', 'duplicate-triangles',
      'opposite-duplicate-triangles', 'same-orientation-duplicate-triangles',
      'exact-duplicate-triangles', 'exact-opposite-duplicate-triangles',
      'exact-same-orientation-duplicate-triangles', 'near-only-duplicate-triangles',
      'near-only-same-orientation-duplicate-triangles',
      'exact-same-group-identical-attribute-triangles',
      'exact-same-group-attribute-variant-triangles',
      'exact-cross-group-same-material-triangles', 'exact-cross-material-triangles',
      'exact-degenerate-same-orientation-triangles', 'topology-triangles',
      'audited-triangles', 'unaudited-topology-triangles',
      'truncated-topology-entries', 'indeterminate-winding', 'topology-offender-count',
      'unexpected-winding', 'normal-aligned-winding', 'normal-opposed-winding',
      'topology-offenders',
      'shadow-boundary-edges', 'shadow-nonmanifold-edges',
      'shadow-winding-conflict-edges',
      'shadow-max-edge-incidence',
    ]),
    culling: pickMetrics(domMetrics, [
      'candidate-light-items', 'culled-light-items',
      'candidate-shadow-items', 'culled-shadow-sphere-items',
      'culled-shadow-frustum-items',
    ]),
    audio: audioPage ? pickMetrics(datasets, [
      'audio-backend', 'audio-module', 'audio-v2-patches', 'audio-v2-events',
      'audio-sample-rate', 'audio-frames', 'audio-nonzero-left',
      'audio-nonzero-right', 'audio-stereo-difference', 'audio-invalid',
      'audio-peak-left', 'audio-peak-right', 'audio-rms-left', 'audio-rms-right',
      'audio-context-delta', 'audio-progress-sample', 'audio-progress-time',
      'audio-paused-sample', 'audio-paused-after', 'audio-resumed-sample',
      'audio-worker-smoke', 'audio-worker-bytes', 'audio-worker-backend',
      'audio-worker-primed', 'audio-worker-blocked-milliseconds',
      'audio-worker-context-delta', 'audio-worker-sample-before',
      'audio-worker-sample-after', 'audio-worker-output-frames-before',
      'audio-worker-output-frames-after', 'audio-worker-output-frame-delta',
      'audio-worker-input-frame-delta', 'audio-worker-nonzero-frames',
      'audio-worker-peak', 'audio-worker-underruns', 'audio-worker-closed',
    ]) : null,
    audioLifecycle: liveProduction ? audioLifecycle : null,
    domMetrics,
    pageDatasets: datasets,
    lastPageProgress,
    output: {
      stdoutCharacters: stdoutCapture.totalCharacters,
      stdoutTruncated: stdoutCapture.truncated,
      stderrCharacters: stderrCapture.totalCharacters,
      stderrTruncated: stderrCapture.truncated,
    },
    monitorError: monitorError ? String(monitorError?.stack || monitorError) : null,
    cdpError: cdpError ? String(cdpError?.stack || cdpError) : null,
    launchError: exitOutcome.error ? String(exitOutcome.error?.stack || exitOutcome.error) : null,
    stderr: stderrCapture.value().trim().split('\n')
      .filter(line => /error|fatal|warning/i.test(line)).slice(-20),
  };
} catch (error) {
  runnerError = error;
} finally {
  if (monitorTimer) clearInterval(monitorTimer);
  if (timeoutTimer) clearTimeout(timeoutTimer);
  if (signalHandler) {
    process.removeListener('SIGINT', signalHandler);
    process.removeListener('SIGTERM', signalHandler);
  }
  try {
    if (killPromise) await killPromise;
    else if (child?.pid) await terminateChromeTree(child, lastTreePids);
    const survivors = await existingSmokeProcesses();
    if (survivors.length) {
      throw new Error(`Debris Chrome cleanup left process(es): ${survivors.join(', ')}`);
    }
  } catch (error) {
    cleanupError = error;
  }
  if (listening) {
    server.closeAllConnections?.();
    await new Promise(resolveClose => server.close(resolveClose)).catch(() => {});
  }
  if (userData) await rm(userData, { recursive: true, force: true }).catch(() => {});
  if (lockHeld && !cleanupError) {
    try { await rm(lockDirectory, { recursive: true, force: true }); }
    catch (error) { cleanupError = error; }
  }
}

if (cleanupError) {
  if (result) {
    result.ok = false;
    result.cleanupError = String(cleanupError?.stack || cleanupError);
  } else {
    runnerError ||= cleanupError;
  }
}

if (runnerError) {
  result = {
    ok: false,
    error: String(runnerError?.stack || runnerError),
    stopReason: stopReason || 'runner-error',
    timedOut, memoryExceeded, interruptedSignal,
    path: smokePath,
    mode: productionPage ? (liveProduction ? 'live' : profileTimes.length
      ? 'profile' : 'fixed-time') : null,
    time: productionPage && !liveProduction && !profileTimes.length ? time : null,
    profileTimes: profileTimes.length ? profileTimes : null,
    profileFrames: profileTimes.length ? profileFrames : null,
    width, height,
    angleBackend,
    textureQuality: productionPage ? textureQuality : null,
    dxt5Mode: productionPage ? dxt5Mode : null,
    diagnostics: productionPage ? diagnostics : null,
    statsOverlay: productionPage ? showStats : null,
    audioLifecycle: liveProduction ? audioLifecycle : null,
    elapsedMilliseconds: Number((performance.now() - started).toFixed(3)),
    cleanupError: cleanupError ? String(cleanupError?.stack || cleanupError) : null,
    memory: {
      physicalMB: Number((totalmem() / MEBIBYTE).toFixed(1)),
      rssLimitMB: Number((rssLimitBytes / MEBIBYTE).toFixed(1)),
      v8HeapLimitMB,
      peakChromeTreeRSSMB: Number((peakRSSBytes / MEBIBYTE).toFixed(1)),
      peakChromeProcessCount: peakProcessCount,
      rssSamples,
    },
  };
}

if (resultArtifactPath) {
  await writeFile(resultArtifactPath, `${JSON.stringify(result, null, 2)}\n`);
}
console.log(JSON.stringify(result, null, 2));
if (!result.ok) process.exitCode = 1;
