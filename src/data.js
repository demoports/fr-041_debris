import { LOADER_V2M_GZIP_BASE64, PARTY_KX_GZIP_BASE64 } from './debris_data.js';

function decodeBase64(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function gunzipBase64(encoded) {
  if (typeof DecompressionStream !== 'function') {
    throw new Error('This browser does not support the gzip DecompressionStream API.');
  }
  const compressed = decodeBase64(encoded);
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

let cached;

function loadProductionData() {
  cached ??= Promise.all([
    gunzipBase64(PARTY_KX_GZIP_BASE64),
    gunzipBase64(LOADER_V2M_GZIP_BASE64),
  ]).then(([kx, loaderSong]) => ({ kx, loaderSong }));
  return cached;
}

export { decodeBase64, gunzipBase64, loadProductionData };
