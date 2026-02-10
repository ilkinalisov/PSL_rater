// PART C3: Centralized API URL + API key configuration for all requests.
const procEnv = (typeof process !== 'undefined' && process && process.env) ? process.env : {};
const viteEnv = (typeof import.meta !== 'undefined' && import.meta && import.meta.env) ? import.meta.env : {};

const pickEnv = (...keys) => {
  for (const key of keys) {
    const v = viteEnv[key];
    if (typeof v === 'string' && v.trim()) return v;
  }
  for (const key of keys) {
    const v = procEnv[key];
    if (typeof v === 'string' && v.trim()) return v;
  }
  return '';
};

const envApiUrl = pickEnv('VITE_API_URL', 'REACT_APP_API_URL');
const envApiKey = pickEnv('VITE_API_KEY', 'REACT_APP_API_KEY');
const envUseV2 = pickEnv('VITE_USE_V2_API', 'REACT_APP_USE_V2_API');
const envBuild = pickEnv(
  'VITE_BUILD_ID',
  'REACT_APP_BUILD_ID',
  'VITE_VERCEL_GIT_COMMIT_SHA',
  'REACT_APP_VERCEL_GIT_COMMIT_SHA',
);

const truthyValues = ['1', 'true', 'yes', 'on'];

let API_URL = '';
let isLocalHost = false;
if (envApiUrl && envApiUrl.trim()) {
  API_URL = envApiUrl.trim().replace(/\/$/, '');
} else if (typeof window !== 'undefined') {
  const host = window.location.hostname;
  isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0';
  API_URL = isLocalHost ? 'http://localhost:8000' : '';
}
if (typeof window !== 'undefined' && !isLocalHost) {
  const host = window.location.hostname;
  isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0';
}

export const API_KEY = envApiKey.trim();
const useV2Raw = String(envUseV2).trim().toLowerCase();
const hasExplicitUseV2 = useV2Raw.length > 0;
export const USE_V2_API = hasExplicitUseV2 ? truthyValues.includes(useV2Raw) : !isLocalHost;
export const CLIENT_BUILD = String(envBuild).trim() || (typeof window !== 'undefined' ? 'web-runtime' : 'local');
export { API_URL };
export default API_URL;
