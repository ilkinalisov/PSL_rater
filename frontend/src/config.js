// PART C3: Centralized API URL + API key configuration for all requests.
const envApiUrl = process.env.REACT_APP_API_URL || process.env.VITE_API_URL;
const envApiKey = process.env.REACT_APP_API_KEY || process.env.VITE_API_KEY || '';
const envUseV2 = process.env.REACT_APP_USE_V2_API || process.env.VITE_USE_V2_API || '';
const envBuild =
  process.env.REACT_APP_BUILD_ID ||
  process.env.VITE_BUILD_ID ||
  process.env.REACT_APP_VERCEL_GIT_COMMIT_SHA ||
  process.env.VITE_VERCEL_GIT_COMMIT_SHA ||
  '';

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
