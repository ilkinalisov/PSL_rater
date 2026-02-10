import axios from 'axios';
import { API_KEY, API_URL, CLIENT_BUILD, USE_V2_API } from './config';

function buildHeaders(extraHeaders = {}) {
  const headers = { ...extraHeaders };
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }
  headers['X-Client-Mode'] = USE_V2_API ? 'v2' : 'v1';
  headers['X-Client-Build'] = CLIENT_BUILD;
  return headers;
}

function normalizeApiError(error) {
  const status = error?.response?.status;
  const detail = error?.response?.data?.detail;
  const apiError = error?.response?.data?.error;

  if (status === 429) {
    return new Error('Too many requests. Please wait a moment and try again.');
  }
  if (status === 413) {
    return new Error('Image file is too large. Maximum 10MB per image.');
  }
  if (status === 403) {
    return new Error('Access denied. API key is missing or invalid.');
  }

  return new Error(detail || apiError || 'Analysis failed. Please try different photos.');
}

export async function analyzePair(frontFile, sideFile) {
  if (!API_URL) {
    throw new Error('API URL is not configured.');
  }

  const formData = new FormData();
  formData.append('front_image', frontFile);
  formData.append('side_image', sideFile);
  const endpoint = USE_V2_API ? '/v2/analyze/pair' : '/analyze/pair';

  try {
    const response = await axios.post(`${API_URL}${endpoint}`, formData, {
      headers: buildHeaders({ 'Content-Type': 'multipart/form-data' }),
    });
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('debug') === '1') {
        const pipeline = response?.data?.debug?.pipeline || {};
        // PART D2: runtime trace to verify v1/v2 path and renderer version.
        console.info('[PSL Debug] analyzePair', {
          endpoint,
          methodSource: response?.data?.side_analysis?.method_source || null,
          endpointVariant: pipeline.endpoint_variant || null,
          overlayRendererVersion: pipeline.overlay_renderer_version || null,
          analysisId: response?.data?.analysis_id || null,
        });
      }
    }
    return response.data;
  } catch (error) {
    throw normalizeApiError(error);
  }
}
