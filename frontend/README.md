# Frontend

This frontend now uses Vite (not `react-scripts`) to avoid deprecated CRA transitive dependencies.

## Scripts

- `npm run dev` (or `npm start`): start local dev server on port `3000`
- `npm run build`: production build to `dist/`
- `npm run preview`: preview production build on port `4173`

## Environment variables

Use Vite-style vars:

- `VITE_API_URL`
- `VITE_API_KEY`
- `VITE_USE_V2_API`
- `VITE_BUILD_ID`

Legacy `REACT_APP_*` variables are still read for backward compatibility.
