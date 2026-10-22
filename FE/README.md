# DAI-Legal-Model Frontend

React 19 and Vite frontend for the Core Gateway legal query API.

## Run locally

```powershell
cd D:\Prj\DAI-Legal-Model\FE
npm.cmd run dev
```

The Vite development server uses port `5173`. On localhost, the frontend calls the Gateway at `http://127.0.0.1:8002`.

Configure a different Gateway base URL with `VITE_GATEWAY_API_URL` in an FE `.env` file:

```dotenv
VITE_GATEWAY_API_URL=http://127.0.0.1:8002
```

The frontend sends `POST /api/v2/legal-query` with `{ "query": "...", "allow_external": false }`. The Gateway response is JSON. A response with HTTP 200 can still have `status: "FAILED"`; the UI preserves and displays that status and the answer supplied by the Gateway. Chat messages are held only in the current browser UI session because the Gateway contract does not expose chat history.

## Checks

```powershell
npm.cmd test
npm.cmd run build
```
