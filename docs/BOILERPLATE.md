# Main connection template

The included application already implements this connection. These small snippets explain the architecture; do not replace the full files with them.

## React -> FastAPI

```tsx
async function generateScene(script: string) {
  const response = await fetch('/api/scene', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ script }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(result.detail));
  const scene = projectSchema.parse(result.scene);
  setProject(scene); // The viewport rebuilds from this data.
}
```

The real handler is in `frontend/src/main.tsx`. React stores scene data; Three.js constructs the scene. It does not execute arbitrary code from an LLM.

## FastAPI endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class SceneRequest(BaseModel):
    script: str = Field(min_length=10, max_length=8000)

@app.post('/api/scene')
async def create_scene(request: SceneRequest):
    raw_json = await invoke(request.script)  # Server-side model adapter
    scene = Scene.model_validate_json(raw_json)
    return {'scene': scene.model_dump()}
```

`invoke` and `Scene` are implemented in `backend/app/main.py` and `backend/app/models.py`. The real endpoint also handles repairs, warnings, error messages and raw run records.

## Frontend development proxy

```ts
import { defineConfig } from 'vite';
export default defineConfig({
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
```

For the single-server Windows setup, FastAPI serves the compiled React `dist` folder. For development, Vite serves React and proxies API requests to FastAPI.

## Where to change things

- Add an object type: update Pydantic, Zod, the provider instructions, and Three.js constructors together.
- Change the model: edit backend `.env`; restart the backend.
- Change appearance: `frontend/src/style.css`.
- Add a test scene: `backend/examples/*.json`.
- Edit model instructions: `SYSTEM` in `backend/app/main.py`.

Versions and exact schemas must agree between frontend and backend. The fixtures are tested in both.
