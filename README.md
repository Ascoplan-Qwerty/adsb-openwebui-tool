Real-Time ADS-B Telemetry & AI Spatial Analytics Tool for OpenWebUI

An edge-native situational awareness tool designed for **OpenWebUI** and **Ollama**. It ingests live ADS-B flight telemetry from the OpenSky Network API, performs in-memory spatial indexing, executes deterministic emergency squawk triage (7500, 7600, 7700), and dynamically routes queries across local LLMs (`qwen2.5`, `qwen2.5-coder:3b`, `llama3.2`, `deepseek-r1`).

Installation & Setup



1. Install Dependencies in OpenWebUI Container

Execute pip install inside your running OpenWebUI container:

```bash

docker exec -it open-webui pip install -r requirements.txt

```



2. Import Tool into OpenWebUI

1. Open **OpenWebUI** → **Workspace** → **Tools** → **+ Create Tool**.

2. Copy and paste the contents of `tools/adsb_telemetry_tool.py`.

3. Click **Save**.



### 3. Enable Artifact Rendering (For Interactive Map Output)

1. Go to **Settings** → **Interface** → **Artifacts**.

2. Toggle **ON** `Artifacts / HTML Rendering`.

3. Toggle **ON** `iframe Sandbox Allow Same Origin`.



### 4. Enable Tools in Chat

Toggle **ON** the Wrench/Tools icon in the chat input controls before submitting spatial queries.

