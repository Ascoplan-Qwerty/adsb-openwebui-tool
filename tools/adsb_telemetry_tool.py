import math
import time
import requests
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM


class Tools:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(
            default="http://172.17.0.1:11434",
            description="Base URL for the local Ollama gateway container.",
        )
        OPENSKY_USERNAME: str = Field(
            default="", description="Optional OpenSky Network account username."
        )
        OPENSKY_PASSWORD: str = Field(
            default="", description="Optional OpenSky Network account password."
        )
        DEFAULT_LATITUDE: float = Field(
            default=12.994,
            description="Default center latitude (e.g., Chennai MAA: 12.994, Delhi DEL: 28.556).",
        )
        DEFAULT_LONGITUDE: float = Field(
            default=80.171,
            description="Default center longitude (e.g., Chennai MAA: 80.171, Delhi DEL: 77.100).",
        )
        DEFAULT_RADIUS_KM: float = Field(
            default=150.0, description="Default spatial bounding radius in kilometers."
        )

    def __init__(self):
        self.valves = self.Valves()

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    def _get_bounding_box(
        self, lat: float, lon: float, radius_km: float
    ) -> Dict[str, float]:
        lat_change = radius_km / 111.0
        lon_change = radius_km / (111.0 * math.cos(math.radians(lat)))
        return {
            "lamin": lat - lat_change,
            "lamax": lat + lat_change,
            "lomin": lon - lon_change,
            "lomax": lon + lon_change,
        }

    def _generate_leaflet_html(
        self, lat: float, lon: float, aircraft_records: List[Dict]
    ) -> str:
        markers_js = []
        for ac in aircraft_records:
            color = "#ef4444" if ac["emergency"] else "#0284c7"
            popup = (
                f"<b>Callsign:</b> {ac['callsign']}<br/>"
                f"<b>ICAO24:</b> {ac['icao24']}<br/>"
                f"<b>Altitude:</b> {ac['alt_ft']:,} ft<br/>"
                f"<b>Speed:</b> {ac['speed_kts']} kts<br/>"
                f"<b>Heading:</b> {ac['heading']}°<br/>"
                f"<b>Squawk:</b> {ac['squawk']}"
            )
            popup_clean = popup.replace("'", "\\'")
            markers_js.append(
                f"L.circleMarker([{ac['lat']}, {ac['lon']}], {{radius: 7, color: '{color}', fillColor: '{color}', fillOpacity: 0.85}})"
                f".bindPopup('{popup_clean}').addTo(map);"
            )

        markers_code = "\n".join(markers_js)

        return f"""
<div id="adsb-map" style="width: 100%; height: 380px; border-radius: 10px; margin-top: 10px; border: 1px solid #334155;"></div>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
    var map = L.map('adsb-map').setView([{lat}, {lon}], 8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        maxZoom: 19,
        attribution: 'OpenSky | OpenStreetMap'
    }}).addTo(map);

    L.marker([{lat}, {lon}]).bindPopup("<b>Center Reference Point</b>").addTo(map);

    {markers_code}
</script>
"""

    def query_live_airspace(
        self,
        center_lat: Optional[float] = Field(
            None,
            description="Optional latitude (e.g., 12.994 for MAA, 28.556 for DEL).",
        ),
        center_lon: Optional[float] = Field(
            None,
            description="Optional longitude (e.g., 80.171 for MAA, 77.100 for DEL).",
        ),
        radius_km: Optional[float] = Field(
            None, description="Search radius in kilometers (default: 150 km)."
        ),
        target_inquiry: Optional[str] = Field(
            "summary",
            description="Inquiry metric/type: 'positional', 'anomaly', 'summary', or 'code_execution'.",
        ),
        callsign_filter: Optional[str] = Field(
            None,
            description="Optional flight callsign or 24-bit ICAO hex to filter (e.g., 'AIC102', '43EA2B').",
        ),
    ) -> str:
        t_start_total = time.perf_counter()

        lat = center_lat if center_lat is not None else self.valves.DEFAULT_LATITUDE
        lon = center_lon if center_lon is not None else self.valves.DEFAULT_LONGITUDE
        radius = radius_km if radius_km is not None else self.valves.DEFAULT_RADIUS_KM

        t_start_ingest = time.perf_counter()
        bbox = self._get_bounding_box(lat, lon, radius)
        url = f"https://opensky-network.org/api/states/all?lamin={bbox['lamin']:.4f}&lamax={bbox['lamax']:.4f}&lomin={bbox['lomin']:.4f}&lomax={bbox['lomax']:.4f}"

        headers = {"User-Agent": "AviationAI-Research/1.0"}
        auth = None
        if self.valves.OPENSKY_USERNAME and self.valves.OPENSKY_PASSWORD:
            auth = (self.valves.OPENSKY_USERNAME, self.valves.OPENSKY_PASSWORD)

        try:
            response = requests.get(url, headers=headers, auth=auth, timeout=8)
            if response.status_code == 429:
                return "⚠️ OpenSky API Rate Limit Reached (429). Please wait before retrying or configure authenticated account credentials in tool Valves."
            response.raise_for_status()
            raw_data = response.json()
        except Exception as e:
            return f"❌ Failed to query OpenSky Network REST API: {str(e)}"

        states = raw_data.get("states")
        t_ingest_ms = round((time.perf_counter() - t_start_ingest) * 1000, 2)

        if not states:
            return f"No active aircraft state vectors detected within {radius} km of target coordinates ({lat:.4f}, {lon:.4f}). Ingest Latency: {t_ingest_ms} ms."

        t_start_rag = time.perf_counter()
        aircraft_records = []
        emergency_events = []

        for s in states:
            icao24 = s[0]
            callsign = (s[1] or "N/A").strip()
            ac_lat = s[6]
            ac_lon = s[5]
            baro_alt_m = (
                s[7] if s[7] is not None else (s[13] if s[13] is not None else 0)
            )
            velocity_ms = s[9] or 0
            heading = s[10] or 0
            vertical_rate_ms = s[11] or 0
            squawk = s[14] or "0000"

            if ac_lat is None or ac_lon is None:
                continue

            dist_km = self._haversine(lat, lon, ac_lat, ac_lon)
            if dist_km > radius:
                continue

            if callsign_filter:
                clean_target = callsign_filter.strip().upper()
                if (
                    clean_target not in icao24.upper()
                    and clean_target not in callsign.upper()
                ):
                    continue

            alt_ft = int(baro_alt_m * 3.28084)
            speed_kts = int(velocity_ms * 1.94384)
            vrate_fpm = int(vertical_rate_ms * 196.85)

            record = {
                "icao24": icao24,
                "callsign": callsign,
                "lat": round(ac_lat, 4),
                "lon": round(ac_lon, 4),
                "dist_km": round(dist_km, 1),
                "alt_ft": alt_ft,
                "speed_kts": speed_kts,
                "heading": int(heading),
                "vrate_fpm": vrate_fpm,
                "squawk": squawk,
                "emergency": squawk in ["7500", "7600", "7700"],
            }

            if record["emergency"]:
                emergency_type = (
                    "7500 (HIJACK)"
                    if squawk == "7500"
                    else (
                        "7600 (RADIO FAILURE)"
                        if squawk == "7600"
                        else "7700 (GENERAL EMERGENCY)"
                    )
                )
                emergency_events.append(
                    f"🚨 **EMERGENCY SQUAWK {emergency_type}**: Flight {callsign} (`{icao24}`) at {alt_ft:,} ft, {dist_km:.1f} km away!"
                )

            aircraft_records.append(record)

        t_rag_ms = round((time.perf_counter() - t_start_rag) * 1000, 2)

        if not aircraft_records:
            return f"No matching active aircraft found within radius {radius} km for filter criteria."

        aircraft_records.sort(key=lambda x: x["dist_km"])

        inquiry_type = (target_inquiry or "summary").lower().strip()
        model_used = ""
        t_start_inference = time.perf_counter()

        if "anomaly" in inquiry_type or emergency_events:
            model_used = "deepseek-r1"
            reasoner_llm = OllamaLLM(
                model=model_used, base_url=self.valves.OLLAMA_BASE_URL
            )
            context_summary = str(aircraft_records[:10])
            prompt = f"""You are an aviation safety and telemetry analyst.
Analyze these live ADS-B flight state vectors for emergency squawks, descent rate anomalies, or airspace risks.
Active Aircraft Context: {context_summary}
Emergency Alerts Active: {emergency_events}

Provide a concise, step-by-step reasoning analysis evaluating potential airspace anomalies."""
            result_body = reasoner_llm.invoke(prompt)

        elif "code" in inquiry_type or "math" in inquiry_type:
            model_used = "qwen2.5-coder:3b"
            coder_llm = OllamaLLM(
                model=model_used, base_url=self.valves.OLLAMA_BASE_URL
            )
            prompt = f"""You are an air traffic data engineer. Formulate exact structured spatial conclusions for these flight vectors:
Flight Vectors: {aircraft_records[:10]}
Identify the closest aircraft, calculate average altitude, and format the output cleanly."""
            result_body = coder_llm.invoke(prompt)

        elif "position" in inquiry_type or "lookup" in inquiry_type:
            model_used = "llama3.2"
            llama_llm = OllamaLLM(
                model=model_used, base_url=self.valves.OLLAMA_BASE_URL
            )
            top_ac = aircraft_records[0]
            prompt = (
                f"""Summarize positional metrics for the closest aircraft: {top_ac}."""
            )
            result_body = llama_llm.invoke(prompt)

        else:
            model_used = "qwen2.5 (Direct Context)"
            lines = [
                f"### ✈️ Live Airspace Table ({len(aircraft_records)} aircraft within {radius} km)\n"
            ]
            if emergency_events:
                lines.append("### 🚨 EMERGENCY ANOMALY DETECTED")
                lines.extend(emergency_events)
                lines.append("\n---\n")

            lines.append(
                "| Callsign | ICAO24 | Distance | Altitude | Speed | Heading | Vertical Rate | Squawk |"
            )
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for ac in aircraft_records[:15]:
                flag = "🚨 " if ac["emergency"] else ""
                lines.append(
                    f"| {flag}{ac['callsign']} | `{ac['icao24']}` | {ac['dist_km']} km | {ac['alt_ft']:,} ft | {ac['speed_kts']} kts | {ac['heading']}° | {ac['vrate_fpm']} ft/min | `{ac['squawk']}` |"
                )
            result_body = "\n".join(lines)

        t_inference_ms = round((time.perf_counter() - t_start_inference) * 1000, 2)
        t_total_ms = round((time.perf_counter() - t_start_total) * 1000, 2)

        map_html = self._generate_leaflet_html(lat, lon, aircraft_records[:15])

        final_output = (
            f"{result_body}\n\n"
            f"### 🌐 Live Spatial Radar Map\n"
            f"{map_html}\n\n"
            f"---\n"
            f"⏱️ **System Latency Benchmarks**:\n"
            f"- **Data Ingestion Latency (OpenSky REST API)**: `{t_ingest_ms} ms`\n"
            f"- **Spatial Haversine Indexing & RAG Filtering**: `{t_rag_ms} ms`\n"
            f"- **LLM Execution Time (`{model_used}`)**: `{t_inference_ms} ms`\n"
            f"- **Total Pipeline Latency**: `{t_total_ms} ms`"
        )

        return final_output
