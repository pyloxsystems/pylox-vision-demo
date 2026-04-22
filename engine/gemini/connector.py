"""Gemini 3.1 Pro Connector — the brain of Pylox Vision V2.

Receives frames from Frigate detections, sends to Gemini with full context
(camera description, time, anomaly score, training file, door sensor state),
and returns a structured threat assessment.

Only called after business hours. During business hours, detections are
logged silently and fed to the 3D twin.

Falls back to V1 rules engine if internet/API is unavailable.
"""

import os
import io
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from PIL import Image

logger = logging.getLogger("pylox-v2.gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
FALLBACK_MODEL = "gemini-2.5-flash"
TRAINING_DIR = Path(__file__).parent.parent.parent / "data" / "training"


class GeminiAssessment:
    """Structured response from Gemini analysis."""
    def __init__(self, raw: dict):
        self.real = raw.get("real", False)
        self.threat = raw.get("threat", 0)
        self.type = raw.get("type", "unknown")
        self.description = raw.get("description", "")
        self.action = raw.get("action", "none")
        self.confidence = raw.get("confidence", 0)
        self.raw = raw

    def to_dict(self) -> dict:
        return {
            "real": self.real,
            "threat": self.threat,
            "type": self.type,
            "description": self.description,
            "action": self.action,
            "confidence": self.confidence,
        }

    @property
    def should_alert(self) -> bool:
        return self.real and self.threat >= 7

    @property
    def should_flag(self) -> bool:
        return self.real and 4 <= self.threat < 7

    @property
    def severity(self) -> str:
        if self.threat >= 7:
            return "critical"
        if self.threat >= 4:
            return "warning"
        return "info"


class GeminiConnector:
    """Connects to Gemini 3.1 Pro for video frame analysis."""

    def __init__(self, daily_cost_cap: float = 5.0):
        self._client = None
        self._available = False
        self._daily_cost_cap = daily_cost_cap
        self._daily_cost = 0.0
        self._cost_reset_date = None
        self._cost_per_call = 0.003  # ~$0.003 per 9-frame analysis
        self._stats = {
            "calls": 0,
            "confirmed_real": 0,
            "false_positives_caught": 0,
            "alerts_sent": 0,
            "errors": 0,
            "avg_response_ms": 0,
            "fallback_activations": 0,
            "daily_cost": 0.0,
            "cost_cap_hit": False,
        }
        self._response_times = []
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._available = True
            logger.info(f"Gemini connector initialized (model: {GEMINI_MODEL})")
        except Exception as e:
            logger.error(f"Gemini client init failed: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    def analyze_video_clip(self, video_bytes: bytes, camera_id: str,
                            camera_name: str, mode: str = "initial",
                            history: list = None,
                            signal_context: str = "",
                            site_config: dict = None) -> Optional[GeminiAssessment]:
        """Analyze a video clip for an incident watch session.

        mode: 'initial' (first call of session) or 'continuation' (subsequent)
        history: list of previous narrations from this session
                 Each: {"timestamp": float, "threat": int, "narration": str}
        """
        if not self.available:
            return None

        # Daily cost cap check
        today = datetime.now().date()
        if self._cost_reset_date != today:
            self._daily_cost = 0.0
            self._cost_reset_date = today
            self._stats["cost_cap_hit"] = False

        if self._daily_cost >= self._daily_cost_cap:
            if not self._stats["cost_cap_hit"]:
                logger.warning(f"Daily cost cap ${self._daily_cost_cap} reached")
                self._stats["cost_cap_hit"] = True
            return None

        try:
            start_time = time.time()

            prompt = self._build_watch_prompt(
                camera_id, camera_name, mode, history or [],
                signal_context, site_config,
            )

            # Send video to Gemini
            from google.genai import types
            contents = [
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                prompt,
            ]

            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
                response_text = response.text if response else None
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    logger.error(f"Gemini quota hit: {err_str[:100]}")
                    self._daily_cost = self._daily_cost_cap
                    return None
                logger.warning(f"Gemini {GEMINI_MODEL} failed: {e}, trying {FALLBACK_MODEL}")
                try:
                    response = self._client.models.generate_content(
                        model=FALLBACK_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=512,
                        ),
                    )
                    response_text = response.text if response else None
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    return None

            if not response_text:
                return None

            assessment = self._parse_response(response_text)

            # Charge cost on success
            self._daily_cost += self._cost_per_call
            self._stats["daily_cost"] = round(self._daily_cost, 4)
            self._stats["calls"] += 1

            elapsed = (time.time() - start_time) * 1000
            self._response_times.append(elapsed)
            if len(self._response_times) > 50:
                self._response_times = self._response_times[-50:]
            self._stats["avg_response_ms"] = round(
                sum(self._response_times) / len(self._response_times), 0
            )

            logger.info(
                f"Gemini watch ({mode}): camera={camera_id} threat={assessment.threat}/10 "
                f"({elapsed:.0f}ms)"
            )

            return assessment

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Gemini watch analysis failed: {e}")
            return None

    def _build_watch_prompt(self, camera_id: str, camera_name: str, mode: str,
                             history: list, signal_context: str,
                             site_config: dict) -> str:
        """Build the prompt for a watch session call."""
        now = datetime.now()
        site = site_config or {}
        site_name = site.get("name", "Commercial Property")
        site_type = site.get("type", "warehouse")

        # Get training context
        training = ""
        if hasattr(self, '_training_manager') and self._training_manager:
            training = self._training_manager.get_prompt_context(camera_id)

        # Build history string
        history_text = ""
        if history:
            history_text = "PREVIOUS OBSERVATIONS THIS INCIDENT:\n"
            for h in history[-10:]:
                t = datetime.fromtimestamp(h.get("timestamp", 0)).strftime("%H:%M:%S")
                history_text += f"  {t} (threat {h.get('threat', '?')}): {h.get('narration', '')}\n"

        if mode == "initial":
            instruction = (
                "This is the FIRST observation of this incident. Determine if this is a real threat.\n"
                "If it's clearly a false positive (animal, shadow, vehicle without person), set real=false."
            )
        else:
            instruction = (
                "This is a CONTINUATION of an active incident. Compare to previous observations.\n"
                "Has the situation escalated, stayed the same, or resolved? Is the person still there?\n"
                "Set should_continue=false if the threat has cleared (person left, false alarm confirmed)."
            )

        prompt = f"""You are Pylox Vision, an AI security guard for {site_name} ({site_type}).

CAMERA: {camera_name} ({camera_id})
TIME: {now.strftime('%I:%M:%S %p, %A %B %d')}
MODE: {mode.upper()} — {('First analysis' if mode == 'initial' else 'Active watch continuation')}

{signal_context}

{training}

{history_text}

VIDEO: 5-second clip from this camera (just received).

{instruction}

RESPOND WITH ONLY THIS JSON (no markdown, no code blocks):
{{"real": true/false, "threat": 0-10, "type": "intruder|suspicious|delivery|employee|animal|false_positive", "narration": "One sentence describing what you see RIGHT NOW", "action": "none|log|alert_owner|deterrent|panic", "should_continue": true/false}}

threat scoring guide:
0-3 = normal/benign (employee, customer, delivery, animal)
4-6 = unusual but not threatening (loitering, unknown person walking past)
7-8 = suspicious (examining vehicles, trying doors, casing the property)
9-10 = active threat (forcing entry, breaking glass, in possession of stolen items)

action mapping:
none/log = just record, no notification
alert_owner = push notification + SMS
deterrent = strobe + push + SMS
panic = strobe + siren + phone call

should_continue = false ONLY if:
- It's confirmed false positive (animal, shadow)
- Person has clearly left and isn't coming back
- Situation is fully resolved"""

        return prompt

    def analyze_detection(self, frames: list, camera_id: str,
                           camera_name: str, label: str, score: float,
                           zones: list, duration: float,
                           anomaly_score: float = None,
                           door_state: dict = None,
                           site_config: dict = None,
                           signal_context: str = None) -> Optional[GeminiAssessment]:
        """Analyze detection frames with Gemini 3.1 Pro.

        Args:
            frames: List of PIL Image or numpy arrays (3-9 frames)
            camera_id: Camera identifier (cam1, cam2, etc.)
            camera_name: Human name ("Front Entrance", "Back Door")
            label: Detection label ("person", "car")
            score: Frigate detection confidence 0-1
            zones: Current zones the detection is in
            duration: How long the person has been visible (seconds)
            anomaly_score: Anomalib visual anomaly score 0-1
            door_state: Dict of door sensor states {"back_door": "open", ...}
            site_config: Client site configuration

        Returns:
            GeminiAssessment or None if analysis failed
        """
        if not self.available:
            logger.warning("Gemini not available, skipping analysis")
            self._stats["fallback_activations"] += 1
            return None

        # Daily cost cap check
        today = datetime.now().date()
        if self._cost_reset_date != today:
            self._daily_cost = 0.0
            self._cost_reset_date = today
            self._stats["cost_cap_hit"] = False

        if self._daily_cost >= self._daily_cost_cap:
            if not self._stats["cost_cap_hit"]:
                logger.warning(f"Daily cost cap ${self._daily_cost_cap} reached. Falling back to V1.")
                self._stats["cost_cap_hit"] = True
            self._stats["fallback_activations"] += 1
            return None

        try:
            start_time = time.time()
            self._stats["calls"] += 1
            # Build the prompt
            prompt = self._build_prompt(
                camera_id, camera_name, label, score,
                zones, duration, anomaly_score, door_state, site_config,
                signal_context,
            )

            # Prepare image parts
            image_parts = self._prepare_frames(frames)

            # Call Gemini
            response = self._call_gemini(prompt, image_parts)

            # Only charge for successful API calls
            if not response:
                return None

            # Parse response
            assessment = self._parse_response(response)

            # Charge cost only after successful call + parse
            self._daily_cost += self._cost_per_call
            self._stats["daily_cost"] = round(self._daily_cost, 4)

            # Update stats
            elapsed = (time.time() - start_time) * 1000
            self._response_times.append(elapsed)
            if len(self._response_times) > 50:
                self._response_times = self._response_times[-50:]
            self._stats["avg_response_ms"] = round(
                sum(self._response_times) / len(self._response_times), 0
            )

            if assessment.real:
                self._stats["confirmed_real"] += 1
                if assessment.should_alert:
                    self._stats["alerts_sent"] += 1
            else:
                self._stats["false_positives_caught"] += 1

            logger.info(
                f"Gemini analysis: camera={camera_id} real={assessment.real} "
                f"threat={assessment.threat}/10 type={assessment.type} "
                f"action={assessment.action} ({elapsed:.0f}ms)"
            )

            return assessment

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Gemini analysis failed: {e}")
            return None

    def _build_prompt(self, camera_id: str, camera_name: str, label: str,
                       score: float, zones: list, duration: float,
                       anomaly_score: float, door_state: dict,
                       site_config: dict, signal_context: str = None) -> str:
        """Build the analysis prompt with full context."""
        now = datetime.now()
        hour = now.hour
        site = site_config or {}
        open_hour = site.get("hours", {}).get("open", 6)
        close_hour = site.get("hours", {}).get("close", 20)
        is_business = open_hour <= hour < close_hour
        site_name = site.get("name", "Commercial Property")
        site_type = site.get("type", "warehouse")

        # Load training file for this camera
        training_context = self._load_training_context(camera_id, site.get("id", "default"))

        # Build door sensor context
        door_context = ""
        if door_state:
            open_doors = [k for k, v in door_state.items() if v == "open"]
            if open_doors:
                door_context = f"\nDoor sensors currently OPEN: {', '.join(open_doors)}"

        prompt = f"""You are a security camera analyst for {site_name} ({site_type}).

Camera: {camera_name} ({camera_id})
Time: {now.strftime('%I:%M %p, %A %B %d')}
Business hours: {open_hour}:00 AM - {close_hour % 12 or 12}:00 {'PM' if close_hour > 12 else 'AM'}
Currently: {'BUSINESS HOURS' if is_business else 'AFTER HOURS'}

Frigate detected: {label} (confidence: {score:.0%})
Person visible for: {duration:.0f} seconds
Zones: {', '.join(zones) if zones else 'none'}

--- INTELLIGENCE SIGNALS ---
{signal_context if signal_context else training_context}
{door_context if not signal_context else ''}

Analyze the following frames from the last few seconds of footage.

You MUST respond with ONLY raw JSON. No markdown, no code blocks, no backticks, no explanation. Just the JSON object:
{{"real":false,"threat":0,"type":"false_positive","description":"No person visible.","action":"none","confidence":8}}

Valid types: employee, visitor, delivery, contractor, suspicious, intruder, animal, vehicle, false_positive
Valid actions: none, log, alert_owner, call_security"""

        return prompt

    def set_training_manager(self, training_manager):
        """Set the training manager for prompt context."""
        self._training_manager = training_manager

    def _load_training_context(self, camera_id: str, site_id: str) -> str:
        """Load the learned patterns for this camera."""
        if hasattr(self, '_training_manager') and self._training_manager:
            return self._training_manager.get_prompt_context(camera_id)

        # Fallback to file-based loading
        training_file = TRAINING_DIR / site_id / "training.json"
        if not training_file.exists():
            return ""
        try:
            data = json.loads(training_file.read_text())
            lines = []
            cam_data = data.get("cameras", {}).get(camera_id, {})
            for fp in cam_data.get("false_positives", [])[-10:]:
                desc = fp.get("description", fp) if isinstance(fp, dict) else fp
                lines.append(f"  - {desc}")
            if lines:
                lines.insert(0, "KNOWN FALSE POSITIVES:")
            return "\n".join(lines)
        except Exception:
            return ""

    def _prepare_frames(self, frames: list) -> list:
        """Convert frames to base64 image parts for Gemini."""
        parts = []
        for frame in frames:
            if hasattr(frame, 'shape'):
                # numpy array
                img = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                img = frame
            else:
                continue

            # Resize to 720px width for cost efficiency
            if img.width > 720:
                ratio = 720 / img.width
                img = img.resize((720, int(img.height * ratio)))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(buf.getvalue()).decode(),
            })

        return parts

    def _call_gemini(self, prompt: str, image_parts: list) -> str:
        """Call Gemini API with prompt and images."""
        from google.genai import types

        # Build content parts: images first, then prompt
        contents = []
        for img in image_parts:
            contents.append(types.Part.from_bytes(
                data=base64.b64decode(img["data"]),
                mime_type=img["mime_type"],
            ))
        contents.append(prompt)

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            return response.text if response else None
        except Exception as e:
            err_str = str(e)
            # Don't retry on rate limit / quota errors
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                logger.error(f"Gemini quota hit, suppressing further calls: {err_str[:100]}")
                self._daily_cost = self._daily_cost_cap  # Trigger cost cap
                return None
            logger.warning(f"Gemini {GEMINI_MODEL} failed, trying {FALLBACK_MODEL}: {e}")
            self._stats["fallback_activations"] += 1
            try:
                response = self._client.models.generate_content(
                    model=FALLBACK_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
                return response.text if response else None
            except Exception as e2:
                logger.error(f"Fallback {FALLBACK_MODEL} also failed: {e2}")
                return None

    def _parse_response(self, response_text: str) -> GeminiAssessment:
        """Parse Gemini's JSON response into a GeminiAssessment."""
        if not response_text:
            return GeminiAssessment({
                "real": False, "threat": 0, "type": "false_positive",
                "description": "Empty response", "action": "none", "confidence": 0,
            })
        try:
            text = response_text.strip()

            # Strip markdown code blocks
            if "```" in text:
                # Extract content between code blocks
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break

            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            data = json.loads(text)

            # Validate and clamp
            if "threat" in data:
                data["threat"] = max(0, min(10, int(data["threat"])))
            else:
                data["threat"] = 0

            if "real" not in data:
                data["real"] = False  # Default to false — don't alert on parse errors

            return GeminiAssessment(data)

        except json.JSONDecodeError:
            logger.warning(f"Gemini returned unparseable: {response_text[:200]}")
            # Default to false positive — never alert on a parse failure
            return GeminiAssessment({
                "real": False,
                "threat": 0,
                "type": "false_positive",
                "description": "AI response could not be parsed",
                "action": "none",
                "confidence": 0,
            })

    def add_false_positive(self, site_id: str, camera_id: str,
                            description: str):
        """Add a learned false positive for a camera."""
        training_file = TRAINING_DIR / site_id / "training.json"
        training_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if training_file.exists():
            data = json.loads(training_file.read_text())

        cameras = data.setdefault("cameras", {})
        cam = cameras.setdefault(camera_id, {})
        fps = cam.setdefault("false_positives", [])

        if description not in fps:
            fps.append(description)
            # Keep max 10 per camera
            if len(fps) > 10:
                fps.pop(0)

        training_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Added false positive for {camera_id}: {description}")

    def add_learned_pattern(self, site_id: str, pattern: str):
        """Add a site-wide learned pattern."""
        training_file = TRAINING_DIR / site_id / "training.json"
        training_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if training_file.exists():
            data = json.loads(training_file.read_text())

        learned = data.setdefault("learned", [])
        if pattern not in learned:
            learned.append(pattern)
            if len(learned) > 20:
                learned.pop(0)

        training_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Added learned pattern: {pattern}")

    def get_stats(self) -> dict:
        return {**self._stats, "available": self.available, "model": GEMINI_MODEL}
