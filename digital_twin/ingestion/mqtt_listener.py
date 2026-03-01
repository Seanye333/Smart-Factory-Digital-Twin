"""
MQTT Listener — subscribes to factory IoT topics and delivers parsed payloads.

Compatible with any standard MQTT broker (Mosquitto, EMQX, HiveMQ, AWS IoT Core).
Falls back gracefully if `paho-mqtt` is not installed.

Topic schema (configurable):
    factory/<plant_id>/<machine_id>/sensors     → JSON sensor readings
    factory/<plant_id>/<machine_id>/events      → fault / alarm events
    factory/<plant_id>/<machine_id>/commands    → control commands (subscribe)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    logger.warning("[MQTT] paho-mqtt not installed — Run: pip install paho-mqtt")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class MQTTConfig:
    host:       str   = "localhost"
    port:       int   = 1883
    username:   str   = ""
    password:   str   = ""
    client_id:  str   = "digital_twin_listener"
    keepalive:  int   = 60
    tls:        bool  = False
    plant_id:   str   = "plant_01"
    qos:        int   = 1


# --------------------------------------------------------------------------
# Listener class
# --------------------------------------------------------------------------

class MQTTListener:
    """
    Subscribes to all machine sensor + event topics.
    Decoded payloads are forwarded to registered callback functions.
    """

    def __init__(self, config: Optional[MQTTConfig] = None):
        self.cfg       = config or MQTTConfig()
        self._client   = None
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            "sensors":  [],
            "events":   [],
            "commands": [],
        }
        self._message_count = 0
        self._last_message_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_sensor_data(self, cb: Callable[[str, Dict], None]) -> None:
        """Register handler for sensor payloads.  Signature: cb(machine_id, payload)"""
        self._callbacks["sensors"].append(cb)

    def on_event(self, cb: Callable[[str, Dict], None]) -> None:
        """Register handler for fault/event payloads."""
        self._callbacks["events"].append(cb)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not PAHO_AVAILABLE:
            logger.error("[MQTT] paho-mqtt not installed.")
            return False

        self._client = mqtt.Client(client_id=self.cfg.client_id, protocol=mqtt.MQTTv5)

        if self.cfg.username:
            self._client.username_pw_set(self.cfg.username, self.cfg.password)
        if self.cfg.tls:
            self._client.tls_set()

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        try:
            self._client.connect(self.cfg.host, self.cfg.port, self.cfg.keepalive)
            self._client.loop_start()
            logger.info(f"[MQTT] Connecting → {self.cfg.host}:{self.cfg.port}")
            return True
        except Exception as exc:
            logger.error(f"[MQTT] Connection error: {exc}")
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("[MQTT] Disconnected")

    # ------------------------------------------------------------------
    # Internal MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            self._connected = True
            logger.info("[MQTT] Connected to broker")
            # Subscribe to all factory topics
            plant = self.cfg.plant_id
            client.subscribe(f"factory/{plant}/+/sensors",  self.cfg.qos)
            client.subscribe(f"factory/{plant}/+/events",   self.cfg.qos)
            client.subscribe(f"factory/{plant}/+/commands", self.cfg.qos)
        else:
            logger.error(f"[MQTT] Connect failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        self._connected = False
        if rc != 0:
            logger.warning(f"[MQTT] Unexpected disconnect rc={rc} — will retry")

    def _on_message(self, client, userdata, msg) -> None:
        self._message_count += 1
        self._last_message_time = time.time()

        try:
            topic_parts = msg.topic.split("/")
            # Expected: factory/<plant>/<machine_id>/<category>
            if len(topic_parts) < 4:
                return
            machine_id = topic_parts[2]
            category   = topic_parts[3]
            payload    = json.loads(msg.payload.decode("utf-8"))
        except Exception as exc:
            logger.warning(f"[MQTT] Parse error on {msg.topic}: {exc}")
            return

        handlers = self._callbacks.get(category, [])
        for cb in handlers:
            try:
                cb(machine_id, payload)
            except Exception as exc:
                logger.error(f"[MQTT] Handler error: {exc}")

    # ------------------------------------------------------------------
    # Publishing (for sending commands back to machines)
    # ------------------------------------------------------------------

    def publish_command(self, machine_id: str, command: Dict[str, Any]) -> bool:
        if not self._connected or not self._client:
            logger.warning("[MQTT] Cannot publish — not connected")
            return False
        topic   = f"factory/{self.cfg.plant_id}/{machine_id}/commands"
        payload = json.dumps(command)
        result  = self._client.publish(topic, payload, qos=self.cfg.qos)
        return result.rc == 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def stats(self) -> Dict[str, Any]:
        return {
            "connected":         self._connected,
            "messages_received": self._message_count,
            "last_message_at":   self._last_message_time,
        }
