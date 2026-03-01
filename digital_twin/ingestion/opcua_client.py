"""
OPC-UA Client — connects to a real PLC / OPC-UA server and reads tag values.
Falls back gracefully if `asyncua` is not installed.

Usage (async):
    client = OPCUAClient("opc.tcp://192.168.0.100:4840")
    await client.connect()
    values = await client.read_tags(["ns=2;i=1001", "ns=2;i=1002"])
    await client.disconnect()
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from asyncua import Client as _AsyncUAClient
    ASYNCUA_AVAILABLE = True
except ImportError:
    ASYNCUA_AVAILABLE = False
    logger.warning("[OPC-UA] asyncua not installed — real PLC connection unavailable. "
                   "Run: pip install asyncua")


@dataclass
class TagConfig:
    node_id:    str
    name:       str
    scale:      float = 1.0
    offset:     float = 0.0
    unit:       str   = ""


class OPCUAClient:
    """
    Async OPC-UA tag reader.

    Parameters
    ----------
    url      : OPC-UA endpoint, e.g. "opc.tcp://192.168.1.10:4840"
    username : optional — leave empty for anonymous access
    password : optional
    tags     : list of TagConfig objects to subscribe to
    """

    def __init__(
        self,
        url:      str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tags:     Optional[List[TagConfig]] = None,
    ):
        self.url      = url
        self.username = username
        self.password = password
        self.tags     = tags or []
        self._client  = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not ASYNCUA_AVAILABLE:
            logger.error("[OPC-UA] Cannot connect — asyncua not installed.")
            return False
        try:
            self._client = _AsyncUAClient(self.url)
            if self.username:
                self._client.set_user(self.username)
                self._client.set_password(self.password or "")
            await self._client.connect()
            self._connected = True
            logger.info(f"[OPC-UA] Connected → {self.url}")
            return True
        except Exception as exc:
            logger.error(f"[OPC-UA] Connection failed: {exc}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("[OPC-UA] Disconnected")

    # ------------------------------------------------------------------
    # Tag reading
    # ------------------------------------------------------------------

    async def read_tags(self, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read a list of node IDs and return {node_id: scaled_value}.
        If node_ids is None, reads all tags in self.tags.
        """
        if not self._connected or not self._client:
            logger.warning("[OPC-UA] Not connected — returning empty dict")
            return {}

        ids = node_ids or [t.node_id for t in self.tags]
        result: Dict[str, Any] = {}

        for node_id in ids:
            try:
                node  = self._client.get_node(node_id)
                value = await node.read_value()
                # Apply scaling if a matching TagConfig exists
                cfg = next((t for t in self.tags if t.node_id == node_id), None)
                if cfg:
                    value = value * cfg.scale + cfg.offset
                result[node_id] = value
            except Exception as exc:
                logger.warning(f"[OPC-UA] Failed to read {node_id}: {exc}")
                result[node_id] = None

        return result

    async def read_tag(self, node_id: str) -> Optional[Any]:
        values = await self.read_tags([node_id])
        return values.get(node_id)

    # ------------------------------------------------------------------
    # Convenience: continuous polling
    # ------------------------------------------------------------------

    async def poll(self, interval: float = 1.0):
        """
        Async generator — yields a full tag dict every `interval` seconds.

        Example:
            async for snapshot in client.poll(interval=2.0):
                process(snapshot)
        """
        while self._connected:
            snapshot = await self.read_tags()
            yield snapshot
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()


# --------------------------------------------------------------------------
# Factory-specific tag map  (adapt to your actual PLC namespace)
# --------------------------------------------------------------------------

FACTORY_TAGS = [
    TagConfig("ns=2;i=1001", "CNC-001.Temperature",   scale=0.1, unit="°C"),
    TagConfig("ns=2;i=1002", "CNC-001.Vibration",     scale=0.01, unit="mm/s"),
    TagConfig("ns=2;i=1003", "CNC-001.Current",       scale=0.1, unit="A"),
    TagConfig("ns=2;i=1004", "CNC-001.Speed",         scale=1.0, unit="RPM"),
    TagConfig("ns=2;i=1011", "CNC-002.Temperature",   scale=0.1, unit="°C"),
    TagConfig("ns=2;i=1012", "CNC-002.Vibration",     scale=0.01, unit="mm/s"),
    TagConfig("ns=2;i=2001", "ROB-001.Temperature",   scale=0.1, unit="°C"),
    TagConfig("ns=2;i=2002", "ROB-001.Current",       scale=0.1, unit="A"),
    TagConfig("ns=2;i=3001", "CNV-001.Speed",         scale=1.0, unit="RPM"),
    TagConfig("ns=2;i=3002", "CNV-001.Current",       scale=0.1, unit="A"),
]
