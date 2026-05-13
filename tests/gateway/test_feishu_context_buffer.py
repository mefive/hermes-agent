"""Tests for the Feishu pre-mention ring buffer (context_window feature)."""

from __future__ import annotations

import asyncio
import unittest
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock

from gateway.platforms.base import MessageType
from gateway.platforms.feishu import (
    BufferedContext,
    FeishuAdapter,
    _format_relative_age,
)
from tests.gateway.feishu_helpers import make_adapter_skeleton


def _install_buffer_state(
    adapter: Any,
    *,
    context_window: int = 50,
    context_max_age: float = 600.0,
) -> None:
    adapter._context_window = context_window
    adapter._context_max_age = context_max_age
    adapter._context_buffer = {}
    adapter._context_buffer_lock = asyncio.Lock()
    adapter._context_window_observed = False
    adapter._context_window_warning_scheduled = False


def _make_message(*, message_id: str = "om_1", chat_id: str = "oc_room") -> Any:
    return SimpleNamespace(
        message_id=message_id,
        chat_id=chat_id,
        chat_type="group",
        content="",
        message_type="text",
        mentions=None,
    )


def _make_sender() -> Any:
    return SimpleNamespace(
        sender_type="user",
        sender_id=SimpleNamespace(open_id="ou_alice", user_id="u_alice", union_id="on_alice"),
    )


def _stub_extract(
    adapter: Any,
    *,
    text: str = "hello",
    media_urls: Optional[List[str]] = None,
    media_types: Optional[List[str]] = None,
    inbound_type: MessageType = MessageType.TEXT,
) -> None:
    async def _ret(_message):
        return (
            text,
            inbound_type,
            list(media_urls or []),
            list(media_types or []),
            [],
        )
    adapter._extract_message_content = _ret  # type: ignore[assignment]


def _stub_sender_profile(adapter: Any, name: str = "Alice") -> None:
    async def _ret(_sender_id, *, is_bot: bool = False):
        return {"user_id": "u_alice", "user_name": name, "user_id_alt": "on_alice"}
    adapter._resolve_sender_profile = _ret  # type: ignore[assignment]


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestRelativeAge(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(_format_relative_age(timedelta(seconds=5)), "5s ago")

    def test_minutes(self):
        self.assertEqual(_format_relative_age(timedelta(seconds=125)), "2m ago")

    def test_hours(self):
        self.assertEqual(_format_relative_age(timedelta(seconds=4000)), "1h ago")

    def test_negative_collapses(self):
        self.assertEqual(_format_relative_age(timedelta(seconds=-3)), "just now")


class TestBufferInbound(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter_skeleton()
        _install_buffer_state(self.adapter)
        _stub_sender_profile(self.adapter)
        _stub_extract(self.adapter, text="image-caption", media_urls=[], media_types=[])

    def test_appends_text_entry_to_per_chat_deque(self):
        msg = _make_message(chat_id="oc_a")
        sender = _make_sender()
        asyncio.run(FeishuAdapter._buffer_inbound(self.adapter, None, msg, sender))
        buf = self.adapter._context_buffer.get("oc_a")
        self.assertIsNotNone(buf)
        self.assertEqual(len(buf), 1)
        entry = buf[0]
        self.assertIsInstance(entry, BufferedContext)
        self.assertEqual(entry.chat_id, "oc_a")
        self.assertEqual(entry.text, "image-caption")
        self.assertEqual(entry.sender_name, "Alice")
        self.assertFalse(entry.is_bot)

    def test_flips_observed_flag_on_first_write(self):
        self.assertFalse(self.adapter._context_window_observed)
        asyncio.run(FeishuAdapter._buffer_inbound(
            self.adapter, None, _make_message(), _make_sender(),
        ))
        self.assertTrue(self.adapter._context_window_observed)

    def test_maxlen_evicts_oldest(self):
        self.adapter._context_window = 3
        for i in range(5):
            _stub_extract(self.adapter, text=f"msg-{i}")
            asyncio.run(FeishuAdapter._buffer_inbound(
                self.adapter, None, _make_message(message_id=f"om_{i}"), _make_sender(),
            ))
        buf = self.adapter._context_buffer["oc_room"]
        self.assertEqual(len(buf), 3)
        self.assertEqual([e.text for e in buf], ["msg-2", "msg-3", "msg-4"])

    def test_drops_bot_senders(self):
        bot_sender = SimpleNamespace(
            sender_type="app",
            sender_id=SimpleNamespace(open_id="ou_bot", user_id=None, union_id=None),
        )
        asyncio.run(FeishuAdapter._buffer_inbound(
            self.adapter, None, _make_message(), bot_sender,
        ))
        self.assertNotIn("oc_room", self.adapter._context_buffer)

    def test_drops_commands(self):
        _stub_extract(self.adapter, text="/help", inbound_type=MessageType.COMMAND)
        asyncio.run(FeishuAdapter._buffer_inbound(
            self.adapter, None, _make_message(), _make_sender(),
        ))
        self.assertNotIn("oc_room", self.adapter._context_buffer)

    def test_drops_empty_payloads(self):
        _stub_extract(self.adapter, text="", media_urls=[], media_types=[])
        asyncio.run(FeishuAdapter._buffer_inbound(
            self.adapter, None, _make_message(), _make_sender(),
        ))
        self.assertNotIn("oc_room", self.adapter._context_buffer)

    def test_filters_non_image_media(self):
        _stub_extract(
            self.adapter,
            text="",
            media_urls=["/cache/a.mp3", "/cache/b.png"],
            media_types=["audio/mpeg", "image/png"],
        )
        asyncio.run(FeishuAdapter._buffer_inbound(
            self.adapter, None, _make_message(), _make_sender(),
        ))
        entry = self.adapter._context_buffer["oc_room"][0]
        self.assertEqual(entry.media_paths, ["/cache/b.png"])
        self.assertEqual(entry.media_types, ["image/png"])


class TestConsumeContext(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter_skeleton()
        _install_buffer_state(self.adapter)

    def _seed(self, chat_id: str, entries: List[BufferedContext]) -> None:
        self.adapter._context_buffer[chat_id] = deque(entries, maxlen=self.adapter._context_window)

    def _entry(
        self,
        *,
        text: str = "hi",
        ago_seconds: float,
        media_paths: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
        sender_name: str = "Alice",
    ) -> BufferedContext:
        return BufferedContext(
            message_id="om",
            chat_id="oc_room",
            sender_name=sender_name,
            sender_user_id="u_alice",
            text=text,
            media_paths=list(media_paths or []),
            media_types=list(media_types or []),
            timestamp=datetime.now() - timedelta(seconds=ago_seconds),
            is_bot=False,
        )

    def test_returns_none_for_empty_buffer(self):
        prefix, paths, types = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=datetime.now(),
        ))
        self.assertIsNone(prefix)
        self.assertEqual(paths, [])
        self.assertEqual(types, [])

    def test_formats_prefix_with_sender_age_and_media_marker(self):
        self._seed("oc_room", [
            self._entry(text="看看这个图", ago_seconds=30),
            self._entry(
                text="",
                ago_seconds=20,
                media_paths=["/cache/x.png"],
                media_types=["image/png"],
                sender_name="Bob",
            ),
        ])
        prefix, paths, types = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=datetime.now(),
        ))
        self.assertIsNotNone(prefix)
        self.assertIn("Recent group context — 2 messages", prefix)
        self.assertIn("Alice 30s ago: 看看这个图", prefix)
        self.assertIn("Bob 20s ago: [image 1 attached]", prefix)
        self.assertEqual(paths, ["/cache/x.png"])
        self.assertEqual(types, ["image/png"])

    def test_consume_removes_items(self):
        self._seed("oc_room", [self._entry(text="one", ago_seconds=5)])
        prefix1, _, _ = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=datetime.now(),
        ))
        self.assertIsNotNone(prefix1)
        prefix2, _, _ = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=datetime.now(),
        ))
        self.assertIsNone(prefix2)

    def test_ttl_drops_old_items_silently(self):
        self.adapter._context_max_age = 60.0
        self._seed("oc_room", [
            self._entry(text="recent", ago_seconds=10),
            self._entry(text="stale", ago_seconds=600),
        ])
        prefix, _, _ = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=datetime.now(),
        ))
        self.assertIsNotNone(prefix)
        self.assertIn("recent", prefix)
        self.assertNotIn("stale", prefix)
        # And the stale one is consumed (removed), not lingering
        self.assertEqual(len(self.adapter._context_buffer["oc_room"]), 0)

    def test_future_items_left_in_buffer(self):
        future = self._entry(text="future", ago_seconds=-5)
        past = self._entry(text="past", ago_seconds=10)
        self._seed("oc_room", [past, future])
        # Use a fixed until_ts so we compare against the buffer's frozen
        # timestamps rather than racing wall-clock.
        until = datetime.now()
        # Adjust seeded timestamps relative to `until` for deterministic comparison.
        past.timestamp = until - timedelta(seconds=10)
        future.timestamp = until + timedelta(seconds=5)
        prefix, _, _ = asyncio.run(FeishuAdapter._consume_context(
            self.adapter, chat_id="oc_room", until_ts=until,
        ))
        self.assertIsNotNone(prefix)
        self.assertIn("past", prefix)
        self.assertNotIn("future", prefix)
        # Future item should still be in the buffer for the next consume.
        remaining = list(self.adapter._context_buffer["oc_room"])
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].text, "future")


if __name__ == "__main__":
    unittest.main()
