"""
Async pub/sub message bus for inter-subsystem communication.

The message bus provides topic-based routing with wildcard support
for flexible subscription patterns.
"""

from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import anyio
import structlog
from ulid import ULID

from autogenrec.bus.topics import Topic
from autogenrec.core.signals import Message

logger = structlog.get_logger()

MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """A subscription to a topic pattern."""

    id: str
    pattern: str
    handler: MessageHandler
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def matches(self, topic: str) -> bool:
        """Check if this subscription matches a topic."""
        return Topic(topic).matches(self.pattern)


@dataclass
class MessageBusStats:
    """Statistics for the message bus."""

    total_messages_published: int = 0
    total_messages_delivered: int = 0
    total_subscriptions: int = 0
    total_errors: int = 0


class MessageBus:
    """
    Async pub/sub message bus with topic-based routing.

    Features:
    - Hierarchical topics with dot-separated segments
    - Wildcard subscriptions (* for single segment, # for multiple)
    - Concurrent message delivery
    - Error isolation (one handler failure doesn't affect others)
    """

    def __init__(self, max_concurrent_handlers: int = 100) -> None:
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._subscriptions_by_id: dict[str, Subscription] = {}
        self._stats = MessageBusStats()
        self._max_concurrent = max_concurrent_handlers
        self._log = logger.bind(component="message_bus")

    @property
    def stats(self) -> MessageBusStats:
        return self._stats

    async def subscribe(
        self,
        pattern: str | Topic,
        handler: MessageHandler,
    ) -> str:
        """
        Subscribe to a topic pattern.

        Args:
            pattern: Topic pattern (supports wildcards)
            handler: Async function to handle messages

        Returns:
            Subscription ID for later unsubscription
        """
        pattern_str = str(pattern) if isinstance(pattern, Topic) else pattern
        sub_id = str(ULID())

        subscription = Subscription(
            id=sub_id,
            pattern=pattern_str,
            handler=handler,
        )

        self._subscriptions[pattern_str].append(subscription)
        self._subscriptions_by_id[sub_id] = subscription
        self._stats.total_subscriptions += 1

        self._log.debug("subscribed", pattern=pattern_str, subscription_id=sub_id)

        return sub_id

    async def unsubscribe(
        self,
        pattern_or_id: str | Topic,
        handler: MessageHandler | None = None,
    ) -> bool:
        """
        Unsubscribe from a topic pattern.

        Can unsubscribe by:
        - Subscription ID (exact match)
        - Pattern + handler (remove specific handler from pattern)
        - Pattern only (remove all handlers for pattern)

        Returns:
            True if any subscriptions were removed
        """
        pattern_str = str(pattern_or_id) if isinstance(pattern_or_id, Topic) else pattern_or_id

        # Check if it's a subscription ID
        if pattern_str in self._subscriptions_by_id:
            sub = self._subscriptions_by_id.pop(pattern_str)
            self._subscriptions[sub.pattern] = [
                s for s in self._subscriptions[sub.pattern] if s.id != sub.id
            ]
            self._stats.total_subscriptions -= 1
            self._log.debug("unsubscribed", subscription_id=pattern_str)
            return True

        # Otherwise treat as pattern
        if pattern_str not in self._subscriptions:
            return False

        if handler is None:
            # Remove all subscriptions for pattern
            removed = len(self._subscriptions[pattern_str])
            for sub in self._subscriptions[pattern_str]:
                self._subscriptions_by_id.pop(sub.id, None)
            del self._subscriptions[pattern_str]
            self._stats.total_subscriptions -= removed
            self._log.debug("unsubscribed_all", pattern=pattern_str, count=removed)
            return removed > 0
        else:
            # Remove specific handler
            original = self._subscriptions[pattern_str]
            self._subscriptions[pattern_str] = [
                s for s in original if s.handler != handler
            ]
            removed = len(original) - len(self._subscriptions[pattern_str])
            for sub in original:
                if sub.handler == handler:
                    self._subscriptions_by_id.pop(sub.id, None)
            self._stats.total_subscriptions -= removed
            self._log.debug("unsubscribed_handler", pattern=pattern_str, count=removed)
            return removed > 0

    async def publish(self, message: Message) -> int:
        """
        Publish a message to all matching subscribers.

        Messages are delivered concurrently to all matching handlers.
        Handler failures are isolated and logged.

        Args:
            message: Message to publish

        Returns:
            Number of handlers that received the message
        """
        self._stats.total_messages_published += 1
        topic = Topic(message.topic)

        # Find all matching subscriptions
        matching: list[Subscription] = []
        for pattern, subs in self._subscriptions.items():
            if topic.matches(pattern):
                matching.extend(subs)

        if not matching:
            self._log.debug("no_subscribers", topic=message.topic)
            return 0

        # Deliver to all handlers concurrently
        delivered = 0

        async def deliver(sub: Subscription) -> bool:
            try:
                await sub.handler(message)
                return True
            except Exception:
                self._stats.total_errors += 1
                self._log.exception(
                    "handler_failed",
                    topic=message.topic,
                    subscription_id=sub.id,
                )
                return False

        async with anyio.create_task_group() as tg:
            results: list[bool] = []

            async def track_delivery(sub: Subscription) -> None:
                result = await deliver(sub)
                results.append(result)

            for sub in matching[:self._max_concurrent]:
                tg.start_soon(track_delivery, sub)

        delivered = sum(1 for r in results if r)
        self._stats.total_messages_delivered += delivered

        self._log.debug(
            "published",
            topic=message.topic,
            message_id=message.id,
            delivered=delivered,
            total_handlers=len(matching),
        )

        return delivered

    async def publish_many(self, messages: list[Message]) -> int:
        """
        Publish multiple messages.

        Args:
            messages: Messages to publish

        Returns:
            Total number of deliveries
        """
        total = 0
        for message in messages:
            total += await self.publish(message)
        return total

    def get_subscriptions(self, pattern: str | None = None) -> list[Subscription]:
        """
        Get current subscriptions.

        Args:
            pattern: Optional pattern to filter by

        Returns:
            List of matching subscriptions
        """
        if pattern is None:
            return list(self._subscriptions_by_id.values())
        return list(self._subscriptions.get(pattern, []))

    def clear(self) -> None:
        """Remove all subscriptions."""
        count = len(self._subscriptions_by_id)
        self._subscriptions.clear()
        self._subscriptions_by_id.clear()
        self._stats.total_subscriptions = 0
        self._log.info("cleared", removed_subscriptions=count)
