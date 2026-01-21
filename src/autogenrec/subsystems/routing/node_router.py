"""
NodeRouter: Manages connections and routing between symbolic nodes.

Provides adaptive pathways and dynamic routing configuration across
the symbolic system.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from autogenrec.bus.topics import SubsystemTopics
from autogenrec.core.process import ProcessContext
from autogenrec.core.signals import Message, Signal, SignalPriority
from autogenrec.core.subsystem import Subsystem, SubsystemMetadata, SubsystemType
from autogenrec.core.symbolic import (
    SymbolicInput,
    SymbolicOutput,
    SymbolicValue,
    SymbolicValueType,
)

logger = structlog.get_logger()


class NodeType(Enum):
    """Types of nodes in the routing system."""

    SUBSYSTEM = auto()  # A subsystem node
    ENDPOINT = auto()  # External endpoint
    RELAY = auto()  # Relay/proxy node
    BROADCAST = auto()  # Broadcast node
    FILTER = auto()  # Filtering node
    AGGREGATOR = auto()  # Aggregation node
    VIRTUAL = auto()  # Virtual/logical node


class NodeStatus(Enum):
    """Status of a node."""

    ACTIVE = auto()  # Node is active and routing
    INACTIVE = auto()  # Node is registered but not active
    DEGRADED = auto()  # Node is experiencing issues
    OFFLINE = auto()  # Node is offline
    MAINTENANCE = auto()  # Node is under maintenance


class RouteType(Enum):
    """Types of routes between nodes."""

    DIRECT = auto()  # Direct point-to-point
    BROADCAST = auto()  # One-to-many broadcast
    MULTICAST = auto()  # Selective multicast
    LOAD_BALANCED = auto()  # Load-balanced across targets
    FAILOVER = auto()  # Primary with failover targets
    CONDITIONAL = auto()  # Conditional routing based on rules


class RouteStatus(Enum):
    """Status of a route."""

    ACTIVE = auto()  # Route is active
    DISABLED = auto()  # Route is disabled
    BLOCKED = auto()  # Route is blocked
    TESTING = auto()  # Route is being tested


class Node(BaseModel):
    """A node in the routing system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    node_type: NodeType = NodeType.SUBSYSTEM
    status: NodeStatus = NodeStatus.ACTIVE

    # Metadata
    description: str = ""
    subsystem_name: str | None = None  # If node represents a subsystem
    address: str | None = None  # For external endpoints
    tags: frozenset[str] = Field(default_factory=frozenset)

    # Capabilities
    can_receive: bool = True
    can_send: bool = True
    max_connections: int = 100
    supported_topics: frozenset[str] = Field(default_factory=frozenset)

    # Statistics
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages_received: int = 0
    messages_sent: int = 0
    last_activity: datetime | None = None


class Route(BaseModel):
    """A route between nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(ULID()))
    name: str
    route_type: RouteType = RouteType.DIRECT
    status: RouteStatus = RouteStatus.ACTIVE

    # Routing
    source_id: str
    target_ids: tuple[str, ...] = Field(default_factory=tuple)  # Multiple for broadcast/multicast
    topic_pattern: str = "*"  # Topic pattern to match

    # Configuration
    priority: int = Field(default=50, ge=0, le=100)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)  # For load balancing
    condition: str | None = None  # Condition expression for conditional routing

    # Metadata
    description: str = ""
    tags: frozenset[str] = Field(default_factory=frozenset)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Statistics
    messages_routed: int = 0
    last_used: datetime | None = None


@dataclass
class RoutingDecision:
    """Decision about where to route a signal."""

    source_node: str
    target_nodes: list[str]
    route_used: str | None = None
    reason: str = ""


@dataclass
class RouteMatch:
    """Result of matching a signal to routes."""

    route: Route
    priority: int
    weight: float


class RoutingTable:
    """Maintains and manages the routing table."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._routes: dict[str, Route] = {}
        self._by_source: dict[str, set[str]] = {}  # source_id -> route_ids
        self._by_target: dict[str, set[str]] = {}  # target_id -> route_ids
        self._by_topic: dict[str, set[str]] = {}  # topic -> route_ids
        self._log = logger.bind(component="routing_table")

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def route_count(self) -> int:
        return len(self._routes)

    def add_node(self, node: Node) -> None:
        """Add a node to the routing table."""
        self._nodes[node.id] = node
        self._log.debug("node_added", node_id=node.id, name=node.name)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Node | None:
        """Get a node by name."""
        for node in self._nodes.values():
            if node.name == name:
                return node
        return None

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its routes."""
        if node_id not in self._nodes:
            return False

        self._nodes.pop(node_id)

        # Remove routes involving this node
        routes_to_remove = set()
        for route_id, route in self._routes.items():
            if route.source_id == node_id or node_id in route.target_ids:
                routes_to_remove.add(route_id)

        for route_id in routes_to_remove:
            self.remove_route(route_id)

        self._log.debug("node_removed", node_id=node_id)
        return True

    def add_route(self, route: Route) -> None:
        """Add a route to the routing table."""
        # Validate nodes exist
        if route.source_id not in self._nodes:
            raise ValueError(f"Source node not found: {route.source_id}")
        for target_id in route.target_ids:
            if target_id not in self._nodes:
                raise ValueError(f"Target node not found: {target_id}")

        self._routes[route.id] = route

        # Index
        self._by_source.setdefault(route.source_id, set()).add(route.id)
        for target_id in route.target_ids:
            self._by_target.setdefault(target_id, set()).add(route.id)

        # Index by topic pattern
        self._by_topic.setdefault(route.topic_pattern, set()).add(route.id)

        self._log.debug(
            "route_added",
            route_id=route.id,
            source=route.source_id,
            targets=route.target_ids,
        )

    def get_route(self, route_id: str) -> Route | None:
        """Get a route by ID."""
        return self._routes.get(route_id)

    def remove_route(self, route_id: str) -> bool:
        """Remove a route."""
        if route_id not in self._routes:
            return False

        route = self._routes.pop(route_id)

        # Remove from indices
        if route.source_id in self._by_source:
            self._by_source[route.source_id].discard(route_id)
        for target_id in route.target_ids:
            if target_id in self._by_target:
                self._by_target[target_id].discard(route_id)
        if route.topic_pattern in self._by_topic:
            self._by_topic[route.topic_pattern].discard(route_id)

        self._log.debug("route_removed", route_id=route_id)
        return True

    def find_routes(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        topic: str | None = None,
        status: RouteStatus | None = None,
    ) -> list[Route]:
        """Find routes matching criteria."""
        candidates = set(self._routes.keys())

        if source_id:
            candidates &= self._by_source.get(source_id, set())
        if target_id:
            candidates &= self._by_target.get(target_id, set())

        routes = [self._routes[rid] for rid in candidates if rid in self._routes]

        # Filter by status
        if status:
            routes = [r for r in routes if r.status == status]

        # Filter by topic (simple wildcard matching)
        if topic:
            routes = [r for r in routes if self._topic_matches(r.topic_pattern, topic)]

        # Sort by priority
        routes.sort(key=lambda r: r.priority, reverse=True)

        return routes

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a pattern."""
        if pattern == "*" or pattern == "#":
            return True
        if pattern == topic:
            return True
        # Simple wildcard matching
        if "*" in pattern:
            parts = pattern.split("*")
            if len(parts) == 2:
                return topic.startswith(parts[0]) and topic.endswith(parts[1])
        if "#" in pattern:
            prefix = pattern.replace("#", "")
            return topic.startswith(prefix)
        return False

    def get_active_nodes(self) -> list[Node]:
        """Get all active nodes."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.ACTIVE]

    def get_active_routes(self) -> list[Route]:
        """Get all active routes."""
        return [r for r in self._routes.values() if r.status == RouteStatus.ACTIVE]


class RouteOptimizer:
    """Optimizes routing decisions."""

    def __init__(self) -> None:
        self._log = logger.bind(component="route_optimizer")

    def select_route(
        self,
        routes: list[Route],
        signal: Signal | None = None,
    ) -> Route | None:
        """Select the best route from candidates."""
        if not routes:
            return None

        # Filter active routes
        active = [r for r in routes if r.status == RouteStatus.ACTIVE]
        if not active:
            return None

        # For failover, check primary first
        failover_routes = [r for r in active if r.route_type == RouteType.FAILOVER]
        if failover_routes:
            # Return highest priority
            return max(failover_routes, key=lambda r: r.priority)

        # For load balanced, select by weight
        lb_routes = [r for r in active if r.route_type == RouteType.LOAD_BALANCED]
        if lb_routes:
            # Simple weighted selection (in production, would track load)
            return max(lb_routes, key=lambda r: r.weight)

        # Default: highest priority
        return max(active, key=lambda r: r.priority)

    def evaluate_route_health(self, route: Route, table: RoutingTable) -> float:
        """Evaluate the health of a route (0.0-1.0)."""
        # Check if source node is active
        source = table.get_node(route.source_id)
        if not source or source.status != NodeStatus.ACTIVE:
            return 0.0

        # Check target nodes
        active_targets = 0
        for target_id in route.target_ids:
            target = table.get_node(target_id)
            if target and target.status == NodeStatus.ACTIVE:
                active_targets += 1

        if not route.target_ids:
            return 0.0

        return active_targets / len(route.target_ids)


class NodeRouter(Subsystem):
    """
    Manages connections and routing between symbolic nodes.

    Process Loop:
    1. Register: Add nodes into the routing system
    2. Configure: Establish connections based on rules or requests
    3. Evaluate: Monitor routes for efficiency and integrity
    4. Adjust: Optimize connections and adapt to changes
    """

    def __init__(self) -> None:
        metadata = SubsystemMetadata(
            name="node_router",
            display_name="Node Router",
            description="Manages connections and routing between symbolic nodes",
            type=SubsystemType.ROUTING,
            tags=frozenset(["routing", "connections", "nodes", "pathways"]),
            input_types=frozenset(["SIGNAL", "MESSAGE", "SCHEMA"]),
            output_types=frozenset(["SIGNAL", "MESSAGE"]),
            subscribed_topics=frozenset([
                str(SubsystemTopics.ALL),
                "routing.#",
                "node.#",
            ]),
            published_topics=frozenset([
                "routing.node.registered",
                "routing.route.created",
                "routing.signal.routed",
                "routing.route.blocked",
            ]),
        )
        super().__init__(metadata)

        self._table = RoutingTable()
        self._optimizer = RouteOptimizer()

    @property
    def node_count(self) -> int:
        return self._table.node_count

    @property
    def route_count(self) -> int:
        return self._table.route_count

    async def intake(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput:
        """Phase 1: Register nodes and receive routing requests."""
        if not input_data.values:
            self._log.debug("empty_input")
            return input_data

        self._log.debug("intake_complete", value_count=len(input_data.values))
        return input_data

    async def process(
        self, input_data: SymbolicInput, ctx: ProcessContext[dict[str, Any]]
    ) -> list[RoutingDecision]:
        """Phase 2: Configure routes and make routing decisions."""
        decisions: list[RoutingDecision] = []

        for value in input_data.values:
            if value.type == SymbolicValueType.SCHEMA:
                # Registration request
                self._handle_registration(value)
            elif value.type == SymbolicValueType.SIGNAL:
                # Routing request
                decision = self._route_signal(value)
                if decision:
                    decisions.append(decision)

        return decisions

    async def evaluate(
        self, intermediate: list[RoutingDecision], ctx: ProcessContext[dict[str, Any]]
    ) -> tuple[SymbolicOutput, bool]:
        """Phase 3: Create output with routing decisions."""
        values: list[SymbolicValue] = []

        for decision in intermediate:
            value = SymbolicValue(
                type=SymbolicValueType.MESSAGE,
                content={
                    "source_node": decision.source_node,
                    "target_nodes": decision.target_nodes,
                    "route_used": decision.route_used,
                    "reason": decision.reason,
                },
                source_subsystem=self.name,
                tags=frozenset(["routing", "decision"]),
                meaning=f"Routed from {decision.source_node} to {len(decision.target_nodes)} targets",
                confidence=1.0 if decision.target_nodes else 0.0,
            )
            values.append(value)

        output = self.create_output(
            values=values,
            input_id=ctx.metadata.get("input_id"),
        )

        return output, False

    async def integrate(
        self, output: SymbolicOutput, ctx: ProcessContext[dict[str, Any]]
    ) -> SymbolicInput | None:
        """Phase 4: Emit routing events."""
        if self._message_bus and output.values:
            for value in output.values:
                if value.content.get("target_nodes"):
                    await self.emit_event(
                        "routing.signal.routed",
                        {
                            "source": value.content.get("source_node"),
                            "targets": value.content.get("target_nodes"),
                            "route": value.content.get("route_used"),
                        },
                    )
                else:
                    await self.emit_event(
                        "routing.route.blocked",
                        {
                            "source": value.content.get("source_node"),
                            "reason": value.content.get("reason"),
                        },
                    )

        return None

    def _handle_registration(self, value: SymbolicValue) -> None:
        """Handle node or route registration."""
        content = value.content
        if not isinstance(content, dict):
            return

        reg_type = content.get("type", "node")

        if reg_type == "node":
            node = self._parse_node(content)
            if node:
                self._table.add_node(node)
        elif reg_type == "route":
            route = self._parse_route(content)
            if route:
                try:
                    self._table.add_route(route)
                except ValueError as e:
                    self._log.warning("route_registration_failed", error=str(e))

    def _parse_node(self, content: dict[str, Any]) -> Node | None:
        """Parse a node from content."""
        try:
            node_type_str = content.get("node_type", "SUBSYSTEM")
            try:
                node_type = NodeType[node_type_str.upper()]
            except KeyError:
                node_type = NodeType.SUBSYSTEM

            return Node(
                id=content.get("id", str(ULID())),
                name=content.get("name", "unnamed"),
                node_type=node_type,
                description=content.get("description", ""),
                subsystem_name=content.get("subsystem_name"),
                address=content.get("address"),
                tags=frozenset(content.get("tags", [])),
                can_receive=content.get("can_receive", True),
                can_send=content.get("can_send", True),
                max_connections=content.get("max_connections", 100),
                supported_topics=frozenset(content.get("supported_topics", [])),
            )
        except Exception as e:
            self._log.warning("node_parse_failed", error=str(e))
            return None

    def _parse_route(self, content: dict[str, Any]) -> Route | None:
        """Parse a route from content."""
        try:
            route_type_str = content.get("route_type", "DIRECT")
            try:
                route_type = RouteType[route_type_str.upper()]
            except KeyError:
                route_type = RouteType.DIRECT

            return Route(
                id=content.get("id", str(ULID())),
                name=content.get("name", "unnamed"),
                route_type=route_type,
                source_id=content.get("source_id", ""),
                target_ids=tuple(content.get("target_ids", [])),
                topic_pattern=content.get("topic_pattern", "*"),
                priority=content.get("priority", 50),
                weight=content.get("weight", 1.0),
                condition=content.get("condition"),
                description=content.get("description", ""),
                tags=frozenset(content.get("tags", [])),
            )
        except Exception as e:
            self._log.warning("route_parse_failed", error=str(e))
            return None

    def _route_signal(self, value: SymbolicValue) -> RoutingDecision | None:
        """Route a signal through the routing table."""
        content = value.content
        if not isinstance(content, dict):
            return None

        source_name = content.get("source", value.source_subsystem)
        topic = content.get("topic", "")

        # Find source node
        source_node = self._table.get_node_by_name(source_name) if source_name else None
        if not source_node:
            return RoutingDecision(
                source_node=source_name or "unknown",
                target_nodes=[],
                reason="Source node not found",
            )

        # Find applicable routes
        routes = self._table.find_routes(
            source_id=source_node.id,
            topic=topic,
            status=RouteStatus.ACTIVE,
        )

        if not routes:
            return RoutingDecision(
                source_node=source_node.name,
                target_nodes=[],
                reason="No matching routes found",
            )

        # Select best route
        route = self._optimizer.select_route(routes)
        if not route:
            return RoutingDecision(
                source_node=source_node.name,
                target_nodes=[],
                reason="No active routes available",
            )

        # Get target node names
        target_names = []
        for target_id in route.target_ids:
            target = self._table.get_node(target_id)
            if target:
                target_names.append(target.name)

        return RoutingDecision(
            source_node=source_node.name,
            target_nodes=target_names,
            route_used=route.name,
            reason="Route matched",
        )

    # --- Message handlers ---

    async def handle_event(self, message: Message) -> None:
        """Handle incoming events."""
        if message.topic.startswith("routing."):
            self._log.debug("routing_event_received", topic=message.topic)

    async def handle_signal(self, signal: Signal) -> None:
        """Handle incoming signals for routing."""
        self._log.debug("signal_received", signal_id=signal.id)

    # --- Public API ---

    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.SUBSYSTEM,
        **kwargs: Any,
    ) -> Node:
        """Add a node to the routing system."""
        node = Node(
            name=name,
            node_type=node_type,
            description=kwargs.get("description", ""),
            subsystem_name=kwargs.get("subsystem_name"),
            address=kwargs.get("address"),
            tags=frozenset(kwargs.get("tags", [])),
            can_receive=kwargs.get("can_receive", True),
            can_send=kwargs.get("can_send", True),
            supported_topics=frozenset(kwargs.get("supported_topics", [])),
        )
        self._table.add_node(node)
        return node

    def add_route(
        self,
        name: str,
        source_id: str,
        target_ids: list[str],
        route_type: RouteType = RouteType.DIRECT,
        **kwargs: Any,
    ) -> Route:
        """Add a route between nodes."""
        route = Route(
            name=name,
            route_type=route_type,
            source_id=source_id,
            target_ids=tuple(target_ids),
            topic_pattern=kwargs.get("topic_pattern", "*"),
            priority=kwargs.get("priority", 50),
            weight=kwargs.get("weight", 1.0),
            condition=kwargs.get("condition"),
            description=kwargs.get("description", ""),
            tags=frozenset(kwargs.get("tags", [])),
        )
        self._table.add_route(route)
        return route

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._table.get_node(node_id)

    def get_node_by_name(self, name: str) -> Node | None:
        """Get a node by name."""
        return self._table.get_node_by_name(name)

    def get_route(self, route_id: str) -> Route | None:
        """Get a route by ID."""
        return self._table.get_route(route_id)

    def find_routes(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        topic: str | None = None,
    ) -> list[Route]:
        """Find routes matching criteria."""
        return self._table.find_routes(source_id, target_id, topic)

    def route(self, source_name: str, topic: str) -> list[str]:
        """Route a message and return target node names."""
        node = self._table.get_node_by_name(source_name)
        if not node:
            return []

        routes = self._table.find_routes(
            source_id=node.id,
            topic=topic,
            status=RouteStatus.ACTIVE,
        )

        route = self._optimizer.select_route(routes)
        if not route:
            return []

        targets = []
        for target_id in route.target_ids:
            target = self._table.get_node(target_id)
            if target:
                targets.append(target.name)

        return targets

    async def broadcast(self, source_name: str, topic: str, payload: Any) -> int:
        """Broadcast a message from source to all matching targets."""
        targets = self.route(source_name, topic)

        if self._message_bus and targets:
            for target in targets:
                signal = Signal(
                    source=source_name,
                    target=target,
                    payload=payload,
                    payload_type=type(payload).__name__,
                )
                await self.send_signal(f"routed.{target}", signal)

        return len(targets)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the routing system."""
        return self._table.remove_node(node_id)

    def remove_route(self, route_id: str) -> bool:
        """Remove a route."""
        return self._table.remove_route(route_id)

    def disable_route(self, route_id: str) -> bool:
        """Disable a route."""
        route = self._table.get_route(route_id)
        if not route:
            return False

        # Create new route with disabled status
        self._table.remove_route(route_id)
        new_route = Route(
            id=route.id,
            name=route.name,
            route_type=route.route_type,
            status=RouteStatus.DISABLED,
            source_id=route.source_id,
            target_ids=route.target_ids,
            topic_pattern=route.topic_pattern,
            priority=route.priority,
            weight=route.weight,
            condition=route.condition,
            description=route.description,
            tags=route.tags,
            created_at=route.created_at,
            updated_at=datetime.now(UTC),
        )
        self._table.add_route(new_route)
        return True

    def get_active_nodes(self) -> list[Node]:
        """Get all active nodes."""
        return self._table.get_active_nodes()

    def get_active_routes(self) -> list[Route]:
        """Get all active routes."""
        return self._table.get_active_routes()

    def evaluate_route_health(self, route_id: str) -> float:
        """Evaluate the health of a route."""
        route = self._table.get_route(route_id)
        if not route:
            return 0.0
        return self._optimizer.evaluate_route_health(route, self._table)

    def clear(self) -> tuple[int, int]:
        """Clear all nodes and routes. Returns (nodes_cleared, routes_cleared)."""
        nodes = self._table.node_count
        routes = self._table.route_count
        self._table = RoutingTable()
        return nodes, routes
