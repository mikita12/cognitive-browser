"""
Domain models and data contracts for Cognitive Browser.
Defines strongly-typed structures connecting the ARIA Parser, Neural Scorer/Planner,
PayloadBinder, RouteCache, BrowserOrchestrator, and Playwright BrowserEngine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

ActionType = Literal[
    "click",
    "fill",
    "fill_and_submit",
    "select",
    "check",
    "uncheck",
    "upload_file",
    "scroll",
    "press",
    "hover",
    "goto",
]

SectionType = Literal["main", "header", "footer", "body"]


@dataclass
class CandidateElement:
    """
    Represents a single interactive element extracted from the page's Accessibility Tree (ARIA).
    Supports both attribute access (cand.role) and dictionary-style access (cand['role'], cand.get('role'))
    for seamless backward compatibility.
    """

    ref: str
    role: str
    name: str
    text: str = ""
    context: str = ""
    nth: int = 0
    url: Optional[str] = None
    placeholder: Optional[str] = None
    state: Optional[str] = None
    raw: str = ""
    section: SectionType = "body"
    confidence_score: Optional[float] = None

    def __post_init__(self) -> None:
        self.role = (self.role or "button").lower().strip()
        self.name = (self.name or "").strip()
        if not self.text:
            self.text = self.name or self.role
        if self.section == "body" and self.context:
            ctx_low = self.context.lower()
            if "contentinfo" in ctx_low or "footer" in ctx_low:
                self.section = "footer"
            elif "banner" in ctx_low or "header" in ctx_low:
                self.section = "header"
            elif "main" in ctx_low:
                self.section = "main"

    @property
    def tag(self) -> str:
        return self.role

    @property
    def is_main(self) -> bool:
        return self.section == "main" or "main" in self.context.lower()

    @property
    def is_footer(self) -> bool:
        return self.section == "footer" or any(k in self.context.lower() for k in ("contentinfo", "footer"))

    @property
    def is_header(self) -> bool:
        return self.section == "header" or any(k in self.context.lower() for k in ("banner", "header"))

    @property
    def is_input(self) -> bool:
        return self.role in ("textbox", "searchbox", "combobox", "spinbutton")

    @property
    def is_boolean_control(self) -> bool:
        return self.role in ("checkbox", "radio", "switch")

    def format_for_model(self) -> str:
        """
        Returns the standardized text representation of this element for Cross-Encoder / NLI scoring.
        """
        label = self.text or self.name or self.role
        if self.context:
            return f"[{self.role}] {label} (context: {self.context})"
        return f"[{self.role}] {label}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the candidate into a plain dictionary compatible with MCP/REST responses."""
        data: Dict[str, Any] = {
            "ref": self.ref,
            "role": self.role,
            "tag": self.role,
            "name": self.name,
            "text": self.text,
            "context": self.context,
            "nth": self.nth,
            "raw": self.raw,
            "section": self.section,
            "is_main": self.is_main,
            "is_footer": self.is_footer,
            "is_header": self.is_header,
        }
        if self.url is not None:
            data["url"] = self.url
        if self.placeholder is not None:
            data["placeholder"] = self.placeholder
        if self.state:
            data["state"] = self.state
        if self.confidence_score is not None:
            data["confidence_score"] = self.confidence_score
        return data

    @classmethod
    def from_dict(cls, data: Union["CandidateElement", Dict[str, Any]]) -> "CandidateElement":
        """Creates a CandidateElement from a dictionary or returns it if already an instance."""
        if isinstance(data, cls):
            return data
        ctx = str(data.get("context") or "")
        section: SectionType = "body"
        if data.get("section") in ("main", "header", "footer", "body"):
            section = data["section"]
        elif data.get("is_footer") or "contentinfo" in ctx.lower() or "footer" in ctx.lower():
            section = "footer"
        elif data.get("is_header") or "banner" in ctx.lower() or "header" in ctx.lower():
            section = "header"
        elif data.get("is_main") or "main" in ctx.lower():
            section = "main"

        return cls(
            ref=str(data.get("ref") or "aria-0"),
            role=str(data.get("role") or data.get("tag") or "button"),
            name=str(data.get("name") or ""),
            text=str(data.get("text") or data.get("name") or ""),
            context=ctx,
            nth=int(data.get("nth", 0) or 0),
            url=data.get("url"),
            placeholder=data.get("placeholder"),
            state=data.get("state"),
            raw=str(data.get("raw") or ""),
            section=section,
            confidence_score=data.get("confidence_score"),
        )

    # Dictionary compatibility protocol
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.to_dict().keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(self.to_dict().items())

    def values(self) -> Iterator[Any]:
        return iter(self.to_dict().values())


@dataclass
class ActionCommand:
    """
    Unified command contract between decision layers (Planner, PayloadBinder, RouteCache)
    and the Playwright execution layer (BrowserEngine).
    """

    action: str
    target: Optional[CandidateElement] = None
    role: str = "button"
    name: str = ""
    nth: int = 0
    value: Optional[str] = None
    url: Optional[str] = None
    files: Optional[List[str]] = None
    direction: str = "down"
    amount: int = 600
    key: str = "Enter"
    confidence: float = 1.0
    source: str = "reflex"
    payload_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target is not None:
            if isinstance(self.target, dict):
                self.target = CandidateElement.from_dict(self.target)
            self.role = self.target.role
            self.name = self.target.name
            self.nth = self.target.nth
            if self.url is None:
                self.url = self.target.url

    def to_dict(self) -> Dict[str, Any]:
        """Converts ActionCommand into the legacy action_plan dictionary format."""
        plan: Dict[str, Any] = {
            "action": self.action,
            "role": self.role,
            "name": self.name,
            "nth": self.nth,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }
        if self.value is not None:
            plan["value"] = self.value
        if self.url is not None:
            plan["url"] = self.url
        if self.files is not None:
            plan["file_paths"] = self.files
            plan["files"] = self.files
        if self.action == "scroll":
            plan["direction"] = self.direction
            plan["amount"] = self.amount
        if self.action == "press":
            plan["key"] = self.key
        if self.payload_key is not None:
            plan["payload_key"] = self.payload_key
        return plan

    @classmethod
    def from_dict(cls, data: Union["ActionCommand", Dict[str, Any]]) -> "ActionCommand":
        if isinstance(data, cls):
            return data
        files = data.get("files") or data.get("file_paths")
        if isinstance(files, str):
            files = [files]
        return cls(
            action=data.get("action", "click"),
            role=str(data.get("role") or "button"),
            name=str(data.get("name") or ""),
            nth=int(data.get("nth", 0) or 0),
            value=data.get("value") if data.get("value") is not None else data.get("fill_value"),
            url=data.get("url"),
            files=list(files) if files else None,
            direction=str(data.get("direction") or "down"),
            amount=int(data.get("amount", 600) or 600),
            key=str(data.get("key") or "Enter"),
            confidence=float(data.get("confidence", 1.0) or 1.0),
            source=str(data.get("source") or "reflex"),
            payload_key=data.get("payload_key"),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()


@dataclass
class PageState:
    """Snapshot of the current browser page state."""

    url: str
    title: str
    snapshot_yaml: str
    candidates: List[CandidateElement] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of executing a single step or hop in BrowserOrchestrator."""

    instruction: str
    executed: bool
    status: str = "success"
    command: Optional[ActionCommand] = None
    target_desc: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    from_cache: bool = False
    error: Optional[str] = None
    clarification_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "instruction": self.instruction,
            "executed": self.executed,
            "status": self.status,
            "target": self.target_desc or (f"[{self.command.role}] {self.command.name}" if self.command else "NONE"),
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.command:
            res["action"] = self.command.action
        if self.from_cache:
            res["from_cache"] = True
        if self.error:
            res["error"] = self.error
        if self.clarification_data:
            res.update(self.clarification_data)
        return res
