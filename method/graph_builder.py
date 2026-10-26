"""
graph_builder.py — Module 0: Deterministic provenance graph construction from events.jsonl + blocks.jsonl.

Implements all 5 edge types (flow, result, file, copy, ref) as described in Section 4 of the plan.
Runtime: ~5 seconds per run, CPU-only.
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ─── Pre-processing helpers ────────────────────────────────────────────────

def normalise_line(line: str) -> str:
    """Remove leading line numbers and collapse whitespace."""
    line = re.sub(r"^\d+: ?", "", line)
    return re.sub(r"\s+", " ", line).strip()


def hash_block(block_text: str) -> str:
    """SHA-1 content hash (16 hex chars) of a normalised block."""
    normalised = normalise_line(block_text)
    return hashlib.sha1(normalised.encode()).hexdigest()[:16]


# ─── Edge 1: flow ──────────────────────────────────────────────────────────

def build_flow_edges(events: list) -> list:
    """Connect consecutive events within the same conversation thread."""
    by_conv = defaultdict(list)
    for e in events:
        by_conv[e["conv"]].append(e)
    edges = []
    for conv_events in by_conv.values():
        conv_events.sort(key=lambda e: e["ts"])
        for i in range(len(conv_events) - 1):
            edges.append((conv_events[i]["id"], conv_events[i + 1]["id"]))
    return edges


# ─── Edge 2: result ────────────────────────────────────────────────────────

def build_result_edges(events: list) -> list:
    """Link each tool_call to its direct tool_result via the conv id."""
    pending = {}  # conv_id → tool_call event id
    edges = []
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["kind"] == "tool_call":
            pending[e["conv"]] = e["id"]
        elif e["kind"] == "tool_result" and e["conv"] in pending:
            edges.append((pending.pop(e["conv"]), e["id"]))
    return edges


# ─── Edge 3: file ──────────────────────────────────────────────────────────

def parse_write_path(event: dict):
    """Return the file path written by this tool_call, or None."""
    if event["kind"] != "tool_call":
        return None
    text = event["text"]
    paren = text.find("(")
    if paren == -1:
        return None
    try:
        args = json.loads(text[paren + 1: -1])
    except (ValueError, IndexError):
        return None
    return args.get("TargetFile")


def parse_echo_path(event: dict):
    """Return path from write-confirmation messages in tool_result."""
    if event["kind"] != "tool_result":
        return None
    m = re.search(r"tool to: (\S+?)\.( If|$)", event["text"])
    return m.group(1) if m else None


def parse_read_path(event: dict):
    """Return path from view_file tool_result headers."""
    if event["kind"] != "tool_result":
        return None
    if not event["text"].startswith("File Path:"):
        return None
    m = re.search(r"File Path: `([^`]+)`", event["text"])
    return m.group(1) if m else None


SHELL_READ = re.compile(r"^\s*(cat|head|tail|sed|grep|less|git\s+(diff|show|log))\s")
PATH_IN_CMD = re.compile(
    r"(?:/workspace/)?((?:repo|tracker|board|notebooks|inbox|outbox|ci)/[\w./-]+\.\w+)"
)


def parse_shell_read_paths(event: dict) -> list:
    """Return file paths read by shell commands (cat, head, tail, grep, etc.)."""
    if event["kind"] != "tool_call":
        return []
    paren = event["text"].find("(")
    try:
        args = json.loads(event["text"][paren + 1: -1])
    except (ValueError, IndexError):
        return []
    cmd = args.get("CommandLine", "")
    if not SHELL_READ.match(cmd):
        return []
    return PATH_IN_CMD.findall(cmd)


def build_file_edges(events: list) -> Tuple[list, Set[str], Set[str]]:
    """
    Build cross-agent file edges. Returns (edges, write_source_ids, read_target_ids).
    """
    def norm(path: str) -> str:
        return path.split("/workspace/")[-1]

    last_write = {}  # normalised_path → event_id
    edges = []
    write_sources: Set[str] = set()
    read_targets: Set[str] = set()

    for e in sorted(events, key=lambda e: e["ts"]):
        # Check for writes
        for path in filter(None, [parse_write_path(e), parse_echo_path(e)]):
            np = norm(path)
            last_write[np] = e["id"]
            write_sources.add(e["id"])

        # Check for reads
        read_paths = list(filter(None, [parse_read_path(e)])) + parse_shell_read_paths(e)
        for path in read_paths:
            np = norm(path)
            if np in last_write and last_write[np] != e["id"]:
                edges.append((last_write[np], e["id"]))
                read_targets.add(e["id"])

    return edges, write_sources, read_targets


# ─── Edge 4: copy ──────────────────────────────────────────────────────────

MIN_BLOCK_CHARS = 30
BOILERPLATE_THRESHOLD = 12


def build_copy_edges(events: list, blocks: list) -> Tuple[list, Set[str]]:
    """
    Build copy edges: events that quote/repeat content from earlier events.
    Returns (edges, copy_target_ids).
    """
    event_ts = {e["id"]: e["ts"] for e in events}
    event_kind = {e["id"]: e["kind"] for e in events}

    by_hash = defaultdict(list)  # hash → [(ts, event_id)]
    for b in blocks:
        if len(b["text"]) < MIN_BLOCK_CHARS:
            continue
        h = hash_block(b["text"])
        eid = b["event"]
        by_hash[h].append((event_ts.get(eid, 0), eid))

    edges = []
    copy_targets: Set[str] = set()
    for h, occurrences in by_hash.items():
        if len(occurrences) > BOILERPLATE_THRESHOLD:
            continue  # boilerplate
        occurrences.sort(key=lambda x: x[0])
        first_ts, first_eid = occurrences[0]
        if event_kind.get(first_eid) not in ("model", "tool_call"):
            continue
        for later_ts, later_eid in occurrences[1:]:
            if later_eid != first_eid:
                edges.append((first_eid, later_eid))
                copy_targets.add(later_eid)
    return edges, copy_targets


# ─── Edge 5: ref ───────────────────────────────────────────────────────────

LOCATOR_PATTERNS = {
    "path": re.compile(
        r"(?:/workspace/)?((?:repo|tracker|board|notebooks|inbox|outbox|ci)/[\w./-]+\.\w+)"
    ),
    "issue": re.compile(r"(?:#|issues?/|pulls?/|PR ?#?|issue ?#?)(\d{3,4})\b", re.I),
    "branch": re.compile(r"\b((?:fix|feat|update|chore|docs|ci|release)[-/][\w.-]{3,60})\b"),
    "commit": re.compile(r"\b([0-9a-f]{7,40})\b"),
    "test": re.compile(r"\b(test_\w{4,}|Test\w{4,})\b"),
    "error": re.compile(r"\b(\w+(?:Error|Exception|Warning)\b[^\n]{0,60})"),
}

COMMON_LOCATOR_THRESHOLD = 15
LISTING_THRESHOLD = 15


def extract_locators(text: str) -> set:
    found = set()
    for kind, pattern in LOCATOR_PATTERNS.items():
        for m in pattern.findall(text):
            val = m if isinstance(m, str) else m[0]
            if kind == "commit":
                if not (re.search(r"[a-f]", val) and re.search(r"\d", val)):
                    continue
            found.add((kind, val.strip()[:70]))
    return found


def build_ref_edges(events: list) -> list:
    """Build ref edges: events sharing specific locator mentions."""
    loc_count = defaultdict(int)
    for e in events:
        for loc in extract_locators(e["text"]):
            loc_count[loc] += 1
    common = {loc for loc, n in loc_count.items() if n > COMMON_LOCATOR_THRESHOLD}

    first_mention = {}
    edges = []
    for e in sorted(events, key=lambda e: e["ts"]):
        locs = extract_locators(e["text"]) - common
        if len(locs) > LISTING_THRESHOLD:
            continue
        for loc in locs:
            if loc not in first_mention:
                first_mention[loc] = e["id"]
            elif first_mention[loc] != e["id"]:
                edges.append((first_mention[loc], e["id"]))
    return edges


# ─── Main build_graph() ────────────────────────────────────────────────────

def build_graph(events: list, blocks: list) -> dict:
    """
    Build provenance graph from raw events.jsonl and blocks.jsonl records.

    Args:
        events: list of dicts per event (id, agent, kind, ts, shift, conv, text, tokens)
        blocks: list of dicts per block (id, event, tokens, text)

    Returns dict with:
        edges             : {type: [(src_id, dst_id), ...]} for flow/result/file/copy/ref
        in_degree         : {event_id: int}
        out_degree        : {event_id: int}
        file_write_sources: set of event_ids that wrote files
        file_read_targets : set of event_ids that read files
        copy_targets      : set of event_ids with copied-in content
        temporal_frac     : {event_id: float} in [0,1], 0=earliest, 1=latest
        agents_by_conv    : {conv_id: set of agent names}
    """
    # Build each edge type
    flow_edges = build_flow_edges(events)
    result_edges = build_result_edges(events)
    file_edges, write_sources, read_targets = build_file_edges(events)
    copy_edges, copy_targets = build_copy_edges(events, blocks)
    ref_edges = build_ref_edges(events)

    all_edges = {
        "flow":   flow_edges,
        "result": result_edges,
        "file":   file_edges,
        "copy":   copy_edges,
        "ref":    ref_edges,
    }

    # Compute in/out degrees
    in_degree: Dict[str, int] = defaultdict(int)
    out_degree: Dict[str, int] = defaultdict(int)
    for edge_list in all_edges.values():
        for src, dst in edge_list:
            out_degree[src] += 1
            in_degree[dst] += 1

    # Temporal fraction
    sorted_events = sorted(events, key=lambda e: e["ts"])
    n = len(sorted_events)
    temporal_frac = {e["id"]: i / max(1, n - 1) for i, e in enumerate(sorted_events)}

    # Agents per conversation
    agents_by_conv: Dict[str, Set[str]] = defaultdict(set)
    for e in events:
        agents_by_conv[e["conv"]].add(e["agent"])

    return dict(
        edges=all_edges,
        in_degree=dict(in_degree),
        out_degree=dict(out_degree),
        file_write_sources=write_sources,
        file_read_targets=read_targets,
        copy_targets=copy_targets,
        temporal_frac=temporal_frac,
        agents_by_conv=dict(agents_by_conv),
    )


def load_run(run_dir: Path) -> Tuple[list, list, list]:
    """Load events, blocks, and questions from a run directory."""
    events = [json.loads(l) for l in open(run_dir / "events.jsonl", encoding="utf-8")]
    blocks = [json.loads(l) for l in open(run_dir / "blocks.jsonl", encoding="utf-8")]
    questions = [json.loads(l) for l in open(run_dir / "questions.jsonl", encoding="utf-8")]
    return events, blocks, questions


def load_answer_key(run_dir: Path) -> dict:
    return json.load(open(run_dir / "answer_key.json", encoding="utf-8"))


if __name__ == "__main__":
    import sys, time
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/on_call")
    t0 = time.time()
    events, blocks, questions = load_run(run_dir)
    g = build_graph(events, blocks)
    elapsed = time.time() - t0
    total_edges = sum(len(v) for v in g["edges"].values())
    print(f"Built graph in {elapsed:.1f}s")
    print(f"  Events: {len(events)}, Blocks: {len(blocks)}")
    for etype, elist in g["edges"].items():
        print(f"  {etype:8s}: {len(elist):5d} edges")
    print(f"  Total edges: {total_edges}")
    print(f"  Write sources: {len(g['file_write_sources'])}, Read targets: {len(g['file_read_targets'])}")
    print(f"  Copy targets: {len(g['copy_targets'])}")
