"""Contextual pieces: every retrievable piece (block, or event without blocks) with the context a reader of the
record would have when looking at it: who wrote it, what kind of step, which command or file produced it,
and the head of its event. Built only from the record, never from keys."""
import json, re
from collections import defaultdict
import snowballstemmer

_stem = snowballstemmer.stemmer("english")
STOP = set("a an the of to in on at for by with from and or is are was were be been it its this that these those what which who whom when where why how did do does done we our us i my me you your they their them he she his the original as into than then there here about after before during over under up out not no yes so if any all each per vs".split())


def toks(text):
    """Words with identifiers split (prices_backup_shift28 -> prices backup shift 28, camelCase too), stemmed."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    out = []
    for w in _WRE.findall(text):
        w = w.lower()
        if w in STOP:
            continue
        s = _CACHE.get(w)
        if s is None:
            s = _CACHE[w] = _stem.stemWord(w)
        out.append(s)
    return out


_WRE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_CACHE = {}  # the vocabulary is small, so stemming each distinct word once removes most of the tokenizing time


def call_summary(text, n=300):
    """Compact form of a tool call: name plus the argument values that say what it touched."""
    m = re.match(r"\s*([\w.]*)\((\{.*)", text, re.S)
    if not m:
        return text[:n]
    name, rest = m.groups()
    try:
        args = json.loads(rest[:-1] if rest.endswith(")") else rest)
        keep = [f"{k}={v}" for k, v in args.items() if isinstance(v, str) and k not in ("Cwd", "WaitMsBeforeAsync")]
        return (name + " " + " | ".join(keep))[:n]
    except Exception:
        return (name + " " + rest)[:n]


def build(events, blocks):
    """-> list of dict(id, event, text, ctx, tokens, kind, agent, shift, ts, order)."""
    by_event = defaultdict(list)
    for b in blocks:
        by_event[b["event"]].append(b)
    last_call = {}
    producer = {}
    for e in events:  # events are in time order
        if e["kind"] == "tool_call":
            last_call[e["conv"]] = e
        elif e["kind"] == "tool_result" and e["conv"] in last_call:
            producer[e["id"]] = last_call[e["conv"]]
    pieces = []
    for k, e in enumerate(events):
        head = f"{e['agent']} {e['kind']} shift {e['shift']}"
        if e["kind"] == "tool_call":
            head += " | " + call_summary(e["text"], 200)
        elif e["id"] in producer:
            head += " | result of " + call_summary(producer[e["id"]]["text"], 200)
        bl = by_event.get(e["id"]) or [dict(id=e["id"], text=e["text"], tokens=e["tokens"])]
        ev_head = e["text"][:160]
        for j, b in enumerate(bl):
            prev = bl[j - 1]["text"][-200:] if j else ""
            ctx = head + "\n" + (ev_head + "\n" if j else "") + prev + "\n" + b["text"]
            pieces.append(dict(id=b["id"], event=e["id"], text=b["text"], ctx=ctx, tokens=b["tokens"], kind=e["kind"],
                               agent=e["agent"], shift=e["shift"], ts=e["ts"], order=k, j=j))
    return pieces
