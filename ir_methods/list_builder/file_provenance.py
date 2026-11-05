"""File provenance for location questions, generic to any agent record with file tools and a shell:
which file a piece of content belongs to (the path its tool call wrote or read), where that file went later
(shell mv / cp / git mv, globs and directory targets, the call's cwd or a leading cd), and which pieces state
the final path. Nothing here names a run, an agent, a folder or an argument key."""
import fnmatch, json, os, re, shlex

PATH = re.compile(r"(?<![\w.:/-])(/?(?:[\w.\-*]+/)+[\w.\-*]+/?)")


def _args(text):
    m = re.match(r"\s*[\w.]*\((\{.*\})\)\s*$", text, re.S)
    if not m:
        return {}
    try:
        a = json.loads(m.group(1))
        return a if isinstance(a, dict) else {}
    except Exception:
        return {}


def _abs(p, cwd):
    p = p.strip("'\"")
    if not p.startswith("/") and cwd:
        p = os.path.join(cwd, p)
    return os.path.normpath(p) + ("/" if p.endswith("/") else "")


def _shell_moves(cmd, cwd):
    """[(src_abs, dst_abs, is_dir_target)] for mv / cp / git mv in a shell command line."""
    out = []
    for seg in re.split(r"&&|\|\||;|\n|\|", cmd):
        try:
            argv = shlex.split(seg, posix=True)
        except ValueError:
            argv = seg.split()
        if not argv:
            continue
        if argv[0] == "cd" and len(argv) > 1:
            cwd = _abs(argv[1], cwd).rstrip("/")
            continue
        if argv[:2] == ["git", "mv"]:
            argv = argv[1:]
        if argv[0] not in ("mv", "cp"):
            continue
        args = [a for a in argv[1:] if not a.startswith("-")]
        if len(args) < 2:
            continue
        srcs, dst = args[:-1], args[-1]
        dirt = len(srcs) > 1 or dst.endswith("/") or "." not in os.path.basename(dst.rstrip("/"))
        for s in srcs:
            out.append((_abs(s, cwd).rstrip("/"), _abs(dst, cwd).rstrip("/"), dirt, argv[0]))
    return out


class Files:
    def __init__(self, events, pieces):
        self.events = events
        self.idx = {e["id"]: k for k, e in enumerate(events)}
        last_call, self.call_of = {}, {}
        for e in events:
            if e["kind"] == "tool_call":
                last_call[e["conv"]] = e
            elif e["kind"] == "tool_result" and e["conv"] in last_call:
                self.call_of[e["id"]] = last_call[e["conv"]]
        self.paths, self.moves = {}, []
        for e in events:
            if e["kind"] != "tool_call":
                continue
            a = _args(e["text"])
            cwd = a.get("Cwd") if isinstance(a.get("Cwd"), str) else None
            vals = [v for k, v in a.items() if isinstance(v, str) and len(v) < 400]
            cands = []
            for v in vals:
                cands += [m for m in PATH.findall(v) if "." in os.path.basename(m.rstrip("/"))]
            self.paths[e["id"]] = [_abs(p, cwd).rstrip("/") for p in cands][:6]
            for v in vals:
                if re.search(r"(^|[\s;&|])(mv|cp|git mv)\s", v):
                    for s, d, dirt, op in _shell_moves(v, cwd):
                        self.moves.append((self.idx[e["id"]], s, d, dirt, op, e["id"]))
        self.moves.sort()
        # pieces by the path tails they contain, for finding cheap statements of a path
        self.pieces = pieces
        self.text = {p["id"]: p["text"] for p in pieces}

    def files_of(self, event_id):
        e = self.events[self.idx[event_id]]
        call = e if e["kind"] == "tool_call" else self.call_of.get(event_id)
        return self.paths.get(call["id"], []) if call else []

    def final(self, path, after_event):
        """Where `path` ended up: follow every later move whose source matches it. -> (path, move event ids)."""
        k0, cur, via = self.idx[after_event], path, []
        for k, s, d, dirt, op, eid in self.moves:
            if k < k0:
                continue
            if s == cur or ("*" in s and fnmatch.fnmatch(cur, s)) or (s.endswith(cur) or cur.endswith(s)) and os.path.basename(s) == os.path.basename(cur):
                new = d + "/" + os.path.basename(cur) if dirt else d
                if op == "mv":
                    cur = new
                via.append(eid)
        return cur, via

    def statements(self, path, limit=4):
        """Cheapest pieces whose text holds the path's last components, longest tail first."""
        parts = path.strip("/").split("/")
        found = []
        for n in (3, 2):
            if len(parts) < n:
                continue
            tail = "/".join(parts[-n:])
            hit = [p for p in self.pieces if tail in p["text"]]
            if hit:
                found = sorted(hit, key=lambda p: p["tokens"])[:limit]
                break
        return [p["id"] for p in found]
