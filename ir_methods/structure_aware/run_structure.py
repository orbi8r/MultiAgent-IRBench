"""Structure of a run for the full3 study: nodes (event, file, agent+shift, section) over the pieces and prefix strings that put that structure in front of the reader. Keyless: uses only pieces and events, never an answer key. Works the same on real runs (data/<run>/) and public pseudo runs (load_pub).

    S = load_run("online_shop")            # data/<run>/pieces.jsonl and data/<run>/events.jsonl
    S = load_pub(piece_dir, event_dir)     # pseudo run, the two files may live in different dirs
    S.info[pid]          event order j nsib agent shift files paths heading pos decile prev next
    S.members(ntype, nid) -> [pid]         S.nodes_of(pid) -> {ntype: [nid]}
    S.prefix(pid, "P1"|"P2"|"P3")          one line, no newline; the scored text is prefix + newline + ctx
"""
import json, os, re
from collections import defaultdict
from file_provenance import Files

NODE_TYPES = ("event", "file", "as", "sec")
STYLES = ("P1", "P2", "P3")
HEAD = re.compile(r"^\s{0,3}#{1,6}\s+(\S.*)$")
WIN = 40  # events of one (agent, shift) per section window when a block has no heading


def ws(s):
    return " ".join(s.split())


def _jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def last_heading(text):
    h = ""
    for line in text.split("\n"):
        m = HEAD.match(line)
        if m:
            h = ws(m.group(1))
    return h


class Struct:
    def __init__(self, pieces, events):
        P = sorted(pieces, key=lambda p: (p["order"], p["j"]))
        nev = max(len(events), max((p["order"] for p in P), default=0) + 1)
        F = Files(events, P)
        by_ev, stream = defaultdict(list), defaultdict(list)
        for p in P:
            by_ev[p["event"]].append(p)
            stream[(p["agent"], p["shift"])].append(p)
        self.P, self.nev = P, nev
        self.text = {p["id"]: p["text"] for p in P}
        self.info = {}
        for ev, ps in by_ev.items():
            paths = list(dict.fromkeys(F.files_of(ev))) if ev in F.idx else []
            head = ""
            for p in ps:
                head = last_heading(p["text"]) or head  # a heading holds for the later blocks of its event
                self.info[p["id"]] = dict(event=ev, order=p["order"], j=p["j"], nsib=len(ps), agent=p["agent"], shift=p["shift"],
                                          paths=paths, files=[os.path.basename(x) for x in paths], heading=head,
                                          pos=p["order"] / nev, decile=min(9, int(10 * p["order"] / nev)))
        self.mem = {t: defaultdict(list) for t in NODE_TYPES}
        self.nod = {p["id"]: {} for p in P}
        for key, ps in stream.items():
            rank, last = -1, None
            for k, p in enumerate(ps):
                i = self.info[p["id"]]
                if p["event"] != last:
                    rank, last = rank + 1, p["event"]
                i["prev"] = ps[k - 1]["id"] if k else None
                i["next"] = ps[k + 1]["id"] if k + 1 < len(ps) else None
                sec = f"{i['event']}#{i['heading'][:80]}" if i["heading"] else f"w|{key[0]}|{key[1]}|{rank // WIN}"
                nodes = dict(event=[i["event"]], file=i["paths"], **{"as": [f"{key[0]}|{key[1]}"], "sec": [sec]})
                for t, ids in nodes.items():
                    self.nod[p["id"]][t] = ids
                    for n in ids:
                        self.mem[t][n].append(p["id"])

    def members(self, ntype, nid):
        return self.mem[ntype].get(nid, [])

    def nodes_of(self, pid):
        return self.nod[pid]

    def prefix(self, pid, style):
        i = self.info[pid]
        s = f"[file={i['files'][0][:40] if i['files'] else 'none'} part={i['j'] + 1}/{i['nsib']} t={i['decile']}]"
        if style == "P1":
            return s
        s += f" sec={i['heading'][:40] or 'none'} ev={i['order']}"
        if style == "P2":
            return s
        nxt = self.text.get(i["next"], "") if i["next"] else ""
        return s + f" next={ws(nxt)[:60] or 'none'}"

    def stats(self):
        out = dict(pieces=len(self.P), events=self.nev, with_heading=sum(1 for i in self.info.values() if i["heading"]),
                   with_file=sum(1 for i in self.info.values() if i["paths"]))
        for t in NODE_TYPES:
            sz = sorted(len(v) for v in self.mem[t].values())
            out[t] = dict(nodes=len(sz), multi=sum(1 for x in sz if x > 1), max=sz[-1] if sz else 0, mean=round(sum(sz) / len(sz), 2) if sz else 0)
        return out


def load_run(name, root=None):
    d = f"{root or os.environ.get('DATAROOT') or 'data'}/{name}"
    return Struct(_jl(f"{d}/pieces.jsonl"), _jl(f"{d}/events.jsonl"))


def load_pub(dirpath, evdir=None):
    """dirpath may be a (piece_dir, event_dir) pair; otherwise events are read from evdir or from dirpath."""
    if isinstance(dirpath, (list, tuple)):
        dirpath, evdir = dirpath
    return Struct(_jl(os.path.join(dirpath, "pieces.jsonl")), _jl(os.path.join(evdir or dirpath, "events.jsonl")))
