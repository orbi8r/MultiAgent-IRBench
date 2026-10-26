# Updated RL-IR Plan: Closing the Gap to the Benchmark Leader

## Goal

Our current best system (RL Reranker M2, 10-fold LOO-CV) achieves **above_floor = +0.300**. The benchmark leader (`c_rrf_kind_field_neigh_moves_dedup`) achieves **above_floor = +0.508**. This plan identifies the root causes of the **0.208-point gap** and proposes concrete changes to close it.

## The Gap at a Glance

| System | Score | Above Floor | Full Support |
|:---|:---:|:---:|:---:|
| **Benchmark Leader** (#1) | 0.590 | **+0.508** | — |
| **Benchmark #2** | 0.563 | +0.480 | — |
| **Structure-only baseline** (kind+dedup, no ML) | 0.466 | +0.383 | — |
| **Tuned BM25** (stemstop) | 0.356 | +0.274 | — |
| **Our M2 RL Reranker** (current best) | 0.383 | +0.300 | 40.5% |
| **Our M0+M1 First-Stage** | 0.363 | +0.280 | 36.7% |
| **Our Full System (M4)** | 0.354 | +0.271 | 35.5% |

> [!CAUTION]
> Our RL reranker (+0.300) is **below** even the non-ML structure-only baseline (+0.383) from `previous_baselines.pdf`. The problem is NOT that RL can't help — it's that our first-stage retrieval is missing the critical heuristic components that the baselines all build on.

---

## Root-Cause Diagnosis: Why We Lose 0.208 Points

The benchmark leader stacks 5 components on top of BM25+Dense RRF. Our system has **none** of them. Each component's marginal contribution (from the baselines PDF ablation):

| Component | What it does | Marginal Δ above_floor | Our system |
|:---|:---|:---:|:---:|
| **Kind multipliers** | `tool_call × 2.0`, `tool_result × 0.3`, `prompt × 0.0` | +0.10 to +0.12 | ❌ Missing. We boost `model` events instead — the opposite of what works |
| **Deduplication** | Drop blocks with identical normalised text | +0.04 to +0.06 | ❌ Missing. Duplicate boilerplate wastes budget slots |
| **Neighbour expansion** | Share the best score in the same event + agent's prev/next step | +0.03 to +0.05 | ❌ Missing. Tool results next to matching tool calls are never recovered |
| **Field-based BM25** | Second BM25 index on `agent_name + kind + file_touched` metadata | +0.02 to +0.03 | ❌ Missing |
| **Move following** | Track `mv` commands for location questions (170/498 = 34% of queries) | +0.02 to +0.04 | ❌ Missing |
| **Better dense model** | BGE-base (768-d) instead of MiniLM (384-d) | +0.02 | ❌ Using weaker model |

**Total explained gap: ~0.17 to 0.30**, which fully accounts for the observed 0.208 deficit.

> [!IMPORTANT]
> The baselines PDF states: *"Knowing the log's structure matters more than the text matcher. Swapping BM25 for dense or fused retrieval adds 0.02 to 0.06, while the step type prior, dedup and move following add 0.20 on top of BM25."*
>
> Our system tried to learn these structural patterns from scratch via RL with only ~450 training questions. The RL reranker managed +0.020 over first-stage, but it can't overcome the **recall ceiling** — if the right blocks aren't in the top-200 candidates, no reranker can fix it.

---

## Six Specific Failure Modes

### Failure 1: Kind multipliers are inverted
Our structural bonus gives `kind_is_model = +0.15` (highest), but empirically **tool calls** hold 62% of answer credit (file writes) and should be boosted ×2. Tool results should get ×0.3 for their source role. Model messages are often just reasoning that doesn't directly answer questions — the baselines give them ×0.

### Failure 2: No deduplication
Boilerplate blocks (status files, harness headers, repeated output) appear identically in many events. Without dedup, 20–40 of our top-200 candidates are duplicates that waste budget tokens and dilute recall.

### Failure 3: No neighbour expansion
When BM25 matches a `tool_call` (e.g., `write_file(TargetFile="/workspace/POSTMORTEM.md", ...)`), the adjacent `tool_result` in the same conversation is the **source** event (0.5 credit). The baselines share the tool_call's score with its result and the agent's previous/next step, effectively pulling in the entire information chain. We don't.

### Failure 4: No move following for location questions
34% of questions (170/498) are `qtype=location`. The benchmark credits the **current** location of a file after `mv` commands. Our system retrieves the original write event, but the answer is the final `mv` destination. The baselines track this chain.

### Failure 5: RL reranker gets no first-stage match signals
The scoring head input is `[q_emb ‖ b_emb ‖ struct_feats(15-d)]`. It does NOT include the BM25 score, dense cosine similarity, or RRF rank of each candidate. The reranker must re-derive text match strength from raw embeddings through a single 256-unit hidden layer with only ~450 training samples — an extremely data-inefficient task.

### Failure 6: GNN training is broken
The GNN was trained on an **identity adjacency matrix** (`torch.eye(N)`) — no edges, no message passing. The loss function (`-(blended.mean() * reward)`) pushes all logits uniformly without any ranking signal. Furthermore, expanded neighbours are never added to the output list. This is why M3 slightly degrades performance.

---

## Proposed Changes (Phased)

### Phase 1: Fix the First-Stage Retrieval (Expected: +0.15 to +0.20)

These changes implement the proven heuristic stack from the baselines, which alone should reach ~+0.43 to +0.48 above_floor — matching or approaching the leader without any RL.

---

#### Change 1: Kind Multipliers on Scores
**File**: [first_stage.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/first_stage.py)

After computing BM25 + Dense scores for each block, multiply by the step type of the parent event:

```python
KIND_MULTIPLIER = {"tool_call": 2.0, "tool_result": 0.3, "model": 0.0}

def apply_kind_prior(block_scores, blocks, events):
    """Multiply each block's retrieval score by its event's kind multiplier."""
    event_kind = {e["id"]: e["kind"] for e in events}
    for i, block in enumerate(blocks):
        eid = block["event"] if "event" in block else block["id"].split("/")[0]
        kind = event_kind.get(eid, "model")
        block_scores[i] *= KIND_MULTIPLIER.get(kind, 0.0)
    return block_scores
```

**Why `model = 0.0`?** This seems counterintuitive, but 62% of answer credit sits in file writes (tool_call events) and their results. Model messages that match the query lexically are usually the agent's reasoning, not the answer itself. Setting model to 0 forces the system to retrieve the **action** (the write/read) rather than the **commentary**.

**Expected impact**: +0.10 to +0.12 (the single largest component).

---

#### Change 2: Text Deduplication
**Files**: [first_stage.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/first_stage.py), [pipeline.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/pipeline.py)

After ranking, remove blocks whose normalised text hash matches an already-selected block:

```python
def dedup_ranked(ranked_block_ids, blocks_by_id):
    """Remove blocks with identical normalised text, keeping first occurrence."""
    seen_hashes = set()
    deduped = []
    for bid in ranked_block_ids:
        block = blocks_by_id[bid]
        h = hash_normalised(block["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(bid)
    return deduped
```

Use the same `normalise_line` + SHA-1 hashing already in [graph_builder.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/graph_builder.py). Apply both at candidate selection (top-200) and at final submission building.

**Expected impact**: +0.04 to +0.06.

---

#### Change 3: Neighbour Score Sharing / Expansion
**File**: [first_stage.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/first_stage.py) (new function)

After BM25+Dense scoring, share credit with structural neighbours:

```python
def expand_neighbours(block_scores, blocks, events):
    """
    For each block, add a fraction of the best score among:
    1. Other blocks in the same event
    2. Blocks in the agent's previous and next events (by timestamp)
    """
    # Group blocks by event
    event_blocks = defaultdict(list)  # event_id -> [(block_idx, score)]
    for i, block in enumerate(blocks):
        eid = block["event"] if "event" in block else block["id"].split("/")[0]
        event_blocks[eid].append((i, block_scores[i]))

    # Same-event sharing: each block gets max(0.5 * best_sibling_score, own_score)
    for eid, group in event_blocks.items():
        if len(group) <= 1:
            continue
        best = max(s for _, s in group)
        for idx, _ in group:
            block_scores[idx] = max(block_scores[idx], 0.5 * best)

    # Agent prev/next step sharing
    # Sort events by agent + timestamp, link consecutive pairs
    agent_events = defaultdict(list)
    for e in events:
        agent_events[e["agent"]].append(e)
    for agent, evts in agent_events.items():
        evts.sort(key=lambda e: e["ts"])
        for j in range(len(evts)):
            eid = evts[j]["id"]
            if eid not in event_blocks:
                continue
            # Look at prev and next events for this agent
            for delta in [-1, 1]:
                k = j + delta
                if 0 <= k < len(evts):
                    neid = evts[k]["id"]
                    if neid in event_blocks:
                        best_neigh = max(s for _, s in event_blocks[neid])
                        for idx, _ in event_blocks[eid]:
                            block_scores[idx] = max(block_scores[idx], 0.3 * best_neigh)

    return block_scores
```

**Expected impact**: +0.03 to +0.05. This is the mechanism that recovers tool results sitting next to matched tool calls.

---

#### Change 4: Move Following for Location Questions
**File**: New function in [first_stage.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/first_stage.py) or [pipeline.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/pipeline.py)

For questions with `qtype == "location"`, trace `mv` / `cp` commands in tool_call events to find the final file location:

```python
MV_PATTERN = re.compile(r'\b(?:mv|cp|rename|shutil\.move)\s+["\']?(\S+)["\']?\s+["\']?(\S+)')

def follow_moves(events):
    """Build a mapping: original_path -> final_path after all mv/cp commands."""
    moves = {}  # path -> final_path
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["kind"] != "tool_call":
            continue
        for m in MV_PATTERN.finditer(e["text"]):
            src, dst = m.group(1), m.group(2)
            # Follow chain: if src was already moved from somewhere, update
            original = next((k for k, v in moves.items() if v == src), src)
            moves[original] = dst
    return moves

def boost_move_targets(ranked_ids, block_scores, blocks, events, question):
    """For location questions, boost blocks mentioning the final destination of moved files."""
    if question.get("qtype") != "location":
        return block_scores
    move_map = follow_moves(events)
    final_paths = set(move_map.values())
    for i, block in enumerate(blocks):
        # If block text mentions a final destination path, boost it
        for path in final_paths:
            if path in block["text"] or path.split("/")[-1] in block["text"]:
                block_scores[i] = max(block_scores[i], block_scores[i] * 1.5)
                break
    return block_scores
```

**Expected impact**: +0.02 to +0.04 (affects 170/498 = 34% of questions).

---

#### Change 5: Field-Based BM25 Index
**File**: [first_stage.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/first_stage.py)

Build a second BM25 index on structured metadata fields per block:

```python
def build_field_text(block, event):
    """Construct a metadata field string for field-based BM25."""
    parts = [event["agent"]]                    # agent name
    parts.append(event["kind"])                  # step type
    if event["kind"] == "tool_call":
        # Extract tool name and target file
        paren = event["text"].find("(")
        if paren > 0:
            parts.append(event["text"][:paren])  # tool name
        # Extract file path from args
        path = parse_write_path(event) or ""
        if path:
            parts.append(path.split("/")[-1])    # filename
    elif event["kind"] == "tool_result":
        path = parse_read_path(event) or ""
        if path:
            parts.append(path.split("/")[-1])
    return " ".join(parts)
```

Then fuse this field BM25 as a 4th signal in RRF:
```python
rrf_fuse([bm25_ranked, dense_ranked, field_ranked, struct_ranked],
         weights=[1.0, 1.0, 0.5, 0.5])
```

**Expected impact**: +0.02 to +0.03.

---

### Phase 2: Fix the RL Reranker (Expected: +0.02 to +0.05 on top of Phase 1)

With Phase 1 providing a strong ~+0.45 first-stage, the RL reranker can now learn to fine-tune the ranking rather than trying to learn everything from scratch.

---

#### Change 6: Add First-Stage Match Features to the Scoring Head
**Files**: [reranker.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/reranker.py), [pipeline.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/pipeline.py)

Augment the 15-d structural features with 4 additional retrieval-match signals per candidate:

| Feature | Description |
|:---|:---|
| `bm25_score_norm` | Normalised BM25 score (divided by max in candidate set) |
| `dense_cosine` | Raw cosine similarity between query and block embeddings |
| `rrf_rank_norm` | `1.0 - rank/200` (RRF position signal) |
| `kind_multiplied_score` | Post-kind-multiplier score (from Change 1) |

This gives the reranker explicit text-match signals instead of forcing it to re-derive them from raw 384-d embeddings. The struct dimension changes from 15 → 19 (update `RL_STRUCT_DIM` in config).

Additionally, add **explicit query-document interaction features**:
```python
# Element-wise product captures "how aligned are these vectors"
interaction = q_emb * b_emb  # (384,)
# Concatenate: [q_emb | b_emb | interaction | struct_feats]
# New input dim: 384 + 384 + 384 + 19 = 1171
```

**Expected impact**: +0.02 to +0.03 (better use of limited training signal).

---

#### Change 7: Fix RL Training Stability
**File**: [rl_trainer.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/rl_trainer.py)

Three fixes:

**7a. Mini-batch gradient accumulation** (instead of per-question SGD):
```python
# Accumulate gradients over 8 questions before stepping
ACCUM_STEPS = 8
for i, item in enumerate(train_data):
    loss = compute_rl_loss(item)
    (loss / ACCUM_STEPS).backward()
    if (i + 1) % ACCUM_STEPS == 0:
        clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
```

**7b. Faster EMA baseline warm-up**: Change momentum from 0.99 → 0.9 for the first 10 epochs, then 0.99. This lets the baseline track the policy's rapid early improvement.

**7c. Top-K truncated reward**: Only score the top-50 items of each sampled permutation (the rest never fit in any budget tier and add pure noise to the gradient):
```python
# Only rank the top-50 items; the rest have zero budget impact
ranked_ids = [block_ids[i] for i in perm.detach().tolist()[:50]]
result = score_query(answer_entry, ranked_ids, tokens_map)
```

**Expected impact**: +0.01 to +0.02 (reduces gradient variance, prevents the reward oscillation we observed in the 80-epoch master training).

---

### Phase 3: Fix or Remove GNN and Budget Selector

#### Change 8: Fix GNN Training (if keeping M3)
**File**: [train.py](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/train.py) (`_train_gnn` function)

The GNN training has **three critical bugs**:
1. Adjacency matrix is `torch.eye(N)` — no edges, no message passing during training
2. Loss function `-(blended.mean() * reward)` has no ranking signal
3. Expanded neighbours are discarded from the output

Fix: Build the real adjacency from provenance graph edges. Use a listwise ranking loss (e.g., ListMLE or the same REINFORCE objective as M2). Add expanded neighbours to the final candidate output.

**Alternatively: remove M3 entirely** for now and focus Phase 1+2 effort, since the baselines show that neighbour expansion at the first-stage level (Change 3) captures the same signal more reliably.

#### Change 9: Fix Budget Selector (M4) or Remove It
The baselines PDF states: *"Budget aware ordering adds nothing once duplicates are removed."*

After adding deduplication (Change 2), the budget selector should either:
- **Become a no-op** (just pass through the reranked order), or
- **Use proper per-budget greedy packing** — run the `cut()` function from `score.py` at each budget tier and optimise ordering to maximise multi-tier coverage.

Current M4 hurts by 0.030. Simplest fix: disable it (already the case when `use_budget_selection=False`).

---

## Implementation Priority and Expected Cumulative Impact

| Priority | Change | Estimated Δ above_floor | Cumulative | Effort |
|:---:|:---|:---:|:---:|:---:|
| **P0** | 1. Kind multipliers | +0.10 to +0.12 | ~0.39 | 30 min |
| **P0** | 2. Deduplication | +0.04 to +0.06 | ~0.44 | 1 hour |
| **P0** | 3. Neighbour expansion | +0.03 to +0.05 | ~0.48 | 1 hour |
| **P1** | 4. Move following | +0.02 to +0.04 | ~0.50 | 1.5 hours |
| **P1** | 5. Field-based BM25 | +0.02 to +0.03 | ~0.52 | 1 hour |
| **P2** | 6. Reranker match features | +0.02 to +0.03 | ~0.54 | 1.5 hours |
| **P2** | 7. RL training stability | +0.01 to +0.02 | ~0.55 | 1 hour |
| **P3** | 8. Fix/remove GNN | — | — | 1 hour |
| **P3** | 9. Fix/remove budget selector | — | — | 30 min |

**Optimistic target**: above_floor **~0.50 to 0.55** (matching or slightly exceeding the leader at +0.508).

**Conservative target**: above_floor **~0.43 to 0.48** (within the top-10 leaderboard).

> [!IMPORTANT]
> Phase 1 (Changes 1–5) alone should get us to **~0.48 to 0.52** — matching or approaching the leader — without any RL at all. Phase 2 then adds RL fine-tuning on top of the strong first-stage to push beyond what hand-tuned heuristics can achieve.

---

## User Review Required

> [!WARNING]
> **Breaking change: Kind multiplier `model = 0.0`**
> Setting model messages to score 0.0 is counterintuitive but empirically validated by the baselines sweep. This means the first-stage retriever will never retrieve model messages on their own — only through neighbour expansion from adjacent tool calls/results. If this feels too aggressive, an alternative is `model = 0.1` (the baselines tried this; it's 0.01 worse).

> [!IMPORTANT]
> **Dense model upgrade question**: The baseline uses `bge-base-en-v1.5` (768-d, ~440MB). Our current MiniLM (384-d, ~120MB) fits easily on the RTX 3050's 3.7GB VRAM. BGE-base would roughly double embedding memory but should still fit. Should we upgrade? Expected gain: ~+0.02 but with higher memory cost and slower encoding. **Recommendation**: Keep MiniLM for now; the structural changes provide 10× more impact per effort.

---

## Open Questions

> [!IMPORTANT]
> 1. **Should we keep the GNN (M3)?** It's currently broken and net-negative. Options:
>    - (a) Fix it properly with real adjacency and ranking loss (1+ hour effort, uncertain gain)
>    - (b) Remove it entirely and rely on neighbour expansion at the first stage (simpler, proven by baselines)
>    - (c) Keep it disabled for now, revisit after Phase 1+2 results

> [!IMPORTANT]
> 2. **Should we keep the budget selector (M4)?** The baselines show it adds nothing once dedup is in place. Options:
>    - (a) Remove it (pass through reranked order)
>    - (b) Redesign it to do proper multi-tier greedy packing

> [!IMPORTANT]
> 3. **Training protocol**: After implementing Phase 1, do we:
>    - (a) Re-run 10-fold CV with RL on the improved first stage (~15 min)
>    - (b) First evaluate the improved first stage alone (no RL) to measure the heuristic gain, then add RL
>    - (c) Both: evaluate first-stage-only first, then add RL

---

## Verification Plan

### Automated Tests
After each phase:
```bash
# Quick single-run test (on_call, ~2 min)
python3 -m method.train --fold on_call --epochs 1 --quick

# Full 10-fold CV evaluation (~15 min)
python3 -m method.train --epochs 40 --samples 8

# Score a single submission against the key
python3 score.py runs/on_call method/results/submission_on_call.json
```

### Key Metrics to Track
For each change, measure:
1. **First-stage Recall@200**: What fraction of primary (1.0) and source (0.5) nugget items are in the top-200 candidates?
2. **First-stage above_floor** (no RL): Does the heuristic stack alone beat the baselines?
3. **RL reranker above_floor**: Does RL add on top of the improved first stage?
4. **Per-stratum breakdown**: Especially `qtype=location` (should improve with move following) and `agents=several` (should improve with neighbour expansion)

### Manual Verification
- Inspect 5–10 submissions side-by-side with the answer key to verify that the kind multiplier is surfacing tool calls/results over model messages
- Check that dedup is removing genuine duplicates (not distinct blocks)
- Verify move following traces the correct mv chains on at least 3 location questions
