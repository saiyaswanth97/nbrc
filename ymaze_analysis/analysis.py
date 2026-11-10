"""Y-maze scoring: arm state per frame, arm entries, alternation, novel-arm preference.

Pure computation, no I/O. Positions are smoothed centroids in video pixels.
"""

import numpy as np

from .geometry import ARMS, arm_frames


def arm_depths(x, y, maze):
    """Per frame and arm: how far along that arm the point is, as a fraction of
    centre-to-tip length (0 = junction centre, 1 = tip). Returns dict arm -> array."""
    c = np.asarray(maze["center"], float)
    p = np.stack([x, y], 1) - c
    return {a: (p @ u) / L for a, (u, L) in arm_frames(maze).items()}


def arm_states(x, y, maze, entry_frac=0.33, exit_frac=0.2):
    """Classify every frame as 'A', 'B', 'C' or 'center' with hysteresis.

    An arm is entered when the body centre is at least ``entry_frac`` of the way
    from the junction centre to that arm's tip. The animal counts as back in the
    centre only once it is below ``exit_frac`` of the way along the arm it was in,
    so wobble around a single line can't create extra entries. Frames with no
    position (before the animal is first seen) are None.
    """
    depth = arm_depths(x, y, maze)
    n = len(x)
    states = [None] * n
    cur = None
    for i in range(n):
        if np.isnan(x[i]):
            states[i] = cur
            continue
        d = {a: depth[a][i] for a in ARMS}
        deepest = max(ARMS, key=lambda a: d[a])
        if cur in ARMS:
            if deepest != cur and d[deepest] >= entry_frac:
                cur = deepest  # ran straight into another arm
            elif d[cur] < exit_frac:
                cur = "center"
        else:
            cur = deepest if d[deepest] >= entry_frac else "center"
        states[i] = cur
    return states


def arm_entries(states, fps):
    """List of entries: dicts with arm, start frame/time and duration (s)."""
    entries = []
    prev = None
    for i, s in enumerate(states):
        if s in ARMS and s != prev:
            entries.append({"arm": s, "frame": i, "time_s": i / fps})
        if entries and prev == entries[-1]["arm"] and s != prev:
            entries[-1]["duration_s"] = (i - entries[-1]["frame"]) / fps
        prev = s
    if entries and "duration_s" not in entries[-1]:
        entries[-1]["duration_s"] = (len(states) - entries[-1]["frame"]) / fps
    return entries


def spontaneous_alternation(sequence):
    """Alternation = three consecutive entries into three different arms.

    Returns (alternations, possible, percent) where possible = entries - 2.
    Re-entering the same arm counts as an entry (and breaks alternation).
    """
    triples = [sequence[i:i + 3] for i in range(len(sequence) - 2)]
    alternations = sum(1 for t in triples if len(set(t)) == 3)
    possible = len(triples)
    pct = 100.0 * alternations / possible if possible else float("nan")
    return alternations, possible, pct


def novel_arm_metrics(states, entries, fps, novel_arm):
    """Preference for the arm that was blocked during training."""
    arm_frames_n = {a: sum(1 for s in states if s == a) for a in ARMS}
    total_arm = sum(arm_frames_n.values())
    n_entries = {a: sum(1 for e in entries if e["arm"] == a) for a in ARMS}
    total_entries = sum(n_entries.values())
    first_novel = next((e["time_s"] for e in entries if e["arm"] == novel_arm), None)
    t0 = next((i for i, s in enumerate(states) if s is not None), 0) / fps
    choice = first_choice(states, entries)
    return {
        "novel_arm": novel_arm,
        "novel_time_s": arm_frames_n[novel_arm] / fps,
        "familiar_time_s": (total_arm - arm_frames_n[novel_arm]) / fps,
        "novel_time_pct": 100.0 * arm_frames_n[novel_arm] / total_arm if total_arm else float("nan"),
        "novel_entries_pct": 100.0 * n_entries[novel_arm] / total_entries if total_entries else float("nan"),
        "first_choice_is_novel": choice == novel_arm,
        "latency_to_novel_s": (first_novel - t0) if first_novel is not None else None,
    }


def start_arm(states):
    """Arm the rat is in when analysis starts (where it was placed), else None."""
    first = next((s for s in states if s is not None), None)
    return first if first in ARMS else None


def first_choice(states, entries):
    """First arm entered after leaving the start arm."""
    sa = start_arm(states)
    rest = entries[1:] if (entries and sa and entries[0]["arm"] == sa and entries[0]["frame"] == 0) else entries
    return rest[0]["arm"] if rest else None


def score(x, y, maze, fps, entry_frac=0.33, exit_frac=0.2, novel_arm=None):
    """Full scoring for one session. Returns (metrics dict, states, entries)."""
    states = arm_states(x, y, maze, entry_frac, exit_frac)
    entries = arm_entries(states, fps)
    seq = [e["arm"] for e in entries]
    alt, possible, alt_pct = spontaneous_alternation(seq)
    seen = [s for s in states if s is not None]
    m = {
        "analysed_s": len(seen) / fps,
        "total_entries": len(entries),
        "arm_sequence": "".join(seq),
        "alternations": alt,
        "possible_alternations": possible,
        "alternation_pct": alt_pct,
        "center_time_s": sum(1 for s in seen if s == "center") / fps,
        "start_arm": start_arm(states),
        "first_choice": first_choice(states, entries),
    }
    for a in ARMS:
        m[f"time_{a}_s"] = sum(1 for s in seen if s == a) / fps
        m[f"entries_{a}"] = sum(1 for e in entries if e["arm"] == a)
    if novel_arm:
        m.update(novel_arm_metrics(states, entries, fps, novel_arm))
    return m, states, entries
