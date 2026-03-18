# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Logger
# ──────────────────────────────────────────────────────────────────────────────

import json
from datetime import datetime

class TrajectoryLogger:
    """
    Records per-episode failure details and aggregates success rates
    keyed by (instruction, object_type, bowl_type).

    Call .log_episode() after every episode.
    Call .save() to flush everything to disk.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"trajectory_log_{timestamp}.json")
        self._stats: dict  = {}   # key → {attempts, successes, failures:[...]}

    # ── public API ────────────────────────────────────────────────────────────

    def log_episode(
        self,
        *,
        instruction:  str,
        object_type:  str,
        bowl_type:    str,
        seed:         int,
        success:      bool,
        final_state:  int,
        pick_pos:     np.ndarray,
        place_pos:    np.ndarray,
        steps_taken:  int,
    ):
        key = self._key(instruction, object_type, bowl_type)
        if key not in self._stats:
            self._stats[key] = {
                "instruction":  instruction,
                "object_type":  object_type,
                "bowl_type":    bowl_type,
                "attempts":     0,
                "successes":    0,
                "success_rate": 0.0,
                "failures":     [],
            }

        entry = self._stats[key]
        entry["attempts"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"].append({
                "seed":        seed,
                "steps_taken": steps_taken,
                "final_state": final_state,
                "pick_pos":    pick_pos.tolist(),
                "place_pos":   place_pos.tolist(),
            })
        entry["success_rate"] = entry["successes"] / entry["attempts"]

    def save(self):
        # Sort by success_rate ascending so worst performers appear first
        sorted_entries = sorted(
            self._stats.values(), key=lambda e: e["success_rate"]
        )
        payload = {
            "saved_at":    datetime.now().isoformat(),
            "total_keys":  len(self._stats),
            "trajectories": sorted_entries,
        }
        with open(self.log_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[TrajectoryLogger] saved → {self.log_path}")
        return self.log_path

    def print_summary(self):
        print("\n── Trajectory Summary ──────────────────────────────────────")
        for entry in sorted(self._stats.values(), key=lambda e: e["success_rate"]):
            print(
                f"  {entry['success_rate']*100:5.1f}%  "
                f"({entry['successes']}/{entry['attempts']})  "
                f"{entry['object_type']} | {entry['bowl_type']} | "
                f"{entry['instruction'][:60]}"
            )
        print("────────────────────────────────────────────────────────────\n")

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _key(instruction, object_type, bowl_type):
        return f"{instruction}|{object_type}|{bowl_type}"