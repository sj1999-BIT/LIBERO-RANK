import abc
import os
import re
import glob
import random
import tempfile
import torch

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

BENCHMARK_MAPPING = {}


# ── HOW TO REGISTER A NEW BENCHMARK ──────────────────────────────────────────
# A benchmark class is registered by decorating it with @register_benchmark.
# The class name (case-insensitive) becomes its lookup key in BENCHMARK_MAPPING,
# so `get_benchmark("LIBERO_SPATIAL")` and `get_benchmark("libero_spatial")` both work.
#
# Standard requirements for a new benchmark:
#   1. TASK MAP ENTRY: Add the suite name → [task filename list] to libero_task_map
#      in benchmark/libero_suite_task_map.py so _make_benchmark() can find tasks.
#   2. BDDL FILES ON DISK: Each task needs a <task_name>.bddl file located at:
#         get_libero_path("bddl_files") / <suite_name> / <task_name>.bddl
#   3. INIT STATE FILES ON DISK: Each task needs a <task_name>.pruned_init file at:
#         get_libero_path("init_states") / <suite_name> / <task_name>.pruned_init
#   4. TASK OBJECTS: task_maps[suite_name] must be populated with Task NamedTuples
#      at module load time (see the loop over libero_suites above).
#   5. BENCHMARK CLASS: Subclass Benchmark, set self.name = suite_name, decorate
#      with @register_benchmark, and call self._make_benchmark() in __init__.
#
# If your benchmark generates BDDL at runtime instead of reading from disk,
# you must override _make_benchmark(), get_task_bddl_file_path(), and
# get_task_init_states() — see LIBERO_DEPTH_ORDER below for the pattern.
# ─────────────────────────────────────────────────────────────────────────────

def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        # REQUIREMENT: task_maps[self.name] must already be populated at module
        # load time (via libero_suite_task_map + the loop over libero_suites).
        # REQUIREMENT: self.name must match the key used in that loop.
        # REQUIREMENT: task_orders must have a valid entry at self.task_order_index
        # for all 10-task suites; libero_90 uses the natural order instead.
        # OVERRIDE NEEDED: If tasks are not pre-registered in task_maps (e.g. they
        # are generated at runtime), subclasses must override this method entirely.
        tasks = list(task_maps[self.name].values())
        if self.name == "libero_90":
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        # REQUIREMENT: the .bddl file must exist at:
        #   get_libero_path("bddl_files") / problem_folder / bddl_file
        # OVERRIDE NEEDED: If BDDL is generated at runtime and stored at an
        # arbitrary path (e.g. a tempfile), override this to return task.bddl_path.
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        # REQUIREMENT: a .pruned_init file must exist at:
        #   get_libero_path("init_states") / problem_folder / init_states_file
        # OVERRIDE NEEDED: If there are no pre-saved init states (e.g. because
        # the environment is reset from scratch each episode), override this to
        # return None or generate initial states on-the-fly.
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()


# ══════════════════════════════════════════════════════════════════════════════
# LIBERO_DEPTH_ORDER — SUMMARY OF WHAT IS NEEDED / WHAT DIFFERS
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard libero benchmarks (LIBERO_SPATIAL, LIBERO_GOAL, …) follow this flow:
#   1. bddl files are pre-authored and live on disk under bddl_files/<suite>/
#   2. init-state files are pre-saved under init_files/<suite>/
#   3. libero_suite_task_map.py lists every task filename for the suite
#   4. At module load the loop over libero_suites builds task_maps[suite_name]
#      from Task NamedTuples  (name, language, problem, problem_folder, bddl_file,
#      init_states_file)
#   5. A @register_benchmark class sets self.name and calls _make_benchmark()
#      which reads from task_maps[self.name] and reorders by task_orders[index]
#
# LIBERO_DEPTH_ORDER breaks assumptions 1-4 because its BDDL is procedurally
# generated at runtime by generate_random_bddl() in generate_bddl.py.
# The following pieces are therefore required or different:
#
#   [TASK OBJECT] — DepthOrderTask (plain class, NOT a NamedTuple)
#       NamedTuple is immutable and lacks the extra fields needed, so
#       DepthOrderTask is a plain class that mirrors the Task interface
#       plus runtime-only fields:
#           • bddl_path        — full path to the tempfile written by generate_random_bddl()
#           • target_object    — instance name (e.g. "cookies_2") of the correct pick target
#           • target_region_center — (x, y) centroid of the target's table cell, used for eval
#       init_states_file is None (no pre-saved states).
#
#   [NO task_maps ENTRY] — do NOT add "libero_depth_order" to libero_suites or
#       libero_suite_task_map.py.  _make_benchmark() would crash looking it up.
#
#   [OVERRIDE _make_benchmark()] — replace with generate_tasks(), which calls
#       generate_random_bddl() once per task, writes to self._tmpdir, and appends
#       a DepthOrderTask to self.tasks.
#
#   [OVERRIDE get_task_bddl_file_path(i)] — return self.tasks[i].bddl_path
#       (the full tempfile path) instead of assembling a path under bddl_files/.
#
#   [OVERRIDE get_task_init_states(i)] — return None (or raise NotImplementedError).
#       The env must be reset from scratch each episode; there are no .pruned_init
#       files.  Callers that depend on init states must handle None gracefully.
#
#   [TEMP DIRECTORY] — self._tmpdir = tempfile.mkdtemp(prefix="libero_depth_order_")
#       holds the generated .bddl files for the session lifetime.  Callers are
#       responsible for cleanup (or rely on OS temp-dir sweeping).
#
#   [LANGUAGE] — DEPTH_ORDER_LANGUAGES supplies the natural-language instructions
#       (e.g. "Place the object closest to the camera in the bowl.").
#       generate_random_bddl() also resolves the generic word "object" → the
#       sampled object type and stores resolved_language on the returned dict.
#
#   [SEED STRATEGY] — base_seed + task index keeps tasks reproducible within a
#       session while still varying per task.
#
#   [WHAT LIBERO STILL PROVIDES (no changes needed elsewhere)]
#       • OffScreenRenderEnv / bddl_file_name kwarg — pass bddl_path directly
#       • @register_benchmark decorator — works unchanged; class name is the key
#       • get_benchmark("LIBERO_DEPTH_ORDER") — works unchanged via BENCHMARK_MAPPING
#       • get_task(i), get_num_tasks(), get_task_names() — inherited, work unchanged
#         as long as self.tasks and self.n_tasks are populated by generate_tasks()
# ══════════════════════════════════════════════════════════════════════════════

from typing import List, Optional
# from .generate_bddl import generate_rndom_bddl, OBJECT_POOL, BOWL_TYPE, 

from .rank_scripts.bddl_generator import generate_random_rank_task_bddl, INSTRUCTION_TEMPLATES, OBJECT_POOL, BOWL_TYPE

# All object types that can appear as pick targets (everything except the bowl).
NON_BOWL_POOL = [obj for obj in OBJECT_POOL if obj != BOWL_TYPE]


class DepthRankTask:
    """
    A Task-like object for a runtime-generated depth-order task.

    Cannot subclass Task (a NamedTuple) because NamedTuple __new__ requires
    all six fields up front and the instance is immutable.  This plain class
    exposes the same attribute interface so it works everywhere Task does.
    """
    def __init__(self, name, language, bddl_path, target_object):
        self.name = name
        self.language = language
        self.problem = "LIBERO_RANK"
        self.problem_folder = "libero_rank"
        self.bddl_file = os.path.basename(bddl_path)
        self.bddl_path = bddl_path          # full path (temp file)
        self.init_states_file = None        # no pre-saved init states
        self.target_object = target_object


@register_benchmark
class LIBERO_RANK(Benchmark):
    """
    A benchmark suite where tasks are generated at runtime using
    relational depth-order predicates (closest, furthest, 2nd closest, …).
    Each call to generate_tasks() produces a fresh set of randomized tasks,
    stratified evenly across all implemented task categories.
    """

    # Regex patterns that identify each category from the language string
    CATEGORY_PATTERNS = {
        "middle":            lambda l: "middle" in l,
        "egocentric_pick":   lambda l: bool(re.search(r"pick the.*(closest|furthest|furtherest|farthest|\d+\w*)\s+object", l)),
        "allocentric_pick":  lambda l: bool(re.search(r"pick the.*object.*(closest|furthest|furtherest|farthest).*bowl", l)),
        "egocentric_place":  lambda l: bool(re.search(r"place in the.*(closest|furthest|furtherest|farthest|\d+\w*)\s+bowl", l)),
        "allocentric_place": lambda l: bool(re.search(r"bowl.*(closest|furthest|furtherest|farthest).*\bit\b", l)),
        "feature":           lambda l: bool(re.search(r"\b(largest|smallest)\b", l)),
    }

    def __init__(
        self,
        task_order_index: int = 0,
        num_tasks: int = 10,
        num_objects: int = 10,
        grid_size: int = 20,
        seed: Optional[int] = None,
        languages: Optional[List[str]] = None,
        object_types: Optional[List[str]] = None,
    ):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_rank"
        self.num_objects = num_objects
        self.grid_size = grid_size
        self.base_seed = seed if seed is not None else random.randint(0, 10_000)
        self.object_types = object_types or NON_BOWL_POOL

        # Bucket INSTRUCTION_TEMPLATES by category at init time
        all_languages = languages or INSTRUCTION_TEMPLATES
        self.category_templates: dict[str, list[str]] = {k: [] for k in self.CATEGORY_PATTERNS}
        for lang in all_languages:
            l = lang.lower()
            for category, match_fn in self.CATEGORY_PATTERNS.items():
                if match_fn(l):
                    self.category_templates[category].append(lang)
                    break  # each template belongs to exactly one category

        # Drop empty categories (not yet implemented)
        self.category_templates = {k: v for k, v in self.category_templates.items() if v}
        self.categories = list(self.category_templates.keys())
        print(f"[DEBUG INFO LIBERO_RANK]: {len(self.categories)} active categories: {self.categories}")

        self._tmpdir = os.path.join(get_libero_path("bddl_files"), "libero_rank")
        os.makedirs(self._tmpdir, exist_ok=True)

        self.tasks: List[DepthRankTask] = []
        self.n_tasks = 0
        self.generate_tasks(num_tasks)

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def generate_tasks(self, num_tasks: int):
        """
        Generate `num_tasks` tasks stratified evenly across all active
        categories.  Within each category templates and object types cycle
        deterministically, so the suite is reproducible given the same seed.
        """
        # track per-category counters so cycling is independent per category
        category_counters = {k: 0 for k in self.categories}

        for i in range(num_tasks):
            seed = self.base_seed + i

            # round-robin across categories
            category = self.categories[i % len(self.categories)]
            counter  = category_counters[category]
            category_counters[category] += 1

            templates    = self.category_templates[category]
            language     = templates[counter % len(templates)]
            object_type  = self.object_types[counter % len(self.object_types)]
            output_path  = os.path.join(self._tmpdir, f"task_{i:03d}.bddl")

            result = generate_random_rank_task_bddl(
                language=language,
                seed=seed,
                num_objects=self.num_objects,
                grid_size=self.grid_size,
                object_types=object_type,
                output_path=output_path,
                save_bddl=True,
            )

            if result is None:
                print(f"[Warning] task {i:03d} skipped — category '{category}' returned None: {language}")
                continue

            task = DepthRankTask(
                name=f"depth_rank_task_{i:03d}",
                language=language,
                bddl_path=result["bddl_path"],
                target_object=result["target_object"],
            )
            self.tasks.append(task)
            print(f"[DEBUG INFO LIBERO_RANK.generate_tasks]： task {i:03d} [{category}]: {language}")

        self.n_tasks = len(self.tasks)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_task_bddl_file_path(self, i):
        return self.tasks[i].bddl_path

    def get_task_init_states(self, i):
        return None