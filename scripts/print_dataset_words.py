#!/usr/bin/env python3
"""Print decoded samples from the algorithmic/formal-language datasets.

Quick start (from anywhere in the repo):

    python length-generalization-log-analysis/scripts/print_dataset_words.py --list
    python length-generalization-log-analysis/scripts/print_dataset_words.py --task parity
    python length-generalization-log-analysis/scripts/print_dataset_words.py --task all --num 2 --no-pause

Must run in the environment where the training dependencies (torch, transformers,
mambapy) are installed, since the datasets are imported from `algorithmic/`.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# This file lives in `length-generalization-log-analysis/scripts/`, so:
# parents[0] = scripts, parents[1] = length-generalization-log-analysis, parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODULE = "algorithmic.dataset_generators"

# Formal-language corpora are generated up front, and at their real sizes that takes
# minutes per task (tomita_6 alone generates 10k train + 4x10k eval strings). Sampling
# a handful of examples needs far fewer, so cap it unless --full-corpus is passed.
DEFAULT_CORPUS_LIMIT = 200

# Task-name spellings that appear in log filenames, plot configs and papers, mapped to
# the names `algorithmic.dataset_generators` routes on.
_TASK_ALIASES = {
    "tomita1": "tomita_1",
    "tomita2": "tomita_2",
    "tomita3": "tomita_3",
    "tomita4": "tomita_4",
    "tomita5": "tomita_5",
    "tomita6": "tomita_6",
    "tomita7": "tomita_7",
    "d2": "d_2",
    "d3": "d_3",
    "d4": "d_4",
    "d12": "d_12",
    "012star_0_2star": "012_star_0_2_star",
    "aastar": "aa_star",
    "ababstar": "abab_star",
    "anstara2": "an_star_a2",
    "mqar_word_problem": "mqar",
}

# `--task` values that expand to several tasks rather than naming one.
_TASK_GROUPS = ("all", "algorithmic", "formal")


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _import_module_or_exit(module_name: str):
    """Import `module_name`, turning a missing training dependency into advice.

    The datasets pull in the full training stack, so running this script with the
    wrong interpreter fails on an import deep inside `algorithmic/` rather than on
    anything the caller wrote.
    """
    _ensure_repo_on_path()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == module_name.split(".")[0]:
            raise SystemExit(
                f"Could not import `{module_name}` from repo root {REPO_ROOT}.\n"
                "Run this script from a checkout of the length_generalization repo."
            ) from e
        raise SystemExit(
            f"Failed to import dependency `{e.name}`, required by `{module_name}`.\n"
            f"Interpreter in use: {sys.executable}\n"
            "Re-run with the project environment's python, where the training "
            "dependencies (torch, transformers, mambapy) are installed -- e.g. the "
            "conda env used for training, or after `uv pip install -e '.[all]'`."
        ) from e


def _task_registry() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(algorithmic, formal, all) task names, read from the dataset module itself."""
    mod = _import_module_or_exit(DEFAULT_MODULE)
    return mod.ALGORITHMIC_TASKS, mod.FORMAL_TASKS, mod.ALL_TASKS


def _dataset_class_names(module_name: str) -> list[str]:
    mod = _import_module_or_exit(module_name)
    base = getattr(mod, "CustomDataset", None)
    if base is None:
        return []
    return sorted(
        name
        for name, obj in vars(mod).items()
        if inspect.isclass(obj) and issubclass(obj, base) and obj is not base
    )


def _resolve_tasks(task_arg: str) -> list[str]:
    """Expand `--task` into a concrete task list, or exit with the valid options."""
    algorithmic, formal, all_tasks = _task_registry()
    raw = task_arg.strip().lower().replace("-", "_")
    if raw in _TASK_GROUPS:
        return list({"all": all_tasks, "algorithmic": algorithmic, "formal": formal}[raw])

    task = _TASK_ALIASES.get(raw, raw)
    if task in all_tasks:
        return [task]

    hint = ""
    close = difflib.get_close_matches(task, all_tasks, n=3, cutoff=0.6)
    if close:
        hint = "\nDid you mean: " + ", ".join(close) + "?"
    raise SystemExit(
        f"Unknown --task {task_arg!r}.{hint}\n"
        f"Algorithmic tasks: {', '.join(algorithmic)}\n"
        f"Formal tasks: {', '.join(formal)}\n"
        f"Groups: {', '.join(_TASK_GROUPS)}\n"
        "Run with --list for the full listing."
    )


def _print_listing(module_name: str) -> None:
    algorithmic, formal, _ = _task_registry()
    print("Tasks for --task (build mode; these are what training uses):\n")
    print(f"  algorithmic ({len(algorithmic)}):")
    for t in algorithmic:
        print(f"    {t}")
    print(f"\n  formal languages ({len(formal)}):")
    for t in formal:
        print(f"    {t}")
    print(f"\n  groups: {', '.join(_TASK_GROUPS)}")

    classes = _dataset_class_names(module_name)
    print(f"\nDataset classes for --mode class --dataset (from {module_name}):\n")
    for name in classes:
        print(f"    {name}")
    print(
        "\nExamples:\n"
        "  --task sort\n"
        "  --task formal --num 2 --no-pause\n"
        "  --task tomita_3 --split test --test-bin len101-150\n"
        '  --mode class --dataset MQARWordProblemDataset --dataset-kwargs \'{"length_range":[20,30],"max_test_length":100,"key_size":8}\''
    )


def _parse_dict_arg(raw: str, flag: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except Exception as e:
            raise SystemExit(
                f"Failed to parse {flag}.\n"
                "Pass a single quoted shell argument containing JSON or a Python dict literal.\n"
                'Example JSON:   \'{"length_range":[20,30],"max_test_length":100}\'\n'
                'Example Python: \'{"length_range": (20, 30), "max_test_length": 100}\'\n'
                f"Got: {raw!r}\nParse error: {e}"
            ) from e
    if not isinstance(value, dict):
        raise SystemExit(f"{flag} must be a dict, got {type(value).__name__}")
    return value


def _parse_length_range(raw: str) -> tuple[int, int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except Exception as e:
            raise SystemExit(
                f"Failed to parse --train-range.\nPass e.g. [0,50] or (0,50).\n"
                f"Got: {raw!r}\nParse error: {e}"
            ) from e
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SystemExit(f"--train-range must be a length-2 list/tuple, got: {value!r}")
    return (int(value[0]), int(value[1]))


# ── Sample rendering ────────────────────────────────────────────────────────


def _as_int_list(x: Any) -> list[int]:
    return [] if x is None else [int(v) for v in x]


@dataclass(frozen=True)
class SampleView:
    input_ids: list[int]
    pos_ids: list[int]
    labels: list[int]
    tokens: list[str]
    label_tokens: list[str]


def _to_sample_view(dataset: Any, sample: Any) -> SampleView:
    # Datasets in algorithmic/task_datasets.py yield (instance, pos_ids, label).
    input_ids, pos_ids, labels = sample
    tok = getattr(dataset, "tokenizer", None)
    if tok is None or not hasattr(tok, "convert_ids_to_tokens"):
        raise SystemExit(
            f"{type(dataset).__name__} has no compatible `tokenizer.convert_ids_to_tokens()`."
        )
    input_ids = _as_int_list(input_ids)
    labels = _as_int_list(labels)
    return SampleView(
        input_ids=input_ids,
        pos_ids=_as_int_list(pos_ids),
        labels=labels,
        tokens=tok.convert_ids_to_tokens(input_ids, rm_special=False),
        label_tokens=tok.convert_ids_to_tokens(labels, rm_special=False),
    )


def _split_on_sep(tokens: list[str], sep_token: str) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    for t in tokens:
        if t == sep_token:
            parts.append([])
        else:
            parts[-1].append(t)
    return parts


def _mask_pad(label_tokens: Iterable[str], pad_token: str) -> list[str]:
    # "·" marks positions the loss ignores, so the scored region stands out.
    return ["·" if t == pad_token else t for t in label_tokens]


def _print_aligned_rows(rows: list[tuple[str, list[str]]], width: int = 100) -> None:
    """Print equal-length token rows as position-aligned columns, wrapping at `width`.

    Labels are per-position for these datasets, so reading a sample means comparing
    input token `t` against label token `t`; unaligned rows make that guesswork.
    """
    label_w = max(len(name) for name, _ in rows)
    n = len(rows[0][1])
    col_w = [max(len(row[i]) for _, row in rows) for i in range(n)]

    start = 0
    while start < n:
        end, line_w = start, 0
        while end < n and (line_w + col_w[end] + 1) <= width:
            line_w += col_w[end] + 1
            end += 1
        end = max(end, start + 1)  # always make progress, even on an over-wide column
        for name, row in rows:
            cells = " ".join(row[i].rjust(col_w[i]) for i in range(start, end))
            print(f"  {name:<{label_w}}  {cells}")
        start = end
        if start < n:
            print()


def _predicted_tokens(label_tokens: list[str], pad_token: str) -> list[str]:
    """What the logits at each position are scored against.

    Both serializations rely on the autoregressive shift (`ForCausalLMLoss`), i.e. the
    logits at position `t` are compared with `label[t + 1]`. Lining the labels up with
    the inputs by raw index instead would misreport every dataset by one token.
    """
    shifted = [*label_tokens[1:], pad_token]
    return _mask_pad(shifted, pad_token)


def _pretty_print(dataset: Any, sv: SampleView, idx: int, *, show_ids: bool, show_pos: bool) -> None:
    tok = dataset.tokenizer
    print("=" * 100)
    print(f"sample {idx}   len(input_ids)={len(sv.input_ids)}")

    if len(sv.label_tokens) == len(sv.tokens):
        rows = [("input", sv.tokens), ("predict", _predicted_tokens(sv.label_tokens, tok.pad_token))]
        if show_pos and len(sv.pos_ids) == len(sv.tokens):
            rows.append(("pos", [str(p) for p in sv.pos_ids]))
        if show_ids:
            rows.append(("input_id", [str(i) for i in sv.input_ids]))
            rows.append(("label", _mask_pad(sv.label_tokens, tok.pad_token)))
        print("\ninput vs prediction target at each position (· = not scored):")
        _print_aligned_rows(rows)
    else:
        # Should not happen for the current datasets, but never hide a sample.
        print("\ntokens:")
        print(" ".join(sv.tokens))
        print("labels (· = not scored):")
        print(" ".join(_mask_pad(sv.label_tokens, tok.pad_token)))
        if show_pos:
            print("pos_ids:")
            print(" ".join(map(str, sv.pos_ids)))

    parts = _split_on_sep(sv.tokens, tok.sep_token)
    if len(parts) >= 2:
        print("\nsegments (split on <sep>):")
        for i, p in enumerate(parts):
            print(f"  [{i}] " + " ".join(p))

    if show_pos and len(sv.pos_ids) != len(sv.tokens):
        print("\npos_ids:")
        print(" ".join(map(str, sv.pos_ids)))


def _format_vocab(tok: Any, max_shown: int = 40) -> str:
    tokens = list(tok.vocab)
    shown = " ".join(tokens[:max_shown])
    if len(tokens) > max_shown:
        shown += f" ... (+{len(tokens) - max_shown} more)"
    return f"vocab({len(tokens)}): {shown}"


def _print_dataset_header(dataset: Any, description: str) -> None:
    facts = [f"dataset={type(dataset).__name__}", f"n_positions={getattr(dataset, 'n_positions', '?')}"]
    if hasattr(dataset, "__len__"):
        facts.append(f"examples={len(dataset)}")

    print()
    print("#" * 100)
    print(f"# {description}")
    print(f"# {'  '.join(facts)}")
    tok = getattr(dataset, "tokenizer", None)
    if tok is not None and hasattr(tok, "vocab"):
        print(f"# {_format_vocab(tok)}")
    print("#" * 100)


# ── Dataset construction ────────────────────────────────────────────────────


def _make_run_config(*, task: str, train_range: tuple[int, int], num_test_bins: int, test_num: int):
    utils = _import_module_or_exit("algorithmic.utils")
    run_config = utils.default_transformer_sweep()
    run_config.task = task
    run_config.train_length_range = train_range
    run_config.num_test_bins = num_test_bins
    run_config.test_num = test_num
    return run_config


def _build_split(args, task: str):
    """Return (dataset, description) for one task's requested split."""
    mod = _import_module_or_exit(args.module)
    if not hasattr(mod, "build_datasets"):
        raise SystemExit(f"Module {args.module} has no build_datasets().")

    run_config = _make_run_config(
        task=task,
        train_range=_parse_length_range(args.train_range),
        num_test_bins=args.num_test_bins,
        test_num=args.test_num,
    )
    corpus_limit = None if args.full_corpus else max(args.corpus_limit, args.num)
    train_dataset, test_dataset, train_range, test_ranges = mod.build_datasets(
        run_config, corpus_size_limit=corpus_limit
    )

    # The cap only bites on formal tasks, and there it also shrinks the corpus the
    # tokenizer and position budget are derived from, so say so rather than reporting
    # an `n_positions` that training would not have used.
    suffix = ""
    if corpus_limit is not None and mod.is_formal_task(task):
        suffix = f"  [corpus capped at {corpus_limit}; --full-corpus for real sizes]"

    bins = ", ".join(test_dataset.keys())
    if args.split == "train":
        return (
            train_dataset,
            f"task={task} split=train train_lengths={train_range} eval_bins=[{bins}]{suffix}",
        )

    if not test_dataset:
        raise SystemExit(f"task={task}: test dataset dict is empty.")
    if args.test_bin is None:
        bin_key = next(iter(test_dataset))
    else:
        bin_key = args.test_bin
        if bin_key not in test_dataset:
            raise SystemExit(f"task={task}: unknown --test-bin {bin_key!r}. Available: {bins}")
    bin_range = dict(zip(test_dataset.keys(), test_ranges))[bin_key]
    return (
        test_dataset[bin_key],
        f"task={task} split=test bin={bin_key} lengths={bin_range}{suffix}",
    )


def _instantiate_class(args) -> tuple[Any, str]:
    mod = _import_module_or_exit(args.module)
    cls = getattr(mod, args.dataset, None)
    if cls is None:
        available = ", ".join(_dataset_class_names(args.module))
        raise SystemExit(
            f"Unknown dataset class {args.dataset!r} in {args.module}.\nAvailable: {available}"
        )

    kwargs = dict(_parse_dict_arg(args.dataset_kwargs, "--dataset-kwargs"))
    signature = inspect.signature(cls)
    # Nearly every dataset class takes these two required positionals; filling them in
    # lets `--mode class --dataset X` work without boilerplate for the common case.
    defaults = {"length_range": [0, 50], "max_test_length": 150}
    filled = []
    for name, default in defaults.items():
        if name in signature.parameters and name not in kwargs:
            kwargs[name] = default
            filled.append(f"{name}={default}")

    try:
        dataset = cls(**kwargs)
    except TypeError as e:
        missing = [
            name
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty and name not in kwargs and name != "self"
        ]
        hint = f"Still needs: {', '.join(missing)}\n" if missing else ""
        raise SystemExit(
            f"Could not construct {args.dataset}: {e}\n"
            f"Signature: {args.dataset}{signature}\n"
            f"{hint}"
            "Pass the remaining arguments via --dataset-kwargs, or use --mode build "
            "with a --task, which constructs the dataset the way training does."
        ) from e

    description = f"class={args.dataset} kwargs={kwargs}"
    if filled:
        description += f"  (defaulted: {', '.join(filled)})"
    return dataset, description


def _iter_samples(d: Any) -> Iterator[Any]:
    # IterableDataset streams forever; EvalDataset is a finite indexable Dataset.
    return iter(d)


def _print_samples(dataset: Any, description: str, args) -> bool:
    """Print up to `args.num` samples. Returns False if the user asked to quit."""
    _print_dataset_header(dataset, description)
    it = _iter_samples(dataset)
    for i in range(args.num):
        try:
            sample = next(it)
        except StopIteration:
            print(f"\n(dataset exhausted after {i} sample(s); it holds fewer than --num={args.num})")
            break
        _pretty_print(dataset, _to_sample_view(dataset, sample), i, show_ids=args.show_ids, show_pos=args.show_pos)

        if args.no_pause:
            continue
        try:
            answer = input("\n[enter]=next, q=quit > ").strip().lower()
        except EOFError:
            return False
        if answer in {"q", "quit", "exit"}:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List every task and dataset class this script can show, then exit.",
    )
    p.add_argument(
        "--task",
        default="parity",
        help=(
            "Task to inspect, or a group: 'all', 'algorithmic', 'formal'. "
            "Use --list to see the task names. (build mode only)"
        ),
    )
    p.add_argument(
        "--mode",
        choices=["build", "class"],
        default="build",
        help=(
            "'build' (default): inspect exactly what training sees, via "
            "build_datasets() for --task. 'class': instantiate --dataset directly "
            "with --dataset-kwargs."
        ),
    )
    p.add_argument("--num", type=int, default=5, help="Samples to print per dataset.")
    p.add_argument("--seed", type=int, default=None, help="Seed python's RNG for reproducible samples.")
    p.add_argument("--show-ids", action="store_true", help="Also show raw input_ids.")
    p.add_argument("--show-pos", action="store_true", help="Also show pos_ids (randomized during training).")
    p.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not pause between samples (for piping or redirecting output).",
    )

    build = p.add_argument_group("build mode")
    build.add_argument(
        "--split", choices=["train", "test"], default="train", help="Which split to sample from."
    )
    build.add_argument(
        "--test-bin",
        default=None,
        help='Eval bin for --split test, e.g. "len51-100". Defaults to the first (in-distribution) bin.',
    )
    build.add_argument(
        "--train-range",
        default="[0,50]",
        help="Train length range, e.g. [0,50]. Ignored by formal tasks, which use their own windows.",
    )
    build.add_argument("--num-test-bins", type=int, default=3, help="Number of eval bins.")
    build.add_argument("--test-num", type=int, default=2000, help="Examples per eval bin.")
    build.add_argument(
        "--corpus-limit",
        type=int,
        default=DEFAULT_CORPUS_LIMIT,
        help=(
            "Cap on generated examples per split for formal-language tasks, whose full "
            f"corpora take minutes to build (default: {DEFAULT_CORPUS_LIMIT})."
        ),
    )
    build.add_argument(
        "--full-corpus",
        action="store_true",
        help="Generate formal-language corpora at their real training sizes (slow).",
    )

    cls_group = p.add_argument_group("class mode")
    cls_group.add_argument("--dataset", default="ParityDataset", help="Dataset class name.")
    cls_group.add_argument(
        "--dataset-kwargs",
        "--datset-kwargs",  # tolerate the long-standing typo
        dest="dataset_kwargs",
        default="{}",
        help="Constructor kwargs as JSON or a Python dict literal.",
    )
    p.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"Module providing build_datasets()/dataset classes (default: {DEFAULT_MODULE}).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        _print_listing(args.module)
        return 0

    if args.seed is not None:
        import random

        random.seed(args.seed)

    if args.mode == "class":
        dataset, description = _instantiate_class(args)
        _print_samples(dataset, description, args)
        return 0

    tasks = _resolve_tasks(args.task)
    if len(tasks) > 1:
        print(f"Inspecting {len(tasks)} tasks: {', '.join(tasks)}")
    for task in tasks:
        dataset, description = _build_split(args, task)
        if not _print_samples(dataset, description, args):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
