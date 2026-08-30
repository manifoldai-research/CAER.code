import json
import os


def prepare_step_metrics_jsonl(path, resume_global_step=None):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if resume_global_step is None:
        with open(path, "w", encoding="utf-8"):
            pass
        return None

    resume_global_step = int(resume_global_step)
    if resume_global_step < 0:
        raise ValueError(
            f"resume_global_step must be non-negative; got {resume_global_step}"
        )

    records_by_step = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        final_nonempty_index = nonempty_indices[-1] if nonempty_indices else -1
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == final_nonempty_index:
                    break
                raise RuntimeError(
                    f"Invalid metrics JSONL record at {path}:{index + 1}"
                ) from exc
            try:
                global_step = int(record["global_step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Metrics record has no integer global_step at {path}:{index + 1}"
                ) from exc
            if global_step <= resume_global_step:
                records_by_step[global_step] = record

    ordered_records = [
        records_by_step[global_step] for global_step in sorted(records_by_step)
    ]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in ordered_records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(tmp_path, path)
    return ordered_records[-1] if ordered_records else None


def append_jsonl(path, record):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def write_json_atomic(path, record):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)
