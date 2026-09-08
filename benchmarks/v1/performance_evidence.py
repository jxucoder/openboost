"""105 lossless numeric inputs and per-fit evidence; run-10 sources stay unchanged."""

import base64
import gc
import hashlib
import json
import math
import platform
import sys
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.v1.performance_checkpoint import CONFIG, fit, quality
from openboost import NumericData, Problem
from openboost.artifacts import Model

SCHEMA = "openboost-performance-inputs-v1"
MAX_ARRAY_BYTES = 64 * 1024**2


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(record):
    return digest(
        json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    )


def array_record(value):
    array = np.asarray(value)
    if array.dtype.str not in ("<f8", "<i8") or not 0 < array.nbytes <= MAX_ARRAY_BYTES:
        raise ValueError("bounded little-endian float64/int64 evidence array required")
    raw = array.tobytes(order="C")
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=digest(raw),
        data=base64.b64encode(zlib.compress(raw)).decode("ascii"),
    )


def load_array(record):
    shape = record["shape"]
    if (
        record["dtype"] not in ("<f8", "<i8")
        or not isinstance(shape, list)
        or not 1 <= len(shape) <= 2
        or any(type(n) is not int or n <= 0 for n in shape)
    ):
        raise ValueError("invalid evidence array shape/dtype")
    size = math.prod(shape) * 8
    if size > MAX_ARRAY_BYTES or len(record["data"]) > 2 * MAX_ARRAY_BYTES:
        raise ValueError("evidence array byte limit exceeded")
    compressed = base64.b64decode(record["data"], validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, size + 1)
    if (
        len(raw) != size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or digest(raw) != record["sha256"]
    ):
        raise ValueError("evidence array bytes/checksum differ")
    return np.frombuffer(raw, dtype=record["dtype"]).reshape(shape)


def problem_record(problem):
    if (
        not isinstance(problem.data, NumericData)
        or problem.classes is not None
        or problem.structure
        or problem.target_kind != "numeric"
    ):
        raise ValueError("checkpoint supports unstructured numeric squared/Normal problems only")
    return dict(
        identity=problem.identity,
        data_identity=problem.data.identity,
        feature_names=list(problem.data.feature_names),
        raw_width=problem.raw_width,
        arrays={
            name: array_record(value)
            for name, value in (
                ("features", problem.data.values),
                ("row_ids", problem.row_ids),
                ("target", problem.target),
                ("weight", problem.weight),
                ("offset", problem.offset),
            )
        },
    )


def input_record(train, validation, recipe):
    body = dict(
        schema=SCHEMA,
        recipe=recipe,
        config=dict(CONFIG),
        train=problem_record(train),
        validation=problem_record(validation),
    )
    record = dict(body, sha256=identity(body))
    load_inputs(record, record["sha256"])
    return record


def load_inputs(record, expected_sha):
    """Rebuild from stored bytes, without calling a generator or fitting transforms."""
    body = {k: v for k, v in record.items() if k != "sha256"}
    if (
        record.get("schema") != SCHEMA
        or record.get("recipe") not in ("squared", "normal")
        or record.get("sha256") != expected_sha
        or identity(body) != expected_sha
    ):
        raise ValueError("input snapshot identity differs")
    restored = []
    for name in ("train", "validation"):
        problem = record[name]
        if set(problem["arrays"]) != {"features", "row_ids", "target", "weight", "offset"}:
            raise ValueError("input role set differs")
        arrays = {k: load_array(v) for k, v in problem["arrays"].items()}
        data = NumericData(arrays["features"], arrays["row_ids"], problem["feature_names"])
        actual = Problem(
            data,
            arrays["target"],
            arrays["row_ids"],
            weight=arrays["weight"],
            offset=arrays["offset"],
            raw_width=problem["raw_width"],
        )
        if (
            actual.identity != problem["identity"]
            or data.identity != problem["data_identity"]
            or actual.raw_width != (1 if record["recipe"] == "squared" else 2)
        ):
            raise ValueError("input problem identity/schema differs")
        restored.append(actual)
    train, validation = restored
    if (
        train.data.feature_names != validation.data.feature_names
        or np.intersect1d(train.row_ids, validation.row_ids).size
    ):
        raise ValueError("input split schema differs or row IDs overlap")
    return train, validation


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    core = Path(sys.modules["openboost"].__file__).parent
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        os=platform.platform(),
        installed_path=str(core),
        core_sources={
            "src/openboost/" + str(p.relative_to(core)): digest(p.read_bytes())
            for p in sorted(core.rglob("*.py"))
        },
    )


def finish_fit(measured, model_record, validation, recipe, repeat):
    """Replay and score outside the official fit timer, before the next fit starts."""
    started = perf_counter()
    model = Model.from_record(model_record)
    load_seconds = perf_counter() - started
    single = NumericData(
        validation.data.values[:1], validation.row_ids[:1], validation.data.feature_names
    )
    timings = {}
    for name, data in (("single", single), ("batch", validation.data)):
        timings[name] = []
        for _ in range(4):
            tick = perf_counter()
            raw = model.predict(data)
            timings[name].append(perf_counter() - tick)
    replay = Model.from_record(model_record).predict(validation.data)
    if not np.array_equal(raw, replay):
        raise ValueError("saved model replay differs")
    result = dict(
        repeat=repeat,
        phase="first" if repeat == 0 else "warm",
        measurement=measured,
        model=model_record,
        model_sha256=identity(model_record),
        validation_raw=array_record(raw),
        quality=quality(validation, raw, recipe),
        model_load_seconds=load_seconds,
        cpu_prediction_seconds=timings,
    )
    result["post_fit_seconds"] = perf_counter() - started
    return result


def run(inputs, backend, output, *, repetitions=None):
    if backend not in ("cpu", "cuda"):
        raise ValueError("explicit CPU/CUDA backend required")
    train, validation = load_inputs(inputs, inputs["sha256"])
    expected = inputs["config"]["cpu_repetitions" if backend == "cpu" else "repetitions"]
    repetitions = expected if repetitions is None else repetitions
    if type(repetitions) is not int or not 1 <= repetitions <= 8:
        raise ValueError("bounded positive repetition count required")
    result = dict(
        schema="openboost-performance-result-v1",
        input_sha256=inputs["sha256"],
        recipe=inputs["recipe"],
        backend=backend,
        config=inputs["config"],
        expected_repetitions=repetitions,
        status="running",
        fits=[],
        environment=environment(),
    )
    write_json(output, result)
    previous = dict(CONFIG)
    try:
        # The frozen run-10 fit is reused in a single-threaded benchmark child.
        # Restore its configuration so a local caller cannot contaminate another run.
        CONFIG.update(inputs["config"])
        for repeat in range(repetitions):
            gc.collect()
            measured, record = fit(train, validation, inputs["recipe"], backend)
            result["fits"].append(
                finish_fit(measured, record, validation, inputs["recipe"], repeat)
            )
            write_json(output, result)
        result["status"] = "complete"
        write_json(output, result)
        return result
    except BaseException as error:
        result.update(status="error", error=dict(type=type(error).__name__, message=str(error)))
        write_json(output, result)
        raise
    finally:
        CONFIG.clear()
        CONFIG.update(previous)


def verify_report(
    report, inputs, expected_sha, *, expected_repetitions=None, expected_sources=None
):
    """Audit every preserved fit; timeout/error/running reports never become complete."""
    _, validation = load_inputs(inputs, expected_sha)
    if (
        report["input_sha256"] != expected_sha
        or report["recipe"] != inputs["recipe"]
        or report["config"] != inputs["config"]
    ):
        raise ValueError("report input binding differs")
    if expected_sources is not None and report["environment"]["core_sources"] != expected_sources:
        raise ValueError("installed sources differ")
    count = report["expected_repetitions"]
    if (
        type(count) is not int
        or not 1 <= count <= 8
        or report["backend"] not in ("cpu", "cuda")
        or report["status"] not in ("running", "complete", "error", "timeout")
    ):
        raise ValueError("invalid report status/backend/repetition contract")
    if expected_repetitions is not None and count != expected_repetitions:
        raise ValueError("declared repetitions differ")
    if not 0 <= len(report["fits"]) <= count:
        raise ValueError("unexpected repetition count")
    models = []
    for repeat, record in enumerate(report["fits"]):
        if record["repeat"] != repeat or record["model_sha256"] != identity(record["model"]):
            raise ValueError("fit index/model identity differs")
        models.append(record["model_sha256"])
        raw = load_array(record["validation_raw"])
        prediction = Model.from_record(record["model"]).predict(validation.data)
        if not np.array_equal(raw, prediction):
            raise ValueError("saved prediction replay differs")
        scores = quality(validation, raw, inputs["recipe"])
        if scores.keys() != record["quality"].keys() or any(
            not np.isclose(v, record["quality"][k], rtol=1e-12, atol=1e-12)
            for k, v in scores.items()
        ):
            raise ValueError("saved quality differs from exact inputs/predictions")
        measured = record["measurement"]
        rounds = inputs["config"]["rounds"]
        width = 1 if inputs["recipe"] == "squared" else 2
        if (
            measured["state"]["version"] != rounds
            or measured["state"]["terms"] != rounds * width
            or measured["stop"]["completed_rounds"] != rounds
        ):
            raise ValueError("incomplete round/term budget")
        if not np.isfinite(measured["fit_seconds"]) or measured["fit_seconds"] <= 0:
            raise ValueError("invalid fit time")
        if report["backend"] == "cuda" and (
            measured["live_bytes_after_run_close"] != 0
            or measured["final_metrics"]["live_bytes"] != 0
        ):
            raise ValueError("owned GPU buffers leaked")
    if len(set(models)) > 1:
        raise ValueError("model identity differs across repetitions")
    complete = report["status"] == "complete" and len(report["fits"]) == count
    if report["status"] == "complete" and not complete:
        raise ValueError("complete report lacks required repetitions")
    return dict(complete=complete, verified_fits=len(report["fits"]))
