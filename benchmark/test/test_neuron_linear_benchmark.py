from benchmark.binary_kernel.benchmark_neuron_linear import _paired_rounds


def test_paired_rounds_alternate_and_keep_pairs():
    calls = []

    def measure(name):
        calls.append(name)
        return (2.0 if name == "baseline" else 1.0, len(calls))

    pairs = _paired_rounds(measure, 3)

    assert calls == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert [pair["speedup"] for pair in pairs] == [2.0, 2.0, 2.0]
    assert [pair["order"] for pair in pairs] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]
