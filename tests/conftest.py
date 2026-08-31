"""Shared fixtures for the two suites that exercise exported weights.

`test_rust_parity.py` compares the torch forward against the Rust forward in-process, and
`test_rust_cli.py` drives the built binary. Both need the same randomly initialized student
exported to `.safetensors`, and building it twice would double the slowest setup in the suite.
"""

import pytest
import torch

from msspeculator.export import export_safetensors
from msspeculator.models.context import ChromRunbook, MSContextEncoder
from msspeculator.models.registry import build_student, save_checkpoint

#: An acquisition setup addressed by name rather than composed from factors, which is the only
#: thing available for a library that records neither instrument nor collision energy.
NAMED_SETUP = "Evosep60SPD_heron"
#: Dataset name and runbook row used by the chromatography-context cases.
CHROM_DATASET, CHROM_ROW = "dsA", 1


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    """A random-init student with every zero-init overwritten, plus its exported weights.

    No training: the point is that both runtimes agree on whatever the weights say, and a zero
    somewhere would let a code path be trivially correct without ever being exercised.
    """
    tmp = tmp_path_factory.mktemp("rustparity")
    torch.manual_seed(0)
    model = build_student("small")
    enc = MSContextEncoder(model.cfg.context_dim)
    # A named setup before the random init below, so its row is as non-trivial as every other
    # weight: a Rust side that quietly fell back to the neutral row would answer differently.
    enc.ensure_setups([NAMED_SETUP])
    runbook = ChromRunbook(2, model.cfg.context_dim)
    for mod in (model, enc, runbook):
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.3)
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    model.eval(), enc.eval(), runbook.eval()

    ckpt = tmp / "m.ckpt"
    save_checkpoint(
        model,
        ckpt,
        encoder=enc,
        runbook=runbook,
        dataset_index={CHROM_DATASET: CHROM_ROW, "dsB": 2},
    )
    art = tmp / "m.safetensors"
    export_safetensors(ckpt, art)
    # The names travel with the modules they address, so a test never has to import them from
    # here; `tests/` is not a package and the fixture is the only thing pytest shares for us.
    return {
        "path": art,
        "model": model,
        "enc": enc,
        "runbook": runbook,
        "setup": NAMED_SETUP,
        "chrom_dataset": CHROM_DATASET,
        "chrom_row": CHROM_ROW,
    }
