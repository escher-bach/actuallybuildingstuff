from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

from step1_experiments.data import END_TURN, BinaryShard, Sequence, collate
from step1_experiments.runner import _run_checked


class DataContracts(unittest.TestCase):
    def test_collator_uses_standard_minus_100_labels(self) -> None:
        batch = collate([Sequence([1, 2, END_TURN], [0, 1, 1], [0, 1, 1])], context=8)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, END_TURN]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, END_TURN]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True]])

    def test_binary_shard_reader_obeys_layout(self) -> None:
        tokens, loss, channels = [257, 65, 261, 258], [0, 1, 1, 0], [0, 1, 1, 0]
        payload = bytearray(b"BLMSHRD1")
        payload += struct.pack("<IQ", 1, 1) + struct.pack("<QQ", len(tokens), 1)
        payload += struct.pack(f"<{len(tokens)}I", *tokens) + bytes(loss) + bytes(channels)
        payload += struct.pack("<QQQ", 0, 0, len(tokens))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(payload)
            shard = BinaryShard(path)
            self.assertEqual(shard[0], Sequence(tokens, loss, channels))
            shard.close()

    def test_subprocess_failure_surfaces_child_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaisesRegex(RuntimeError, "UNIQUE_CHILD_FAILURE"):
                _run_checked([sys.executable, "-c", "print('UNIQUE_CHILD_FAILURE'); raise SystemExit(7)"], log, Path(directory), 30)


if __name__ == "__main__":
    unittest.main()
