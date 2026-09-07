from __future__ import annotations

import argparse
from pathlib import Path


def patch_trainer(trainer_path: Path) -> bool:
    text = trainer_path.read_text(encoding="utf-8")
    patched = (
        "    try:\n"
        "        writer.close()\n"
        "    except NameError:\n"
        "        pass\n"
    )
    if patched in text:
        return False
    needle = "    writer.close()\n"
    if needle not in text:
        raise ValueError(f"writer.close() not found in {trainer_path}")
    text = text.replace(needle, patched)
    trainer_path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Patch known runtime issues in the MMP reference repo.")
    p.add_argument("--mmp_root", required=True, help="Path to the MMP repository root")
    args = p.parse_args()

    mmp_root = Path(args.mmp_root).resolve()
    trainer_path = mmp_root / "src" / "training" / "trainer.py"
    if not trainer_path.exists():
        raise FileNotFoundError(trainer_path)

    changed = patch_trainer(trainer_path)
    print(f"trainer_patched={changed}")
    print(f"trainer_path={trainer_path}")


if __name__ == "__main__":
    main()
