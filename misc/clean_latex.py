#!/usr/bin/env python3
"""
cleanup_latex.py

Elimina i file temporanei generati da LaTeX.
Uso:
    python cleanup_latex.py           # pulisce la cartella corrente
    python cleanup_latex.py -r       # pulisce ricorsivamente
    python cleanup_latex.py /path    # pulisce la cartella /path
    python cleanup_latex.py -r /path # ricorsivo su /path
"""

import os
import argparse

# Estensioni tipiche dei file temporanei LaTeX
LATEX_TEMP_EXTS = {
    ".aux", ".log", ".out", ".toc", ".lof", ".lot",
    ".nav", ".snm", ".fls", ".fdb_latexmk",
    ".synctex.gz", ".bbl", ".blg", ".brf",
    ".ilg", ".ind", ".idx", ".glo", ".glg", ".gls",
}

def should_delete(filename: str) -> bool:
    for ext in LATEX_TEMP_EXTS:
        if filename.endswith(ext):
            return True
    return False

def cleanup(path: str, recursive: bool = False, dry_run: bool = False) -> None:
    if recursive:
        walker = os.walk(path)
    else:
        # Solo directory corrente
        walker = [(path, [], os.listdir(path))]

    to_delete = []

    for root, dirs, files in walker:
        for name in files:
            if should_delete(name):
                full_path = os.path.join(root, name)
                to_delete.append(full_path)

    if not to_delete:
        print("Nessun file temporaneo LaTeX da eliminare.")
        return

    print("File trovati:")
    for f in to_delete:
        print("  ", f)

    if dry_run:
        print("\nModalità dry-run: nessun file verrà realmente eliminato.")
        return

    ans = input("\nVuoi eliminare questi file? [y/N] ").strip().lower()
    if ans != "y":
        print("Annullato.")
        return

    for f in to_delete:
        try:
            os.remove(f)
            print("Eliminato:", f)
        except OSError as e:
            print("Errore eliminando", f, "->", e)

def main():
    parser = argparse.ArgumentParser(description="Elimina file temporanei LaTeX.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory di partenza (default: cartella corrente)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scansione ricorsiva nelle sottodirectory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra cosa verrebbe cancellato senza eliminare nulla",
    )

    args = parser.parse_args()
    cleanup(args.path, recursive=args.recursive, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
