"""Acceso y verificación al conjunto de 714 imágenes originales."""
import csv
import hashlib
from collections import Counter
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from zipfile import ZipFile

from PIL import Image

from preprocess import CLASSES, prepare

COUNTS = {"train": 571, "val": 72, "test": 71}
FIELDS = {"relative_path", "split", "class_label", "origin_group", "source_file", "sha256"}


class Dataset:
    def __init__(self, archive, expected_sha256=None):
        self.archive = ZipFile(archive)
        try:
            manifest = self.archive.read("manifest.csv")
            self.manifest_sha256 = hashlib.sha256(manifest).hexdigest()
            if expected_sha256 is not None and self.manifest_sha256 != expected_sha256:
                raise ValueError("El manifiesto no corresponde al modelo.")
            reader = csv.DictReader(StringIO(manifest.decode("utf-8-sig")))
            if set(reader.fieldnames or ()) != FIELDS:
                raise ValueError("El manifiesto debe describir las imágenes originales.")
            self.rows = list(reader)
            self.by_path = {row["relative_path"]: row for row in self.rows}
            self.counts = Counter(row["split"] for row in self.rows)
            self.groups = {split: set() for split in COUNTS}
            if self.counts != COUNTS or len(self.by_path) != len(self.rows):
                raise ValueError("Las particiones o rutas no corresponden al conjunto de 714 originales.")
            for row in self.rows:
                path = PurePosixPath(row["relative_path"])
                split, label, source = row["split"], row["class_label"], row["source_file"]
                if (label not in CLASSES or path.parts != (split, label, source)
                        or path.suffix.lower() != ".jpg" or "__aug_" in source
                        or "\\" in source or not row["origin_group"]):
                    raise ValueError(f"Registro de imagen inválido: {path}")
                self.groups[split].add(row["origin_group"])
            groups = tuple(self.groups.values())
            if sum(map(len, groups)) != len(set.union(*groups)):
                raise ValueError("Un grupo de origen aparece en más de una partición.")
            hashes = {row["sha256"] for row in self.rows}
            if len(hashes) != len(self.rows):
                raise ValueError("El manifiesto repite imágenes por SHA-256.")
            members = [entry.filename for entry in self.archive.infolist() if not entry.is_dir()]
            if len(members) != len(set(members)) or set(members) != self.by_path.keys() | {"manifest.csv"}:
                raise ValueError("El ZIP debe contener el manifiesto y exactamente sus 714 imágenes.")
        except BaseException:
            self.archive.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.archive.close()

    def image_bytes(self, relative_path):
        row = self.by_path[relative_path]
        data = self.archive.read(relative_path)
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"El SHA-256 de {relative_path} no coincide.")
        return data

    def tensor(self, row):
        with Image.open(BytesIO(self.image_bytes(row["relative_path"]))) as image:
            return prepare(image)

    def partition(self, split):
        return [row for row in self.rows if row["split"] == split]
