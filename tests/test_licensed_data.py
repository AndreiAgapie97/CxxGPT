"""Check fail-closed licensing and the corpus's split/provenance guarantees."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("licensed", Path(__file__).parents[1] / "scripts/prepare_licensed.py")
licensed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(licensed)


class LicensedDataTests(unittest.TestCase):
    def test_only_explicit_unambiguous_allowlisted_licenses(self):
        self.assertEqual(licensed.license_id("Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"), "CC-BY-4.0")
        self.assertEqual(licensed.license_id("http://creativecommons.org/publicdomain/zero/1.0/"), "CC0-1.0")
        for value in (None, [], "CC BY 4.0", "public domain", "all rights reserved",
                      "https://creativecommons.org/licenses/by-nc/4.0/",
                      "https://creativecommons.org/licenses/by-sa/4.0/",
                      "https://creativecommons.org/licenses/by-nd/4.0/",
                      "Noncommercial https://creativecommons.org/licenses/by/4.0/",
                      "https://creativecommons.org/licenses/by/4.0/ https://creativecommons.org/licenses/by-sa/4.0/"):
            with self.subTest(value=value):
                self.assertIsNone(licensed.license_id(value))

    def test_attribution_offsets_deduplication_and_book_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = {"name":"sample", "repo":"publisher/sample", "revision":"pinned", "file":"sample.gz"}
            raw = root / "raw/sample"
            raw.mkdir(parents=True)
            (raw/'DATASET_CARD.md').write_text('Publisher dataset card')
            base = {"source":"pressbooks", "metadata": {
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "author":"A. Author", "title":"Textbook", "url":"https://example.org/book/ch1", "book_url":"https://example.org/book"}}
            rows = []
            for text in ("A paragraph of clearly readable educational material.",
                         "A second chapter about mathematical concepts and examples.",
                         "  A PARAGRAPH of clearly readable educational material.  "):
                rows.append({**base, "text":text})
            rows.append({**base, "text":"Another chapter", "metadata":{**base['metadata'], "license":"unknown"}})
            with gzip.open(raw/source['file'], "wt") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            result = licensed.prepare(source, root, set(), {}, 5, quality="licensed")
            self.assertEqual(result['counts']['drop_duplicate'], 1)
            self.assertEqual(result['counts']['drop_license'], 1)
            audit = [json.loads(line) for line in (root/'sample/attribution.jsonl').read_text().splitlines()]
            self.assertEqual(len(audit), 2)
            self.assertEqual(len({a['split'] for a in audit}), 1)
            for a in audit:
                text = (root/'sample'/f"{a['split']}.txt").read_bytes()[a['byte_offset']:a['byte_offset']+a['bytes']]
                self.assertEqual(hashlib.sha256(text).hexdigest(), a['text_sha256'])
                self.assertEqual(a['author'], 'A. Author')


if __name__ == '__main__':
    unittest.main()
