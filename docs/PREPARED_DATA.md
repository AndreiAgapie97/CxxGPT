# Prepared CxxGPT data

The complete bundle and split parts contain the same data. Choose either the
complete archive or all numbered parts; do not download both sets unnecessarily.
Extract them into the same parent folder as the supplied CxxGPT repository.
Use 7-Zip (Zstandard support) or GNU tar with zstd installed:

    tar --zstd -xf CxxGPT-prepared-data.tar.zst

For numbered parts, extract every CxxGPT-data-*.tar.zst into the same folder.
Their members have distinct paths. Check the provided SHA256SUMS first.
Uncompressed prepared data needs roughly 23GB; allow extra space for models.

data/licensed-10b/blocks contains native .tok shards for 10,000,016,185 training,
103,058,222 validation and 101,416,161 test tokens. Keep these splits separate.
Use the paired data/licensed-10b/tok.bpe tokenizer (16,384 vocabulary).
The prepared general corpus is tokenized; original downloaded JSON sources are
not duplicated here. Pinned source revisions, original checksums, licenses,
document attribution and token offsets are supplied. Internal dedup SQLite
build caches and superseded pilot datasets are excluded.

data/targeted-math contains the worked-math continuation: 963,089 generated
training problems and 1,000 generated development problems, in readable JSONL,
evaluation TXT and masked token formats. These were reconstructed on the free
cloud CPU from the original generator, seeds and held-out exclusion records;
BOTH token files match the original Pod training SHA-256 byte-for-byte.
original-training-manifest.json preserves the original provenance; manifest.json
records reconstruction. Human GSM8K train/dev/test files and MIT licensing are
also retained. Never mix held-out development/test files into training.

balanced-recipe.json preserves the approximately 53% worked-math /47% general
input sampling mixture, with corpus paths adapted to this download layout.
Sampling weights repeat data; they do not increase unique token counts. The
generated worked-math corpus is additional to the 10B licensed general corpus.

The lossless packaging process verified every archived file against its original
SHA-256 before replacing uncompressed copies in the space-limited workspace.
The complete bundle is independently reverified against every part's file list.
No Runpod resource or GPU spending was needed to prepare these downloads.
