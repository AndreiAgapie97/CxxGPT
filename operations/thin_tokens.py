"""Keeps every K-th problem of a token file, for a lighter share in a mix.

night.sh's MIX repeats a file a whole number of times (file*N), so a smaller
share needs a smaller file. This writes one: the problems (documents, each
starting with the BOS token) number 0, K, 2K, ... of the input, with their
loss mask, in the same CXTOKEN1 format. Taking every K-th problem instead of
the first ones keeps the mix of topics and difficulties, in case the file is
written in order of them.

  python -I operations/thin_tokens.py data/gen-math/eq-train.tok data/thin/eq-train-3.tok 3
"""
import hashlib
import struct
import sys
from array import array

BOS = 1  # kBos in src/common.h


def main():
    src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert k >= 1
    with open(src, 'rb') as f:
        magic, vocab, has_mask, n = struct.unpack('<8sIIQ', f.read(24))
        assert magic == b'CXTOKEN1', f'{src} is not a token file'
        tokens = array('H')
        tokens.fromfile(f, n)
        mask = f.read(n) if has_mask else b''
        assert not has_mask or len(mask) == n, f'{src}: truncated mask'
    assert sys.byteorder == 'little'
    starts = [i for i in range(n) if tokens[i] == BOS]
    assert starts and starts[0] == 0, f'{src} does not start with a BOS token'
    ends = starts[1:] + [n]
    out_tokens, out_mask, kept = array('H'), bytearray(), 0
    for doc in range(0, len(starts), k):
        a, b = starts[doc], ends[doc]
        out_tokens.extend(tokens[a:b])
        if has_mask:
            out_mask.extend(mask[a:b])
        kept += 1
    with open(dst, 'wb') as f:
        f.write(struct.pack('<8sIIQ', b'CXTOKEN1', vocab, has_mask, len(out_tokens)))
        out_tokens.tofile(f)
        f.write(out_mask)
    with open(dst, 'rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    print(f'{dst}: {kept} of {len(starts)} problems (every {k}), {len(out_tokens)} of {n} tokens, sha256 {digest}')


if __name__ == '__main__':
    main()
