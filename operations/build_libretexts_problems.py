"""Word problems with their answer keys from CC BY math textbooks on LibreTexts.

The source is Common Pile's copy of LibreTexts (huggingface.co/datasets/
common-pile/libretexts, one file, 00000_libretext.json.gz), which keeps each
page's licence and address. Only math, statistics and medicine pages (the
nursing books' dosage calculations) licensed CC BY (not share-alike, not
non-commercial) or public domain are read.

OpenStax's own math books (Prealgebra, Intermediate Algebra, Algebra and
Trigonometry, Precalculus, Contemporary Mathematics) were CC BY 4.0 until
OpenStax relicensed them CC BY-NC-SA 4.0 in 2026 (the LICENSE files of
github.com/openstax/osbooks-* on 2026-03-12 to 03-19, their md:license
lines on 2026-04-23). LibreTexts copied them
while they were CC BY, and a CC licence cannot be withdrawn from copies
already made, so those copies stay CC BY: that is what this script uses,
never openstax.org's current books.

What is kept: numbered exercises ("12. There are 16 girls ..."), "Exercise
N" blocks, and OpenStax's "Try It" exercises (their header did not survive
into the text, so the question is the one paragraph before the answer),
each with an "Answer" (or "Answer:") right after it, the answer on the next
line after a lone "-" or on the "- 340 yd" line itself, where
  - the question is one paragraph of prose (at least 10 words, a number,
    a question), with no figure, table, graph, other exercise or equation
    it depends on, and no LaTeX left after the plain numbers, fractions of
    whole numbers ("3 1/2") and degrees are unwrapped;
  - the answer holds exactly one number ("58 hardback books" is 58, "six
    boys" is 6, "$1,250" is 1250, "12%" is 12); answers with two numbers,
    fractions, ratios, times or no number are dropped ("per" is a unit:
    "60 miles per hour" is 60). A wrong pairing is
    worse than a lost problem, so anything unclear is dropped;
  - the question appears in no other kept page (course copies of the same
    book are common; a reworded copy has the same numbers, answer and most
    words), its copies all give the same answer, it shares no 13-word run
    with GSM8K dev or test (heldout.jsonl), and it does not have a GSM8K
    dev/test question's numbers and answer (build_targeted_math.py's rules);
  - it is not one of the problems the hand audit found wrong in the book
    (EXCLUDED), nor an angle answered in radians the question never asks for.
Each problem keeps its page's title, address, book and licence. The
audit-sample.txt it writes holds the 100 problems with the smallest sha256
of the question, for checking by hand (--audit-note puts the result in the
manifest). --previous names an earlier build's problems.jsonl: each problem
then says whether it is new (not the same problem as one there, by the
near-copy rule), and the audit draws from the new ones only.

These problems have answers but no worked solutions, so the token file is
for evaluation and self-train (the model writes the work, the answer key
checks it), not for the training mix as it is. LibreTexts is also in the
training data (textbooks-libretexts), so scores on this set are not a
held-out test. Usage, from the repository root:

  python -I operations/build_libretexts_problems.py --pile data/libretexts-raw/00000_libretext.json.gz \\
      --heldout data/targeted-math/heldout.jsonl --cxxgpt build/cxxgpt.exe \\
      --tok data/licensed-10b/tok.bpe --out data/libretexts-problems-2 \\
      --previous data/libretexts-problems/problems.jsonl

data/libretexts-problems (2026-10-17, 383 problems) was built before the
"Answer:" layouts, the medicine pages, fractions and degrees in questions,
"per" and a sentence's full stop after the answer ("The number is 4.").
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder out
from build_targeted_math import grams, numbers, sha256  # noqa: E402

ITEM = re.compile(r'^(?:\d+\s*[.)]\s+(?=\S)|Exercise \\\(\\PageIndex\{\d+\}\\\)\s*:?\s*)')
ANSWER = re.compile(r'^- ?Answers?\b(.*)$')
UNSEEN = re.compile(r'\b(figures?|tables?|graphs?|graphed|shown|below|above|diagrams?|charts?|pictures?|maps?|'
                    r'following|exercises?|examples?|previous|preceding|data set|sample|spreadsheet|'
                    r'calculator|estimate|approximate(ly)?|about how|trends?|the data)\b', re.I)
NOT_WORD = re.compile(r'^(simplify|evaluate|factor|graph|translate|write|explain|describe|round|estimate|use|'
                      r'name|list|determine whether|identify|convert|express|rewrite|plot|draw|label|sketch|'
                      r'compare|order|classify|verify|prove|show|state)\b', re.I)
SMALL = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
         'sixteen seventeen eighteen nineteen twenty').split()
TENS = {'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
WORD_NUMBERS = {w: i for i, w in enumerate(SMALL)} | TENS
# Math and statistics, and medicine for its dosage calculations ("The client weighs 176 lb. How many
# milligrams ...?"): a question needs two numbers to compute with, so recall questions do not pass.
HOSTS = re.compile(r'https://(math|stats|med)\.libretexts\.org/')
NUMBER = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])')


def allowed_licence(licence):
    """CC BY of any version, or public domain; never share-alike or non-commercial."""
    low = licence.lower()
    if 'public domain' in low or '/publicdomain/' in low:
        return True
    return low.startswith('creative commons - attribution - ') and '/licenses/by/' in low


def unwrap(text):
    """Plain text from LibreTexts' inline LaTeX, or None if anything but plain numbers is left."""
    text = text.replace('−', '-').replace('–', '-').replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"').replace('\u00a0', ' ')

    def inline(m):
        s = m.group(1)
        s = re.sub(r'\\(?:text|mathrm|textbf)\{([^{}]*)\}', r'\1', s)
        s = s.replace('\\$', '$').replace('\\%', '%').replace('\\,', '').replace('\\;', ' ').replace('\\ ', ' ')
        s = s.replace('{,}', ',').replace('\\!', '')
        # Fractions of whole numbers and degrees, as a question would print them: "3\dfrac{1}{2}" is
        # "3 1/2", "\frac{3}{4}" is "3/4", "27^{\circ}" is "27°". An answer with a "/" is still dropped.
        s = re.sub(r'(\d)\s*\\[dt]?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}', r'\1 \2/\3', s)
        s = re.sub(r'\\[dt]?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}', r'\1/\2', s)
        s = re.sub(r'\s*\^\s*(?:\{\s*\\circ\s*\}|\\circ)', '°', s)
        return s
    text = re.sub(r'\\\((.*?)\\\)', inline, text)
    text = text.replace('\\$', '$').replace('\\%', '%')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\s*\$\s*(?=\d)', '$-', text)  # "-$28" is minus 28 dollars
    if '\\' in text or '{' in text or '}' in text or '^' in text:
        return None
    return text


def answer_number(text):
    """The one number an answer key gives, as GSM8K writes it after '####', or None."""
    text = unwrap(text)
    if text is not None and len(text.split()) > 10:
        return None  # a sentence that explains, not a number
    if text is not None and re.match(r'([a-d][.)]\s|(example|exercise|solution|try it|section|chapter|lesson|'
                                     r'problem|step|figure|table)\b)', text, re.I):
        return None  # part a of several, or an empty answer followed by the next heading ("EXAMPLE 11")
    # ("per" is a unit, "1036.73 miles per hour"; a rate of two numbers, "3 per 100", has two numbers)
    if text is None or re.search(r'[/:±×=<>]|\bor\b|\band\b|\bto\b|\bremainder\b|\bR\d', text):
        if text is None or not re.fullmatch(r'[a-z]\s*=\s*-?[\d,.]+', text):  # "x = 5" is fine
            return None
        text = text.split('=')[1]
    plain = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    plain = re.sub(r'(?<=\d)\.$', '', plain.strip())  # a sentence's full stop: "The number is 4."
    found = NUMBER.findall(plain)
    words = [w for w in re.findall(r'[a-z]+(?:-[a-z]+)?', plain.lower())
             if w in WORD_NUMBERS or (w.split('-')[0] in TENS and w.split('-')[-1] in SMALL[1:10])]
    if len(found) == 1 and not words:
        n = found[0]
        if re.search(r'\d\s*%|percent', plain) and n.startswith('-'):
            return None
        # eval-reasoning compares the text after "####" exactly: "57.00" as GSM8K would write it, 57
        return n.rstrip('0').rstrip('.') if '.' in n else n
    if not found and len(words) == 1 and len(plain.split()) <= 4:  # "six boys", not "in two different ways"
        w = words[0]
        if w in WORD_NUMBERS:
            return str(WORD_NUMBERS[w])
        tens, ones = w.split('-')
        return str(TENS[tens] + SMALL.index(ones))
    return None


def question_text(lines):
    """The question of an exercise block, or None if it is not a one-paragraph word problem."""
    first = ITEM.sub('', lines[0]).strip()
    body = [first] if first else []
    body += [l.strip() for l in lines[1:] if l.strip()]
    if len(body) != 1 or body[0].startswith(('-', '|')):
        return None  # sub-parts, tables, two problems sharing one answer (Burzynski and Ellis' pairs)
    q = unwrap(body[0])
    if q is None or UNSEEN.search(q) or NOT_WORD.match(q) or not q[0].isupper():
        return None  # (a lowercase start is the end of a paragraph cut in two: "where t is the child's age")
    if re.search(r'\bdimensions\b|\bhow (many|much)\b.*\band how (many|much)\b', q, re.I):
        return None  # asks for two numbers
    if re.search(r'(^|\s)[a-d][.)] |\([a-d]\)|[ⓐⓑⓒ]', q):
        return None  # parts a, b, c with one answer between them
    if re.search(r'\b(at|of|in) each (rate|type|kind|size|account|investment|bond|fund)\b|\b(many|much) of each\b', q, re.I):
        return None  # asks for several amounts
    if re.search(r'\bfind (an?|the) (equation|expression|formula|function)\b', q, re.I):
        return None  # "Find an equation of the plane ...": the key "z = -2" is not a number to compute
    if re.search(r'\bproblems? \d', q, re.I):
        return None  # "The statistician in problem 78 ...": another problem's numbers
    if re.search(r'\d\s+\.\s|\.\s+\.|\$\s+\d+\s+\.', q):
        return None  # "$ 2 . .65": a broken copy of a number
    if re.search(r'[=<>]', q) or len(re.findall(r'\b[A-Za-z]{2,}\b', q)) < 10:
        return None
    given = re.findall(r'\d+(?:[.,]\d+)*|\b(?:' + '|'.join(SMALL[2:] + list(TENS)) + r')\b', q, re.I)
    if len([n for n in given if not re.fullmatch(r'1[89]\d\d|20[0-3]\d', n)]) < 2:
        return None  # "how long until the population reaches 900?", "how many more in 1960 than in 1975?"
        # need an earlier exercise's numbers or a table; years are not numbers to compute with
    if not (q.endswith('?') or re.search(r'\b(find|how many|how much|what|which)\b', q, re.I)):
        return None
    return q


def problems_of(text):
    """(question, answer) pairs of a page, and the count of answers that could not be used."""
    lines = text.split('\n')
    found, dropped, start = [], 0, None
    for i, line in enumerate(lines):
        if ITEM.match(line):
            start = i
            continue
        m = ANSWER.match(line)
        if not m:
            continue
        if start is None and i > 0:
            start = i - 1  # OpenStax's "Try It" exercises lost their header: the paragraph before the answer
        # "- Answer" or "- Answer:" (the colon in a multivariable calculus remix and Precalculus 1e), not
        # "- Answer a", one part of several.
        if start is None or m.group(1).strip() not in ('', ':') or i + 2 >= len(lines):
            dropped += 1
            start = None
            continue
        # The answer is one line: the next paragraph is already the next exercise or the book's text.
        # It is the line after a lone "-", or the rest of a "- 340 yd" line.
        if lines[i + 1].strip() == '-':
            ans = lines[i + 2]
        elif re.match(r'^- \S', lines[i + 1]) and not lines[i + 2].startswith('-'):
            ans = lines[i + 1][2:]  # (a second "- " line would be the key's second part)
        else:
            dropped += 1
            start = None
            continue
        q, a = question_text(lines[start:i]), None if ITEM.match(ans) else answer_number(ans)
        if q and a is not None:
            found.append((q, a))
        else:
            dropped += 1
        start = None
    return found, dropped


def key(q):
    return ' '.join(re.findall(r'[a-z0-9]+', q.lower()))


def fingerprint(question, final):
    """What two wordings of one problem share: the answer, the numbers in order, and the words."""
    words = key(question).split()
    return final, numbers(' '.join(str(WORD_NUMBERS.get(w, w)) for w in words)), set(words)


def same_problem(a, b):
    """Course remixes reword a little ("has an area of 15 ... 5 feet" and "has area 15 ... five feet"):
    the same answer, the same numbers and mostly the same words is one problem."""
    return a[0] == b[0] and a[1] == b[1] and len(a[2] & b[2]) >= 0.5 * len(a[2] | b[2])


# Problems the hand audit found broken in the books themselves (the start of the question, and why).
EXCLUDED = {
    'The height of a trapezoid is 18 centimeters and the bases are 17 and 8': 'the key says 255; it is 225',
    'Mitzi received some gourmet brownies as a gift': 'a stray "28%" in the question ("each 28% brownie")',
    'Rachael had a 21-ounce strawberry shake': 'the key says 1161; it is 1126',
    'Kurt can ride his bike for 30 miles': 'the key says 14; it is 34',
    'The fuel consumption (mpg) of a car varies inversely with its weight. A Ford Focus': '15.65 is 15.7, not 15.6',
    'The coordinates of an object are (5.682, 2.0521)': 'the key 6 is a rounded 6.04',
    # (the 2026-10-17 audit of the second build)
    'Two airplanes take off in different directions. One travels 300 mph': 'the key says 599.8; it is 292.4',
    'Find the linear speed of the moon': 'the key 2234 is 2234.65 cut short, not rounded',
    'Someone lends you a tire jack': "the wrench's length is in the exercise before",
}


def radians(q, a):
    """An angle answered in radians although the question does not ask for radians."""
    return bool(re.search(r'\bangle\b', q, re.I)) and '.' in a and abs(float(a)) < 6.3 and 'radian' not in q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pile', required=True, help="Common Pile's 00000_libretext.json.gz")
    ap.add_argument('--heldout', required=True, help='GSM8K dev and test questions (heldout.jsonl)')
    ap.add_argument('--cxxgpt', required=True)
    ap.add_argument('--tok', required=True, help="the model's tokenizer")
    ap.add_argument('--out', required=True, help='a new folder')
    ap.add_argument('--audit', type=int, default=100, help='problems drawn for a hand audit')
    ap.add_argument('--audit-note', help="the hand audit's result, for the manifest")
    ap.add_argument('--previous', help="an earlier build's problems.jsonl: each problem says whether it is new "
                                       "there, and the audit draws from the new ones only")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    args.cxxgpt = str(pathlib.Path(args.cxxgpt).resolve())  # Windows does not search relative paths
    assert not out.exists(), f'{out} exists; choose a new folder'

    held, held_numbers = set(), set()
    for line in pathlib.Path(args.heldout).read_text(encoding='utf-8').splitlines():
        problem = json.loads(line)
        held.update(grams(problem['question']))
        nums = numbers(problem['question'])
        if len(nums) >= 2:
            held_numbers.add((tuple(nums), problem['answer'].split('####')[-1].strip().replace(',', '')))

    def same_numbers(question, answer):
        nums = numbers(question)
        return any((tuple(k), answer) in held_numbers for k in [nums] + [nums[:i] + nums[i + 1:]
                                                                         for i in range(len(nums))])

    stats = collections.Counter()
    kept, seen, answers = [], {}, collections.defaultdict(set)
    with gzip.open(args.pile, 'rt', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            meta = doc['metadata']
            url = meta.get('url', '')
            if not re.match(HOSTS, url):
                continue
            stats['pages read'] += 1
            if not allowed_licence(meta.get('license', '')):
                stats['pages with another licence'] += 1
                continue
            pairs, dropped = problems_of(doc['text'])
            stats['answers not usable'] += dropped
            for q, a in pairs:
                stats['found'] += 1
                if any(q.startswith(start) for start in EXCLUDED):
                    stats['dropped: wrong in the book (hand audit)'] += 1
                    continue
                if radians(q, a):
                    stats['dropped: an angle in radians nobody asked for'] += 1
                    continue
                k = key(q)
                answers[k].add(float(a))
                if k in seen:
                    seen[k]['copies'] += 1
                    stats['copies of a kept question'] += 1
                    continue
                if grams(q) & held:
                    stats['dropped: 13-word run of GSM8K dev/test'] += 1
                    continue
                if same_numbers(q, a.replace(',', '')):
                    stats['dropped: GSM8K dev/test numbers and answer'] += 1
                    continue
                book = urllib.parse.unquote('/'.join(url.split('/')[3:6]))  # Bookshelves/Algebra/<book>
                item = {'question': q, 'answer': '#### ' + a, 'final': a, 'title': meta.get('title', ''),
                        'url': url, 'book': book, 'book_url': meta.get('book_url', ''), 'license': meta.get('license', ''),
                        'source': 'LibreTexts via Common Pile (common-pile/libretexts)', 'copies': 1}
                seen[k] = item
                kept.append(item)
    disagree = [item for item in kept if len(answers[key(item['question'])]) > 1]
    stats['dropped: copies of the question disagree on the answer'] = len(disagree)
    kept = [item for item in kept if len(answers[key(item['question'])]) == 1]
    unique = []
    for item in kept:
        mark = fingerprint(item['question'], item['final'])
        twin = next((u for u, u_mark in unique if same_problem(mark, u_mark)), None)
        if twin:
            twin['copies'] += item['copies']
            stats['near-copies of a kept question'] += 1
        else:
            unique.append((item, mark))
    kept = [item for item, _ in unique]
    stats['kept'] = len(kept)
    per_book = collections.Counter(item['book'] for item in kept)  # of the problems written, not those found
    licences = collections.Counter(item['license'] for item in kept)
    if args.previous:
        # The same problem in other words counts as old: the near-copy rule above decides.
        before = [fingerprint(p['question'], p['final']) for p in
                  map(json.loads, pathlib.Path(args.previous).read_text(encoding='utf-8').splitlines())]
        for item, mark in unique:
            item['new'] = not any(same_problem(mark, b) for b in before)
        stats['kept, new since --previous'] = sum(item['new'] for item in kept)
        stats['--previous problems not kept again'] = sum(not any(same_problem(b, mark) for _, mark in unique)
                                                          for b in before)
    print(dict(stats), flush=True)

    out.mkdir(parents=True)
    with open(out / 'problems.jsonl', 'w', encoding='utf-8', newline='\n') as dest:
        for item in kept:
            dest.write(json.dumps(item, ensure_ascii=False) + '\n')
    subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(out / 'problems.jsonl'), '--out', str(out / 'problems'),
                    '--bpe', args.tok, '--task', 'libretexts'], check=True)
    # The audit draws by each question's hash, so a filter that drops a problem leaves the rest of the draw.
    audit = sorted([item for item in kept if item.get('new', True)],
                   key=lambda item: hashlib.sha256(item['question'].encode()).hexdigest())[:args.audit]
    with open(out / 'audit-sample.txt', 'w', encoding='utf-8', newline='\n') as dest:
        for i, item in enumerate(audit, 1):
            dest.write(f"{i}. {item['question']}\n   answer: {item['final']}\n   {item['url']}\n\n")
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': 'huggingface.co/datasets/common-pile/libretexts, 00000_libretext.json.gz',
        'source_sha256': sha256(args.pile),
        'builder': 'operations/build_libretexts_problems.py',
        'builder_sha256': sha256(__file__),  # the script that ran, which can differ from the commit's
        'builder_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                         check=True).stdout.strip(),
        'tokenizer_sha256': sha256(args.tok),
        'previous': {'path': args.previous, 'sha256': sha256(args.previous)} if args.previous else None,
        'licences': dict(licences),
        'licence_note': "OpenStax's books were CC BY 4.0 until 2026 (then CC BY-NC-SA 4.0: the osbooks LICENSE "
                        'files from 2026-03-12, the md:license lines from 2026-04-23); '
                        'the LibreTexts copies used here were made under CC BY and keep it (Common Pile '
                        'uploaded its LibreTexts file on 2025-03-17)',
        'use': 'evaluation and self-train only: final answers without worked solutions; LibreTexts is in '
               'the training mix (textbooks-libretexts), so this is not a held-out test',
        'counts': dict(stats),
        'per_book': dict(per_book.most_common()),
        'problems_sha256': sha256(out / 'problems.jsonl'),
        'audit': args.audit_note or f'{len(audit)} problems (the smallest sha256 of the question) in audit-sample.txt, not yet read',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('done', out, len(kept), 'problems', flush=True)


if __name__ == '__main__':
    main()
