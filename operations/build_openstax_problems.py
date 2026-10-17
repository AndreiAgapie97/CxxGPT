"""Word problems with their answer keys from OpenStax's math books, at commits from while they were CC BY.

The source is OpenStax's own book sources on GitHub (github.com/openstax/
osbooks-*, one CNXML file per section, modules/<id>/index.cnxml, and one
collection file per book), where many exercises carry their answer inline:
<exercise><problem><para>Seven school busses ... How many students were
there?</para></problem><solution><para>There were 308 students.</para>
</solution></exercise>. LibreTexts' copies of these books (build_libretexts_
problems.py) lost those answers.

Licence: OpenStax relicensed these books CC BY-NC-SA 4.0 in 2026 (each
repository's LICENSE file between 2026-03-12 and 2026-04-08, the books'
md:license lines on 2026-04-23). PINS below are, per repository, the last
commit before its LICENSE change; at each of them the LICENSE file and every
collection's md:license are CC BY 4.0, and this script checks both before it
reads anything. A CC licence cannot be withdrawn from copies already made,
and the owner decided on 2026-10-17 that such a commit counts as CC BY for
this project. Never read a later commit: those are non-commercial.

--fetch DIR makes a new folder DIR with one checkout per repository at its
pinned commit: one commit deep, without file contents until checkout
(--filter=blob:none), and only modules/*/index.cnxml and collections/* (no
images): about 90 MB.

What is kept, besides build_libretexts_problems.py's rules for the question
(question_text: one paragraph of prose with a question and at least two
numbers, no figure, table or other exercise it depends on) and the answer
(answer_number: exactly one number), EXCLUDED, the radians check and its
GSM8K dev/test filters (a 13-word run, or the numbers and the answer):
  - an <exercise> outside worked examples, whose <problem> is one <para>
    and whose one <solution> is one <para>; the paragraph holds only text,
    emphasis, and math that is plain numbers ($, %, commas, minus signs,
    a fraction of two whole numbers, degrees): a variable, a power, a list,
    a figure, a table, a link or a footnote drops it;
  - the topic label in bold before the question ("Field trip Seven school
    busses ...") is taken off;
  - the instructions before a group of exercises ("Use the following
    information to answer the next three exercises", "In the following
    exercises, use the table ...") do not point at a table, figure,
    data or another exercise: the question would not be whole without them;
  - the answer is not a bound or an estimate ("More than $42,857.14 worth
    of jewelry", "about 12 years");
  - a rounding instruction among the group's instructions ("In the
    following exercises, solve. Round answers to the nearest hundredth.")
    is added to the question; a decimal answer to compound interest, growth,
    decay, probabilities or trigonometry needs one ("How long will it take
    for her money to double?" is keyed 11.6, it is 11.58...); a circle,
    cylinder, cone or sphere needs the question to say 3.14 or pi
    (Prealgebra keys them with 3.14, said only in its text);
  - the problem is not in data/libretexts-problems or libretexts-problems-2
    (same numbers and most of the same words, whatever the answer): LibreTexts
    holds the first editions of several of these books, and those problems
    are already in the earlier sets. A problem the second edition changed
    the numbers of is a new problem.
Within OpenStax, a module in several books (College Algebra 2e, Algebra and
Trigonometry 2e and Precalculus 2e share most sections) is read once, and
copies of a question across modules are one problem, dropped if they
disagree on the answer, with build_libretexts_problems.py's near-copy rule.
Each problem keeps its repository, commit, module id and title, the books
that hold the module, the licence, and a link to the file at the commit.

These problems have answers but no worked solutions: for evaluation and
self-train (the model writes the work, the answer key checks it), not for
the training mix as they are. OpenStax's books are in the training data
through LibreTexts' copies of the first editions, so this is not a held-out
test either. Usage, from the repository root:

  python -I operations/build_openstax_problems.py --fetch data/openstax-raw
  python -I operations/build_openstax_problems.py --raw data/openstax-raw \\
      --heldout data/targeted-math/heldout.jsonl --cxxgpt build/cxxgpt.exe \\
      --tok data/licensed-10b/tok.bpe --out data/openstax-problems \\
      --libretexts data/libretexts-problems/problems.jsonl,data/libretexts-problems-2/problems.jsonl
"""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder out
from build_libretexts_problems import (EXCLUDED, answer_number, fingerprint, key, question_text,  # noqa: E402
                                       radians, same_problem)
from build_targeted_math import grams, numbers, sha256  # noqa: E402

# The last commit before each repository's LICENSE became CC BY-NC-SA (the date of that change).
PINS = {
    'osbooks-prealgebra-bundle': '77a933c672e9c48c746eb802d588a9a7ddfd6e0d',  # 2026-03-12
    'osbooks-college-algebra-bundle': '4922e46ebc04326979e19392ccc2384cb8b9076c',  # 2026-03-18
    'osbooks-contemporary-mathematics': 'fa7c1a151783958fd2c114da66831e3d57ce2321',  # 2026-03-18
    'osbooks-introductory-statistics-bundle': '50ec91749080bb040b57bb40c34337cacc994cba',  # 2026-03-19
    'osbooks-principles-finance': '3b7fc65c527ef51dbee24689691f19947fb72fb2',  # 2026-04-08
    'osbooks-algebra-1': 'a277f9bb2abcc90d201a66879973463a45e5eaa4',  # 2025-04-07
    # Its LICENSE is still CC BY; md:license changed on 2026-04-23 (7dea80ae): the commit before that.
    'osbooks-statistics': 'ee0efc177a01145d28d9429fc309e331fab9c566',
}
CC_BY = 'http://creativecommons.org/licenses/by/4.0/'
NS = {'c': 'http://cnx.rice.edu/cnxml', 'm': 'http://www.w3.org/1998/Math/MathML',
      'md': 'http://cnx.rice.edu/mdml', 'col': 'http://cnx.rice.edu/collxml'}
C, M = '{%s}' % NS['c'], '{%s}' % NS['m']

# Instructions before a group of exercises that hand them something the question does not hold.
INSTRUCTION_UNSEEN = re.compile(r'\b(figures?|tables?|graphs?|charts?|diagrams?|maps?|pictures?|data|information|'
                                r'scenario|survey|sample|previous|preceding|above|below|shown|examples?|'
                                r'exercises? \d|next \w+ (exercises|problems|questions)|'
                                r'use (the|your) (results?|answers?|equations?|functions?|models?)|'
                                r'distributions?)\b|~', re.I)  # ("X ~ U(1.5, 4.5)": the question's numbers)
ROUNDING = re.compile(r'\b(round|nearest|decimal places?|tenths?|hundredths?|thousandths?)\b', re.I)
# Answers that are rounded unless the question says otherwise: compound interest, growth and decay,
# probabilities, normal distributions, trigonometry ("How long until her money doubles?" is 11.58...).
ROUNDED_TOPIC = re.compile(r'\b(compound\w*|continuous(ly)?|annuity|IRA|doubles?|triples?|half-life|decay\w*|'
                           r'grows?|growth|probability|standard deviations?|normal|binomial|oscillat\w*|'
                           r'amplitude|elevation|depression|sine|cosine|tangent)\b|N\(', re.I)
# Prealgebra's circles, cylinders, cones and spheres use 3.14 for pi (64,108.33 for the silo, not
# 64,140.85), in its text, not in the exercise.
ROUND_SHAPE = re.compile(r'\b(circles?|circular|radius|radii|diameters?|circumference|cones?|cone-shaped|'
                         r'cylind\w*|spheres?|spherical|semicircl\w*)\b', re.I)
# Problems the hand audit found broken in the books themselves (the start of the question, and why), on top
# of build_libretexts_problems.py's EXCLUDED.
EXCLUDED_HERE = {
    'What is the average rate of return for the values that follow': 'growth factors or percents, the '
    'question does not say; the key of 4.2, 4.3, 4.5 says 33 (333 as factors, 4.33 as percents)',
}
BOUND = re.compile(r'\b(more|less|fewer|greater|larger|smaller) than\b|\bat (least|most)\b|\bbetween\b|'
                   r'\b(about|approximately|around|nearly|almost|roughly|over|under|up to)\b|[≈~≤≥]', re.I)
PLAIN_MO = {'(', ')', ',', '.', '−', '-', '–', '$', '%', '°', ''}


class Unusable(Exception):
    """A paragraph that is not plain prose with plain numbers."""


def math_text(el):
    """A MathML number as plain text ("$1,250", "−6", "3 1/2", "27°"), or Unusable."""
    tag = el.tag.replace(M, '')
    if tag in ('math', 'mrow', 'mstyle'):
        return ''.join(math_text(c) for c in el)
    if tag == 'mn':
        return (el.text or '').strip()
    if tag == 'mtext':
        return el.text or ''
    if tag == 'mspace':
        return ' '
    if tag == 'mo':
        s = (el.text or '').strip()
        if s not in PLAIN_MO:
            raise Unusable(f'mo {s}')
        return s
    if tag == 'mfrac' and len(el) == 2 and all(c.tag == M + 'mn' for c in el):
        return ' %s/%s' % (el[0].text.strip(), el[1].text.strip())  # "3 1/2" after a whole number
    if tag == 'msup' and len(el) == 2 and (el[1].text or '').strip() in ('∘', '°'):
        return math_text(el[0]) + '°'
    raise Unusable(tag)


def para_text(para, label=False):
    """A paragraph as plain text; with label, a leading bold topic ("Field trip") is taken off."""
    parts = [para.text or '']
    for i, child in enumerate(para):
        tag = child.tag
        if tag == C + 'emphasis':
            if child.get('effect') == 'bold' and label and i == 0 and not parts[0].strip():
                parts.append(child.tail or '')  # the topic label, not part of the question
                continue
            if len(child):
                raise Unusable('markup inside emphasis')
            parts.append(child.text or '')
        elif tag == M + 'math':
            text = math_text(child)
            if re.search(r'\w$', ''.join(parts)) and re.match(r'[\w$−-]', text):
                text = ' ' + text  # "The temperature is<m:math>−7°" reads "is -7°"
            parts.append(text)
        elif tag == C + 'term' and not len(child):
            parts.append(child.text or '')
        else:
            raise Unusable(tag.split('}')[-1])  # list, figure, table, link, footnote, sub, sup, newline
        parts.append(child.tail or '')
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def only_para(el):
    """The one <para> that is the whole of a <problem> or <solution>, or None."""
    kids = [c for c in el if c.tag != C + 'label' and c.tag != C + 'title']
    if len(kids) != 1 or kids[0].tag != C + 'para' or (el.text or '').strip() or (kids[0].tail or '').strip():
        return None
    return kids[0]


def exercise_pair(ex, instruction):
    """(question, answer) of one <exercise>, or the reason it is not kept."""
    problems, solutions = ex.findall(C + 'problem'), ex.findall(C + 'solution')
    if not solutions:
        return 'no answer'
    if len(problems) != 1 or len(solutions) != 1:
        return 'several problems or solutions'
    p, s = only_para(problems[0]), only_para(solutions[0])
    if p is None or s is None:
        return 'more than one paragraph'
    if instruction and INSTRUCTION_UNSEEN.search(instruction):
        return 'instructions point at a table, figure or other exercise'
    try:
        q_text, a_text = para_text(p, label=True), para_text(s)
    except Unusable:
        return 'not plain text and numbers'
    q = question_text([q_text])
    if q is None:
        return 'not a one-paragraph word problem'
    rounding = re.search(r'[^.]*\bround[^.]*\.', instruction or '', re.I)
    if rounding and not ROUNDING.search(q):
        q += ' ' + rounding.group().strip()  # "In the following exercises, solve. Round to the nearest tenth."
    if BOUND.search(a_text):
        return 'the answer is a bound or an estimate'
    a = answer_number(a_text)
    if a is None:
        return 'the answer is not one number'
    if ROUND_SHAPE.search(q) and not re.search(r'3\.14|π|\bpi\b|radian', q):
        return 'a circle or cone: pi or 3.14, the question does not say'
    if '.' in a and ROUNDED_TOPIC.search(q) and not ROUNDING.search(q):
        return 'a rounded answer, and the question does not say how'
    return q, a


def module_problems(root):
    """(exercise id, question, answer) of a module, and a Counter of why the others were not kept."""
    found, why = [], collections.Counter()

    def walk(node):
        instruction = None  # the last paragraph before an exercise, at the same level
        for child in node:
            if child.tag == C + 'para':
                instruction = ''.join(child.itertext())
            elif child.tag == C + 'exercise':
                got = exercise_pair(child, instruction)
                if isinstance(got, str):
                    why[got] += 1
                else:
                    found.append((child.get('id', ''), *got))
            elif child.tag == C + 'example':
                why['inside a worked example'] += len(child.findall('.//' + C + 'exercise'))
            else:
                walk(child)  # a section, or a note ("Try It"), has its own instructions
    content = root.find(C + 'content')
    if content is not None:
        walk(content)
    return found, why


def collections_of(repo):
    """Module id -> titles of the books that hold it, and each book's licence URL."""
    books, licences = collections.defaultdict(list), {}
    for path in sorted((repo / 'collections').glob('*.collection.xml')):
        root = ET.parse(path).getroot()
        title = root.find('col:metadata/md:title', NS).text.strip()
        licences[title] = root.find('col:metadata/md:license', NS).get('url')
        for mod in root.iter('{%s}module' % NS['col']):
            if title not in books[mod.get('document')]:
                books[mod.get('document')].append(title)
    return books, licences


def git(repo, *args):
    return subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(repo), *args], capture_output=True,
                          text=True, encoding='utf-8', check=True).stdout


def check_pin(repo):
    """The checkout's commit, its date and LICENSE's first line; stops unless it is the pin and CC BY."""
    head = git(repo, 'rev-parse', 'HEAD').strip()
    assert head == PINS[repo.name], f'{repo} is at {head}, not the CC BY commit {PINS[repo.name]}'
    licence = git(repo, 'show', 'HEAD:LICENSE').splitlines()[0].strip()
    assert licence == 'Attribution 4.0 International', f'{repo}: LICENSE is {licence!r}'
    return head, git(repo, 'log', '-1', '--format=%cI').strip(), licence


def fetch(dest):
    dest = pathlib.Path(dest)
    assert not dest.exists(), f'{dest} exists; choose a new folder'
    for name, commit in PINS.items():
        repo = dest / name
        repo.mkdir(parents=True)
        git(repo, 'init', '-q')
        git(repo, 'remote', 'add', 'origin', f'https://github.com/openstax/{name}.git')
        git(repo, 'sparse-checkout', 'set', '--no-cone', '/modules/*/index.cnxml', '/collections/*')
        git(repo, 'fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', commit)
        git(repo, '-c', 'advice.detachedHead=false', 'checkout', '-q', 'FETCH_HEAD')
        print(name, *check_pin(repo), flush=True)


def heldout_filters(path):
    """build_libretexts_problems.py's GSM8K dev/test checks: (shares a 13-word run, has the numbers and answer)."""
    held, held_numbers = set(), set()
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        problem = json.loads(line)
        held.update(grams(problem['question']))
        nums = numbers(problem['question'])
        if len(nums) >= 2:
            held_numbers.add((tuple(nums), problem['answer'].split('####')[-1].strip().replace(',', '')))

    def same_numbers(question, answer):
        nums = numbers(question)
        return any((tuple(k), answer) in held_numbers for k in [nums] + [nums[:i] + nums[i + 1:]
                                                                         for i in range(len(nums))])
    return (lambda q: bool(grams(q) & held)), same_numbers


def words_and_numbers(question):
    """fingerprint() without the answer: a problem LibreTexts has, whatever answer key it gives."""
    _, nums, words = fingerprint(question, '')
    return tuple(nums), words


def in_libretexts(paths):
    """Whether a question is a problem of the earlier LibreTexts sets (same numbers, most of the same words)."""
    by_numbers = collections.defaultdict(list)
    for path in paths:
        for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
            nums, words = words_and_numbers(json.loads(line)['question'])
            by_numbers[nums].append(words)

    def check(question):
        nums, words = words_and_numbers(question)
        return any(len(words & w) >= 0.5 * len(words | w) for w in by_numbers.get(nums, ()))
    return check


def build(raw, held_run, held_numbers, libretexts):
    """The kept problems, the counts, per-repository facts."""
    stats, repos = collections.Counter(), {}
    kept, seen, answers = [], {}, collections.defaultdict(set)
    for name in PINS:
        repo = pathlib.Path(raw) / name
        commit, date, licence = check_pin(repo)
        books, book_licences = collections_of(repo)
        assert all(url == CC_BY for url in book_licences.values()), f'{name}: md:license {book_licences}'
        repos[name] = {'commit': commit, 'commit_date': date, 'LICENSE': licence, 'books': book_licences}
        for module in sorted(books):  # modules in no book (drafts, retired sections) are not read
            path = repo / 'modules' / module / 'index.cnxml'
            if not path.exists():
                stats['modules a book names but the commit lacks'] += 1
                continue
            root = ET.parse(path).getroot()
            title = (root.findtext(C + 'title') or '').strip()
            pairs, why = module_problems(root)
            stats['exercises'] += len(pairs) + sum(why.values())
            for reason, n in why.items():
                stats['not kept: ' + reason] += n
            for ex_id, q, a in pairs:
                stats['found'] += 1
                if any(q.startswith(start) for start in [*EXCLUDED, *EXCLUDED_HERE]):
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
                if held_run(q):
                    stats['dropped: 13-word run of GSM8K dev/test'] += 1
                    continue
                if held_numbers(q, a.replace(',', '')):
                    stats['dropped: GSM8K dev/test numbers and answer'] += 1
                    continue
                item = {'question': q, 'answer': '#### ' + a, 'final': a, 'repo': f'github.com/openstax/{name}',
                        'commit': commit, 'module': module, 'module_title': title, 'exercise': ex_id,
                        'books': books[module], 'license': CC_BY,
                        'url': f'https://github.com/openstax/{name}/blob/{commit}/modules/{module}/index.cnxml',
                        'source': 'OpenStax CNXML at a CC BY commit', 'copies': 1}
                seen[k] = item
                kept.append(item)
    stats['dropped: copies of the question disagree on the answer'] = sum(len(answers[key(i['question'])]) > 1
                                                                          for i in kept)
    kept = [item for item in kept if len(answers[key(item['question'])]) == 1]
    unique, buckets = [], collections.defaultdict(list)  # same_problem needs the same answer and numbers
    for item in kept:
        mark = fingerprint(item['question'], item['final'])
        bucket = buckets[(mark[0], tuple(mark[1]))]
        twin = next((u for u, u_mark in bucket if same_problem(mark, u_mark)), None)
        if twin:
            twin['copies'] += item['copies']
            stats['near-copies of a kept question'] += 1
        else:
            bucket.append((item, mark))
            unique.append(item)
    kept = []
    for item in unique:
        if libretexts(item['question']):
            stats['dropped: already in the LibreTexts sets'] += 1
        else:
            kept.append(item)
    stats['kept'] = len(kept)
    return kept, stats, repos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', help='a new folder: check out every pinned commit there, then stop')
    ap.add_argument('--raw', help='the folder --fetch made')
    ap.add_argument('--heldout', help='GSM8K dev and test questions (heldout.jsonl)')
    ap.add_argument('--cxxgpt')
    ap.add_argument('--tok', help="the model's tokenizer")
    ap.add_argument('--out', help='a new folder')
    ap.add_argument('--libretexts', default='', help='comma-separated problems.jsonl of the LibreTexts sets')
    ap.add_argument('--audit', type=int, default=50, help='problems drawn for a hand audit')
    ap.add_argument('--audit-note', help="the hand audit's result, for the manifest")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.fetch)
        return
    for need in ('raw', 'heldout', 'cxxgpt', 'tok', 'out'):
        assert getattr(args, need), f'--{need} is needed'
    out = pathlib.Path(args.out)
    args.cxxgpt = str(pathlib.Path(args.cxxgpt).resolve())  # Windows does not search relative paths
    assert not out.exists(), f'{out} exists; choose a new folder'
    lt_paths = [p for p in args.libretexts.split(',') if p]
    kept, stats, repos = build(args.raw, *heldout_filters(args.heldout), in_libretexts(lt_paths))
    print(dict(stats), flush=True)

    out.mkdir(parents=True)
    with open(out / 'problems.jsonl', 'w', encoding='utf-8', newline='\n') as dest:
        for item in kept:
            dest.write(json.dumps(item, ensure_ascii=False) + '\n')
    subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(out / 'problems.jsonl'), '--out', str(out / 'problems'),
                    '--bpe', args.tok, '--task', 'openstax'], check=True)
    # The audit draws by each question's hash, so a filter that drops a problem leaves the rest of the draw.
    audit = sorted(kept, key=lambda item: hashlib.sha256(item['question'].encode()).hexdigest())[:args.audit]
    with open(out / 'audit-sample.txt', 'w', encoding='utf-8', newline='\n') as dest:
        for i, item in enumerate(audit, 1):
            dest.write(f"{i}. {item['question']}\n   answer: {item['final']}\n   {item['url']} {item['exercise']}\n\n")
    per_book = collections.Counter(book for item in kept for book in item['books'])
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': 'github.com/openstax/osbooks-* at the commits below (modules/*/index.cnxml, collections/*)',
        'repos': repos,
        'builder': 'operations/build_openstax_problems.py',
        'builder_sha256': sha256(__file__),  # the script that ran, which can differ from the commit's
        'builder_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                         check=True).stdout.strip(),
        'tokenizer_sha256': sha256(args.tok),
        'libretexts_sets': {p: sha256(p) for p in lt_paths},
        'licence_note': 'CC BY 4.0: each commit is the last before its repository relicensed CC BY-NC-SA '
                        '(LICENSE and md:license both CC BY 4.0 there); the owner decided on 2026-10-17 that '
                        'such a commit counts as CC BY for this project',
        'use': 'evaluation and self-train only: final answers without worked solutions; the first editions '
               'of several of these books are in the training mix through LibreTexts, so this is not a '
               'held-out test',
        'counts': dict(stats),
        'per_book': dict(per_book.most_common()),
        'per_repo': dict(collections.Counter(item['repo'] for item in kept).most_common()),
        'problems_sha256': sha256(out / 'problems.jsonl'),
        'audit': args.audit_note or f'{len(audit)} problems (the smallest sha256 of the question) in audit-sample.txt, not yet read',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('done', out, len(kept), 'problems', flush=True)


if __name__ == '__main__':
    main()
