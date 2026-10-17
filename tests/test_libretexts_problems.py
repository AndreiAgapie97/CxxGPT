#!/usr/bin/env python3
"""The rules that pair a LibreTexts exercise with its answer key.

operations/build_libretexts_problems.py cannot check the books' answers (no
solver), so a wrong pairing would go unseen: these pages are made up in the
layouts the real pages have, one rule each.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'operations'))
spec = importlib.util.spec_from_file_location('build_libretexts_problems',
                                              root / 'operations/build_libretexts_problems.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class Answers(unittest.TestCase):
    def test_one_number(self):
        for text, want in [('58 hardback books', '58'), ('six boys', '6'), (r'\(\$1,250\)', '1250'),
                           ('12%', '12'), (r'\(−\$28\)', '-28'), ('x = 5', '5'), ('57.00', '57'),
                           ('0.80', '0.8'), ('The difference in temperatures was 9 degrees.', '9'),
                           (r'The number is \(4\).', '4'), (r'\(1036.73\) miles per hour', '1036.73'),
                           (r'\(43^{\circ}\)', '43'), ('120 mg', '120')]:
            self.assertEqual(B.answer_number(text), want, text)

    def test_unclear_answers_are_dropped(self):
        for text in ['35 adults, 82 children', '8 and 40 gallons', r'\(18, \;24\)', r'\(\frac{3}{4}\)',
                     'Answers will vary.', 'a. $7.20', 'EXAMPLE 11', '3:15',
                     'We will answer this question in two different ways.',
                     'The pole is a focus and e = 1 for a parabola', '3 per 100', r'\(3\dfrac{1}{2}\) miles',
                     r'\(\dfrac{3}{4}\) cup']:
            self.assertIsNone(B.answer_number(text), text)


class Questions(unittest.TestCase):
    def test_word_problem(self):
        q = 'Vanessa brought 8 packs of hot dogs to a family reunion. Each pack has 10 hot dogs. How many hot dogs did Vanessa bring?'
        self.assertEqual(B.question_text(['54. ' + q]), q)

    def test_fractions_and_degrees_in_the_question(self):
        q = (r'A man and a woman standing \(3\dfrac{1}{2}\) miles apart spot a hot air balloon at the same time. '
             r'The angle of elevation from the man is \(27^{\circ}\) and from the woman \(41^{\circ}\). '
             r'Find the altitude of the balloon to the nearest foot, given \(\frac{3}{4}\) of a mile is 3960 feet.')
        self.assertEqual(B.question_text(['69) ' + q]),
                         'A man and a woman standing 3 1/2 miles apart spot a hot air balloon at the same time. '
                         'The angle of elevation from the man is 27° and from the woman 41°. Find the altitude '
                         'of the balloon to the nearest foot, given 3/4 of a mile is 3960 feet.')
        self.assertIsNone(B.question_text([r'A tank holds \(\frac{x}{2}\) gallons more than the 12 gallon tank. '
                                           r'How many gallons does the big tank hold if it is full?']))

    def test_not_word_problems(self):
        for lines in [
            ['In the number 3,392, how many ones are there?', 'In the number 10,046, how many thousands are there?'],
            ['Simplify the expression 4 times 7 plus 3 times 2 and tell me what you get in the end?'],
            ['If these trends were to continue, what would be the median home value in Indiana in 2010?'],
            ['How many more dry holes were drilled in 1960 than in 1975 according to the records kept?'],
            ['where t is the child\'s age in years. What dose should be given to a child 8 years old at 15 units?'],
            ['Elissa earned $ 152.84 this week, which was $ 2 . .65 more than she earned last week. How much was it?'],
            ['a. 250% of 65 is what number? b. 8.2% of what amount is 2.87 dollars in the store today?'],
            ['The length of a rectangle is 5 feet longer than three times its width. Find the dimensions if the area is 138 square feet.'],
            ['Carolyn invests a total of $12,000 in two bonds paying 10.5% and 12%. How much was invested at each rate?'],
        ]:
            self.assertIsNone(B.question_text(lines), lines)


class Pages(unittest.TestCase):
    def test_layouts(self):
        page = '\n'.join([
            'Exercise \\(\\PageIndex{6}\\)',  # Burzynski and Ellis: two problems, one answer
            'In the number 3,392, how many ones are there?',
            'In the number 10,046, how many thousands are there?',
            '- Answer', '-', '0',
            '3. There are \\(16\\) girls in a school club. The number of girls is four more than twice the number of boys. Find the number of boys.',
            '- Answer', '-', 'six boys',
            'Try It: no header survives, so the paragraph before the answer is the question.',
            'The temperature in Denver was −6 degrees at lunchtime. By sunset the temperature had dropped to −15 degrees. What was the difference in the lunchtime and sunset temperatures?',
            '- Answer', '-', 'The difference in temperatures was 9 degrees.',
            'EXAMPLE 10',
            'Carolyn invests a total of $12,000 in two municipal bonds. The interest earned on the two investments was $1,335 last year. How much is in the larger bond?',
            '- Answer', '-', 'EXAMPLE 11',
            'Daria bought a bracelet at original cost $16 to sell in her store. She marked the price up 45%. What was the list price of the bracelet?',
            '- Answer', '-', 'a. $7.20', 'b. $23.20',
        ])
        found, dropped = B.problems_of(page)
        self.assertEqual([a for _, a in found], ['6', '9'])
        self.assertTrue(found[0][0].startswith('There are 16 girls'))
        self.assertTrue(found[1][0].startswith('The temperature in Denver was -6 degrees'))
        self.assertEqual(dropped, 3)  # the pair, the empty answer before "EXAMPLE 11", the two-part key

    def test_answer_with_a_colon_or_on_the_dash_line(self):
        page = '\n'.join([
            r'65) Jamal wants to save \(\$54,000\) for a down payment on a home. How much will he need to invest in an account with \(8.2\%\) APR, compounding daily, in order to reach his goal in \(5\) years?',
            '- Answer:', '-', r'\(\$35,838.76\)',
            'The length of a rectangle is 120 yards and the width is 50 yards. What is the perimeter of the rectangle in yards?',
            '- Answer', '- 340 yd',
            'The length of a rectangle is 62 feet and the width is 48 feet. Find (a) the perimeter and (b) the area of it.',
            '- Answer a', '-', '220 ft', '- Answer b', '- 2976 sq. ft',
        ])
        found, dropped = B.problems_of(page)
        self.assertEqual([a for _, a in found], ['35838.76', '340'])
        self.assertEqual(dropped, 2)  # "Answer a" and "Answer b": parts of one exercise

    def test_unanswered_exercise_does_not_take_the_next_answer(self):
        page = '\n'.join([
            '4. There are \\(18\\) Cub Scouts in Troop 645, three more than five times the number of leaders. How many leaders are there?',
            'Jeff lines up 42 adult bicycles and nine less than three times as many children\'s bicycles. How many children\'s bicycles are there?',
            '- Answer', '-', '117',
        ])
        self.assertEqual(B.problems_of(page), ([], 1))


class Copies(unittest.TestCase):
    def test_reworded_copy_is_the_same_problem(self):
        a = B.fingerprint('A rectangle has an area of 15 square feet and a width of 5 feet. How long is it?', '3')
        b = B.fingerprint('The rectangle has area 15 square feet and width five feet. How long is the rectangle?', '3')
        self.assertTrue(B.same_problem(a, b))
        other = B.fingerprint('A rectangle has an area of 18 square feet and a width of 6 feet. How long is it?', '3')
        self.assertFalse(B.same_problem(a, other))  # other numbers: another problem with the same answer


if __name__ == '__main__':
    unittest.main()
