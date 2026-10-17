#!/usr/bin/env python3
"""The rules that pair an OpenStax CNXML exercise with its answer key.

operations/build_openstax_problems.py cannot check the books' answers (no
solver), so a wrong pairing would go unseen: these modules are made up in the
layouts the real CNXML has, one rule each.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'operations'))
spec = importlib.util.spec_from_file_location('build_openstax_problems', root / 'operations/build_openstax_problems.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

HEAD = '<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"><title>T</title><content>'


def module(body):
    return ET.fromstring(HEAD + body + '</content></document>')


def exercise(problem, solution='<para>There were 308 students.</para>'):
    sol = f'<solution>{solution}</solution>' if solution is not None else ''
    return f'<exercise id="e1"><problem>{problem}</problem>{sol}</exercise>'


BUSSES = ('Seven school busses let off their students in front of a museum in Washington, DC. '
          'Each school bus had 44 students. How many students were there?')


def pairs(body):
    found, why = B.module_problems(module(body))
    return [(q, a) for _, q, a in found], why


class Exercises(unittest.TestCase):
    def test_kept_with_its_label_taken_off(self):
        found, _ = pairs(exercise(f'<para><emphasis effect="bold">Field trip</emphasis> {BUSSES}</para>'))
        self.assertEqual(found, [(BUSSES, '308')])

    def test_math_numbers_are_plain_text(self):
        q = ('<para>Elissa earned <m:math><m:mrow><m:mo>$</m:mo><m:mn>152.84</m:mn></m:mrow></m:math> this week, '
             'which was <m:math><m:mn>$21.65</m:mn></m:math> more than she earned last week. The week before she '
             'earned <m:math><m:mrow><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow></m:math> dollars less, and rode '
             '<m:math><m:mn>2</m:mn><m:mfrac><m:mn>1</m:mn><m:mn>2</m:mn></m:mfrac></m:math> hours. '
             'How much did she earn last week?</para>')
        found, _ = pairs(exercise(q, '<para><m:math><m:mn>$131.19</m:mn></m:math></para>'))
        self.assertEqual(found[0][1], '131.19')
        self.assertIn('earned $152.84 this week', found[0][0])
        self.assertIn('-3 dollars', found[0][0])
        self.assertIn('rode 2 1/2 hours', found[0][0])

    def test_unanswered_or_worked_or_multipart_are_dropped(self):
        body = ''.join([
            exercise(f'<para>{BUSSES}</para>', solution=None),  # odd exercises only have keys
            exercise(f'<para>{BUSSES}</para>', '<para>Step 1. Read the problem.</para><para>There were 308.</para>'),
            exercise(f'<para>{BUSSES}</para><para>How many were girls?</para>'),
            f'<example>{exercise(f"<para>{BUSSES}</para>")}</example>',
        ])
        found, why = pairs(body)
        self.assertEqual(found, [])
        self.assertEqual(why, {'no answer': 1, 'more than one paragraph': 2, 'inside a worked example': 1})

    def test_variables_lists_and_links_are_dropped(self):
        for problem in [
            '<para>Timmy goes to the fair with $40. Each ride costs $2. How much money will he have left after '
            'riding <m:math><m:mi>n</m:mi></m:math> rides?</para>',
            '<para>The polynomial <m:math><m:msup><m:mi>t</m:mi><m:mn>2</m:mn></m:msup></m:math> gives the height '
            'of a ball 3 seconds after it is dropped from a 250-foot building. Find the height?</para>',
            '<para>When hired at a new job selling electronics, you are given two pay options: <list><item>Option A: '
            '$14,000 and 10%</item></list> How much would you need to sell for option A to be larger?</para>',
            '<para>Using the data in <link target-id="t1"/>, find the average rate of change between 2005 and '
            '2010 in dollars per year for the two families?</para>',
        ]:
            found, why = pairs(exercise(problem, '<para>15</para>'))
            self.assertEqual(found, [], problem)
            self.assertEqual(why, {'not plain text and numbers': 1}, problem)

    def test_answers_that_are_bounds_or_several_numbers(self):
        q = ('<para>You are offered a base salary of $14,000 a year with a commission of 10% of your sales, or '
             '$20,000 with 3%. How much jewelry would you need to sell for option A to produce a larger income?</para>')
        for answer, reason in [('More than $42,857.14 worth of jewelry', 'the answer is a bound or an estimate'),
                               ('about 12 years', 'the answer is a bound or an estimate'),
                               ('24.5 cm., 12.5 cm.', 'the answer is not one number')]:
            found, why = pairs(exercise(q, f'<para>{answer}</para>'))
            self.assertEqual((found, why), ([], {reason: 1}), answer)

    def test_instructions_that_point_elsewhere(self):
        story = ('<para>Use the following information to answer the next three exercises: a phone battery lasts '
                 '10 hours on average.</para>')
        q = exercise('<para>Find the probability that the mean time 64 batteries last is greater than 12 hours '
                     'when 8 of them are new?</para>', '<para>0.0003</para>')
        found, why = pairs(story + q)
        self.assertEqual((found, why), ([], {'instructions point at a table, figure or other exercise': 1}))
        # An instruction that only says what to do is no reason; it holds for the exercises after it.
        found, _ = pairs('<para>In the following exercises, solve.</para>' + exercise(f'<para>{BUSSES}</para>'))
        self.assertEqual(len(found), 1)
        found, why = pairs('<section><para>In the following exercises, use the table.</para></section>'
                           + exercise(f'<para>{BUSSES}</para>'))
        self.assertEqual(len(found), 1)  # the instruction of another section
        found, why = pairs('<para>The distribution can be written as X ~ U(1.5, 4.5).</para>' + exercise(
            '<para>Find the probability that a randomly selected home has more than 3,000 square feet given '
            'that it has more than 2,000 square feet.</para>', '<para>0.6</para>'))
        self.assertEqual((found, why), ([], {'instructions point at a table, figure or other exercise': 1}))

    def test_rounding_instruction_joins_the_question(self):
        found, _ = pairs('<para>In the following exercises, solve. Round answers to the nearest hundredth.</para>'
                         + exercise('<para>Find the area of a triangle with base 8.3 meters and height 6.1 '
                                    'meters. What is it?</para>', '<para>25.32 square meters</para>'))
        self.assertEqual(found, [('Find the area of a triangle with base 8.3 meters and height 6.1 meters. What '
                                  'is it? Round answers to the nearest hundredth.', '25.32')])

    def test_rounded_answers_need_the_rounding(self):
        double = ('<para>Elise invests $4500 in an account that compounds interest monthly and earns 6%. How long '
                  'will it take for her money to double{}</para>')
        found, why = pairs(exercise(double.format('?'), '<para>11.6 years</para>'))
        self.assertEqual((found, why), ([], {'a rounded answer, and the question does not say how': 1}))
        found, _ = pairs('<para>In the following exercises, solve. Round to the nearest tenth.</para>'
                         + exercise(double.format('?'), '<para>11.6 years</para>'))
        self.assertEqual(found[0][1], '11.6')
        found, _ = pairs(exercise(double.format(' if it earns $270 a year?'), '<para>17 years</para>'))
        self.assertEqual(found[0][1], '17')  # a whole number is not rounded by the key

    def test_circles_need_to_say_pi(self):
        cone = ('<para>What is the volume of a cone-shaped silo that is 50 feet tall and 70 feet across at the '
                'base{}? Round answers to the nearest hundredth.</para>')
        found, why = pairs(exercise(cone.format(''), '<para>64,108.33 cubic feet</para>'))
        self.assertEqual((found, why), ([], {'a circle or cone: pi or 3.14, the question does not say': 1}))
        found, _ = pairs(exercise(cone.format(', using 3.14 for pi'), '<para>64,108.33 cubic feet</para>'))
        self.assertEqual(found[0][1], '64108.33')

    def test_space_before_math_after_a_word(self):
        found, _ = pairs(exercise('<para>Rene is on an Alpine hike. The temperature is<m:math><m:mrow><m:mo>−</m:mo>'
                                  '<m:mtext>7</m:mtext><m:mtext>°</m:mtext></m:mrow></m:math>. Her sleeping bag is '
                                  'rated to -20°. How much can the temperature change before it is too cold?</para>',
                                  '<para>13°</para>'))
        self.assertIn('The temperature is -7°.', found[0][0])


class Books(unittest.TestCase):
    def test_a_module_in_two_books(self):
        col = ('<col:collection xmlns:col="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml">'
               '<col:metadata><md:title>{}</md:title><md:license url="http://creativecommons.org/licenses/by/4.0/">'
               'CC BY</md:license></col:metadata><col:content><col:module document="m1"/>{}</col:content>'
               '</col:collection>')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'collections').mkdir()
            (Path(tmp) / 'collections/a.collection.xml').write_text(col.format('College Algebra 2e', ''))
            (Path(tmp) / 'collections/b.collection.xml').write_text(
                col.format('Precalculus 2e', '<col:subcollection><col:content><col:module document="m2"/>'
                                             '</col:content></col:subcollection>'))
            books, licences = B.collections_of(Path(tmp))
        self.assertEqual(books, {'m1': ['College Algebra 2e', 'Precalculus 2e'], 'm2': ['Precalculus 2e']})
        self.assertEqual(set(licences.values()), {B.CC_BY})


class Copies(unittest.TestCase):
    def test_libretexts_copy_with_another_answer_is_still_a_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'problems.jsonl'
            path.write_text('{"question": "Seven school busses let off their students at a museum in Washington, '
                            'DC. Each bus had 44 students. How many students were there?", "final": "300"}\n')
            seen = B.in_libretexts([path])
        self.assertTrue(seen(BUSSES.replace('Seven', '7')))
        self.assertFalse(seen(BUSSES.replace('44', '45')))


if __name__ == '__main__':
    unittest.main()
