"""The note checker's evaluator (operations/build_gsm8k_notes.py) agrees with src/calc.h.

The cases are tests/test_calc.h's, so a note the checker counts right is one
the calculator can finish, with the same result.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'operations'))
from build_gsm8k_notes import evaluate, written  # noqa: E402


def calc(expression):
    v = evaluate(expression)
    return None if v is None else written(v)


class CalcNotes(unittest.TestCase):
    def test_exact_as_calc_h(self):
        for expression, expected in [
                ('\r1\t+2\n', '3'), ('\f1 + 2\v', '3'), ('48/2', '24'), ('7/2*2', '7'), ('1/3*3', '1'),
                ('0.1+0.2', '0.3'), ('-(2+3)*4/8', '-2.5'), ('12.5000', '12.5'), ('0/-2', '0'),
                ('3-2-1', '0'), ('2+3*4', '14'), ('.5 + 1.', '1.5'),
                ('9223372036854775807/9223372036854775807', '1'), ('1/8', '0.125'),
                ('1/1000000000000000000', '0.000000000000000001'), ('18*.4', '7.2'),
                ('4611686018427387904+4611686018427387903', '9223372036854775807')]:
            self.assertEqual(calc(expression), expected, expression)

    def test_rejected_as_calc_h(self):
        for expression in ['', ' ', '1/0', '1/3', '2^3', 'sqrt(4)', '2x3', '1,000', '2(3)', '1e3', '1..2',
                           '(2+3', '2+3)', '2+', '1=1', 'nan', 'inf', 'a+2', '9223372036854775808',
                           '9223372036854775807+1', '1/0.0000000000000000001', '560//10',
                           '3037000500*3037000500', '-9223372036854775807-1',
                           '(' * 129 + '1' + ')' * 129, '1' * 4097]:
            self.assertIsNone(calc(expression), expression[:60])

    def test_depth_bound_is_calc_h_s(self):
        self.assertEqual(calc('(' * 127 + '1' + ')' * 127), '1')


if __name__ == '__main__':
    unittest.main()
