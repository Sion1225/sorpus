import pytest

from sorpus.replacer import SpecialWordReplacer

def test_replace_special_char():
    replacer = SpecialWordReplacer(['New York', 'El Nino'], ' ', '•')
    assert replacer.mask_words('New York is affected by El Nino.') == 'New•York is affected by El•Nino.'

def test_replace_special_words_between_same_chars():
    replacer = SpecialWordReplacer(['"apple"'], 'apple', 'green apple')
    assert replacer.mask_words('He said the word "apple" while pointing at an apple.') == 'He said the word "green apple" while pointing at an apple.'

def test_replace_special_words_between_difference_chars():
    replacer = SpecialWordReplacer(['qapplew'], 'apple', 'green apple')
    assert replacer.mask_words('He said the word qapplew while pointing at an apple.') == 'He said the word qgreen applew while pointing at an apple.'

def test_replace_special_words_repr():
    replacer = SpecialWordReplacer(['New York', 'El Nino'], ' ', '•')
    assert repr(replacer) == '<SpecialWordReplacer(special_words=[\'New York\', \'El Nino\'], target_word=\' \', replacement_char=\'•\')>'
