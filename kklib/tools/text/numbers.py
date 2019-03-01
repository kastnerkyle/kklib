# -*- coding: utf-8 -*-
""" from https://github.com/keithito/tacotron """

import re
from .number_to_words import NumberToWords

n2w = NumberToWords()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  """
  WARNING:
  For now this only handles days of the month type numbers...
  """
  # st nd rd th 
  piece = m.group(0)
  if len(piece) > 3:
      # do the dumbest possible thing
      pre = piece[:-3]
      suffix = piece[-3:]
      s = suffix
      num = int(pre + "0")
      prefix = n2w.convert(num)
      if s == "0th":
          r = "th"
      elif s == "1st":
          r = "first"
      elif s == "1th":
          prefix = ""
          r = "eleventh"
      elif s == "2nd":
          r = "second"
      elif s == "2th":
          prefix = ""
          r = "twelth"
      elif s == "3rd":
          r = "third"
      elif s == "3th":
          prefix = ""
          r = "thirteenth"
      elif s == "4th":
          if piece[-4] == "1":
              prefix = ""
              r = "fourteenth"
          else:
              r = "fourth"
      elif s == "5th":
          if piece[-4] == "1":
              prefix = ""
              r = "fifteenth"
          else:
              r = "fifth"
      elif s == "6th":
          if piece[-4] == "1":
              prefix = ""
              r = "sixteenth"
          else:
              r = "sixth"
      elif s == "7th":
          if piece[-4] == "1":
              prefix = ""
              r = "seventeenth"
          else:
              r = "seventh"
      elif s == "8th":
          if piece[-4] == "1":
              prefix = ""
              r = "eighteenth"
          else:
              r = "eighth"
      elif s == "9th":
          if piece[-4] == "1":
              prefix = ""
              r = "nineteenth"
          else:
              r = "ninth"
      else:
          raise ValueError("Unknown ordinal expansion for {}".format(m))
  else:
      prefix = ""
      suffix = piece[-3:]
      s = suffix
      if s == "1st":
          r = "first"
      elif s == "2nd":
          r = "second"
      elif s == "3rd":
          r = "third"
      elif s == "4th":
          r = "fourth"
      elif s == "5th":
          r = "fifth"
      elif s == "6th":
          r = "sixth"
      elif s == "7th":
          r = "seventh"
      elif s == "8th":
          r = "eighth"
      elif s == "9th":
          r = "ninth"
      else:
          raise ValueError("Unknown ordinal expansion for {}".format(m))

  if prefix != "":
      return prefix + suffix
  else:
      return suffix


def _expand_number(m):
  num = int(m.group(0))
  return n2w.convert(num)

'''
_inflect = inflect.engine()
def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))


def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')
'''

def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text
