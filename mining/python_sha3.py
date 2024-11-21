#! /usr/bin/env python
# coding: utf-8

# The Keccak sponge function was designed by Guido Bertoni, Joan Daemen,
# Michaël Peeters and Gilles Van Assche. For more information, feedback or
# questions, please refer to their website: http://keccak.noekeon.org/
#
# Based on the implementation by Renaud Bauvin,
# from http://keccak.noekeon.org/KeccakInPython-3.0.zip
#
# Modified by Moshe Kaplan to be hashlib-compliant
#
# To the extent possible under law, the implementer has waived all copyright
# and related or neighboring rights to the source code in this file.
# http://creativecommons.org/publicdomain/zero/1.0/


import math


def sha3_224(data=None):
  return Keccak(c=448, r=1152, n=224, data=data)


def sha3_256(data=None):
  return Keccak(c=512, r=1088, n=256, data=data)


def sha3_384(data=None):
  return Keccak(c=768, r=832, n=384, data=data)


def sha3_512(data=None):
  return Keccak(c=1024, r=576, n=512, data=data)


class KeccakError(Exception):
  """Custom error Class used in the Keccak implementation"""

  def __init__(self, value):
    self.value = value

  def __str__(self):
    return repr(self.value)


class Keccak:
  def __init__(self, r, c, n, data=None):
    # Initialize the constants used throughout Keccak
    # bitrate
    self.r = r
    # capacity
    self.c = c
    # output size
    self.n = n

    self.b = r + c
    # b = 25*w
    self.w = self.b // 25
     # 2**l = w
    self.l = int(math.log(self.w, 2))

    self.n_r = 12 + 2 * self.l

    self.block_size = r
    self.digest_size = n

    # Initialize the state of the sponge
    # The state is made up of 25 words, each word being w bits.
    self.S = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]

    # A string of hexchars, where each char represents 4 bits.
    self.buffered_data = ""

    # Store the calculated digest.
    # We'll only apply padding and recalculate the hash if it's modified.
    self.last_digest = None

    if data:
      self.update(data)

  # Constants

  ## Round constants
  RC = [0x0000000000000001,
        0x0000000000008082,
        0x800000000000808A,
        0x8000000080008000,
        0x000000000000808B,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008A,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000A,
        0x000000008000808B,
        0x800000000000008B,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800A,
        0x800000008000000A,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008]

  ## Rotation offsets
  r = [[0,  36,   3,  41,  18],
       [1,  44,  10,  45,   2],
       [62,  6,  43,  15,  61],
       [28, 55,  25,  21,  56],
       [27, 20,  39,   8,  14]]

  @staticmethod
  def Round(A, RCfixed, w):
    """Perform one round of computation as defined in the Keccak-f permutation

    A: current state (5x5 matrix)
    RCfixed: value of round constant to use (integer)
    """

    #Initialization of temporary variables
    B = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    C = [0, 0, 0, 0, 0]
    D = [0, 0, 0, 0, 0]

    #Theta step
    for x in range(5):
      C[x] = A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4]

    for x in range(5):
      D[x] = C[(x - 1) % 5] ^ _rot(C[(x + 1) % 5], 1, w)

    for x in range(5):
      for y in range(5):
        A[x][y] = A[x][y] ^ D[x]

    #Rho and Pi steps
    for x in range(5):
      for y in range(5):
        B[y][(2 * x + 3 * y) % 5] = _rot(A[x][y], Keccak.r[x][y], w)

    #Chi step
    for x in range(5):
      for y in range(5):
        A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])

    #Iota step
    A[0][0] = A[0][0] ^ RCfixed

    return A

  @staticmethod
  def KeccakF(A, n_r, w):
    """Perform Keccak-f function on the state A

    A: 5x5 matrix containing the state, where each entry is a string of hexchars that is 'w' bits long
    n_r: number of rounds
    w: word size
    """

    for i in xrange(n_r):
      A = Keccak.Round(A, Keccak.RC[i] % (1 << w), w)

    return A

  ### Padding rule
  # This is a disgusting piece of code. Clean it.
  @staticmethod
  def pad10star1(M, n):
    """Pad M with the pad10*1 padding rule to reach a length multiple of r bits

    M: message pair (length in bits, string of hex characters ('9AFC...')
    n: length in bits (must be a multiple of 8)
    Example: pad10star1([60, 'BA594E0FB9EBBD30'],8) returns 'BA594E0FB9EBBD93'
    """

    [my_string_length, my_string] = M

    # Check the parameter n
    if n % 8 != 0:
      raise KeccakError.KeccakError("n must be a multiple of 8")

    # Check the length of the provided string
    if len(my_string) % 2 != 0:
      #Pad with one '0' to reach correct length (don't know test
      #vectors coding)
      my_string += '0'
    if my_string_length > (len(my_string) // 2 * 8):
      raise KeccakError.KeccakError("the string is too short to contain the number of bits announced")

    nr_bytes_filled = my_string_length // 8
    nbr_bits_filled = my_string_length % 8
    l = my_string_length % n
    if ((n - 8) <= l <= (n - 2)):
      if (nbr_bits_filled == 0):
        my_byte = 0
      else:
        my_byte = int(my_string[nr_bytes_filled * 2:nr_bytes_filled * 2 + 2], 16)
      my_byte = (my_byte >> (8 - nbr_bits_filled))
      my_byte = my_byte + 2 ** (nbr_bits_filled) + 2 ** 7
      my_byte = "%02X" % my_byte
      my_string = my_string[0:nr_bytes_filled * 2] + my_byte
    else:
      if (nbr_bits_filled == 0):
        my_byte = 0
      else:
        my_byte = int(my_string[nr_bytes_filled * 2:nr_bytes_filled * 2 + 2], 16)
      my_byte = (my_byte >> (8 - nbr_bits_filled))
      my_byte = my_byte + 2 ** (nbr_bits_filled)
      my_byte = "%02X" % my_byte
      my_string = my_string[0:nr_bytes_filled * 2] + my_byte
      while((8 * len(my_string) // 2) % n < (n - 8)):
        my_string = my_string + '00'
      my_string = my_string + '80'

    return my_string

  def update(self, arg):
    # Update the hash object with the string arg. Repeated calls are equivalent to a single call with the concatenation of all the arguments: m.update(a); m.update(b) is equivalent to m.update(a+b). arg is a normal bytestring.

    self.last_digest = None
    # Convert the data into a workable format, and add it to the buffer
    self.buffered_data += arg.encode('hex')

    # Absorb any blocks we can:
    if len(self.buffered_data) * 4 >= self.r:
      extra_bits = len(self.buffered_data) * 4 % self.r

      # An exact fit!
      if extra_bits == 0:
        P = self.buffered_data
        self.buffered_data = ""
      else:
        # Slice it up into the first r*a bits, for some constant a>=1, and the remaining total-r*a bits.
        P = self.buffered_data[:-extra_bits // 4]
        self.buffered_data = self.buffered_data[-extra_bits // 4:]

      #Absorbing phase
      for i in xrange((len(P) * 8 // 2) // self.r):
        to_convert = P[i * (2 * self.r // 8):(i + 1) * (2 * self.r // 8)] + '00' * (self.c // 8)
        P_i = _convertStrToTable(to_convert, self.w, self.b)

        # First apply the XOR to the state + block
        for y in xrange(5):
          for x in xrange(5):
            self.S[x][y] = self.S[x][y] ^ P_i[x][y]
        # Then apply the block permutation, Keccak-F
        self.S = Keccak.KeccakF(self.S, self.n_r, self.w)

  def digest(self):
    """Return the digest of the strings passed to the update() method so far.

    This is a string of digest_size bytes which may contain non-ASCII
    characters, including null bytes."""

    if self.last_digest:
      return self.last_digest

    # UGLY WARNING
    # Handle bytestring/hexstring conversions
    M = _build_message_pair(self.buffered_data.decode('hex'))

    # First finish the padding and force the final update:
    self.buffered_data = Keccak.pad10star1(M, self.r)
    self.update('')
    # UGLY WARNING over

    assert len(self.buffered_data) == 0, "Why is there data left in the buffer? %s with length %d" % (self.buffered_data, len(self.buffered_data) * 4)

    # Squeezing time!
    Z = ''
    outputLength = self.n
    while outputLength > 0:
      string = _convertTableToStr(self.S, self.w)
      # Read the first 'r' bits of the state
      Z = Z + string[:self.r * 2 // 8]
      outputLength -= self.r
      if outputLength > 0:
        S = KeccakF(S, verbose)

    self.last_digest = Z[:2 * self.n // 8].decode('hex')
    return self.last_digest

  def hexdigest(self):
    """Like digest() except the digest is returned as a string of hex digits

    This may be used to exchange the value safely in email or other
    non-binary environments."""
    return self.digest().encode('hex')

  def copy(self):
    # First initialize whatever can be done normally
    duplicate = Keccak(c=self.c, r=self.r, n=self.n)
    # Then copy over the state.
    for i in xrange(5):
      for j in xrange(5):
        duplicate.S[i][j] = self.S[i][j]
    # and any other stored data
    duplicate.buffered_data = self.buffered_data
    duplicate.last_digest = self.last_digest
    return duplicate


## Generic utility functions

def _build_message_pair(data):
  hex_data = data.encode('hex')
  size = len(hex_data) * 4
  return (size, hex_data)


def _rot(x, shift_amount, length):
  """Rotate x shift_amount bits to the left, considering the \
  string of bits is length bits long"""

  shift_amount = shift_amount % length
  return ((x >> (length - shift_amount)) + (x << shift_amount)) % (1 << length)

### Conversion functions String <-> Table (and vice-versa)


def _fromHexStringToLane(string):
  """Convert a string of bytes written in hexadecimal to a lane value"""

  #Check that the string has an even number of characters i.e. whole number of bytes
  if len(string) % 2 != 0:
    raise KeccakError.KeccakError("The provided string does not end with a full byte")

  #Perform the conversion
  temp = ''
  nrBytes = len(string) // 2
  for i in xrange(nrBytes):
    offset = (nrBytes - i - 1) * 2
    temp += string[offset:offset + 2]
  return int(temp, 16)


def _fromLaneToHexString(lane, w):
  """Convert a lane value to a string of bytes written in hexadecimal"""

  laneHexBE = (("%%0%dX" % (w // 4)) % lane)
  #Perform the conversion
  temp = ''
  nrBytes = len(laneHexBE) // 2
  for i in xrange(nrBytes):
    offset = (nrBytes - i - 1) * 2
    temp += laneHexBE[offset:offset + 2]
  return temp.upper()


def _convertStrToTable(string, w, b):
  """Convert a string of hex-chars to its 5x5 matrix representation

  string: string of bytes of hex-coded bytes (e.g. '9A2C...')"""

  # Check that the input paramaters are expected
  if w % 8 != 0:
    raise KeccakError("w is not a multiple of 8")

  # Each character in the string represents 4 bits.
  # The string should have exactly 'b' bits.
  if len(string) * 4 != b:
    raise KeccakError.KeccakError("string can't be divided in 25 blocks of w bits\
    i.e. string must have exactly b bits")

  #Convert
  output = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]

  bits_per_char = 2 * w // 8
  for x in xrange(5):
    for y in xrange(5):
      # Each entry will have b/25=w bits.
      offset = (5 * y + x) * bits_per_char
      # Store the data into the associated word.
      hexstring = string[offset:offset + bits_per_char]
      output[x][y] = _fromHexStringToLane(hexstring)
  return output


def _convertTableToStr(table, w):
  """Convert a 5x5 matrix representation to its string representation"""

  #Check input format
  if w % 8 != 0:
    raise KeccakError.KeccakError("w is not a multiple of 8")
  if (len(table) != 5) or any(len(row) != 5 for row in table):
    raise KeccakError.KeccakError("table must be 5x5")

  #Convert
  output = [''] * 25
  for x in xrange(5):
    for y in xrange(5):
      output[5 * y + x] = _fromLaneToHexString(table[x][y], w)
  output = ''.join(output).upper()
  return output
