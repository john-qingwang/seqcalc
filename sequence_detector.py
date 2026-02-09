from collections import namedtuple
import copy
import enum
import itertools
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
import numpy as np

INCREMENT = 1.0

MASS_LIB_FILENAME = 'aa_mass.txt'

_DEFAULT_TOL = 10.0


ClippedSeq = namedtuple('ClippedSeq', ['mass', 'indices'])


class ClipMode(enum.Enum):
  """Defines the mode of clipping."""
  LEFT = 'left'
  RIGHT = 'right'
  BOTH = 'both'


def _find_sequence_for_target_mass_legacy(
    seq: np.ndarray,
	target: float,
	tol: float,
) -> Optional[ClippedSeq]:
  """Finds start/end indices in an ascending seq that equals target in tol.

  Args:
    seq: A floating point array in ascending order.
    target: Target floating point number to be found in the sequence.
    tol: The tolerance for determining if the sequence hits the target.

  Returns:
    The start and end indices (inclusive) of the sequence that fits the
    target. If there are multiple subsequences that meets the target, only
    the first one from the left will be returned.
    If nothing found within the tolerance, `None` will be returned.
  """
  buf = seq
  for i in range(len(seq) - 1):
    j = np.searchsorted(buf, target)
    if j > 0 and np.abs(buf[j - 1] - target) <= tol:
      return ClippedSeq(mass=buf[j - 1], indices=((i, i + j - 1)))
    elif j < len(buf) and np.abs(buf[j] - target) <= tol:
      return ClippedSeq(mass=buf[j], indices=((i, i + j)))
    buf = seq[i + 1:] - seq[i]
  return None


def _find_sequence_for_target_mass(
    mass: np.ndarray,
	target: float,
	tol: float,
    mode: ClipMode = ClipMode.LEFT,
) -> Optional[ClippedSeq]:
  """Finds start/end indices in a protein mass sequence that equals target in tol.

  Looks for sequence(s) breaks starting from the 2 ends.

  Args:
    mass: A floating point array specifying the mass of a protein sequence.
    target: Target floating point number to be found in the sequence.
    tol: The tolerance for determining if the sequence hits the target.

  Returns:
    The start and end indices (inclusive) of the sequence that fits the
    target. If there are multiple subsequences that meets the target, only
    the first one from the left will be returned.
    If nothing found within the tolerance, `None` will be returned.
  """
  def search_target_index(seq, t):
    j = np.searchsorted(seq, t)
    if j > 0 and np.abs(seq[j - 1] - target) < tol:
      return j - 1
    elif j < len(seq) and np.abs(seq[j] - target) < tol:
      return j
    else:
      return None

  if mode == ClipMode.BOTH:
    left_cumsum = _cumsum_mass_sequence(mass, 'left')
    right_cumsum = _cumsum_mass_sequence(mass, 'right')

    left_target = 0.0
    
    while (left_target := left_target + INCREMENT) < target:
      l = search_target_index(left_cumsum, left_target)
      if l is None:
        continue
      right_target = target - left_cumsum[l]
      r = search_target_index(right_cumsum, right_target)
      if r is None:
        continue
      return ClippedSeq(
              mass=left_cumsum[l] + right_cumsum[r],
              indices=[(0, l), (r, len(mass) - 1)])

  else:
    mass_cumsum = _cumsum_mass_sequence(mass, mode.value)
    if mode == ClipMode.RIGHT:
      target = -target
      mass_cumsum = -mass_cumsum

    j = search_target_index(mass_cumsum, target)

    if j is None:
      return None
    else:
      if mode == ClipMode.LEFT:
        idx = [(0, j),]
      elif mode == ClipMode.RIGHT:
        idx = [(j, len(mass) - 1),]
      else:
        raise ValueError(f'Unsupported mode {mode.value}')
      return ClippedSeq(mass=np.abs(mass_cumsum[j]), indices=idx)

  return None


def _find_subsequence_combinations_for_total_mass(
  seqs: Sequence[np.ndarray],
	total_mass: float,
	tol: float,
    modes: Optional[Sequence[ClipMode]] = None,
) -> Optional[List[List[Tuple[int, int]]]]:
  """Finds all possible combinations of sequences that sums to target."""
  # Find the possible sub-sequences from the all sequences.
  seq_info = []
  with tqdm(total = len(seqs) * (total_mass // INCREMENT)) as pbar:
    pbar.set_description('Populate')
    for i in range(len(seqs)):
      seq_info_current = []

      target = INCREMENT

      while target < total_mass:
        mode = modes[i] if modes is not None else None
        clipped_seq = _find_sequence_for_target_mass(seqs[i], target, tol, mode)
        target += INCREMENT
        if clipped_seq is not None:
          seq_info_current.append(clipped_seq)
        
        pbar.update(1)

      # This function assumes that the combination of subsequences have to
      # come from all sequences in the input.
      if not seq_info_current:
        return None

      seq_info.append(seq_info_current)

  # Find all possible sub-sequences that the sum of them equals the total mass within
  # the tolerance.
  seq_idx = []
  with tqdm(total = np.product([len(clips) for clips in seq_info])) as pbar:
    pbar.set_description('Confirm')
    for seq_info_i in itertools.product(*seq_info):
        mass = np.sum([info.mass for info in seq_info_i])
        if np.abs(mass - total_mass) <= tol:
            seq_idx.append(seq_info_i)

        pbar.update(1)

  return seq_idx if seq_idx else None


def _find_all_subsequence_combinations(
    seqs: Sequence[np.ndarray],
	total_mass: float,
	tol: float,
    modes: Optional[Sequence[ClipMode]] = None,
) -> Optional[List[List[Tuple[int, int]]]]:
  """Finds all possible combinations of sequences that sums equals target."""
  seqs_buf = [[seq, None] for seq in seqs]
  seq_idx = []
  for seq_comb in tqdm(itertools.product(*seqs_buf)):
    valid_idx = [i for i in range(len(seq_comb)) if seq_comb[i] is not None]
    buf = _find_subsequence_combinations_for_total_mass(
        [seq for seq in seq_comb if seq is not None], total_mass, tol, modes)
    if buf is not None:
      for seq in buf:
        res = [None,] * len(seq_comb)
        for i in range(len(valid_idx)):
          res[valid_idx[i]] = seq[i]
        seq_idx.append(res)

  return seq_idx if seq_idx else None


def _convert_protein_to_mass_sequence(
    seq: str,
    mass_lib: Dict[str, float],
) -> np.ndarray:
  """Converts a protein sequence into an array of molecular mass."""
  i = 0
  mass = []
  while i < len(seq):
    m = seq[i]
    if m == '[':
      while (i := i + 1) < len(seq) and seq[i] != ']':
        m += seq[i]
      m += ']'
    mass.append(mass_lib[m])
    i += 1

  return np.array(mass)


def _cumsum_mass_sequence(seq: np.ndarray, side: str = 'left') -> np.ndarray:
  """Finds the cummulative sum of a  from a side."""
  if side == 'left':
    return np.cumsum(seq)
  elif side == 'right':
    return np.flip(np.cumsum(np.flip(seq)))
  else:
    raise ValueError(f'Unknown side {side}. Available options are: "left", "right".')


def _parse_mass_library(filename: str = MASS_LIB_FILENAME) -> Dict[str, float]:
  """Parses a mass library file and converted into dictionary."""
  lib = {}
  with open(filename, 'r') as f:
    while l := f.readline():
      buf = [v.replace(' ', '') for v in l.split('=')]
      lib.update({buf[0]: float(buf[1].replace(',', ''))})

  return lib


def _parse_sequences(filename: str) -> Sequence[str]:
  """Parses sequences from a text file."""
  seqs = []
  modes = []
  with open(filename, 'r') as f:
    while l := f.readline():
      l = l.replace('\n', '')
      val = l.split(':')

      m_mode = val[0].replace(' ', '')
      if m_mode == 'left':
        mode = ClipMode.LEFT
      elif m_mode == 'right':
        mode = ClipMode.RIGHT
      elif m_mode == 'both':
        mode = ClipMode.BOTH
      else:
        raise ValueError(
            f'Unknown mode: {m_mode}. Available options: "left", "right", "both"')

      modes.append(mode)
      seqs.append(val[1].replace(' ', ''))

  return seqs, modes


def main(argv) -> None:
  """Writes detected subsequences to file."""

  # Parse the inputs.
  kwargs = dict(arg.split('=') for arg in argv)

  seqs_filename = kwargs.get('sequence_filename', None)
  if seqs_filename is None:
    raise ValueError('The sequence file is not provided.')
  seqs, modes = _parse_sequences(seqs_filename)

  mass_lib = _parse_mass_library(kwargs.get('mass_lib_filename', MASS_LIB_FILENAME))

  output_filename = kwargs.get('output_filename', 'output.txt')

  tol = float(kwargs.get('tol', _DEFAULT_TOL))

  target_mass = float(kwargs.get('target_mass', -1))
  if target_mass < 0.0:
    raise ValueError('A positive target mass need to be provided.')

  # Computes the subsequence combinations.
  mass = [_convert_protein_to_mass_sequence(seq, mass_lib) for seq in seqs]

  output = _find_all_subsequence_combinations(mass, target_mass, tol, modes)

  # Get back the molecule sequences and dump it to file.
  if output is not None:
    all_m_seqs = set()
    for seq_comb in output:
      m_seqs = ''
      total = 0.0
      for i in range(len(seq_comb)):
        seq = seq_comb[i]
        m_sub_seq = '('
        if seq is not None:
          total += seq.mass
          for idx in seq.indices:
            m_sub_seq += (seqs[i][idx[0]:idx[1] + 1] + ',')
        m_sub_seq += ')'
        m_seqs += (m_sub_seq + ',')
      m_seqs = (
          f'{target_mass:7.2f},{total:7.2f},{total - target_mass:7.2f}:' +
          m_seqs)

      all_m_seqs.add(m_seqs)

    with open(output_filename, 'w') as f:
      for m_seqs in all_m_seqs:
        m_seqs += '\n'
        f.write(m_seqs)

    print(f'Output file written: {output_filename}.')
  else:
    print('No clipped sequence is found matching the target mass.')


if __name__ == '__main__':
  main(sys.argv[1:])

