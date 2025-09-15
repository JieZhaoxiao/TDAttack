import torch
import numpy as np

class ScoreParams:
    """Parameters for sequence alignment scoring system"""

    def __init__(self, gap, match, mismatch):
        """
        Initialize scoring parameters

        Args:
            gap: Penalty for gap insertion
            match: Reward for character match
            mismatch: Penalty for character mismatch
        """
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        """Calculate score for character comparison"""
        return self.mismatch if x != y else self.match


def get_matrix(size_x, size_y, gap):
    """
    Create initial alignment matrix with gap penalties

    Args:
        size_x: Length of sequence X
        size_y: Length of sequence Y
        gap: Gap penalty value

    Returns:
        Initialized alignment matrix
    """
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    """
    Create matrix for tracking alignment operations

    Args:
        size_x: Length of sequence X
        size_y: Length of sequence Y

    Returns:
        Initialized traceback matrix
    """
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1  # Left direction
    matrix[1:, 0] = 2  # Up direction
    matrix[0, 0] = 4  # Start position
    return matrix


def global_align(x, y, score_params):
    """
    Perform global sequence alignment using Needleman-Wunsch algorithm

    Args:
        x: First sequence
        y: Second sequence
        score_params: ScoreParams object with alignment scores

    Returns:
        alignment_matrix: Matrix with alignment scores
        trace_back: Matrix with alignment directions
    """
    alignment_matrix = get_matrix(len(x), len(y), score_params.gap)
    trace_back = get_traceback_matrix(len(x), len(y))

    # Fill alignment matrices
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left_score = alignment_matrix[i, j - 1] + score_params.gap
            up_score = alignment_matrix[i - 1, j] + score_params.gap
            diag_score = alignment_matrix[i - 1, j - 1] + score_params.mis_match_char(x[i - 1], y[j - 1])

            alignment_matrix[i, j] = max(left_score, up_score, diag_score)

            # Store direction in traceback matrix
            if alignment_matrix[i, j] == left_score:
                trace_back[i, j] = 1  # Left
            elif alignment_matrix[i, j] == up_score:
                trace_back[i, j] = 2  # Up
            else:
                trace_back[i, j] = 3  # Diagonal


def get_aligned_sequences(x, y, trace_back):
    """
    Generate aligned sequences from traceback matrix

    Args:
        x: First sequence
        y: Second sequence
        trace_back: Traceback matrix with alignment directions

    Returns:
        x_seq: Aligned first sequence
        y_seq: Aligned second sequence
        mapper_y_to_x: Mapping between sequence positions
    """
    x_seq, y_seq = [], []
    mapper_y_to_x = []
    i, j = len(x), len(y)

    # Trace back from end to start of matrix
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            # Diagonal move - match/mismatch
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i, j = i - 1, j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i, j] == 1:
            # Left move - gap in x
            x_seq.append('-')
            y_seq.append(y[j - 1])
            j -= 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i, j] == 2:
            # Up move - gap in y
            x_seq.append(x[i - 1])
            y_seq.append('-')
            i -= 1
        elif trace_back[i, j] == 4:
            break

    mapper_y_to_x.reverse()
    return x_seq[::-1], y_seq[::-1], torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    """
    Create token mapping between two sequences

    Args:
        x: Source sequence
        y: Target sequence
        tokenizer: Tokenizer for sequence encoding
        max_len: Maximum length for output sequences

    Returns:
        mapper: Token position mapping
        alphas: Attention weight indicators
    """
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)  # Alignment scoring parameters

    # Perform global alignment
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]

    # Create output tensors
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()

    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))

    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    """
    Generate token mappers for prompt refinement

    Args:
        prompts: List of prompt sequences
        tokenizer: Tokenizer for sequence encoding
        max_len: Maximum sequence length

    Returns:
        mappers: Tensor of token position mappers
        alphas: Tensor of attention weight indicators
    """
    x_seq = prompts[0]
    mappers, alphas = [], []

    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)

    return torch.stack(mappers), torch.stack(alphas)


def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Get token indices corresponding to specific words

    Args:
        text: Input text sequence
        word_place: Word position(s) to find
        tokenizer: Tokenizer for sequence encoding

    Returns:
        Array of token indices corresponding to specified words
    """
    split_text = text.split(" ")

    # Handle string or integer word positions
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]

    out = []
    if len(word_place) > 0:
        # Get tokenized words
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]

        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0

    return np.array(out)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    """
    Create token replacement mapping between sequences

    Args:
        x: Original prompt
        y: Modified prompt
        tokenizer: Tokenizer for sequence encoding
        max_len: Maximum sequence length

    Returns:
        Mapper matrix for token replacements
    """
    words_x = x.split(' ')
    words_y = y.split(' ')

    # Validate prompt lengths
    if len(words_x) != len(words_y):
        raise ValueError(f"Prompts must have equal word count. Prompt A: {len(words_x)}, Prompt B: {len(words_y)}")

    # Find differing word positions
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]

    # Initialize mapping matrix
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0

    # Build mapping between token positions
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    """
    Generate token replacement mappers for prompt sequences

    Args:
        prompts: List of prompt sequences
        tokenizer: Tokenizer for sequence encoding
        max_len: Maximum sequence length

    Returns:
        Stacked tensor of replacement mappers
    """
    x_seq = prompts[0]
    mappers = []

    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)

    return torch.stack(mappers)