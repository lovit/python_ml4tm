from collections import Counter
from collections import defaultdict
from soynlp.normalizer import normalize


class SubwordTokenizer:
    """
    어절의 왼쪽에 위치하는 모든 subwords 중 등록된 subwords 만을 return 하는 토크나이저 입니다.
    Tokenizer 는 반드시 str 을 포함하는 set 혹은 dictionary 를 입력해야 합니다.
        >>> subwords = {'엔진', '엔진오일', '교체'}
        >>> tokenizer = SubwordTokenizer(subwords)
    '엔진오일'과 '교체'가 모두 포함되어 있더라도 '엔진오일교체'를 유의미한 subword 로 인식하지는 않습니다.
        >>> tokenizer.tokenize('엔진오일을 교체하려 합니다 엔진오일교체')
        $ ['엔진', '엔진오일', '교체', '엔진오일']
    tokenize 함수에 입력되는 문장은 soynlp.normalizer.normalize 함수를 거쳐 알파벳, 숫자,
    그리고 한글을 제외한 다른 글자는 모두 제거 됩니다.
    """
    def __init__(self, subwords=None):
        if not subwords:
            raise ValueError('subwords must be non-empty dictionary')
        self.subwords = subwords

    def tokenize(self, sent, flatten=True):
        """
        Arguments
        ---------
        sent : str
            Input sentence
        flatten : Boolean
            If True, it returns list of str
            Else, it returns nested list.
        Returns
        -------
        sent : "list of str" or "list of list of str"
        Usage
        -----
            >>> subwords = {'엔진', '엔진오일', '교체'}
            >>> tokenizer = SubwordTokenizer(subwords)
            >>> tokenizer.tokenize('엔진오일을 교체하려 합니다 엔진오일교체')
            $ ['엔진', '엔진오일', '교체', '엔진오일']
        """
        def tokenize(word):
            subwords = find_subwords(word)
            return [sub for sub in subwords if sub in self.subwords]
        sent = normalize(sent, alphabet=True, number=True)
        sent = [tokenize(word) for word in sent.split()]
        if flatten:
            sent = [sub for subwords in sent for sub in subwords]
        return sent

def find_subwords(word):
    """
    Argument
    --------
    word : str
        Input token
    Returns
    -------
    subwords : list of str
        All possible left-side subwords
    Usage
    -----
        >>> find_subword('abcde')
        $ ['a', 'ab', 'abc', 'abcd', 'abcde']
    """
    return [word[:i] for i in range(1, len(word)+1)]

def count_subword(documents, min_count=50):
    """
    Arguments
    ---------
    documents : list of str
        Document set
    min_count : int
        Minumum frequency of subword
        Default is 50
    Returns
    -------
    subword counter : {str:int}
    """
    counter = defaultdict(int)
    for doc in documents:
        for word in doc.split():
            for sub in find_subwords(word):
                counter[sub] += 1
    counter = {sub:count for sub, count in counter.items() if count >= min_count}
    return counter

def train_droprate(subword_counter):
    """
    droprate(s) = 1 - max (#(s + c)) / #(s)
    Argument
    --------
    subword_counter : {str:int}
    Returns
    -------
    droprates : {str:float}
        Dictionary of droprate score
    """
    droprates = defaultdict(lambda: 0)
    for longer, longer_count in subword_counter.items():
        if len(longer) <= 2:
            continue
        shorter = longer[:-1]
        shorter_count = subword_counter.get(shorter, 0)
        if shorter_count == 0:
            continue
        droprate = longer_count / shorter_count
        droprates[shorter] = max(droprates[shorter], droprate)
    return dict(droprates)

def droprate_to_word_score(droprates):
    """
    word score (s) = 1 - droprate(s)
    Argument
    --------
    droprates : {str:float}
        Dictionary of droprate score
    Returns
    -------
    word score dictionary : {str:float}
    """
    return {sub:1-dr for sub, dr in droprates.items()}

def word_score_by_droprate(documents, min_count=50):
    """
    Training function for word score by droprate
    Arguments
    ---------
    documents : list of str
        Document set
    min_count : int
        Minumum frequency of subword
        Default is 50
    Returns
    -------
    word score dictionary : {str:float}
    """
    counter = count_subword(documents, min_count)
    droprates = train_droprate(counter)
    word_scores = droprate_to_word_score(droprates)
    return word_scores
