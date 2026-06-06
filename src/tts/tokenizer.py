import cmudict
import re

# Load dictionary
arpabet = cmudict.dict()

# Map to tokens
PUNCT_MAP = {
    ".": ["<PERIOD>"],
    ",": ["<COMMA>"],
    "!": ["<EXCLAMATION>"],
    "?": ["<QUESTION>"],
    ";": ["<SEMICOLON>"],
    ":": ["<COLON>"],
    "(": ["<LPAREN>"],
    ")": ["<RPAREN>"],
}


def normalize(text):
    text = text.lower()

    # remove unwanted chars
    text = re.sub(r"[^a-z0-9\-\'\.\,\!\?\;\:\(\)\[\]\{\} ]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


    
def tokenize(text):
    # words, hyphen words, apostrophes, punctuation, but split everything else
    # return re.findall(r"[A-z0-9]+(?:[-'][a-z0-9]+)*|[.,!?;:(){}\[\]]", text)
    # return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[a-z0-9]+|[-]|[.,!?;:(){}\[\]]", text)
    return re.findall(r"[A-z0-9]+(?:['][a-z0-9]+)*|[-.,!?;:(){}\[\]]", text)



def phonemize(text, single_unknown = True):
    '''
    Takes input text and splits into sounds (via ARPABET) along with
    punctiation tokens. Unknown characters are ignored, and unknown
    words are replaced with 1 token per character. Words are also
    seperated by a | to denote end of word.

    Example input: 
        "Hi! ROFL?"

    Example output:
        ['HH',
         'AY1',
         '|',
         '<EXCLAMATION>',
         '<UNK_R>',
         '<UNK_O>',
         '<UNK_F>',
         '<UNK_L>',
         '|'
         '<QUESTION>']
    '''
    text = normalize(text) 
    tokens = tokenize(text)

    out = []


    for n in range(len(tokens)):

        # Get current token and next token
        tok = tokens[n]
        if n==(len(tokens)-1): # No next token for last one
            next_token = ''
        else:
            next_token = tokens[n+1]

        # normal hyphen (compound word)
        if tok == "-":
            out.append("<HYPHEN>")
            continue

        # punctuation
        if tok in PUNCT_MAP:
            out.extend(PUNCT_MAP[tok])
            continue

        # CMUdict lookup
        if tok in arpabet:
            out.extend(arpabet[tok][0])
        else:
            # Just pass to single unknown token
            if single_unknown:
                out.append("<UNK_WORD>")
            # Pass all characters individually
            else:
                for ch in tok:
                    if ch.isalnum():
                        out.append("<UNK_" + ch.upper() + ">")

        # Only append if no hyphen next
        if next_token != '-':
            out.append("|")  # word boundary

    return out