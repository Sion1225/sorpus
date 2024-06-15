from typing import List, Union, Tuple, Iterator

try:
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize, pos_tag
    from nltk.tag import StanfordNERTagger
except ImportError:
    raise ImportError(
        "nltk is not installed. Please install it using 'pip install nltk'"
    )


class TokenizeNLemmatize:
    """
    Tokenize And Lemmatize

    Tokenize and Lemmatize the text using Stanford NER and NLTK

    If you want to tokenize and lemmatize compound words containing spaces as single words,
    replace the spaces in those words with a specific character.
    The default character is '•'.

    Methods:
        - text_lemmatizer(text, space_token='•', return_pos=False)
            tokenize and lemmatizes the text
        - texts_lemmatizer(texts, space_token='•', return_pos=False)
            tokenize and lemmatizes the list of texts
        - auto_lemmatizer(text, space_token='•', return_pos=False)
            autometically chooses the method based on input type
            if input type is List[str], multiprocesses the list of texts
    """

    def __init__(self, model_path: str, jar_path: str, encoding="utf8"):
        try:
            self.__st_ner = StanfordNERTagger(model_path, jar_path, encoding)
        except LookupError:
            raise LookupError(
                "Stanford NER model not found.\nCheck paths or Download it from 'https://nlp.stanford.edu/software/CRF-NER.html'"
                ""
            )

        self.__lemmatizer = WordNetLemmatizer()

    def __pos2tag(self, pos: str) -> str:
        """Convert POS tags to WordNet tags

        :param pos: POS tag

        :return: WordNet tag
        """
        return pos[0].lower() if pos[0] in ["A", "N", "R", "V"] else "n"

    def text_lemmatizer(
        self, text: str, space_token="•", return_pos=False
    ) -> Union[List[str], List[Tuple[str, str]]]:  # Takes around .5 seconds
        """this method conduct Tokenize, POS-tag, and lemmatize the input sentence and return the processed tokens.
        Specifically, check if the first token of the sentence is a proper noun or 'I', and if so, retain its capitalization; otherwise, convert it to lowercase.
        Additionally, replace the space tokens in compound words with spaces and lemmatize plural forms to their singular forms.

        :param text: input sentence
        :param space_token: character that replaced space instead in compound words
        :param return_pos: if True, return POS tags along with lemmatized words

        :return: lemmatized words or lemmatized words with POS tags

        >>> lemmatizer = TokenizeNLemmatize(model_path, jar_path)
        >>> lemmatizer.text_lemmatizer("Proverbs are short sentences drawn from long experience.")
        ['proverb', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.']
        >>> lemmatizer.text_lemmatizer("Proverbs are short sentences drawn from long experience.", return_pos=True)
        [('proverb', 'NNS'), ('be', 'VBP'), ('short', 'JJ'), ('sentence', 'NNS'), ('draw', 'VBP'), ('from', 'IN'), ('long', 'JJ'), ('experience', 'NN'), ('.', '.')]

        >>> from sorpus import SpecialWordReplacer, TokenizeNLemmatize
        >>> replacer = SpecialWordReplacer(['New York', 'police officer'], ' ', '•')
        >>> replaced_text = replacer.mask_words("In New York, I saw police officers and firefighters.")
        >>> print(replaced_text)
        >>> print(lemmatizer.text_lemmatizer(replaced_text, space_token='•'))
        ['in', 'New York', ',', 'I', 'saw', 'police officer', 'and', 'firefighter', '.']
        """
        # Tokenize the sentence
        words = word_tokenize(text)

        # replace space_token to space
        new_words = []
        for j, word in enumerate(words):
            if space_token in word:
                compounds = word.split(space_token)
                # lemmaitze for last word of the compound word
                last_compound_pos = pos_tag([compounds[-1]])
                compounds[-1] = self.__lemmatizer.lemmatize(
                    compounds[-1], pos=self.__pos2tag(last_compound_pos[0][1])
                )
                new_words.append(" ".join(compounds))
            else:
                new_words.append(word)
        words = new_words

        # Lower the words
        if (
            self.__st_ner.tag(words)[0][1] == "O" and words[0] != "I"
        ):  # If first word is not proper noun
            words[0] = words[0].lower()

        # POS tagging
        pos_tags = pos_tag(words)

        # Lemmatize the words
        if return_pos:
            return [
                (
                    self.__lemmatizer.lemmatize(
                        word, pos=self.__pos2tag(pos_tags[i][1])
                    ),
                    pos_tags[i][1],
                )
                for i, word in enumerate(words)
            ]

        return [
            self.__lemmatizer.lemmatize(word, pos=self.__pos2tag(pos_tags[i][1]))
            for i, word in enumerate(words)
        ]

    def texts_lemmatizer(
        self, texts: List[str], space_token="•", return_pos=False
    ) -> Iterator[Union[List[str], List[Tuple[str, str]]]]:
        """this method yeilds text_lemmatizer for each text in the list of texts.
        Conduct Tokenize, POS-tag, and lemmatize the input sentences and return the processed tokens.
        Specifically, check if the first token of the sentence is a proper noun or 'I', and if so, retain its capitalization; otherwise, convert it to lowercase.
        Additionally, replace the space tokens in compound words with spaces and lemmatize plural forms to their singular forms.

        :param texts: list of input sentences
        :param space_token: character that replaced space instead in compound words
        :param return_pos: if True, return POS tags along with lemmatized words

        :return: Iterator of lemmatized words or lemmatized words with POS tags

        >>> lemmatizer = TokenizeNLemmatize(model_path, jar_path)
        >>> texts = ["Proverbs are short sentences drawn from long experience.",
        ... "In New•York, I saw police•officers and firefighters."]
        >>> list(lemmatizer.texts_lemmatizer(texts, space_token='•'))
        [['proverb', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.'], ['in', 'New York', ',', 'I', 'saw', 'police officer', 'and', 'firefighter', '.']]
        """
        yield from (
            self.text_lemmatizer(text, space_token, return_pos) for text in texts
        )

    def auto_lemmatizer(
        self, text: Union[str, List[str]], space_token="•", return_pos=False
    ):
        """this method autometically chooses the method based on input type.
        If input type is List[str], multiprocesses the list of texts.

        :param text: input sentence or list of input sentences
        :param space_token: character that replaced space instead in compound words
        :param return_pos: if True, return POS tags along with lemmatized words

        :return: lemmatized words or lemmatized words with POS tags
        """
        if isinstance(text, str):
            return self.text_lemmatizer(text, space_token, return_pos)
        elif isinstance(text, list):
            import multiprocessing

            with multiprocessing.Pool() as pool:
                yield from pool.imap(self.text_lemmatizer, text)
        else:
            raise TypeError("Input should be either string or list of strings")

    def __repr__(self) -> str:
        return "<TokenizeNLemmatize>"
