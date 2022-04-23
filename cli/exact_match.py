import datasets


class Spider(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features={'predictions': datasets.Value(dtype='string', id='sequence'), 
                        'references': datasets.features.Sequence(feature=datasets.Value(dtype='string', id='sequence'), length=-1, id='references')},
            codebase_urls=[""],
            reference_urls=[""],
        )
    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    def _compute(self, predictions, references):
        return {'score' :self.compute_exact_match(predictions, references)}