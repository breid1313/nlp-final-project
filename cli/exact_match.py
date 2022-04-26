import datasets


class ExactMatch(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                }
            ),
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

        # def remove_punc(text):
        #     exclude = set(string.punctuation)
        #     return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(lower(s)))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    def _compute(self, predictions, references):
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("ExactMatch requires the same number of references for each prediction")
            
        transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]
        return {'score' :self.compute_exact_match(predictions, references)}