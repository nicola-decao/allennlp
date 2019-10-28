import pytest

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestLanguageModelingDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = LanguageModelingReader(tokens_per_instance=3, lazy=lazy)

        instances = ensure_list(
            reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "language_modeling.txt")
        )
        # The last potential instance is left out, which is ok, because we don't have an end token
        # in here, anyway.
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields["source"].tokens] == ["This", "is", "a"]

        assert [t.text for t in instances[1].fields["source"].tokens] == [
            "sentence",
            "for",
            "language",
        ]

        assert [t.text for t in instances[2].fields["source"].tokens] == ["modelling", ".", "Here"]

        assert [t.text for t in instances[3].fields["source"].tokens] == ["'s", "another", "one"]

        assert [t.text for t in instances[4].fields["source"].tokens] == [
            "for",
            "extra",
            "language",
        ]
