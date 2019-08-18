import numpy as np
from src.main.python.preprocessing.label_encoding import encode, num2class, class2num
from sklearn.preprocessing import OneHotEncoder
import pytest


class TestLabelEncoding:

    @pytest.fixture()
    def init_encoder(self):
        classes = np.array([[0], [1], [2], [3]])

        encoder = encode(classes)

        return encoder

    def test_encode_is_OneHotEncoder(self, init_encoder):
        encoder = init_encoder

        assert isinstance(encoder, OneHotEncoder)

    def test_encode_return_correct_result(self, init_encoder):
        example = np.array([[3], [2]])
        expected_result = [[0., 0., 0., 1.], [0., 0., 1., 0.]]

        encoder = init_encoder

        result = encoder.transform(example).toarray().tolist()

        assert result == expected_result

    def test_encode_throw_exception_on_unknown_class(self, init_encoder):
        example = np.array([[4]])

        encoder = init_encoder

        with pytest.raises(ValueError):
            encoder.transform(example).toarray().tolist()

    def test_num2class_return_correct_result(self):
        classes = np.array([[0], [1], [2], [3]])

        result = num2class(classes)

        assert result == ["bmp-1", "btr-80", "t-55", "t-72b"]

    def test_num2class_return_empty_list_on_unkown_class(self):
        classes = np.array([[4]])

        result = num2class(classes)

        assert result == []

    def test_class2num_return_correct_result(self):
        classes = np.array([["bmp-1"], ["btr-80"], ["t-55"], ["t-72b"]])

        result = class2num(classes)

        assert result == [0, 1, 2, 3]

    def test_class2num_return_empty_list_on_unkown_class(self):
        classes = np.array([["abc"]])

        result = class2num(classes)

        assert result == []

if __name__ == "__main__":
    pass
