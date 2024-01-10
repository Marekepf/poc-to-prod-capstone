# import unittest
# from unittest.mock import MagicMock, patch
# import json
# import os
# import pandas as pd
# from preprocessing.preprocessing import utils

# from predict.predict import run

# # Assuming embed is in a file in the preprocessing/preprocessing directory
# from preprocessing.preprocessing.embeddings import embed  


# def load_dataset_mock():
#     titles = [
#         "Is it possible to execute the procedure of a function in the scope of the caller?",
#         "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
#         "Is it possible to execute the procedure of a function in the scope of the caller?",
#         "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
#         "Is it possible to execute the procedure of a function in the scope of the caller?",
#         "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
#         "Is it possible to execute the procedure of a function in the scope of the caller?",
#         "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
#         "Is it possible to execute the procedure of a function in the scope of the caller?",
#         "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
#     ]
#     tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
#             "php", "ruby-on-rails"]

#     return pd.DataFrame({
#         'title': titles,
#         'tag_name': tags
#     })


# # Mocking the dependencies
# def load_model_mock():
#     model_mock = MagicMock()
#     model_mock.predict = MagicMock(return_value=[[0.1, 0.9], [0.8, 0.2]])
#     return model_mock

# def embed_mock(text_list, params):
#     return ["mock_embedding" for _ in text_list]

# class TestTextPredictionModel(unittest.TestCase):
#     utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    
#     def test_from_artefacts_and_predict(self, mock_load_model, mock_embed):
#         # Mocking artefact files
#         artefacts_path = 'fake_artefacts_path'
#         model_path = os.path.join(artefacts_path, "model.h5")
#         params_path = os.path.join(artefacts_path, "params.json")
#         labels_path = os.path.join(artefacts_path, "labels_to_index.json")

#         with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({"dummy": "params"}))) as mock_file:
#             # Instantiate the TextPredictionModel
#             model = run.from_artefacts(artefacts_path)

#             # Ensure artefacts were attempted to be read
#             mock_file.assert_called_with(params_path, 'r')
#             mock_file.assert_called_with(labels_path, 'r')

#             # Test predict method
#             predictions = model.predict(["test text"])
#             self.assertIsNotNone(predictions)
#             # Add more assertions here to validate the predictions

# if __name__ == '__main__':
#     unittest.main()


import unittest
from unittest.mock import MagicMock, patch
import json
import os
import pandas as pd
from preprocessing.preprocessing import utils

from predict.predict import run
from preprocessing.preprocessing.embeddings import embed  

def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


# Mocking the dependencies
def load_model_mock():
    model_mock = MagicMock()
    model_mock.predict = MagicMock(return_value=[[0.1, 0.9], [0.8, 0.2]])
    return model_mock

def embed_mock(text_list, params):
    return ["mock_embedding" for _ in text_list]

class TestTextPredictionModel(unittest.TestCase):
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    @patch("predict.predict.load_model", side_effect=load_model_mock)
    @patch("predict.predict.embed", side_effect=embed_mock)
    def test_from_artefacts_and_predict(self, mock_load_model, mock_embed):
        # Mocking artefact files
        artefacts_path = 'fake_artefacts_path'
        model_path = os.path.join(artefacts_path, "model.h5")
        params_path = os.path.join(artefacts_path, "params.json")
        labels_path = os.path.join(artefacts_path, "labels_to_index.json")

        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({"dummy": "params"}))) as mock_file:
            # Instantiate the TextPredictionModel
            model = run.from_artefacts(artefacts_path)

            # Ensure artefacts were attempted to be read
            mock_file.assert_called_with(params_path, 'r')
            mock_file.assert_called_with(labels_path, 'r')

            # Test predict method
            predictions = model.predict(["test text"])
            self.assertIsNotNone(predictions)
            # Add more assertions here to validate the predictions

if __name__ == '__main__':
    unittest.main()