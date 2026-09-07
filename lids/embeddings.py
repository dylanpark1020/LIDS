import numpy as np
import torch
from transformers import BertTokenizer, BertModel


DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 512


class BERTEmbedder:
    """
    Converts text into a matrix of BERT token embeddings.

    Each row corresponds to a token and each column corresponds
    to one dimension of the BERT embedding.
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        max_length=DEFAULT_MAX_LENGTH,
    ):
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 3:
            raise ValueError("max_length must be an integer of at least 3.")

        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        """
        Convert text into a token embedding matrix.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (number_of_tokens, embedding_dimension).
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            raise ValueError("Input text contains no tokens.")

        tokens = self.tokenizer.tokenize(text)

        all_token_embeddings = []

        # Leave room for [CLS] and [SEP].
        chunk_size = self.max_length - 2

        for i in range(0, len(tokens), chunk_size):

            chunk_tokens = tokens[i:i + chunk_size]

            chunk_tokens = ["[CLS]"] + chunk_tokens + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)

            input_ids_tensor = torch.tensor([input_ids])

            with torch.no_grad():
                outputs = self.model(input_ids_tensor)

            hidden_states = outputs.last_hidden_state

            # Remove [CLS] and [SEP].
            chunk_embeddings = hidden_states[0][1:-1]

            all_token_embeddings.extend(chunk_embeddings)

        if not all_token_embeddings:
            raise ValueError("Input text contains no tokens.")

        return np.asarray(
            [embedding.cpu().numpy() for embedding in all_token_embeddings]
        )
