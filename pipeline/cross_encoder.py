
import numpy as np

from huggingface_hub import hf_hub_download


# -----------------------------------------
# CROSS-ENCODER
# Reads two pieces of text together, so every word in one can attend to every word in the other.
# Runs on onnxruntime, so no PyTorch is needed. Weights are downloaded once into the Hugging Face cache.
# -----------------------------------------

MAX_TOKENS = 256
BATCH = 16


class CrossEncoder:

    def __init__(self, repo, onnx_file="onnx/model.onnx"):

        import onnxruntime as ort

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()

        self.session = ort.InferenceSession(hf_hub_download(repo, onnx_file), providers=["CPUExecutionProvider"])

        self.wanted = {i.name for i in self.session.get_inputs()}

    def score(self, pairs):

        # One raw score per (text a, text b) pair; higher means the two belong together more
        out = []

        for i in range(0, len(pairs), BATCH):

            encoded = self.tokenizer.encode_batch(pairs[i:i + BATCH])

            feeds = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64)
            }

            logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self.wanted})[0]

            out.extend(logits.reshape(len(encoded), -1)[:, 0].tolist())

        return np.array(out)
