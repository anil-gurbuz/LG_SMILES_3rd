import asyncio
import ray
import time
from torch import nn
import torch.optim
import torch.utils.data
from model.Network import Encoder, PredictiveDecoder

@ray.remote(num_cpus=1, num_gpus=0.2)
class Predict():
    """
    A predict class that receives image data and return decoded sequence
    """
    def __init__(self, config, device, gpu_non_block, decode_length, load_path):
        """
        :param config: configure data
        :param reversed_token_map: converts prediction to readable format
        :param decode_length: maximum length of the decoded SMILES format sequence
        :param load_path: loading path of model
        """

        self._vocab_size = 70
        self._decode_length = decode_length
        self._emb_dim = int(config['emb_dim'])
        self._attention_dim = int(config['attention_dim'])
        self._decoder_dim = int(config['decoder_dim'])
        self._device = device

        self._model_load_name = config['load_model_name']
        self._model_load_path = load_path

        self._encoder = Encoder(model_type=config['encoder_type']).to(non_blocking=gpu_non_block)
        self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
                                          embed_dim=self._emb_dim,
                                          decoder_dim=self._decoder_dim,
                                          vocab_size=self._vocab_size,
                                          device=self._device).to(non_blocking=gpu_non_block)

        self._encoder.to(self._device, non_blocking=gpu_non_block)
        self._decoder.to(self._device, non_blocking=gpu_non_block)

        self.model_load()
        print(self._model_load_name, 'load successed!')

    def decode(self, img):
        """
        :param img: preprocessed image data
        :return: the decoded sequence of molecule image with sequence token value
        """

        self._encoder.eval()
        self._decoder.eval()

        # image to latent vecotr
        encoded_img = self._encoder(img.unsqueeze(0))
        # predicted sequence vector
        predictions = self._decoder(encoded_img, self._decode_length)

        return predictions

    def model_load(self):
        self._decoder.load_state_dict(
            torch.load('{}/decoder{}.pkl'.format(self._model_load_path, self._model_load_name),
                       map_location=self._device)
        )

        weight_data = torch.load('{}/encoder{}.pkl'.format(self._model_load_path, self._model_load_name),
                                 map_location=self._device)
        new_keys = [x[7:] for x in list(weight_data.keys())]
        encoder_weight = {}
        for key, n_key in zip(weight_data.keys(), new_keys):
            encoder_weight[n_key] = weight_data[key]
        self._encoder.load_state_dict(encoder_weight)
