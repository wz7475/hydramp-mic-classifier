""" module for model specific concretisation of abstract interfaces """
import os

import numpy as np
import pandas as pd

from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model
        expects tsv file with columns:
        classifier: Prediction, Probability_score
        regressor: Prediction"""
        df = pd.read_csv(filepath)
        df.rename(columns={"Prediction": "Probability_score"}, inplace=True)
        df["Prediction"] = np.where((df["Probability_score"] >= 0.5), "AMP", "non-AMP")
        df.to_csv(output_filename, sep="\t")


class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        command = f"python -m amp.inference.scripts.predict_if_amp --model_path models/mic_classifier/ \
--sequence_path {filepath} --format fasta --output_csv {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        pass
