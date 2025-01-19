# This file contains utilities that can be used by any submodule
import pandas as pd
from tqdm import tqdm

from pathlib import Path

def find_images(dir):
    """
    find list of images in the given directory.
    """
    images_extensions = ['.jpg', '.jpeg', '.png']
    #[yield p for p in Path(dir).rglob('*') if p.suffix.lower() in images_extensions]
    for p in Path(dir).rglob('*'):
        if p.suffix in images_extensions:
            yield p

def convert_jsons_to_latexes(jsons_dir):
    """
    Convert all json files into latex within a directory.
    The output is saved within output directory.
    """
    jsons_dir = Path(jsons_dir)
    for in_file in tqdm(jsons_dir.rglob('*.json')):
        out_file = in_file.parent / (str(in_file.stem) + '.tex')
        pd.read_json(in_file).to_latex(out_file)

def convert_plotqa_jsons_to_latexes(jsons_dir):
    """
    function tailored to convert plotqa jsons into latex
    """
    jsons_dir = Path(jsons_dir)
    for in_file in tqdm(jsons_dir.rglob('*.json')):
        out_file = in_file.parent / (str(in_file.stem) + '.tex')
        data = read_json_file(in_file)['models']
        pd.DataFrame.from_dict(data).to_latex(out_file)

def read_markdown_as_dataframe(file):
    return pd.read_table(
            file, sep='|', skipinitialspace=True, header=0,
            ).dropna(axis=1).iloc[1:]


def convert_mds_to_latexes(mds_dir):
    """
    Convert all md files into latex within a directory.
    The output is saved within output directory.
    """
    mds_dir = Path(mds_dir)
    for in_file in tqdm(mds_dir.rglob('*.md')):
        out_file = in_file.parent / (str(in_file.stem) + '.tex')
        read_markdown_as_dataframe(in_file).to_latex(out_file)
