# coding: utf-8
from pathlib import Path
import json

"""
get_ipython().run_line_magic('run', '-i ~/path_scripts/my_helping_functions.py')
get_ipython().run_line_magic('run', '-i ~/important_repos/GraphIngestionEngineFall2024/GIE-Parser-GenAI-Approach/evaluate.py')
plotqa_sample_jsons = {str(path): read_json_file(path) for path in sorted(list(Path('test-sample/plotqa/jsons/').rglob('*.json')))}
plotqa_predictions_sample_jsons = {str(path): read_json_file(path) for path in sorted(list(Path('test-sample/plotqa/predictions/gemini-1.5-flash/').rglob('*.json')))}
icpr2_sample_jsons = {str(path): read_json_file(path) for path in sorted(list(Path('test-sample/icpr22/jsons/').rglob('*.json')))}
icpr22_predictions_sample_jsons = {str(path): read_json_file(path) for path in sorted(list(Path('test-sample/icpr22//predictions/gemini-1.5-flash/').rglob('*.json')))}
"""

# pretty print for ease of read
def pretty_print_ref_and_pred_plotqa(
        true_dict, pred_dict, ref_json_path, model='gemini-1.5-flash', output_dir_suffix='1st_trial',
        ):
    ref_json_path = Path(ref_json_path)
    #pred_json_path = ref_json_path.replace('jsons', f'predictions/{model}/{output_dir_suffix}')
    try:
       pred_path = list(filter(lambda x: ref_json_path.stem in x, list(pred_dict.keys())))[0]
    except IndexError:
        print(f"can not find key including {ref_json_path.stem} in predictions")
        return
    ref_json_path = str(ref_json_path)
    print('reference:')
    _ = [print(model['name'], '\n', model['x'], '\n', model['y']) for model in true_dict[ref_json_path]['models']]
    print('prediction:')
    _ = [print(f"{k}: {v}") for k, v in pred_dict[pred_path].items()]
