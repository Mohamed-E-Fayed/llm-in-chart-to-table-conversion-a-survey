#!/usr/bin/env python3

import os
import json
import time
import random
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv


safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
    #HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
}
system_message = """
    You are an EXPERT in scientific visualizations analysis and chart understanding and description.
    You should help people with disabilities in giving an overview of the chart and converting the chart into its original data point form.
"""
sleep_per_response = lambda x=4: time.sleep(x)

def configure_gemini():
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

extract_json_from_string = lambda text: json.loads(text.replace('```json', '').replace('`', '').replace('\n', ''))

def identify_graph_type(img_path):
    global safety_settings
    prompt1 = """
    For the graph image that is uploaded, I want you to do the following:

    1) first identify what type of graph it is (line, bar, pie, etc)

    Just return the answer as a string. Here are the possible responses:
    - Bar Graph
    - Scatter Plot
    - Line Graph
    - Not a graph
    """
    prompt2 = """
    System Message:
    You are an EXPERT in scientific visualizations analysis and chart understanding and description.
    You should help people with disabilities in giving an overview of the chart and converting the chart into its original data point form.
    Prompt:
    Analyze the given image and determine what kind of chart is it.
    Just reply with one of the following choices:

    - Bar Graph
    - Scatter Plot
    - Line Graph
    - Not a graph
    - Other:Correct Type
    if Other, just reply with TWO WORDS MAXIMUM.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    img = Image.open(img_path)
    response = model.generate_content([prompt1, img], safety_settings=safety_settings)
    return response.text.strip()

def extract_axes_names_and_values(
        img_path,
        prompt=None,
            model = genai.GenerativeModel('gemini-1.5-flash'),
            ):
    global system_message, safety_settings
    if not prompt:
        prompt = f"""
        system message: {system_message}
        prompt: For the following image, tell me the name of x and y axes, and the values of the each axis.
        The output should be in format of 
        """ + "{x_axis_name: x_axis_vavlues_list, y_xis_name: y_axis_values_list}"
    img = Image.open(img_path)
    response = model.generate_content([prompt, img], safety_settings=safety_settings)
    try:
        dictionary = extract_json_from_string(response.text.strip())
    except Exception as e:
        sleep_per_response()
        prompt = f"""{prompt.replace('prompt:', 'user:')}
        assistant: {response.text.strip()}
        user: Thank you so much for explanation.
        I want to convert it into python dictionary. So, could you please reply with JSON object only?
        """
        response = model.generate_content([prompt, img], safety_settings=safety_settings)
        dictionary = extract_json_from_string(response.text.strip())

    return dictionary


def convert_chart_to_table_end_to_end(
        img_path,
        prompt1=None,
        prompt2=None,
            model = genai.GenerativeModel('gemini-1.5-flash'),
                                ):
    global system_message, safety_settings
    img = Image.open(img_path)
    if not prompt1:
        prompt1 = f"""
        system message: {system_message}
        prompt:
        "Please analyze the chart in the image using a step-by-step approach:

Identify the title, x-axis label, and y-axis label, along with any legend information.
Describe the overall structure, including chart type (e.g., line, bar, scatter) and any key areas of focus.
Extract each data point in the format [x-value, y-value] by scanning along the x-axis. If there are multiple data series, group points by their series and include the legend category if relevant. NOTE: please include 6 digits in numbers in generated tables.
Summarize any observable trends or patterns in the data, noting key points, outliers, or any relationship between data series if present.
Return all data points and the summary, ensuring all information is organized based on these steps.
        """
    img = Image.open(img_path)
    response = model.generate_content([prompt1, img], safety_settings=safety_settings)
    prompt2 = f"""
    assistant: {response.text.strip()}
    user:
    Please,  the table in json format to create pandas data frame. Could you please extract the table only from your response, double check it and return the final table in json format without explaining any edits you make?
    """
    response = model.generate_content([prompt1, img, prompt2], safety_settings=safety_settings)
    try:
        output = extract_json_from_string(response.text.strip())
    except Exception as e:
        output = response.text.strip()
    sleep_per_response(5)
    return output


def load_plotqa_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def map_plotqa_type_to_our_type(plotqa_type):
    type_mapping = {
        'vbar_categorical': 'Bar Graph',
        'hbar_categorical': 'Bar Graph',
        'dot_line': 'Scatter Plot',
        'line': 'Line Graph',
    }
    return type_mapping.get(plotqa_type, 'Not a graph')

def evaluate_graph_classification(img_dir, annotations_path, num_samples=100):
    configure_gemini()
    annotations = load_plotqa_annotations(annotations_path)

    # the overall dataset is very large, so just sample a few
    sampled_annotations = random.sample(annotations, min(num_samples, len(annotations)))

    true_labels = []
    predicted_labels = []

    for i, item in enumerate(tqdm(sampled_annotations, desc="Processing images")):
        # pause after every 15 requests
        # Gemini Flash Free Tier has a rate limit of 15 requests/minute
        #if i % 15 == 0:
        #    time.sleep(60)  # Wait for 60 seconds
        # alternatively, decrease the rate to avoid exceeding this limit
        time.sleep(3.9) #emperically, the request is handled within a second on average. So, totalit would be 4.9s/request, which is more than 4s/request limit.

        img_path = os.path.join(img_dir, f"{item['image_index']}.png")
        true_type = map_plotqa_type_to_our_type(item['type'])
        predicted_type = identify_graph_type(img_path)

        true_labels.append(true_type)
        predicted_labels.append(predicted_type)

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

def main():
    # default path assuming repository as a submodule of the GraphIngestionEngine repo
    # if not, replace this with the path to PlotQA dataset on your system
    plotqa_data_dir = "../datasets/PlotQA/test"
    annotations_path = os.path.join(plotqa_data_dir, "annotations.json")
    img_dir = os.path.join(plotqa_data_dir, "png/")
    evaluate_graph_classification(img_dir, annotations_path)


#if __name__ == "__main__":
#    main()
