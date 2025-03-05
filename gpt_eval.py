import os
import time
import base64
from openai import OpenAI
#import openai
import json
import argparse

def read_example_data(readfrom_folder, hash_model_folder):
    doc = ""
    summary = ""
    source_img_list = [] # list of directory to the img file
    summary_img_list = []

    with open(os.path.join(readfrom_folder, hash_model_folder, "source_doc.txt")) as file:
        doc = file.read().rstrip()

    with open(os.path.join(readfrom_folder, hash_model_folder, "summary.txt")) as file:
        summary = file.read().rstrip()

    source_img_folder = os.path.join(readfrom_folder, hash_model_folder, "source_img")
    for source_img in os.listdir(source_img_folder):
        source_img_list.append(os.path.join(source_img_folder, source_img))

    summary_img_folder = os.path.join(readfrom_folder, hash_model_folder, "summary_img")
    for summary_img in os.listdir(summary_img_folder):
        summary_img_list.append(os.path.join(summary_img_folder, summary_img))

    return doc, summary, source_img_list, summary_img_list


def summary_img_id(source_imgpath_list_sorted, summary_imgpath_list_sorted):

    #output the summary index (starting from 1)
    #e.g.: source = [img_1, img_2, img_3, img_4, img_6, img_7, img_8] summary = [img_2, img_4, img_8] then should output a string of "2, 4, 7"

    source_jpg_list = [os.path.split(x)[1] for x in source_imgpath_list_sorted]
    summary_jpg_dic = {os.path.split(x)[1]:1 for x in summary_imgpath_list_sorted}

    summary_img_id_list = [str(i+1) for i, source_jpg in enumerate(source_jpg_list) if source_jpg in summary_jpg_dic.keys()]

    return ", ".join(summary_img_id_list)



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gptvision_api_eval(prompt, article, summary, source_img_list, summary_img_list):
    #time.sleep(2)
    print()
    source_imgpath_list_sorted = sorted(source_img_list, key = lambda x: int(os.path.split(x)[1][41:-4]))
    summary_imgpath_list_sorted = sorted(summary_img_list, key = lambda x: int(os.path.split(x)[1][41:-4]))
    summary_img_id_prompt = summary_img_id(source_imgpath_list_sorted, summary_imgpath_list_sorted)

    content = []
    for i, img_path in enumerate(source_imgpath_list_sorted):
        content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}", "detail": "low"}})
        content.append({ "type": "text", "text": f"Source Image ID: {i + 1}"})

    prompt_text = open(prompt).read()
    txt_content = prompt_text.replace('{{Document}}', article).replace('{{Summary}}', summary).replace('{{Summary Image ID}}', summary_img_id_prompt)
    content.append({ "type": "text", "text": txt_content})
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": content,
        }
      ],
      max_tokens=7,
    )

    #return response.choices[0].message.content
    return response




if __name__ == '__main__':

    #data_folder = "./all_data"

    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--prompt_fp', type=str, default='prompts/con_detailed.txt')
    #argparser.add_argument('--save_fp', type=str, default='results/gpt4_con_detailed_openai.json')
    #argparser.add_argument('--key', type=str, required=True)
    #argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()
    #openai.api_key = args.key

    #summeval = json.load(open(args.summeval_fp))
    #prompt = open(args.prompt_fp).read()

    examples_list = os.listdir(data_folder)

    for example in examples_list[:5]:
        doc, summary, source_img_list, summary_img_list = read_example_data(data_folder, example)
        response = gptvision_api_eval("./test_prompt.txt", doc, summary, source_img_list, summary_img_list)
        print(response)
        print("****************************************")












