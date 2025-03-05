import pickle
import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr

#prj_path = "."

#table1_path = "./label1.csv"
#table2_path = "./label2.csv"
#table3_path = "./label3.csv"
#table_avg_path = "./label_avg.csv"
#table1_overall_path = "./label1_overall.csv"
#table2_overall_path = "./label2_overall.csv"
#table3_overall_path = "./label3_overall.csv"
#table_avg_overall_path = "./label_overall_avg.csv"

ratingtable_avg_path = "./ratingtable_avg.csv"
evalres1_path = "./eval_results_pkl_01"
evalres2_path = "./eval_results_pkl_02"
mergedpkl_path = "./eval_results_pkl_merged"

ifm = "informativeness"
coh = "coherence"
cst = "consistency"
flu = "fluency"
imq = "image_quality"
rel = "relevance"
ova = "overall"
header_names = ["id", "folder_name", ifm, coh, cst, flu, imq, rel]
header_names_ova = ["id", "folder_name", ifm, coh, cst, flu, imq, rel, ova]


def csv_to_dic(table_path, header_names):
    # a table mapping to a dict
    asp_list = [ifm, coh, cst, flu, imq, rel, ova] if ova in header_names else [ifm, coh, cst, flu, imq, rel]
    out = {}
    df = pd.read_csv(table_path, header=None, names=header_names)
    for index, row in df.iterrows():
        out[row["folder_name"]] = {}
        for asp in asp_list:
            out[row["folder_name"]][asp] = row[asp]

    return out

#dic = csv_to_dic(ratingtable_avg_path, header_names_ova)
#print(len(dic))
#print((dic["f581a0702769a82f6338561b95a70e36ce1af02c_gpt4"]))



def load_pkl_to_dic(pkl_file):
    with open(pkl_file, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict

#print(load_pkl_to_dic("text_r2_dic.pkl"))
#print(len(load_pkl_to_dic("text_rl_dic.pkl")))


def merge_two_pkl_path(pkl_folder1, pkl_folder2):
    pkl_list1 = os.listdir(pkl_folder1)
    pkl_list2 = os.listdir(pkl_folder2)

    #print(len(pkl_list1))
    #print(len(pkl_list2))
    print((pkl_list1 == pkl_list2))

    #print(pkl_list1[4])
    #print(pkl_list2[5])

    for pkl_file in pkl_list1:
        dic1 = load_pkl_to_dic(os.path.join(pkl_folder1, pkl_file))
        dic2 = load_pkl_to_dic(os.path.join(pkl_folder2, pkl_file))

        #print(len(dic1))
        #print(len(dic2))

        merged_dic =  {**dic1, **dic2}
        with open(pkl_file, 'wb') as f1:
            pickle.dump(merged_dic, f1)

    print("======= successfully merged! ============")

#merge_two_pkl_path(evalres1_path, evalres2_path)


def corr_3_psk(x, y):
    p = pearsonr(x, y)[0]
    s = spearmanr(x, y)[0]
    k = kendalltau(x, y)[0]

    return p,s,k


def corr_cal(table_path, header_names, pkl_file, asp):

    table_dic = csv_to_dic(table_path, header_names)
    metric_result_dic = load_pkl_to_dic(pkl_file)

    #print(len(metric_result_dic))
    #print(list(table_dic.keys()).sort() == list(metric_result_dic.keys()).sort())
    example_list = list(table_dic.keys())
    human_anno_list = []
    metric_result_list = []
    for example in example_list:
        human_anno_list.append(table_dic[example][asp])
        metric_result_list.append(metric_result_dic[example])
    p,s,k = corr_3_psk(human_anno_list, metric_result_list)
    #print(p)
    #print(s)
    #print(k)
    #print("=+++++++++++++=====")
    return p,s,k

#corr_cal(ratingtable_avg_path, header_names_ova, "img_ref_precision.pkl", imq)


def calculate_all_files(mergedpkl_path, human_table_path, header_names):
    asp_list = [ifm, coh, cst, flu, imq, rel]
    file_list = os.listdir(mergedpkl_path)
    text_pkl_list = []
    img_pkl_list = []
    tir_pkl_list = []

    for each_file in file_list:
        if each_file.endswith(".pkl") and each_file.startswith("text"):
            text_pkl_list.append(each_file)
        if each_file.endswith(".pkl") and each_file.startswith("img"):
            img_pkl_list.append(each_file)
        if each_file.endswith(".pkl") and each_file.startswith("tir"):
            tir_pkl_list.append(each_file)

    print(text_pkl_list)
    print(len(text_pkl_list))
    print(img_pkl_list)
    print(len(img_pkl_list))
    print(tir_pkl_list)
    print(len(tir_pkl_list))

    print("the total number of pkl files")
    print(len(text_pkl_list) + len(img_pkl_list) + len(tir_pkl_list))


    print("======================== TEXT =========================")

    for text_pkl in text_pkl_list:
        print(text_pkl)

        sco_list = []
        for asp in asp_list[:4]:
            #print(asp)
            p,s,k = corr_cal(human_table_path, header_names, os.path.join(mergedpkl_path, text_pkl), asp)
            sco_list.append(str(round(p, 2)))
            sco_list.append(str(round(s, 2)))
            sco_list.append(str(round(k, 2)))
            #print(p, s, k, sep='&')
            #print()
        print("&".join(sco_list))

        print()


    print("======================== TEXT =========================")
    print()
    print()
    print()
    print("======================== Image =========================")

    for img_pkl in img_pkl_list:
        print(img_pkl)
        sco_img_list = []
        p,s,k = corr_cal(human_table_path, header_names, os.path.join(mergedpkl_path, img_pkl), asp_list[4])
        #print(p, s, k, sep='&')
        sco_img_list.append(str(round(p, 2)))
        sco_img_list.append(str(round(s, 2)))
        sco_img_list.append(str(round(k, 2)))
        #print(p, s, k, sep='&')
        #print()
        print("&".join(sco_img_list))
        print()

    print("======================== Image =========================")
    print()
    print()
    print()

    print("======================== TIR =========================")

    for tir_pkl in tir_pkl_list:
        print(tir_pkl)
        sco_tir_list = []
        p,s,k = corr_cal(human_table_path, header_names, os.path.join(mergedpkl_path, tir_pkl), asp_list[5])
        #print(p, s, k, sep='&')
        sco_tir_list.append(str(round(p, 2)))
        sco_tir_list.append(str(round(s, 2)))
        sco_tir_list.append(str(round(k, 2)))
        #print(p, s, k, sep='&')
        #print()
        print("&".join(sco_tir_list))

        print()

    print("======================== TIR =========================")
    print()
    print()
    print()



    return 0

calculate_all_files(mergedpkl_path, ratingtable_avg_path, header_names_ova)
