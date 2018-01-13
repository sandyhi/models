#coding=utf-8
import sys
import random
import cPickle as pk

def create_pickle(in_file, out_file):
    train_img_list = []
    train_label_list = []
    test_img_list = []
    test_label_list = []
    label_dict = {}
    with open(in_file, 'r') as fr:
        line_list = fr.readlines()

    print "total input number ", len(line_list)

    for line in line_list:
        line_cut = line.strip().split(" ")
        if len(line_cut) != 2:
            print "illegal line content : ", line_cut
            continue
        img_name = line_cut[0]
        label = line_cut[1]

        if label not in label_dict.keys():
            label_dict[label] = len(label_dict)

        if random.random() >= 0.1:
            train_img_list.append(img_name)
            train_label_list.append(label_dict[label])
        else:
            test_img_list.append(img_name)
            test_label_list.append(label_dict[label])


    # write down pickle
    obj = {"train_data":train_img_list, "train_label":train_label_list,
           "valid_data":test_img_list, "valid_label":test_label_list}
    pk.dump(obj, open(out_file, 'w'))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage : python xx.py [in_file] [out_file]"
        sys.exit(1)
    else:
        print "create pickle begin : "
        create_pickle(sys.argv[1], sys.argv[2])
        print "done !"