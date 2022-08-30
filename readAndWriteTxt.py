path = "data/imgs.txt"

write_path = "data/npz.txt"

ifs = open(path)
for line in ifs.readlines():
    line = line.replace(".jpg",".jpg.rd")
    with open(write_path,"a") as f:
        f.write(line)

# import re
# import numpy as np
#
# def getDigital(string):
#     assert(isinstance(string,str))
#     return re.sub('\D','',string)
#
# def bubble_sort(our_list):
#     for i in range(len(our_list)-1):
#         for j in range(len(our_list)-1):
#             if our_list[j] > our_list[j+1]:
#                 our_list[j],our_list[j+1] = our_list[j+1],our_list[j]
#     return our_list
#
#
# if __name__ == '__main__':
#     data = np.loadtxt("data/imgs_npz.txt")
#     print(data)
#     print(data.shape)
#     sortData = bubble_sort(data)
#     print(sortData)
#
#     write_path = "data/npz.txt"
#
#     with open(write_path,"a") as f:
#         for i in range(len(sortData)):
#             path = str(sortData[i]).replace('.0','') + ".jpg" + "\n"
#             f.write(path)