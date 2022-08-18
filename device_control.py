import torch
import os

def number_counter(string):
    num=0
    for ch in string :
        if ch.isdigit() :
            num+=1
    return num

print(torch.cuda.is_available())
# all_visible_devices=input("Please input which devices you want to use:\nFor example: 0,1,2,3\n: ")
all_visible_devices='0'
os.environ["CUDA_VISIBLE_DEVICES"] = all_visible_devices
print("Using real gpu:{:>6s}".format(all_visible_devices)) # print(f"using gpu:{all_devices:>6s}") 
 
n_gpu = torch.cuda.device_count()
print('Total number of visible gpu:', n_gpu)

# if n_gpu!=number_counter(all_visible_devices):
#     raise Exception('assigned gpu index error')
assert n_gpu==number_counter(all_visible_devices),"assigned gpu index error"




device_ids_visual=range(n_gpu)
main_device = torch.device('cuda:'+str(device_ids_visual[0])) #用于存放模型类，nn.module