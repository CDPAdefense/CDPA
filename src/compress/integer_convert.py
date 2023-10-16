import struct
from xmlrpc.client import Binary
import numpy as np
import torch

def convert_to_binary(num):
    num1 = num
    num = round(num.item(), 5) 

        
    
    num = int(num * 100000) 
    if num >= 0:
        binary_number = bin(num)[2:].zfill(32)
    else:
        binary_number = bin(num)[3:].zfill(32)
        binary_number = '1' + binary_number[1:]
    return binary_number



def binary_to_decimal(binary):
    
    if binary[0] == '1':
        binary1 = '0' + binary[1:]
        decimal_number = -int(binary1, 2) 
    else:
        decimal_number = int(binary, 2)
    decimal_number /= 100000
    return decimal_number


# 下面的function把任意一个tensor转化成内部为32 binary
def binary_convert(tensor, p):
    binary_tensor = np.array([convert_to_binary(x) for x in tensor.flatten()])
    binary_tensor = list(binary_tensor)
    binary_tensor2 = []
    for i, to_binary in enumerate(binary_tensor):
        binary_list = list(to_binary)
        for j, str_ in enumerate(binary_list):
            if np.random.random() > p and j in {21,22}:
                binary_list[j] = str(1-int(str_))
        new_str = ''.join(binary_list)

        binary_tensor2.append(new_str)
    binary_tensor2 = np.array(binary_tensor2).reshape(tensor.shape)
    return binary_tensor2

def rearrange_tensor_list(tensor_list):
    new_list = []
    stacked_tensors = np.stack(tensor_list , axis=0)
    if np.array(tensor_list).ndim == 5:
        for i in range(stacked_tensors.shape[1]):
            for j in range(stacked_tensors.shape[2]):
                for k in range(stacked_tensors.shape[3]):
                    for l in range(stacked_tensors.shape[4]):
                    # 获取相同位置的元素并添加到新列表中
                        new_list.append(stacked_tensors[:, i, j, k, l])  
    if np.array(tensor_list).ndim == 2:
        for i in range(stacked_tensors.shape[1]):
            new_list.append(stacked_tensors[:, i])
    
    if np.array(tensor_list).ndim == 4:
        for i in range(stacked_tensors.shape[1]):
            for j in range(stacked_tensors.shape[2]):
                for k in range(stacked_tensors.shape[3]):
                    new_list.append(stacked_tensors[:, i, j, k])
    
    if np.array(tensor_list).ndim == 3:
        for i in range(stacked_tensors.shape[1]):
            for j in range(stacked_tensors.shape[2]):
                new_list.append(stacked_tensors[:, i, j])
    
    return new_list

def binary_to_decimal2(tensor):
    tensor_tem = tensor.flatten()
    for i,binary in enumerate(tensor_tem):
        if binary[0] == '1':
            binary1 = '0' + binary[1:]
            decimal_number = -int(binary1, 2) 
        else:
            decimal_number = int(binary, 2)
        decimal_number /= 10000
        tensor_tem[i] = decimal_number
    tensor_tem = tensor_tem.astype(float)
    
    # tensor_tem = torch.from_numpy(tensor_tem)
    tensor_tem = np.resize(tensor_tem, tensor.shape)
    return tensor_tem

# 把binary 32的tensor还原
def process_tensors(tensor_list,p):
    num_tensors = len(tensor_list)
    num_bits = 32
    new_list = rearrange_tensor_list(tensor_list)
    result = []

    for sublist in new_list:
        new_values = []
        for bit_index in range(num_bits):
            
            new_num = 0
            new_int = 0
            new_str = ''
            new_char = ''
            value = 0.0
            
            for j in range(len(sublist)):
                try: 

                    sublist[j][bit_index] == '1'
                except Exception as e:
                    print('j:', j, "bit_index: ", bit_index)
                    print(sublist)
                
                if sublist[j][bit_index] == '1':
                    new_num += 1
                if j == len(sublist) - 1:
                    new_num = new_num / ( p  * num_tensors)
                    if new_num < 0.5:
                        new_char = '0'
                    else:
                        new_char = '1'

          
        #   new_str += new_char
            new_values.append(new_char)
          
        new = ''.join(new_values)
        result.append(new)
    
    result2 = []
    for item in result:
        
        result2.append(binary_to_decimal(item))
    
    source_tensor = torch.tensor(result2)
    if source_tensor.numel() != 1 and source_tensor.numel() != 0:
        reshaped_tensor = source_tensor.reshape(tensor_list[0].shape)
        reshaped_tensor.requires_grad = True
    else:
        reshaped_tensor = source_tensor 
        reshaped_tensor.requires_grad = True
    return reshaped_tensor


