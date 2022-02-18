
# Function to Read data from txt file
import random


def read_from_txt(path):
    file = open(path,'r',encoding="utf8")
    data = file.readlines()
    file.close()
    return data

# Function to Generate wrong word pair for training
# 1. Define Characters
UNI_KA = 0X1780 # the first Khmer character
UNI_LAST = 0x17ff #t he last Khmer character
C_START = UNI_LAST - UNI_KA + 1 # Special character C_START
C_STOP = C_START + 1 # Special character C_STOP
C_UNK = C_START + 2 # Special character C_UNK (unknown)
N_CHAR = UNI_LAST - UNI_KA + 1 + 3

# 2. Error: insertion
def str_insert(str):
    pos = random.randint(0, len(str)) # random position to insert character in a word
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3)) # random character to insert in a word
    return str[:pos] + rand_c + str[pos:]

# 3. Error: deletion
def str_delete(str):
    pos = random.randint(0, len(str) - 1)
    return str[:pos] + str[pos + 1:]

# 4. Error: substition/replace
def str_replace(str):
    pos = random.randint(0, len(str) - 1)
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3))
    return str[:pos] + rand_c + str[pos + 1:]

# 5. Generate error word pair with random error methods
def str_rand_err(str):
    t = random.randint(0, 2)
    if t == 0:
        return str_insert(str)
    if t == 1:
        return str_delete(str)
    else:
        return str_replace(str)

str = "Institute"
print()and_err(str)