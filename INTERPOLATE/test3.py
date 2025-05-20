def xor(a, b):
    # XOR 2 chuỗi bit
    return ''.join(['0' if i == j else '1' for i, j in zip(a, b)])

def mod2div(dividend, divisor):
    pick = len(divisor)
    tmp = dividend[:pick]

    while pick < len(dividend):
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor('0'*pick, tmp) + dividend[pick]
        tmp = tmp[1:]
        pick += 1

    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)
    return tmp[1:]  # Bỏ bit đầu tiên

def encode_crc(data, generator):
    l_gen = len(generator)
    appended_data = data + '0'*(l_gen-1)
    remainder = mod2div(appended_data, generator)
    return data + remainder

# Ví dụ
data = "11010011101100"
generator = "1011"
crc_code = encode_crc(data, generator)
print("Dữ liệu sau khi mã hóa CRC:", crc_code)
