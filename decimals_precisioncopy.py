from decimal import *
print(0.1+0.2)
# 0.3000 0000 0000 0000 4



def bin_to_dec(b):
    b1 = b[:b.index('.')]
    b2 = b[b.index('.')+1:]
    d1 = int(b1,2)
    d2 = int(b2,2)/2**(len(b2))
    return d1+d2

# 计算float的最大范围
# print(bin_to_dec('1.1111111111111111111111111111111111111111111111111111')*pow(2,1023))
# print('%.40f'%bin_to_dec('1.1111111111111111111111111111111111111111111111111111'))

# 编码备忘录：
# 0.0001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1010  # 0.1的双精度表示
# 0.00 1 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1001 1010 # 0.2的双精度表示
# 0.0100 1100 1100 1100 1100 1100 1100 1100 1100 1100 1100 1100 1101 00  #两二进制相加并保留52位有效数字
print("双精度存储，64位情况")
ddd=bin_to_dec('0.00011001100110011001100110011001100110011001100110011010')
print('%.60f'%(ddd))
eee=bin_to_dec('0.0011001100110011001100110011001100110011001100110011010')
print('%.60f'%(eee))
print('%.60f'%(eee+ddd))
fff=bin_to_dec('0.010011001100110011001100110011001100110011001100110100')
print('%.60f'%(fff))
print('%.60f'%(0.1+0.2))
print((0.1+0.2)==fff)

print("Decimal模块使用")
print(Decimal(0.1))
print(Decimal(0.2))
print(Decimal(0.1)+Decimal(0.2))
print(Decimal(0.1)+Decimal(0.2))

print(round(1.135,2),'pass') # 1.14
print(Decimal(1.135))
print(round(1.235,2),'pass') # 1.24 
print(Decimal(1.235))
print(round(1.335,2),'error') # 1.33 error
print(Decimal(1.335))
print(round(1.435,2),'pass') # 1.44
print(Decimal(1.435))
print(round(1.535,2),'error') # 1.53 error
print(Decimal(1.535))
print(round(1.635,2),'pass') # 1.64
print(Decimal(1.635))




#奇进偶舍现象
print(round(4.5)) # 4
print(round(5.5)) # 6


print("\n\nFraction模块的使用")
from fractions import Fraction
print(Fraction(10,3))


# res = Decimal('1')/Decimal('8')
# print(res)      
# res = res.quantize(Decimal('0.00'),rounding=ROUND_HALF_UP)
# print(res) 




# print("++++++++  solution 2  +++++++++")
# from decimal import Decimal,ROUND_HALF_UP
# a = 1.311
# b = 1.212
# print(a)
# print(b)
# c=a+b
# print(c) # 2.522999999999999 error


# aa=Decimal(str(a)) #借助字符串达到绝对精确
# bb=Decimal(str(b))
# print(aa)
# print(bb)
# cc=aa+bb
# print(cc)
# print(' aa+bb= %.20f'%(cc)) # 挑战cc的精度极限

# print("++++++++  solution 3  +++++++++")
# aaa=Decimal(a) 
# bbb=Decimal(b)

# Decimal('1.41421356').quantize(Decimal('1.000'))

