# 自注意力和位置编码
# 对一个给定序列 X1-Xn, 其中 X是长为d的序列
# 自注意力为 Yi(f(Xi,(X1,X1),....,(Xn,Xn))) 即 X1-Xn，（X1,X1）表示 即为key 又是value，把 Xi当做query，求它和其他 X1-Xn的注意力 就得到Yi