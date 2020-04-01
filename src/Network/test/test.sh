#!/usr/bin/env bash
# 确定 mutant是否使用 以及 确定最优参数组合(没有残差块)
## neighbor 30
#python ./test1/wild_VS_wild_mutant_1_neighbor30_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor30_test01.log # wild,参数顺序1
#python ./test1/wild_VS_wild_mutant_1_neighbor30_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor30_test02.log # wild,参数顺序2
#python ./test3/wild_VS_wild_mutant_3_neighbor30_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor30_test01.log # mutant,参数顺序1
#python ./test3/wild_VS_wild_mutant_3_neighbor30_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor30_test02.log # mutant,参数顺序2
#python ./test2/wild_VS_wild_mutant_2_neighbor30_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor30_test01.log # wild and mutant,参数顺序1
#python ./test2/wild_VS_wild_mutant_2_neighbor30_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor30_test02.log # wild and mutant,参数顺序2
### neighbor 40
#python ./test1/wild_VS_wild_mutant_1_neighbor40_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor40_test01.log # wild,参数顺序1
#python ./test1/wild_VS_wild_mutant_1_neighbor40_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor40_test02.log # wild,参数顺序2
#python ./test3/wild_VS_wild_mutant_3_neighbor40_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor40_test01.log # mutant,参数顺序1
#python ./test3/wild_VS_wild_mutant_3_neighbor40_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor40_test02.log # mutant,参数顺序2
#python ./test2/wild_VS_wild_mutant_2_neighbor40_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor40_test01.log # wild and mutant,参数顺序1
#python ./test2/wild_VS_wild_mutant_2_neighbor40_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor40_test02.log # wild and mutant,参数顺序2
### neighbor 50
#python ./test1/wild_VS_wild_mutant_1_neighbor50_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor50_test01.log # wild,参数顺序1
#python ./test1/wild_VS_wild_mutant_1_neighbor50_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor50_test02.log # wild,参数顺序2
#python ./test3/wild_VS_wild_mutant_3_neighbor50_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor50_test01.log # mutant,参数顺序1
#python ./test3/wild_VS_wild_mutant_3_neighbor50_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor50_test02.log # mutant,参数顺序2
#python ./test2/wild_VS_wild_mutant_2_neighbor50_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor50_test01.log # wild and mutant,参数顺序1
#python ./test2/wild_VS_wild_mutant_2_neighbor50_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor50_test02.log # wild and mutant,参数顺序2
### neighbor 60
#python ./test1/wild_VS_wild_mutant_1_neighbor60_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor60_test01.log # wild,参数顺序1
#python ./test1/wild_VS_wild_mutant_1_neighbor60_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor60_test02.log # wild,参数顺序2
#python ./test3/wild_VS_wild_mutant_3_neighbor60_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor60_test01.log # mutant,参数顺序1
#python ./test3/wild_VS_wild_mutant_3_neighbor60_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor60_test02.log # mutant,参数顺序2
#python ./test2/wild_VS_wild_mutant_2_neighbor60_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor60_test01.log # wild and mutant,参数顺序1
#python ./test2/wild_VS_wild_mutant_2_neighbor60_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor60_test02.log # wild and mutant,参数顺序2
## neighbor 70
time python ./test1/wild_VS_wild_mutant_1_neighbor70_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor70_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor70_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor70_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor70_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor70_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor70_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor70_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor70_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor70_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor70_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor70_test02.log # wild and mutant,参数顺序2
## neighbor 80
time python ./test1/wild_VS_wild_mutant_1_neighbor80_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor80_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor80_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor80_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor80_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor80_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor80_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor80_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor80_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor80_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor80_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor80_test02.log # wild and mutant,参数顺序2
## neighbor 90
time python ./test1/wild_VS_wild_mutant_1_neighbor90_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor90_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor90_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor90_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor90_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor90_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor90_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor90_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor90_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor90_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor90_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor90_test02.log # wild and mutant,参数顺序2
## neighbor 100
time python ./test1/wild_VS_wild_mutant_1_neighbor100_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor100_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor100_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor100_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor100_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor100_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor100_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor100_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor100_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor100_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor100_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor100_test02.log # wild and mutant,参数顺序2
## neighbor 110
time python ./test1/wild_VS_wild_mutant_1_neighbor110_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor110_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor110_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor110_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor110_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor110_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor110_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor110_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor110_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor110_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor110_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor110_test02.log # wild and mutant,参数顺序2
## neighbor 120
time python ./test1/wild_VS_wild_mutant_1_neighbor120_test01.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor120_test01.log # wild,参数顺序1
#time python ./test1/wild_VS_wild_mutant_1_neighbor120_test02.py 0 50 > ./test1/wild_VS_wild_mutant_1_neighbor120_test02.log # wild,参数顺序2
time python ./test3/wild_VS_wild_mutant_3_neighbor120_test01.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor120_test01.log # mutant,参数顺序1
#time python ./test3/wild_VS_wild_mutant_3_neighbor120_test02.py 0 50 > ./test3/wild_VS_wild_mutant_3_neighbor120_test02.log # mutant,参数顺序2
time python ./test2/wild_VS_wild_mutant_2_neighbor120_test01.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor120_test01.log # wild and mutant,参数顺序1
#time python ./test2/wild_VS_wild_mutant_2_neighbor120_test02.py 0 50 > ./test2/wild_VS_wild_mutant_2_neighbor120_test02.log # wild and mutant,参数顺序2