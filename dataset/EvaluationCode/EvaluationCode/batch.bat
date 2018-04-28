for /l %%i in (1 1 9) do (python angle.py 00%%i.txt)
for /l %%i in (10 1 99) do (python angle.py 0%%i.txt)
for /l %%i in (100 1 150) do (python angle.py %%i.txt)
