Python 3.12.1 (v3.12.1:2305ca5144, Dec  7 2023, 17:23:38) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
================ RESTART: /Users/shashankbasty/Documents/knn.py ================
>>> 
=============== RESTART: /Users/shashankbasty/Documents/train.py ===============
Traceback (most recent call last):
  File "/Users/shashankbasty/Documents/train.py", line 6, in <module>
    from KNN import KNN
ModuleNotFoundError: No module named 'KNN'
>>> 
=============== RESTART: /Users/shashankbasty/Documents/train.py ===============
Traceback (most recent call last):
  File "/Users/shashankbasty/Documents/train.py", line 6, in <module>
    from knn import knn
ImportError: cannot import name 'knn' from 'knn' (/Users/shashankbasty/Documents/knn.py)
>>> 
=============== RESTART: /Users/shashankbasty/Documents/train.py ===============

=============== RESTART: /Users/shashankbasty/Documents/train.py ===============
Traceback (most recent call last):
  File "/Users/shashankbasty/Documents/train.py", line 20, in <module>
    clf = KNN(k=5)
NameError: name 'KNN' is not defined
>>> 
=============== RESTART: /Users/shashankbasty/Documents/train.py ===============
[1, 2, 2, 0, 1, 0, 0, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 2, 1, 1, 1, 1, 1, 2, 0, 2, 1, 2, 0]
0.9666666666666667
