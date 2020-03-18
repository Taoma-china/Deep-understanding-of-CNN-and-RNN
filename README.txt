In order to get the cnn model, run the 'cnn.py' in problem 1 file. 

And for the digits I use in all problem, I create the python file to get the picture and save them to use. You can run the 'getpicture.py' to get the picture you want, but the only thing is you need to change a little bit code.


For problem 1: (1) visualize the filters: please run the "visualize_filter.py' to get the result. I only output the first six in every layer.

		(2) visualize the feature of digit '0' and '8'. For this problem, you need to run 'visualize_featuremap.py'. Ps: if you want to get '8', you need to change the "img = Image.open('8.jpg') or ('0.jpg')".

		(3) Shift the digit '1': you need to run 'visualize_changepixels.py'. you will get the prediction of result before shift and after shift.

For problem 2: (1) you need to run 'cover.py'. you will get the  the probability of ‘6’ of the partially covered image in map 1, the highest probability (among the 10 classes) in map 2, and classified label (‘0’ to ‘9’) in map 3. And also, you can get the picture of covering important part.

		(2) Analyzing in report.

		(3) I cover the important part using digit '8'. You can run the 'coveranother.py' to save the picture. But every time, you need to change the code in "draw.rectangle(j,i,j+8,i+8). (i=12,j=6/8; i=13,j=6,8; i=14,j=8; i=15, j=8).
After that, you need to run the 'coveranother_predict.py' to get the result. You can see all the label predicted correctly. 


Problem 3: you need to run 'compute_y&loss.py' to get all result from problem3. Details in the report.
