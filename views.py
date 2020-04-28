

from django.http import HttpResponseRedirect
from django.shortcuts import render
import cv2
import numpy as np
from .train import *
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from django.templatetags.static import static



# def simple_upload(request):
#     if request.method == 'POST' and request.FILES['myfile']:
#         myfile = request.FILES['myfile']
#         fs = FileSystemStorage()
#         filename = fs.save(myfile.name, myfile)
#         uploaded_file_url = fs.url(filename)
#         return render(request, 'index.html', {
#             'uploaded_file_url': uploaded_file_url
#         })
#     return render(request,'index.html')


def evaluate_ocr(request):
    act_model = model_ocr()

    url = static('my_model.h5')


    act_model.load_weights(url)

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        im = uploaded_file_url
        print(im)
        # path of test image

        img = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 32))  # resizing


        prediction = act_model.predict(img.reshape(1, 32, 128, 1))  # reshaping the img to pass in model

        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                       greedy=False)[0][0])

        # see the results
        preds = []
        for x in out:
            print("original_text = ", im.split('.')[0])
            print("predicted text = ", end='')
            for p in x:
                if int(p) != -1:
                    pred = char_list[int(p)]
                    preds.append(pred)

        result = ''.join(preds)
        return render(request, 'index.html', {
            'predictions': result,
            'uploaded_file_url': uploaded_file_url
        })
            # print('\n')

    return render(request,'index.html')