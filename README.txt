##DATA COLLECTION

I collected the data from paper titled "Holistic Recognition of Low Quality License Plates by CNN using Track Annotated Data [IWT4S-AVSS 2017]" which is created by Jakub Špaňhel, Jakub Sochor, Roman Juránek, Adam Herout, Lukáš Maršík, Pavel Zemčík under 2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). 


##DATA PREPROCESSING

In this task, I have used Crnn(Convolutional Recurrent Neural Network) to extract the text from license plate. The dataset is small mainly, 652 images in which I split 521 images for training and 131 images for testing. As i checked the images, almost all images are of different sizes. In order to pass into the model, all images should be of same size. Therefore, I resized the images to width=128 and height=32. After resizing, I added padding with the help of pad sequence in keras. Doing padding means that all the label should be of same size. It prevent the problem of invariable sequence.For example, a:1 , b:2, c:3, d:4 and there are 'abccd' and 'dba' words. After passing the encoding_label function (define in the code) it output will be '[12334]' and '[421]'. Both these words are of different length. Therefore, padding will be add in second word which result in '[42100]',where padding:0 and in this case, we do two times padding so that it can become the same length of first word ('abccd').


##Building Model


I used 6 convolutional layers with 3 Maxpooling layers and 3 Batch normalization layers for feature extraction. The output of convolutional layer goes to lambda function which squeeze the output and make it compatible with Bidirectional LSTM layer. There are 2 bidirectional LSTM layer of 128 and 64 units with return sequence = True so that it can backpropagate and also dropping 20% weigth in these two layers. The output of LSTM layer goes to Dense layer where 63 is the total number of output classes including blank character.




##Defining Loss function

I have used CTC loss funtion for recognizing license plate. CTC overcome the problem of sequence generation.
Secondly, It take care of multiple time steps. for example, word is 'texture' and RNN model predits ['t','e','e','x','t','t','u','r','e'].so, to predict the output, we need to merge the character
adjacent to each other. But this can also predict the word ['text'] which is wrong.so CTC take care of this things.Basically,
it place blank between the predicted output ['t','e','e','x','-','t','t','u','r','e'] and then merge and can predict correctly['texture'].


So CTC take 4 inputs:

1. y_true: your ground truth data. The data you are going to compare with the model's outputs in training. 
2. y_pred: is the model's calculated output
3. input_length:the length (in steps, or chars) of each sample (sentence) in the y_pred tensor.
4. label_length:the length (in steps, or chars ) of each sample (sentence) in the y_true (or labels) tensor.


##Compiling Model

Now I compiled the model with RMSPROP optimizer with learning_rate = 0.001, and save the weights in "my_model.h5" on the 
basis of validation loss.At last, Fitting the model with epochs =100 , and batch size = 16.

### PREDICTION

In CTC decoder, it removes the duplicates character (inserting blank) and also take the highest probability 
of the character in each step.


##Deployment

I used Django framework to deploy the model in heroku.
link : https://herokuocrapp.herokuapp.com/