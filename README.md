# MidpointAndShapeDetectModels
This uses convolutional Neural Networks to detect the shape and its midpoints. This project can be broken down into a few parts: first, its preprocessing, model, experiments, loss selection, and results 


# Preprocessing and Model 
To preprocess the data, first, I turn images into grayscale, then scale pixel values, which I do by dividing x and y points by their respective max. Lastly, I one-hot encoded the target y.

<img width="1135" height="460" alt="image" src="https://github.com/user-attachments/assets/352d8ba2-1cd7-4535-8157-a94d7f5f23b9" />
As stated earlier, I used convolutional Neural Networks to do this task, and you can find the model architecture in the notebook files.


# Experiments and Results 
The hyperparameters that were experimented on were the types of activation functions, the number of convolution layers, the number of max pooling layers, the number of filters, and the number of filters with each convolutional layer. The exact hyperparameters used at the end can be found in the notebook files. The results from the best model for the respective task were <br />
Shape Detection: 97% Accuracy <br />
Center/Midpoint Detection: 0.0020 Loss<br />
To validate the results, I ran randomized trials for each task, and the results were <br />
Shape Detection: Mean Accuracy: 0.9780 and Standard Deviation: 0.0010077<br />
Center of Shape Detection: Standard Deviation: 9.4280904158206E-5 and Mean Loss:  0.00156 <br />

