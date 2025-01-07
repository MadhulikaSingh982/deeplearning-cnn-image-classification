# Pytorch

- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

## Convolutional Neural Net

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. 
While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. 
Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.


## Architecture

An image is nothing but a matrix of pixel values, right? So why not just flatten the image (e.g. 3x3 image matrix into a 9x1 vector) and feed it to a Multi-Level Perceptron for classification purposes?
In cases of extremely basic binary images, the method might show an average precision score while performing prediction of classes but would have little to no accuracy when it comes to complex images having pixel dependencies throughout.
A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. 
The architecture performs a better fitting to the image dataset due to the reduction in the number of parameters involved and reusability of weights. In other words, the network can be trained to understand the sophistication of the image better.
 The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction. 
This is important when we are to design an architecture which is not only good at learning features but also is scalable to massive datasets.

- Pooling Layer
    :The Pooling layer is responsible for reducing the spatial size of the Convolved Feature. 
This is to decrease the computational power required to process the data through dimensionality reduction. 
Furthermore, it is useful for extracting dominant features which are rotational and positional invariant, thus maintaining the process of effectively training of the model.
There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. 
On the other hand, Average Pooling returns the average of all the values from the portion of the image covered by the Kernel.

- Classification — Fully Connected Layer:
    Adding a Fully-Connected layer is a (usually) cheap way of learning non-linear combinations of the high-level features as represented by the output of the convolutional layer. 
The Fully-Connected layer is learning a possibly non-linear function in that space.
Now that we have converted our input image into a suitable form for our Multi-Level Perceptron, we shall flatten the image into a column vector. 
The flattened output is fed to a feed-forward neural network and backpropagation applied to every iteration of training. 
Over a series of epochs, the model is able to distinguish between dominating and certain low-level features in images and classify them using the Softmax Classification technique.




## Code Description


    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle


    File Name : CNNNet.py
    File Description : Class of CNN structure
    
    File Name : TrainModel.py
    File Description : Code to train and evaluate the pytorch model


    File Name : CreateDataset.py
    File Description : Code to load and transform the dataset. 
    
    Link to dataset: https://drive.google.com/file/d/1zGi0IQPIP3S2lVIKmND6oEA81nnJIuJw/view



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `CNN.ipynb`

