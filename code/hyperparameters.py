"""
Project 6 - Term Project
CS1430 - Computer Vision
Brown University
"""

"""
Number of epochs. If you experiment with more complex networks you
might need to increase this. Likewise if you add regularization that
slows training.
"""
num_epochs = 30

"""
A critical parameter that can dramatically affect whether training
succeeds or fails. The value for this depends significantly on which
optimizer is used. Refer to the default learning rate parameter
"""
learning_rate = 5e-4

"""
Momentum on the gradient (if you use a momentum-based optimizer)
"""
momentum = 0.01

"""
Resize image size for task 1. Task 3 must have an image size of 224,
so that is hard-coded elsewhere.
"""
img_size = 224

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 400

"""
Maximum number of weight files to save to checkpoint directory. If
set to a number <= 0, then all weight files of every epoch will be
saved. Otherwise, only the weights with highest accuracy will be saved.
"""
max_num_weights = 5

"""
Defines the number of training examples per batch.
You don't need to modify this.
"""
batch_size = 10

"""
The number of image scene classes. Don't change this.
"""
num_classes = 30

"""
Square dimension of an image
"""
image_dim = 96

"""
Indeces for different keypoints
"""
left_eye_center = 0
right_eye_center = 1
left_eye_inner_corner = 2
left_eye_outer_corner = 3
right_eye_inner_corner = 4
right_eye_outer_corner = 5
left_eyebrow_inner_end = 6
left_eyebrow_outer_end = 7
right_eyebrow_inner_end = 8
right_eyebrow_outer_end = 9
nose_tip = 10
mouth_left_corner = 11
mouth_right_corner = 12
mouth_center_top_lip = 13
mouth_center_bottom_lip = 14
