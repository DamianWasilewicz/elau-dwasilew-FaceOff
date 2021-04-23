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
Defines the number of training examples per batch.
You don't need to modify this.
"""
batch_size = 10

"""
The number of facial key points, x and y classes.
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
